# -*- coding: utf-8 -*-
# Batched evaluation (compat with eval_old prompt path) + AUC + feather + optional label→rationale.

import os, numpy as np, torch
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from src.llm.config import CHAT_SYSTEM, PROMPT_TMPL
from src.llm.load import load_finetuned_model
from src.llm.modeling import enable_fast_generate

from src.clinvar import build_context, clinsig_to_binary


def sanitize_model_text(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        parts = s.split("```", 2)
        s = parts[-1].strip() if len(parts) == 3 else s
    s = s.replace("<think>", "").replace("</think>", "").strip()
    return s


# -----------------------
# Tokenizer helpers
# -----------------------
def _encode_label_variants(tokenizer, word: str):
    a = tokenizer.encode(word, add_special_tokens=False)
    b = tokenizer.encode(" " + word, add_special_tokens=False)
    out, seen = [], set()
    for ids in (a, b):
        t = tuple(ids)
        if ids and t not in seen:
            out.append(ids)
            seen.add(t)
    return out


@torch.inference_mode()
def _apply_chat_template(tokenizer, prompts: List[str]) -> Dict[str, torch.Tensor]:
    msgs_list = []
    for p in prompts:
        msgs = [
            {"role": "system", "content": CHAT_SYSTEM},
            {"role": "user", "content": p},
        ]
        chat = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        msgs_list.append(chat)
    return tokenizer(
        msgs_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )


@torch.inference_mode()
def _prep_inputs_embeds(model, tokenizer, prompts: List[str], cond_vecs: np.ndarray):
    enc = _apply_chat_template(tokenizer, prompts)
    dev = next(model.parameters()).device
    enc = {k: v.to(dev) for k, v in enc.items()}

    E = model.get_input_embeddings()
    txt_emb = E(enc["input_ids"])  # [B,T,H]

    cond = cond_vecs.astype("float32")
    cond = cond / (np.linalg.norm(cond, axis=1, keepdims=True) + 1e-6)
    cond = torch.from_numpy(cond).to(dev, dtype=txt_emb.dtype)  # [B,D]
    cond_emb = model.cond_projector(cond)  # [B,K,H]

    inputs_embeds = torch.cat([cond_emb, txt_emb], dim=1)
    attn = enc["attention_mask"]
    B, K = attn.size(0), cond_emb.size(1)
    ones = torch.ones((B, K), dtype=attn.dtype, device=attn.device)
    attn = torch.cat([ones, attn], dim=1)
    return inputs_embeds, attn


@torch.inference_mode()
def _batched_label_logprobs(
    model, tokenizer, inputs_embeds, attn, label_ids: List[int]
) -> np.ndarray:
    """
    Teacher-forced scoring on the PROMPT side (append candidate label tokens
    to prompt+cond embeddings; score next-token logprobs). This is the robust path
    that matched your eval_old implementation.
    """
    dev = inputs_embeds.device
    E = model.get_input_embeddings()
    ids = torch.tensor([label_ids], device=dev)  # [1,L]
    emb = E(ids)  # [1,L,H]
    B = inputs_embeds.size(0)
    emb = emb.expand(B, -1, -1)

    full_emb = torch.cat([inputs_embeds, emb], dim=1)
    L = emb.size(1)
    full_attn = torch.cat(
        [attn, torch.ones((B, L), dtype=attn.dtype, device=attn.device)], dim=1
    )

    out = model(inputs_embeds=full_emb, attention_mask=full_attn, use_cache=False)
    logits = out.logits  # [B, S+L, V]

    label_logits = logits[:, -L - 1 : -1, :]
    logprobs = torch.log_softmax(label_logits, dim=-1)  # [B,L,V]
    ids_batch = ids.expand(B, -1)  # [B,L]
    token_lp = logprobs.gather(-1, ids_batch.unsqueeze(-1)).squeeze(-1)  # [B,L]
    lp = token_lp.sum(dim=1)  # [B]
    return lp.detach().float().cpu().numpy()


# -----------------------
# Main evaluator (compat)
# -----------------------
@torch.inference_mode()
def evaluate_split_batched(
    seq_ds: Dict[str, Any],
    out_dir: str,
    split: str = "val",
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507",
    max_n: Optional[int] = None,
    batch_size: int = 64,
    sample_rationales: int = 0,  # generate AFTER predicting label
    rationale_tokens: int = 48,
    rationale_temp: float = 0.7,
    rationale_top_p: float = 0.9,
    guard_prompts: bool = True,
) -> Dict[str, Any]:
    """
    IMPORTANT: We do NOT insert ground-truth anywhere into the prompt or scoring.
    Ground-truth labels are read only AFTER predictions to compute metrics.
    Rationales (if requested) are generated AFTER label prediction and are
    conditioned only on the PREDICTED label.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
        classification_report,
        roc_auc_score,
        average_precision_score,
    )

    adapter_dir = os.path.join(out_dir, "adapter")
    projector_path = os.path.join(out_dir, "projector.pt")
    D_cond = seq_ds["train"].D_eff + seq_ds["train"].D_go

    tok, model = load_finetuned_model(model_id, D_cond, adapter_dir, projector_path)
    enable_fast_generate(model, tok)

    ds = seq_ds[split]
    N_all = len(ds)

    # Collect labeled idxs WITHOUT touching ground truth later than needed
    idxs = []
    for i in range(N_all):
        row = ds.meta.iloc[i]
        y = row.get("_y", None)
        if y is None:
            y = clinsig_to_binary(row.get("ClinicalSignificance", ""))
        if y is not None:
            idxs.append(i)

    if max_n is not None:
        idxs = idxs[:max_n]
    N = len(idxs)
    if N == 0:
        return {"n": 0, "note": f"No labeled samples in split '{split}'."}

    # Optional: sanity guard — no label words inside prompts
    if guard_prompts:
        leaks = 0
        for i in idxs[: min(2000, len(idxs))]:
            p = PROMPT_TMPL.format(ctx=build_context(ds.meta.iloc[i]))
            if ("Benign" in p) or ("Pathogenic" in p):
                leaks += 1
        assert leaks == 0, f"Prompt leakage: {leaks} prompts include label words."

    # Prepare tokenizations for labels once
    cand = {
        "Benign": _encode_label_variants(tok, "Benign"),
        "Pathogenic": _encode_label_variants(tok, "Pathogenic"),
    }

    y_true: List[int] = []
    y_pred: List[int] = []
    prob_pathogenic: List[float] = []
    examples: List[Dict[str, Any]] = []

    for bi in tqdm(range(0, N, batch_size), desc=f"Scoring [{split}]", unit="batch"):
        batch_idxs = idxs[bi : bi + batch_size]

        # Build prompts/conds WITHOUT ground-truth
        prompts, conds = [], []
        for i in batch_idxs:
            x, _, _ = ds[i]
            conds.append(x.numpy())
            ctx = build_ctx_from_row(ds.meta.iloc[i])
            prompts.append(PROMPT_TMPL.format(ctx=ctx))
        conds = np.stack(conds, axis=0)

        inp_emb, attn = _prep_inputs_embeds(model, tok, prompts, conds)

        # Teacher-forced label logprobs
        lp_b_list, lp_p_list = [], []
        for ids in cand["Benign"]:
            lp_b_list.append(_batched_label_logprobs(model, tok, inp_emb, attn, ids))
        for ids in cand["Pathogenic"]:
            lp_p_list.append(_batched_label_logprobs(model, tok, inp_emb, attn, ids))

        lp_b = (
            np.max(np.vstack(lp_b_list), axis=0)
            if lp_b_list
            else np.full((len(batch_idxs),), -1e9, dtype=np.float32)
        )
        lp_p = (
            np.max(np.vstack(lp_p_list), axis=0)
            if lp_p_list
            else np.full((len(batch_idxs),), -1e9, dtype=np.float32)
        )

        denom = np.logaddexp(lp_b, lp_p)
        prob_b = np.exp(lp_b - denom)
        prob_p = np.exp(lp_p - denom)
        yhat_batch = (lp_p > lp_b).astype(np.int32)

        # Now fetch ground-truth ONLY to score
        y_batch = []
        for i in batch_idxs:
            row = ds.meta.iloc[i]
            y = row.get("_y", None)
            if y is None:
                y = clinsig_to_binary(row.get("ClinicalSignificance", ""))
            y_batch.append(int(y))

        y_true.extend(y_batch)
        y_pred.extend(yhat_batch.tolist())
        prob_pathogenic.extend(prob_p.tolist())

        if len(examples) < 10:
            for j in range(min(len(batch_idxs), 10 - len(examples))):
                examples.append(
                    {
                        "idx": int(batch_idxs[j]),
                        "truth": int(y_batch[j]),
                        "pred": int(yhat_batch[j]),
                        "probs": {
                            "Benign": float(prob_b[j]),
                            "Pathogenic": float(prob_p[j]),
                        },
                    }
                )

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=BIN_LABELS, digits=3
    )
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score

        auc_roc = float(roc_auc_score(y_true, prob_pathogenic))
        auc_pr = float(average_precision_score(y_true, prob_pathogenic))
    except Exception:
        auc_roc, auc_pr = float("nan"), float("nan")

    # Save predictions alongside the original meta
    try:
        import pyarrow.feather as feather

        df = ds.meta.iloc[idxs].copy().reset_index(drop=True)
        df["_pred_pathogenic_prob"] = prob_pathogenic
        df["_pred_label"] = [int(x) for x in y_pred]
        out_path = os.path.join(out_dir, f"{split}_scored.feather")
        feather.write_feather(df, out_path)
        preds_path = out_path
    except Exception as e:
        preds_path = f"(save failed: {e})"

    out = {
        "split": split,
        "n": len(y_true),
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "confusion_matrix": cm,
        "report": report,
        "examples": examples,
        "predictions_path": preds_path,
    }

    # Optional: rationales AFTER prediction, conditioned on the *predicted* label
    if sample_rationales > 0:
        rng = np.random.default_rng(0)
        pool = rng.choice(idxs, size=min(sample_rationales, len(idxs)), replace=False)
        rats = []

        def _gen_rationale(user_ctx: str, cond_vec: np.ndarray, label_word: str) -> str:
            dev = next(model.parameters()).device
            msgs = [
                {"role": "system", "content": CHAT_SYSTEM},
                {
                    "role": "user",
                    "content": f"{user_ctx}\n\nPredicted label: {label_word}.\n"
                    f"In one sentence, explain why this classification is plausible based only on the given context. "
                    f"Do not restate the label.",
                },
            ]
            chat = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            enc = tok([chat], return_tensors="pt").to(dev)
            E = model.get_input_embeddings()
            txt_emb = E(enc["input_ids"])
            v = cond_vec.astype("float32")
            v = v / (np.linalg.norm(v) + 1e-6)
            v = torch.from_numpy(v).unsqueeze(0).to(dev, dtype=txt_emb.dtype)
            cond_emb = model.cond_projector(v)
            inputs_embeds = torch.cat([cond_emb, txt_emb], dim=1)
            attn = enc["attention_mask"]
            K = cond_emb.size(1)
            ones = torch.ones((attn.size(0), K), dtype=attn.dtype, device=attn.device)
            attn = torch.cat([ones, attn], dim=1)

            # Ban label words so it doesn't just echo them
            bad_words_ids = []
            for w in ("Benign", "Pathogenic"):
                a = tok.encode(w, add_special_tokens=False)
                b = tok.encode(" " + w, add_special_tokens=False)
                if a:
                    bad_words_ids.append(a)
                if b:
                    bad_words_ids.append(b)

            gen = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                do_sample=True,
                temperature=rationale_temp,
                top_p=rationale_top_p,
                max_new_tokens=rationale_tokens,
                bad_words_ids=bad_words_ids,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
            raw = tok.decode(gen[0], skip_special_tokens=True)
            return sanitize_model_text(raw)

        # reuse tokenizer + model from outer scope
        tok = tok  # explicit, just to be clear this is the same tokenizer
        for i in pool:
            x, _, _ = ds[i]
            cond_vec = x.numpy()
            row = ds.meta.iloc[i]
            ctx = build_context(row)
            prompt = PROMPT_TMPL.format(ctx=ctx)

            # decide predicted label quickly
            inp_emb, attn = _prep_inputs_embeds(
                model, tok, [prompt], cond_vec[np.newaxis, :]
            )
            best_b = max(
                (
                    _batched_label_logprobs(model, tok, inp_emb, attn, ids)[0]
                    for ids in _encode_label_variants(tok, "Benign")
                ),
                default=-1e9,
            )
            best_p = max(
                (
                    _batched_label_logprobs(model, tok, inp_emb, attn, ids)[0]
                    for ids in _encode_label_variants(tok, "Pathogenic")
                ),
                default=-1e9,
            )
            label_word = "Pathogenic" if best_p > best_b else "Benign"
            rat = _gen_rationale(prompt, cond_vec, label_word)
            rats.append({"idx": int(i), "pred_label": label_word, "rationale": rat})

        out["rationales"] = rats

    return out


# -----------------------
# Convenience driver
# -----------------------
def run_all(
    seq_ds,
    out_dir: str = "artifacts/qwen3_hpo_lora",
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507",
    max_n_val: Optional[int] = None,
    max_n_test: Optional[int] = None,
    batch_size: int = 64,
    sample_rationales: int = 4,
):
    res_val = evaluate_split_batched(
        seq_ds,
        out_dir,
        split="val",
        model_id=model_id,
        max_n=max_n_val,
        batch_size=batch_size,
        sample_rationales=sample_rationales,
    )
    print("\nVAL metrics:")
    print(
        {
            "n": res_val["n"],
            "accuracy": res_val["accuracy"],
            "precision": res_val["precision"],
            "recall": res_val["recall"],
            "f1": res_val["f1"],
            "auc_roc": res_val.get("auc_roc"),
            "auc_pr": res_val.get("auc_pr"),
        }
    )
    print("Confusion:", res_val["confusion_matrix"])
    if res_val.get("predictions_path"):
        print("Saved VAL predictions to:", res_val["predictions_path"])

    res_test = evaluate_split_batched(
        seq_ds,
        out_dir,
        split="test",
        model_id=model_id,
        max_n=max_n_test,
        batch_size=batch_size,
        sample_rationales=sample_rationales,
    )
    print("\nTEST metrics:")
    print(
        {
            "n": res_test["n"],
            "accuracy": res_test["accuracy"],
            "precision": res_test["precision"],
            "recall": res_test["recall"],
            "f1": res_test["f1"],
            "auc_roc": res_test.get("auc_roc"),
            "auc_pr": res_test.get("auc_pr"),
        }
    )
    print("Confusion:", res_test["confusion_matrix"])
    if res_test.get("predictions_path"):
        print("Saved TEST predictions to:", res_test["predictions_path"])

    return {"val": res_val, "test": res_test}
