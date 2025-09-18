# src/llm/eval.py
# -*- coding: utf-8 -*-
# Batched evaluation + AUC + feather + optional label→rationale.
# Notes:
# - Ground truth is NEVER inserted into prompts or scoring. It is read only to compute metrics.
# - Rationales are generated AFTER predicting the label, and are conditioned on the PREDICTED label only.
# - LoRA + conditioning projector remain active for both scoring and rationale generation.

import os
import re
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from src.llm.config import CHAT_SYSTEM, PROMPT_TMPL, BIN_LABELS
from src.llm.load import load_finetuned_model
from src.llm.modeling import enable_fast_generate

from src.clinvar import (
    pick_gene,
    extract_hgvsc,
    extract_hgvsp,
    classify_protein_change,
    parse_hgvsc_features,
)

# --- Debug env toggles ---
_ENV_PROBE = os.environ.get("PATHOLENS_PROBE_PROMPTS", "0") == "1"

# Regexes used to sanity-check that the prompt actually looks blank
_RX_HGVS_P = re.compile(r"(?i)\bp\.(?!\[)[A-Za-z0-9_=*]+")
_RX_HGVS_C = re.compile(r"(?i)\bc\.(?!\[)[A-Za-z0-9_+\-*>=<]+")
_RX_LABELS = re.compile(r"(?i)\b(benign|pathogenic)\b")


# -----------------------
# Context building & hygiene
# -----------------------
def build_ctx_from_row(row) -> str:
    """
    Build the user-visible context string for the prompt.
    IMPORTANT: If no context can be built, returns the EMPTY string (""),
    not a fallback phrase. This is critical for 'cond only' ablations.
    """
    ctx = row.get("context", "")
    if isinstance(ctx, str) and ctx.strip():
        return ctx

    name = row.get("Name", "") or ""
    gene = pick_gene(row)
    hgvsc = extract_hgvsc(name)
    hgvsp = extract_hgvsp(name)
    pcls = classify_protein_change(hgvsp)
    cfeat = parse_hgvsc_features(hgvsc)

    lines: List[str] = []
    if gene:
        lines.append(f"Gene: {gene}")
    if hgvsc:
        lines.append(f"HGVS.c: {hgvsc}")
    if hgvsp:
        lines.append(f"HGVS.p: {hgvsp}")
    cons = pcls or (
        "splice"
        if cfeat["splice"]
        else (cfeat["kind"] if cfeat["kind"] != "other" else "")
    )
    if cons:
        lines.append(f"Consequence: {cons}")
    if cfeat["region"] != "unknown":
        lines.append(f"Region: {cfeat['region']}")
    if cfeat["splice"]:
        lines.append("Splice-site: yes")
    if cfeat["indel_len"] > 0:
        lines.append(f"Indel length: {cfeat['indel_len']}")
    if cfeat["frameshift_guess"] is not None:
        lines.append(
            f"Frameshift (heuristic): {'yes' if cfeat['frameshift_guess'] else 'no'}"
        )
    if cfeat["snv_change"]:
        lines.append(f"SNV change: {cfeat['snv_change']}")

    # CRITICAL: if nothing could be built, return truly empty
    return "\n".join(lines) if lines else ""


def sanitize_model_text(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        parts = s.split("```", 2)
        s = parts[-1].strip() if len(parts) == 3 else s
    s = s.replace("<think>", "").replace("</think>", "").strip()
    return s


# -----------------------
# Tokenizer helpers (label variants + chat)
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


def _probe_prompt_strings(prompts: List[str], tag: str = "probe"):
    """Print stats that prove emptiness (or not) of prompts."""
    n = len(prompts)
    if n == 0:
        print(f"[{tag}] no prompts")
        return
    # Basic stats
    lengths = [len(p) for p in prompts]
    n_blank = sum(1 for p in prompts if p.strip() == "")
    n_hgvsc = sum(1 for p in prompts if _RX_HGVS_C.search(p or ""))
    n_hgvsp = sum(1 for p in prompts if _RX_HGVS_P.search(p or ""))
    n_labels = sum(1 for p in prompts if _RX_LABELS.search(p or ""))

    print(
        f"[{tag}] prompts={n} blank={n_blank} "
        f"len[min/med/max]={min(lengths)}/{int(np.median(lengths))}/{max(lengths)} "
        f"HGVS.c hits={n_hgvsc} HGVS.p hits={n_hgvsp} label-word hits={n_labels}"
    )
    # Show up to 3 examples (hash + visible)
    for i in range(min(3, n)):
        s = prompts[i] or ""
        sha = _sha10(s)
        head = s.replace("\n", "\\n")[:180]
        print(f"[{tag}] i={i} sha={sha} text='{head}{'…' if len(s) > 180 else ''}'")


def _sha10(s: str) -> str:
    import hashlib

    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:10]


@torch.inference_mode()
def _prep_inputs_embeds(
    model, tokenizer, prompts: List[str], cond_vecs: np.ndarray, *, debug_probe=False
):
    # Optional: probe raw prompts before tokenization
    if debug_probe or _ENV_PROBE:
        _probe_prompt_strings(prompts, tag="prompt-probe(raw)")

    enc = _apply_chat_template(tokenizer, prompts)
    dev = next(model.parameters()).device
    enc = {k: v.to(dev) for k, v in enc.items()}

    # Optional: probe token lengths
    if debug_probe or _ENV_PROBE:
        attn = enc["attention_mask"].cpu().numpy()
        toks = enc["input_ids"].cpu().numpy()
        tok_lens = attn.sum(axis=1).tolist()
        print(
            f"[prompt-probe(tok)] batches={len(tok_lens)} "
            f"tok_len[min/med/max]={min(tok_lens)}/{int(np.median(tok_lens))}/{max(tok_lens)}"
        )

    E = model.get_input_embeddings()
    txt_emb = E(enc["input_ids"])  # [B,T,H]

    cond = cond_vecs.astype("float32")
    cond = cond / (np.linalg.norm(cond, axis=1, keepdims=True) + 1e-6)
    cond = torch.from_numpy(cond).to(dev, dtype=txt_emb.dtype)  # [B,D]
    cond_emb, _ = model.cond_projector(cond)  # [B,K,H]

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
    to prompt+cond embeddings; score next-token logprobs). Matches eval_old.
    """
    dev = inputs_embeds.device
    E = model.get_input_embeddings()
    ids = torch.tensor([label_ids], device=dev)  # [1,L]
    emb = E(ids).expand(inputs_embeds.size(0), -1, -1)  # [B,L,H]

    full_emb = torch.cat([inputs_embeds, emb], dim=1)
    L = emb.size(1)
    full_attn = torch.cat(
        [attn, torch.ones((attn.size(0), L), dtype=attn.dtype, device=attn.device)],
        dim=1,
    )

    out = model(inputs_embeds=full_emb, attention_mask=full_attn, use_cache=False)
    logits = out.logits  # [B, S+L, V]

    label_logits = logits[:, -L - 1 : -1, :]  # next-token scores over label span
    logprobs = torch.log_softmax(label_logits, dim=-1)  # [B,L,V]
    ids_batch = ids.expand(inputs_embeds.size(0), -1)  # [B,L]
    token_lp = logprobs.gather(-1, ids_batch.unsqueeze(-1)).squeeze(-1)  # [B,L]
    s = token_lp.sum(dim=1).detach().float().cpu().numpy()  # [B]
    # guard against any weird values
    s = np.where(np.isfinite(s), s, -1e9).astype(np.float32)
    return s


# -----------------------
# Rationale helpers (LoRA kept ON, assistant-seeded with " Because")
# -----------------------
def _light_badwords_for_labels(tok) -> list[list[int]]:
    out = []
    for w in ("Benign", "Pathogenic"):
        for form in (w, " " + w):
            ids = tok.encode(form, add_special_tokens=False)
            if ids:
                out.append(ids)
    return out


def _rationale_instruction_for(label_word: str) -> str:
    # Use synonyms so we don't introduce the literal label tokens.
    if label_word == "Pathogenic":
        hint = "deleterious (disease-causing)"
    else:
        hint = "tolerated/neutral (not disease-causing)"
    return (
        f"Assume the predicted class is {hint}. "
        "Write one short sentence explaining why this classification is plausible based only on the given features. "
        "Do not use the words 'Benign' or 'Pathogenic'."
    )


@torch.inference_mode()
def _generate_rationale_with_seed(
    model,
    tok,
    user_prompt: str,
    cond_vec: np.ndarray,
    label_word: str,
    *,
    seed_text: str = " Because",
    temperature: float = 0.9,
    top_p: float = 0.92,
    max_new_tokens: int = 48,
    min_new_tokens: int = 12,
    repetition_penalty: float = 1.05,
) -> str:
    """Generate a one-sentence rationale while LoRA + conditioning are active."""
    dev = next(model.parameters()).device
    msgs = [
        {"role": "system", "content": CHAT_SYSTEM},
        {
            "role": "user",
            "content": f"{user_prompt}\n\n{_rationale_instruction_for(label_word)}\nRationale:",
        },
    ]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok([chat], return_tensors="pt").to(dev)

    E = model.get_input_embeddings()
    txt_emb = E(enc["input_ids"])  # [1,T,H]

    # Cond embeddings
    v = cond_vec.astype("float32")
    v /= np.linalg.norm(v) + 1e-6
    v = torch.from_numpy(v).unsqueeze(0).to(dev, dtype=txt_emb.dtype)  # [1,D]
    cond_emb, _ = model.cond_projector(v)  # [1,K,H]

    # Assistant prefix seed (e.g., " Because")
    seed_ids = tok.encode(seed_text, add_special_tokens=False)
    seed_emb = E(torch.tensor([seed_ids], device=dev)) if seed_ids else None

    # Concatenate: [cond | prompt | seed] → then generate
    if seed_emb is not None:
        inputs_embeds = torch.cat([cond_emb, txt_emb, seed_emb], dim=1)
        seed_len = seed_emb.size(1)
    else:
        inputs_embeds = torch.cat([cond_emb, txt_emb], dim=1)
        seed_len = 0

    attn = enc["attention_mask"]
    K = cond_emb.size(1)
    ones = torch.ones(
        (attn.size(0), K + seed_len), dtype=attn.dtype, device=attn.device
    )
    attn = torch.cat([ones, attn], dim=1)

    bad_words_ids = _light_badwords_for_labels(tok)

    gen = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        repetition_penalty=repetition_penalty,
        bad_words_ids=bad_words_ids,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    raw = tok.decode(gen[0], skip_special_tokens=True)
    txt = sanitize_model_text(raw)

    # Guards: no label words; require a bit of substance. No fallback text.
    if re.search(r"(?i)\b(benign|pathogenic)\b", txt):
        return ""
    if len(txt.strip()) < 10:
        return ""
    # Trim to a single sentence.
    txt = re.split(r"(?<=[.!?])\s", txt.strip())[0]
    return txt


# -----------------------
# Main evaluator (compat with eval_old)
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
    rationale_temp: float = 0.9,
    rationale_top_p: float = 0.92,
    guard_prompts: bool = True,
    debug_prompt_probe: bool = False,
) -> Dict[str, Any]:
    """
    IMPORTANT: We do NOT insert ground-truth anywhere into the prompt or scoring.
    """
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        roc_auc_score,
        average_precision_score,
    )

    adapter_dir = os.path.join(out_dir, "adapter")
    projector_path = os.path.join(out_dir, "projector.pt")

    D_eff = int(seq_ds["train"].D_eff)
    D_go = int(seq_ds["train"].D_go)
    D_prot = int(getattr(seq_ds["train"], "D_prot", 0) or 0)
    D_cond = D_eff + D_go + D_prot
    print(f"[INFO] Eval: D_cond={D_cond} (eff={D_eff} go={D_go} prot={D_prot})")

    tok, model = load_finetuned_model(model_id, D_cond, adapter_dir, projector_path)
    enable_fast_generate(model, tok)

    ds = seq_ds[split]
    N_all = len(ds)

    # Evaluate all rows
    idxs = list(range(N_all))
    if max_n is not None:
        idxs = idxs[:max_n]
    N = len(idxs)
    if N == 0:
        return {"n": 0, "note": f"No samples in split '{split}'."}

    # (Optional) sanity guard: no explicit label words in prompts; also probe HGVS content
    if guard_prompts or debug_prompt_probe or _ENV_PROBE:
        leaks = 0
        hg_hits = 0
        probe_prompts: List[str] = []
        cap = min(2000, len(idxs))
        for i in idxs[:cap]:
            ctx = build_ctx_from_row(ds.meta.iloc[i])
            p = PROMPT_TMPL.format(ctx=ctx)
            if ("Benign" in p) or ("Pathogenic" in p):
                leaks += 1
            if _RX_HGVS_C.search(p) or _RX_HGVS_P.search(p):
                hg_hits += 1
            probe_prompts.append(p)
        assert leaks == 0, f"Prompt leakage: {leaks} prompts include label words."
        if debug_prompt_probe or _ENV_PROBE:
            _probe_prompt_strings(probe_prompts[:64], tag="prompt-probe(prebatch)")

    # Prepare tokenizations for labels once
    cand = {
        "Benign": _encode_label_variants(tok, "Benign"),
        "Pathogenic": _encode_label_variants(tok, "Pathogenic"),
    }

    y_true: List[int] = []
    y_pred: List[int] = []
    prob_pathogenic: List[float] = []
    examples: List[Dict[str, Any]] = []

    # Batched scoring — matches eval_old path
    first_batch_done = False
    for bi in tqdm(range(0, N, batch_size), desc=f"Scoring [{split}]", unit="batch"):
        batch_idxs = idxs[bi : bi + batch_size]

        # Build prompts/conds WITHOUT ground-truth
        prompts, conds, y_batch = [], [], []
        for i in batch_idxs:
            x, *_ = ds[i]
            conds.append(x.numpy())

            # normal context build
            ctx = build_ctx_from_row(ds.meta.iloc[i])

            # if flagged by ablation: force ctx blank
            if getattr(ds, "_force_blank_prompts", False):
                ctx = ""

            prompts.append(PROMPT_TMPL.format(ctx=ctx))
            y_batch.append(int(ds.meta.iloc[i]["_y"]))

        # One-off: deeper probe for the first batch actually used
        if (debug_prompt_probe or _ENV_PROBE) and not first_batch_done:
            _probe_prompt_strings(prompts, tag="prompt-probe(batch0)")
            first_batch_done = True

        conds = np.stack(conds, axis=0)
        inp_emb, attn = _prep_inputs_embeds(
            model, tok, prompts, conds, debug_probe=(debug_prompt_probe or _ENV_PROBE)
        )

        # Teacher-forced label logprobs
        lp_b_list, lp_p_list = [], []
        for ids in cand["Benign"]:
            lp_b_list.append(_batched_label_logprobs(model, tok, inp_emb, attn, ids))
        for ids in cand["Pathogenic"]:
            lp_p_list.append(_batched_label_logprobs(model, tok, inp_emb, attn, ids))

        # Max over whitespace/no-whitespace variants
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

        # Two-class normalization → calibrated probability for Pathogenic
        denom = np.logaddexp(lp_b, lp_p)
        prob_b = np.exp(lp_b - denom)
        prob_p = np.exp(lp_p - denom)

        yhat_batch = (lp_p > lp_b).astype(np.int32)

        y_true.extend(y_batch)
        y_pred.extend(yhat_batch.tolist())
        prob_pathogenic.extend(np.clip(prob_p, 0.0, 1.0).tolist())

        # log a few examples
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
    acc = float(np.mean(np.array(y_pred) == np.array(y_true))) if y_true else 0.0
    p, r, f1 = torchmetrics_safe_prf(y_true, y_pred)


    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=BIN_LABELS, digits=3
    )
    try:
        auc_roc = float(roc_auc_score(y_true, prob_pathogenic))
    except Exception:
        auc_roc = float("nan")
    try:
        auc_pr = float(average_precision_score(y_true, prob_pathogenic))
    except Exception:
        auc_pr = float("nan")

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

    # Optional: rationales AFTER prediction (LoRA ON, cond used)
    if sample_rationales > 0:
        rng = np.random.default_rng(0)
        pool = rng.choice(idxs, size=min(sample_rationales, len(idxs)), replace=False)
        rats = []
        for i in pool:
            x, *_ = ds[i]
            cond_vec = x.numpy()
            row = ds.meta.iloc[i]
            ctx = build_ctx_from_row(row)
            prompt = PROMPT_TMPL.format(ctx=ctx)

            inp_emb, attn = _prep_inputs_embeds(
                model, tok, [prompt], cond_vec[np.newaxis, :], debug_probe=False
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

            rat = _generate_rationale_with_seed(
                model,
                tok,
                prompt,
                cond_vec,
                label_word,
                seed_text=" Because",
                temperature=rationale_temp,
                top_p=rationale_top_p,
                max_new_tokens=rationale_tokens,
                min_new_tokens=12,
                repetition_penalty=1.05,
            )
            rats.append({"idx": int(i), "pred_label": label_word, "rationale": rat})
        out["rationales"] = rats

    return out


def torchmetrics_safe_prf(y_true, y_pred):
    """Helper: precision/recall/F1 with zero_division=0 and floats."""
    from sklearn.metrics import precision_recall_fscore_support

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return float(p), float(r), float(f1)


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
