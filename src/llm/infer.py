# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np, torch
from .config import CHAT_SYSTEM, BIN_LABELS
from .chat import sanitize_model_text
from .modeling import enable_fast_generate


@torch.inference_mode()
def _build_prompt_inputs_embeds(
    model, tokenizer, prompt_text: str, cond_vec_np: np.ndarray
):
    dev = next(model.parameters()).device
    msgs = [
        {"role": "system", "content": CHAT_SYSTEM},
        {"role": "user", "content": prompt_text},
    ]
    chat_str = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer([chat_str], return_tensors="pt").to(dev)
    txt_emb = model.get_input_embeddings()(enc["input_ids"])
    cond = cond_vec_np.astype("float32")
    cond /= np.linalg.norm(cond) + 1e-6
    cond = torch.from_numpy(cond).unsqueeze(0).to(dev, dtype=txt_emb.dtype)
    cond_emb = model.cond_projector(cond)
    inputs_embeds = torch.cat([cond_emb, txt_emb], dim=1)
    attn = enc["attention_mask"]
    K = cond_emb.size(1)
    ones = torch.ones((attn.size(0), K), dtype=attn.dtype, device=attn.device)
    attn = torch.cat([ones, attn], dim=1)
    return inputs_embeds, attn, enc["input_ids"].size(1)


def _encode_label_variants(tokenizer, label_text: str):
    ids_a = tokenizer.encode(label_text, add_special_tokens=False)
    ids_b = tokenizer.encode(" " + label_text, add_special_tokens=False)
    out, seen = [], set()
    for ids in (ids_a, ids_b):
        t = tuple(ids)
        if ids and t not in seen:
            out.append(ids)
            seen.add(t)
    return out


@torch.inference_mode()
def score_label_logprob(
    model, tokenizer, prompt_text: str, cond_vec_np: np.ndarray, label_text: str
) -> float:
    inputs_embeds, attn, _ = _build_prompt_inputs_embeds(
        model, tokenizer, prompt_text, cond_vec_np
    )
    dev = inputs_embeds.device
    E = model.get_input_embeddings()
    best_lp = None
    for label_ids in _encode_label_variants(tokenizer, label_text):
        lab_ids = torch.tensor([label_ids], device=dev)
        lab_emb = E(lab_ids)
        full_emb = torch.cat([inputs_embeds, lab_emb], dim=1)
        full_attn = torch.cat(
            [
                attn,
                torch.ones(
                    (attn.size(0), lab_emb.size(1)),
                    dtype=attn.dtype,
                    device=attn.device,
                ),
            ],
            dim=1,
        )
        out = model(inputs_embeds=full_emb, attention_mask=full_attn, use_cache=False)
        logits = out.logits
        L = lab_emb.size(1)
        label_logits = logits[:, -L - 1 : -1, :]
        logprobs = torch.log_softmax(label_logits, dim=-1)
        token_lp = logprobs.gather(-1, lab_ids.unsqueeze(-1)).squeeze(-1)
        lp = float(token_lp.sum().item())
        if (best_lp is None) or (lp > best_lp):
            best_lp = lp
    return best_lp if best_lp is not None else -1e9


@torch.inference_mode()
def predict_label_with_probs(
    model, tokenizer, prompt_text: str, cond_vec_np: np.ndarray
) -> Dict[str, Any]:
    lp_b = score_label_logprob(model, tokenizer, prompt_text, cond_vec_np, "Benign")
    lp_p = score_label_logprob(model, tokenizer, prompt_text, cond_vec_np, "Pathogenic")
    logits = torch.tensor([lp_b, lp_p])
    probs = torch.softmax(logits, dim=-1).tolist()
    label = "Benign" if probs[0] >= probs[1] else "Pathogenic"
    return {
        "label": label,
        "probs": {"Benign": probs[0], "Pathogenic": probs[1]},
        "logprobs": {"Benign": float(lp_b), "Pathogenic": float(lp_p)},
    }


def _bad_words_ids(tokenizer, words: List[str]):
    out = []
    for w in words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if ids:
            out.append(ids)
        ids2 = tokenizer.encode(" " + w, add_special_tokens=False)
        if ids2:
            out.append(ids2)
    return out


def _make_label_prefix_fn(tokenizer):
    benign_ids = tokenizer.encode("Benign", add_special_tokens=False)
    path_ids = tokenizer.encode("Pathogenic", add_special_tokens=False)
    cand = [benign_ids, path_ids]

    def fn(batch_id, input_ids):
        max_k = max(len(x) for x in cand)
        suffix = (
            input_ids[-max_k:].tolist()
            if hasattr(input_ids, "tolist")
            else list(input_ids[-max_k:])
        )
        allowed_next = set()
        for seq in cand:
            for k in range(len(seq) + 1):
                prefix = seq[:k]
                if len(suffix) >= len(prefix) and suffix[-len(prefix) :] == prefix:
                    if k < len(seq):
                        allowed_next.add(seq[k])
                    else:
                        allowed_next.add(tokenizer.eos_token_id)
                    break
        if not allowed_next:
            allowed_next.update([seq[0] for seq in cand])
        return list(allowed_next)

    fn._target_len = max(len(x) for x in cand)
    return fn


@torch.inference_mode()
def generate_rationale_then_label(
    model,
    tokenizer,
    user_ctx_text: str,
    cond_vec_np: np.ndarray,
    *,
    rationale_tokens: int = 64,
    label_tokens: int = 6,
    rationale_temp: float = 0.5,
    rationale_top_p: float = 0.9,
):
    dev = next(model.parameters()).device

    def _embeds_for(user_text: str):
        msgs = [
            {"role": "system", "content": CHAT_SYSTEM},
            {
                "role": "user",
                "content": f"{user_text}\n\nGive a one-sentence rationale ONLY. Do not output the final label.",
            },
        ]
        chat = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer([chat], return_tensors="pt").to(dev)
        txt_emb = model.get_input_embeddings()(enc["input_ids"])
        cond = cond_vec_np.astype("float32")
        cond /= np.linalg.norm(cond) + 1e-6
        cond = torch.from_numpy(cond).unsqueeze(0).to(dev, dtype=txt_emb.dtype)
        cond_emb = model.cond_projector(cond)
        inputs_embeds = torch.cat([cond_emb, txt_emb], dim=1)
        attn = enc["attention_mask"]
        K = cond_emb.size(1)
        ones = torch.ones((attn.size(0), K), dtype=attn.dtype, device=attn.device)
        attn = torch.cat([ones, attn], dim=1)
        return inputs_embeds, attn

    # Pass 1: rationale (ban label words)
    inp_emb, attn = _embeds_for(user_ctx_text)
    bad_ids = _bad_words_ids(tokenizer, ["Benign", "Pathogenic"])
    gen_rat = model.generate(
        inputs_embeds=inp_emb,
        attention_mask=attn,
        do_sample=True,
        temperature=rationale_temp,
        top_p=rationale_top_p,
        max_new_tokens=rationale_tokens,
        bad_words_ids=bad_ids,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    rationale_raw = tokenizer.decode(gen_rat[0], skip_special_tokens=True)
    rationale = sanitize_model_text(rationale_raw)

    # Pass 2: label (prefix-constrained)
    msgs_lbl = [
        {"role": "system", "content": CHAT_SYSTEM},
        {
            "role": "user",
            "content": f"{user_ctx_text}\n\nRationale: {rationale}\n\nNow output ONLY one word: Benign or Pathogenic.",
        },
    ]
    chat_lbl = tokenizer.apply_chat_template(
        msgs_lbl, tokenize=False, add_generation_prompt=True
    )
    enc_lbl = tokenizer([chat_lbl], return_tensors="pt").to(dev)
    txt_emb_lbl = model.get_input_embeddings()(enc_lbl["input_ids"])
    cond = cond_vec_np.astype("float32")
    cond /= np.linalg.norm(cond) + 1e-6
    cond = torch.from_numpy(cond).unsqueeze(0).to(dev, dtype=txt_emb_lbl.dtype)
    cond_emb = model.cond_projector(cond)
    inputs_embeds_lbl = torch.cat([cond_emb, txt_emb_lbl], dim=1)
    attn_lbl = enc_lbl["attention_mask"]
    K = cond_emb.size(1)
    ones = torch.ones(
        (attn_lbl.size(0), K), dtype=attn_lbl.dtype, device=attn_lbl.device
    )
    attn_lbl = torch.cat([ones, attn_lbl], dim=1)

    prefix_fn = _make_label_prefix_fn(tokenizer)
    cap = getattr(prefix_fn, "_target_len", 6)
    gen_lab = model.generate(
        inputs_embeds=inputs_embeds_lbl,
        attention_mask=attn_lbl,
        do_sample=False,
        max_new_tokens=min(label_tokens, cap),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        prefix_allowed_tokens_fn=prefix_fn,
    )
    label_raw = tokenizer.decode(gen_lab[0], skip_special_tokens=True)
    label_txt = sanitize_model_text(label_raw)
    if "Pathogenic" in label_txt:
        label = "Pathogenic"
    elif "Benign" in label_txt:
        label = "Benign"
    else:
        label = "Pathogenic" if "Patho" in label_txt else "Benign"

    return {
        "label": label,
        "rationale": rationale,
        "raw_label": label_raw,
        "raw_rationale": rationale_raw,
    }


# ---- Quick evaluation (uses scoring path; no JSON) ----
def quick_eval_pathogenicity_binary(
    model, tokenizer, val_seq_ds, n: int = 512, seed: int = 0
) -> Dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
        classification_report,
    )

    rng = np.random.default_rng(seed)
    idxs = rng.choice(
        len(val_seq_ds), size=min(n, len(val_seq_ds)), replace=False
    ).tolist()
    enable_fast_generate(model, tokenizer)

    def _true_label_from_meta(row) -> int | None:
        from .config import clinsig_to_binary

        y = row.get("_y", None)
        if y is not None:
            try:
                return int(y)
            except Exception:
                pass
        return clinsig_to_binary(row.get("ClinicalSignificance", ""))

    y_true, y_pred, examples = [], [], []
    for i in idxs:
        x, _, _ = val_seq_ds[i]
        cond = x.numpy()
        row = val_seq_ds.meta.iloc[i]
        y = _true_label_from_meta(row)
        if y is None:
            continue
        from .data import build_ctx_from_name

        ctx = build_ctx_from_name(row)
        prompt = f"Variant context (no phenotype):\n{ctx}\n\nReturn label now:"
        out = predict_label_with_probs(model, tokenizer, prompt, cond)
        cls = out["label"]
        yhat = 1 if cls == "Pathogenic" else 0
        y_true.append(int(y))
        y_pred.append(int(yhat))
        if len(examples) < 8:
            examples.append(
                {"idx": i, "truth": int(y), "pred": int(yhat), "probs": out["probs"]}
            )

    if not y_true:
        return {"n": 0, "note": "No labeled samples in val meta."}

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=BIN_LABELS, digits=3
    )
    return {
        "n": len(y_true),
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": cm,
        "report": report,
        "samples": examples,
    }


# ---- Tiny inference smoke (optionally shows rationale) ----
def quick_smoke_generate_labels(seq_ds, out_dir, n=8, seed=0, with_rationale=True):
    from collections import Counter
    import os
    from .load import load_finetuned_model

    rng = np.random.default_rng(seed)

    adapter_dir = os.path.join(out_dir, "adapter")
    projector_path = os.path.join(out_dir, "projector.pt")
    D_cond = seq_ds["train"].D_eff + seq_ds["train"].D_go

    tok, model = load_finetuned_model(
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        D_cond=D_cond,
        adapter_dir=adapter_dir,
        projector_path=projector_path,
    )
    enable_fast_generate(model, tok)

    val = seq_ds["val"]
    idxs = rng.choice(len(val), size=min(n, len(val)), replace=False).tolist()

    preds = []
    for i in idxs:
        x, _, _ = val[i]
        cond = x.numpy()
        row = val.meta.iloc[i]
        from .config import clinsig_to_binary

        y = row.get("_y", None)
        if y is None:
            y = clinsig_to_binary(row.get("ClinicalSignificance", ""))
        from .data import build_ctx_from_name

        ctx = build_ctx_from_name(row)
        prompt = f"Variant context (no phenotype):\n{ctx}\n\nReturn label now:"

        if with_rationale:
            out = generate_rationale_then_label(
                model,
                tok,
                prompt,
                cond,
                rationale_tokens=64,
                label_tokens=6,
                rationale_temp=0.5,
                rationale_top_p=0.9,
            )
            label, rat = out["label"], out["rationale"]
        else:
            # label-only constrained decode
            msgs = [
                {"role": "system", "content": CHAT_SYSTEM},
                {
                    "role": "user",
                    "content": f"{prompt}\n\nOutput ONLY one word: Benign or Pathogenic.",
                },
            ]
            chat = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            enc = tok([chat], return_tensors="pt").to(next(model.parameters()).device)
            txt_emb = model.get_input_embeddings()(enc["input_ids"])
            condf = cond.astype("float32")
            condf /= np.linalg.norm(condf) + 1e-6
            condf = (
                torch.from_numpy(condf)
                .unsqueeze(0)
                .to(txt_emb.device, dtype=txt_emb.dtype)
            )
            cond_emb = model.cond_projector(condf)
            inputs_embeds = torch.cat([cond_emb, txt_emb], dim=1)
            attn = enc["attention_mask"]
            K = cond_emb.size(1)
            ones = torch.ones((attn.size(0), K), dtype=attn.dtype, device=attn.device)
            attn = torch.cat([ones, attn], dim=1)
            prefix_fn = _make_label_prefix_fn(tok)
            gen = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                do_sample=False,
                max_new_tokens=min(6, getattr(prefix_fn, "_target_len", 6)),
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
                prefix_allowed_tokens_fn=prefix_fn,
            )
            raw = tok.decode(gen[0], skip_special_tokens=True)
            out_txt = sanitize_model_text(raw)
            label = "Pathogenic" if "Pathogenic" in out_txt else "Benign"
            rat = ""

        preds.append((i, y, label, rat))

    print("[SMOKE] Pred label counts:", Counter([p[2] for p in preds]))
    for i, y, label, rat in preds:
        print(f"- idx={i} truth={y} pred={label}  rationale[:120]={rat[:120]!r}")
    return {"n": len(preds), "preds": preds}
