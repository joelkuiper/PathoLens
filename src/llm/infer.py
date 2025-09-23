# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import torch
from .config import CHAT_SYSTEM
from .chat import sanitize_model_text


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
    cond = torch.from_numpy(cond).unsqueeze(0).to(dev, dtype=txt_emb.dtype)
    cond_emb, _ = model.cond_projector(cond)
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


from transformers import GenerationConfig


@torch.inference_mode()
def generate_label_then_rationale(
    model,
    tokenizer,
    user_ctx_text: str,
    cond_vec_np: np.ndarray,
    *,
    rationale_tokens: int = 64,
    rationale_temp: float = 0.7,
    rationale_top_p: float = 0.9,
    ban_label_in_rationale: bool = True,
):
    """
    1) Predict label (teacher-forced scoring).
    2) Ask for a one-sentence explanation for that label.

    Returns: {
      "label": "Benign" | "Pathogenic",
      "probs": {"Benign": p0, "Pathogenic": p1},
      "logprobs": {...},
      "rationale": <str>,               # may be empty
      "raw_rationale": <str>,           # before sanitization
    }
    """
    # ---- Pass 1: score label probabilities (deterministic) ----
    label_out = predict_label_with_probs(model, tokenizer, user_ctx_text, cond_vec_np)
    label = label_out["label"]

    # ---- Pass 2: generate rationale conditioned on predicted label ----
    dev = next(model.parameters()).device

    msgs = [
        {"role": "system", "content": CHAT_SYSTEM},
        {
            "role": "user",
            "content": (
                f"{user_ctx_text}\n\n"
                f"Predicted label: {label}.\n"
                "Explain in one short sentence why this label is plausible, "
                "using cues from the context (gene, HGVS, consequence). "
                "Do not restate the label word; focus on mechanism."
            ),
        },
    ]
    chat = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer([chat], return_tensors="pt").to(dev)

    txt_emb = model.get_input_embeddings()(enc["input_ids"])  # [1, T, H]

    cond = cond_vec_np.astype("float32")
    cond = torch.from_numpy(cond).unsqueeze(0).to(dev, dtype=txt_emb.dtype)  # [1, D]
    cond_emb, _ = model.cond_projector(cond)  # [1, K, H]

    inputs_embeds = torch.cat([cond_emb, txt_emb], dim=1)
    attn = enc["attention_mask"]
    K = cond_emb.size(1)
    ones = torch.ones((attn.size(0), K), dtype=attn.dtype, device=attn.device)
    attn = torch.cat([ones, attn], dim=1)

    # Ensure sampling flags actually apply
    gen_cfg = GenerationConfig(
        do_sample=True,
        temperature=rationale_temp,
        top_p=rationale_top_p,
        max_new_tokens=rationale_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    bad_ids = (
        _bad_words_ids(tokenizer, ["Benign", "Pathogenic"])
        if ban_label_in_rationale
        else None
    )

    gen_ids = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        generation_config=gen_cfg,
        bad_words_ids=bad_ids,
    )
    rationale_raw = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    rationale = sanitize_model_text(rationale_raw)

    # Optional: if the model only produced whitespace or nothing useful, keep it empty
    if rationale.strip().lower() in {"", "n/a", "no rationale"}:
        rationale = ""

    return {
        "label": label,
        "probs": label_out["probs"],
        "logprobs": label_out["logprobs"],
        "rationale": rationale,
        "raw_rationale": rationale_raw,
    }
