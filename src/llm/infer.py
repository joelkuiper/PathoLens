# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence, Match

import numpy as np
import torch

from .config import BIN_LABELS, CHAT_SYSTEM
from .chat import build_chat_strings, sanitize_model_text


def encode_label_variants(tokenizer, label_word: str) -> list[list[int]]:
    """Return token ID variants for a label with/without leading space."""
    ids_a = tokenizer.encode(label_word, add_special_tokens=False)
    ids_b = tokenizer.encode(" " + label_word, add_special_tokens=False)
    out, seen = [], set()
    for ids in (ids_a, ids_b):
        t = tuple(ids)
        if ids and t not in seen:
            out.append(list(ids))
            seen.add(t)
    return out


def _to_numpy_cond(cond_vecs: Any) -> np.ndarray:
    if isinstance(cond_vecs, torch.Tensor):
        arr = cond_vecs.detach().cpu().numpy()
    else:
        arr = np.asarray(cond_vecs)
    arr = arr.astype(np.float32, copy=False)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError("cond_vecs must be 1D or 2D array-like")
    return arr


@torch.inference_mode()
def prepare_prompt_embeddings(
    model,
    tokenizer,
    prompt_texts: Sequence[str],
    cond_vecs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(prompt_texts, str):
        prompts = [prompt_texts]
    else:
        prompts = [p or "" for p in prompt_texts]

    cond = _to_numpy_cond(cond_vecs)
    if cond.shape[0] != len(prompts):
        raise ValueError(
            "Number of prompts and conditioning vectors must match: "
            f"{len(prompts)} vs {cond.shape[0]}"
        )

    prompt_strings, _ = build_chat_strings(tokenizer, prompts)

    dev = next(model.parameters()).device
    enc = tokenizer(
        prompt_strings,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    enc = {k: v.to(dev) for k, v in enc.items()}

    E = model.get_input_embeddings()
    txt_emb = E(enc["input_ids"])  # [B, T, H]

    cond_tensor = torch.from_numpy(cond).to(dev, dtype=txt_emb.dtype)  # [B, D]
    cond_emb, _ = model.cond_projector(cond_tensor)  # [B, K, H]

    inputs_embeds = torch.cat([cond_emb, txt_emb], dim=1)
    attn = enc["attention_mask"]
    B, K = attn.size(0), cond_emb.size(1)
    ones = torch.ones((B, K), dtype=attn.dtype, device=attn.device)
    attn = torch.cat([ones, attn], dim=1)
    return inputs_embeds, attn


@torch.inference_mode()
def batched_label_logprobs(
    model,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    label_ids: Sequence[int],
) -> np.ndarray:
    dev = inputs_embeds.device
    E = model.get_input_embeddings()
    ids = torch.tensor([label_ids], device=dev)  # [1, L]
    emb = E(ids).expand(inputs_embeds.size(0), -1, -1)  # [B, L, H]

    full_emb = torch.cat([inputs_embeds, emb], dim=1)
    L = emb.size(1)
    full_attn = torch.cat(
        [
            attention_mask,
            torch.ones(
                (attention_mask.size(0), L),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            ),
        ],
        dim=1,
    )

    out = model(inputs_embeds=full_emb, attention_mask=full_attn, use_cache=False)
    logits = out.logits  # [B, S+L, V]

    label_logits = logits[:, -L - 1 : -1, :]
    logprobs = torch.log_softmax(label_logits, dim=-1)  # [B, L, V]
    ids_batch = ids.expand(inputs_embeds.size(0), -1)  # [B, L]
    token_lp = logprobs.gather(-1, ids_batch.unsqueeze(-1)).squeeze(-1)  # [B, L]
    scores = token_lp.sum(dim=1).detach().float().cpu().numpy()
    scores = np.where(np.isfinite(scores), scores, -1e9).astype(np.float32)
    return scores


@torch.inference_mode()
def score_label_word_logprobs(
    model,
    tokenizer,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    label_word: str,
    *,
    label_variants: Optional[Sequence[Sequence[int]]] = None,
) -> np.ndarray:
    if label_variants is None:
        label_variants = encode_label_variants(tokenizer, label_word)

    best: Optional[np.ndarray] = None
    for ids in label_variants:
        if not ids:
            continue
        lp = batched_label_logprobs(model, inputs_embeds, attention_mask, ids)
        best = lp if best is None else np.maximum(best, lp)

    if best is None:
        B = inputs_embeds.size(0)
        return np.full((B,), -1e9, dtype=np.float32)
    return best.astype(np.float32, copy=False)


@torch.inference_mode()
def predict_label_with_probs(
    model,
    tokenizer,
    prompt_text: str,
    cond_vec_np,
) -> Dict[str, Any]:
    inputs_embeds, attn = prepare_prompt_embeddings(
        model, tokenizer, [prompt_text], cond_vec_np
    )

    logprobs = {}
    for label in BIN_LABELS:
        lp = score_label_word_logprobs(model, tokenizer, inputs_embeds, attn, label)
        logprobs[label] = float(lp[0])

    lp_values = np.array([logprobs[label] for label in BIN_LABELS], dtype=np.float32)
    denom = np.logaddexp.reduce(lp_values)
    probs = np.exp(lp_values - denom)
    label_idx = int(np.argmax(probs))
    label = BIN_LABELS[label_idx]

    return {
        "label": label,
        "probs": {label: float(prob) for label, prob in zip(BIN_LABELS, probs)},
        "logprobs": {label: float(logprobs[label]) for label in BIN_LABELS},
    }


def _light_badwords_for_labels(tokenizer) -> list[list[int]]:
    out = []
    for word in BIN_LABELS:
        for form in (word, " " + word):
            ids = tokenizer.encode(form, add_special_tokens=False)
            if ids:
                out.append(ids)
    return out


def _rationale_instruction_for(label_word: str) -> str:
    if label_word == "Pathogenic":
        hint = "deleterious (disease-causing)"
    else:
        hint = "tolerated/neutral (not disease-causing)"
    return (
        f"Assume the predicted class is {hint}. "
        "Write one short sentence explaining why this classification is plausible based only on the given features. "
        "Do not use the words 'Benign' or 'Pathogenic'."
    )


_LABEL_WORD_RX = re.compile(r"(?i)\b(benign|pathogenic)\b")
_LABEL_WORD_REPLACEMENTS = {
    "benign": "likely harmless",
    "pathogenic": "likely disease-causing",
}


def _replace_label_words(text: str) -> str:
    def repl(match: Match[str]) -> str:
        word = match.group(0)
        key = match.group(1).lower()
        replacement = _LABEL_WORD_REPLACEMENTS.get(key, "")
        if not replacement:
            return ""
        if word.isupper():
            return replacement.upper()
        if word[0].isupper():
            return replacement.capitalize()
        return replacement

    return _LABEL_WORD_RX.sub(repl, text)


def _combine_seed_with_body(seed_text: str, body: str) -> str:
    seed_clean = (seed_text or "").strip()
    body_clean = (body or "").strip()
    if not seed_clean:
        return body_clean
    if not body_clean:
        return seed_clean
    if body_clean.lower().startswith(seed_clean.lower()):
        return body_clean
    joiner = "" if body_clean[:1] in ",.!?:;" else " "
    return f"{seed_clean}{joiner}{body_clean}"


def _clean_generated_rationale(
    raw_text: str,
    *,
    seed_text: str,
    ban_label_words: bool,
    min_chars: int,
) -> tuple[str, str]:
    combined_raw = _combine_seed_with_body(seed_text, raw_text.strip())
    txt = sanitize_model_text(raw_text).strip()
    if not txt:
        return "", combined_raw

    txt = re.sub(r"\s+", " ", txt).strip()
    txt = re.split(r"(?<=[.!?])\s", txt)[0].strip()
    if not txt:
        return "", combined_raw

    if ban_label_words:
        txt = _replace_label_words(txt)
        txt = re.sub(r"\s+", " ", txt).strip()

    if min_chars > 0 and len(txt) < min_chars:
        return "", combined_raw

    final = _combine_seed_with_body(seed_text, txt)
    return final, combined_raw


@torch.inference_mode()
def generate_rationale_with_seed(
    model,
    tokenizer,
    user_prompt: str,
    cond_vec_np,
    label_word: str,
    *,
    seed_text: str = " Because",
    temperature: float = 0.9,
    top_p: float = 0.92,
    max_new_tokens: int = 48,
    min_new_tokens: int = 12,
    repetition_penalty: float = 1.05,
    ban_label_words: bool = True,
    min_chars: int = 12,
) -> tuple[str, str]:
    dev = next(model.parameters()).device
    msgs = [
        {"role": "system", "content": CHAT_SYSTEM},
        {
            "role": "user",
            "content": (
                f"{user_prompt}\n\n"
                f"{_rationale_instruction_for(label_word)}\n"
                "Rationale:"
            ),
        },
    ]
    chat = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tokenizer([chat], return_tensors="pt").to(dev)

    E = model.get_input_embeddings()
    txt_emb = E(enc["input_ids"])  # [1, T, H]

    cond = _to_numpy_cond(cond_vec_np)
    cond_tensor = torch.from_numpy(cond).to(dev, dtype=txt_emb.dtype)  # [1, D]
    cond_emb, _ = model.cond_projector(cond_tensor)  # [1, K, H]

    seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if seed_ids:
        seed_emb = E(torch.tensor([seed_ids], device=dev))
        inputs_embeds = torch.cat([cond_emb, txt_emb, seed_emb], dim=1)
        seed_len = seed_emb.size(1)
    else:
        inputs_embeds = torch.cat([cond_emb, txt_emb], dim=1)
        seed_len = 0

    attn = enc["attention_mask"]
    K = cond_emb.size(1)
    ones = torch.ones((attn.size(0), K + seed_len), dtype=attn.dtype, device=attn.device)
    attn = torch.cat([ones, attn], dim=1)

    bad_words = _light_badwords_for_labels(tokenizer) if ban_label_words else None

    gen = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        repetition_penalty=repetition_penalty,
        bad_words_ids=bad_words,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    raw = tokenizer.decode(gen[0], skip_special_tokens=True)

    cleaned, combined_raw = _clean_generated_rationale(
        raw,
        seed_text=seed_text,
        ban_label_words=ban_label_words,
        min_chars=min_chars,
    )
    if cleaned:
        return cleaned, combined_raw

    fallback_text, fallback_raw = _clean_generated_rationale(
        raw,
        seed_text=seed_text,
        ban_label_words=ban_label_words,
        min_chars=0,
    )
    if fallback_text:
        return fallback_text, fallback_raw

    return "", combined_raw or _combine_seed_with_body(seed_text, "")


@torch.inference_mode()
def generate_label_then_rationale(
    model,
    tokenizer,
    prompt_text: str,
    cond_vec_np,
    *,
    rationale_tokens: int = 48,
    rationale_temp: float = 0.9,
    rationale_top_p: float = 0.92,
    rationale_min_tokens: int = 12,
    repetition_penalty: float = 1.05,
    ban_label_in_rationale: bool = True,
    rationale_seed_text: str = " Because",
    rationale_min_chars: int = 12,
) -> Dict[str, Any]:
    label_out = predict_label_with_probs(model, tokenizer, prompt_text, cond_vec_np)
    label_word = label_out["label"]

    rationale, raw = generate_rationale_with_seed(
        model,
        tokenizer,
        prompt_text,
        cond_vec_np,
        label_word,
        seed_text=rationale_seed_text,
        temperature=rationale_temp,
        top_p=rationale_top_p,
        max_new_tokens=rationale_tokens,
        min_new_tokens=rationale_min_tokens,
        repetition_penalty=repetition_penalty,
        ban_label_words=ban_label_in_rationale,
        min_chars=rationale_min_chars,
    )

    return {
        "label": label_word,
        "probs": label_out["probs"],
        "logprobs": label_out["logprobs"],
        "rationale": rationale,
        "raw_rationale": raw,
    }


__all__ = [
    "encode_label_variants",
    "prepare_prompt_embeddings",
    "batched_label_logprobs",
    "score_label_word_logprobs",
    "predict_label_with_probs",
    "generate_rationale_with_seed",
    "generate_label_then_rationale",
]
