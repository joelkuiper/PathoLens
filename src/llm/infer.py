# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from .config import (
    BIN_LABELS,
    COND_START_TOKEN,
    COND_END_TOKEN,
)
from .chat import build_chat_strings


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

    def _single_token_embedding(token: str) -> torch.Tensor:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Tokenizer must map '{token}' to exactly one id; got {ids}"
            )
        token_id = torch.tensor(ids, device=dev)
        emb = E(token_id).unsqueeze(0)  # [1, 1, H]
        return emb.expand(txt_emb.size(0), -1, -1)

    start_emb = _single_token_embedding(COND_START_TOKEN)
    end_emb = _single_token_embedding(COND_END_TOKEN)

    inputs_embeds = torch.cat([start_emb, cond_emb, end_emb, txt_emb], dim=1)
    attn = enc["attention_mask"]
    B = attn.size(0)
    ones_start = torch.ones(
        (B, start_emb.size(1)), dtype=attn.dtype, device=attn.device
    )
    ones_cond = torch.ones((B, cond_emb.size(1)), dtype=attn.dtype, device=attn.device)
    ones_end = torch.ones((B, end_emb.size(1)), dtype=attn.dtype, device=attn.device)
    attn = torch.cat([ones_start, ones_cond, ones_end, attn], dim=1)
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


__all__ = [
    "encode_label_variants",
    "prepare_prompt_embeddings",
    "batched_label_logprobs",
    "score_label_word_logprobs",
    "predict_label_with_probs",
]
