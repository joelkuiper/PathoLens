# src/llm/data.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from .config import PROMPT_TMPL
from .chat import build_chat_strings
from .context import build_ctx_from_row


def row_to_example(meta_row) -> Tuple[str, str]:
    """Training target is just the label: 'Benign' or 'Pathogenic'."""
    ctx = build_ctx_from_row(meta_row)
    prompt = PROMPT_TMPL.format(ctx=ctx)
    y = int(meta_row.get("_y"))
    cls = "Pathogenic" if y == 1 else "Benign"
    return prompt, cls


# ---------- Dataset ----------
class CondTextDataset(Dataset):
    """
    Builds conditioning vectors from SequenceTowerDataset samples.

    Assumes each sample returns:
        v, *_  where v packs  [ dna_eff | prot_eff ]

    We *explicitly* slice to D_cond = D_eff + D_prot
    so any future additions to v won't silently leak in.
    """

    def __init__(self, seq_ds, tokenizer=None):
        t0 = time.time()
        self.seq_ds = seq_ds
        self.meta = seq_ds.meta
        # include protein block alongside DNA features
        D_eff = int(seq_ds.D_eff)
        D_prot = int(seq_ds.D_prot)
        self.D_eff, self.D_prot = D_eff, D_prot
        self.D_go = int(getattr(seq_ds, "D_go", 0) or 0)
        self.D_cond = D_eff + D_prot
        print(
            f"[DEBUG] CondTextDataset: D_eff={D_eff} D_prot={D_prot} "
            f"=> D_cond={self.D_cond} | Build time: {time.time() - t0:.2f}s"
        )

    def __len__(self):
        return len(self.seq_ds)

    def __getitem__(self, i: int):
        x, *_ = self.seq_ds[i]  # v contains [dna | prot]
        x = x if isinstance(x, np.ndarray) else x.numpy()
        x = x.astype("float32", copy=False)

        # Slice *exactly* the conditioning span
        cond = x[: self.D_cond]

        meta_row = self.meta.iloc[i]
        prompt, target = row_to_example(meta_row)
        if "_y" not in meta_row:
            raise KeyError("CondTextDataset requires '_y' label in metadata.")
        y = int(meta_row.get("_y"))

        return {"cond_vec": cond, "prompt": prompt, "target": target, "_y": y}


def make_collator(
    tokenizer, max_len: int = 384, label_boost: float = 5.0, mask_think: bool = True
):
    # tiny helpers for label token upweight (optional)
    tok_B = tokenizer.encode("Benign", add_special_tokens=False)
    tok_P = tokenizer.encode("Pathogenic", add_special_tokens=False)

    # tokens for <think> and </think>
    tok_TOPEN = tokenizer.encode("<think>", add_special_tokens=False)
    tok_TCLOSE = tokenizer.encode("</think>", add_special_tokens=False)

    def _find_subseq(where_ids, what_ids, start_at):
        n, m = len(where_ids), len(what_ids)
        if m == 0 or n == 0 or n < m:
            return -1
        for i in range(start_at, n - m + 1):
            if where_ids[i : i + m] == what_ids:
                return i
        return -1

    def collate(batch):
        t0 = time.time()
        prompts = [str(b.get("prompt", "")) for b in batch]
        targets = [
            str(b.get("target", "") or "") for b in batch
        ]  # "Benign"/"Pathogenic"

        prompt_strings, full_strings = build_chat_strings(tokenizer, prompts, targets)

        enc_prompt = tokenizer(
            prompt_strings,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        enc_full = tokenizer(
            full_strings,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )

        input_ids = enc_full["input_ids"]
        attn_mask = enc_full["attention_mask"]
        labels = input_ids.clone()

        prompt_lens = [
            int(x) for x in enc_prompt["attention_mask"].sum(dim=1).tolist()
        ]
        pad_lens = [
            int(x)
            for x in (input_ids.shape[1] - attn_mask.sum(dim=1)).tolist()
        ]
        label_starts = [pad + prompt for pad, prompt in zip(pad_lens, prompt_lens)]

        pad_token_mask = attn_mask == 0
        labels = labels.masked_fill(pad_token_mask, -100)

        # --- label token upweighting ---
        label_weights = torch.ones_like(labels, dtype=torch.float32)
        label_weights = label_weights.masked_fill(pad_token_mask, 0.0)

        for i, (prompt_len, prompt_start, label_start) in enumerate(
            zip(prompt_lens, pad_lens, label_starts)
        ):
            prompt_end = label_start
            labels[i, prompt_start:prompt_end] = -100  # mask system+user
            label_weights[i, prompt_start:prompt_end] = 0.0  # no weight for prompt

            ids = input_ids[i].tolist()
            start = label_start
            jB = _find_subseq(ids, tok_B, start)
            jP = _find_subseq(ids, tok_P, start)
            j = jB if (jB != -1 and (jP == -1 or jB <= jP)) else jP
            if j != -1:
                L = len(tok_B) if j == jB else len(tok_P)
                label_weights[i, j : j + L] *= label_boost

        # --- mask <think>…</think> spans ---
        if mask_think and tok_TOPEN and tok_TCLOSE:
            ids_list = input_ids.tolist()
            for i, ids in enumerate(ids_list):
                seek_from = label_starts[i]  # only consider assistant region
                while True:
                    j = _find_subseq(ids, tok_TOPEN, seek_from)
                    if j == -1:
                        break
                    k = _find_subseq(ids, tok_TCLOSE, j + len(tok_TOPEN))
                    if k == -1:
                        break
                    k2 = k + len(tok_TCLOSE)
                    labels[i, j:k2] = -100
                    label_weights[i, j:k2] = 0.0
                    seek_from = k2

        cond_vec = torch.from_numpy(
            np.stack([b["cond_vec"] for b in batch]).astype("float32")
        )
        try:
            gold_y = torch.tensor([int(b["_y"]) for b in batch], dtype=torch.long)
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise KeyError("Collator expects '_y' labels from dataset.") from exc

        dt = time.time() - t0
        if dt > 0.2:
            print(
                f"[DEBUG] Collator latency: {dt:.3f}s batch_size={len(batch)} "
                f"(tokenizer likely the bottleneck)"
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
            "label_weights": label_weights,
            "cond_vec": cond_vec,
            "_y": gold_y,
        }

    return collate
