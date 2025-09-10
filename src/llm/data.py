# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple
import time, numpy as np, torch
from torch.utils.data import Dataset
from .config import PROMPT_TMPL, clinsig_to_binary
from .chat import build_chat_strings
from src.clinvar import (
    pick_gene,
    extract_hgvsc,
    extract_hgvsp,
    classify_protein_change,
    parse_hgvsc_features,
)


# ---------- Context builder ----------
def build_ctx_from_name(row) -> str:
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
    return "\n".join(lines) if lines else "Variant context provided."


def row_to_example(meta_row) -> Tuple[str, str]:
    """Training target is just the label: 'Benign' or 'Pathogenic'."""
    ctx = build_ctx_from_name(meta_row)
    prompt = PROMPT_TMPL.format(ctx=ctx)
    y = meta_row.get("_y", None)
    if y is None:
        y = clinsig_to_binary(meta_row.get("ClinicalSignificance", ""))
    cls = "Pathogenic" if y == 1 else "Benign"
    return prompt, cls


# ---------- Dataset ----------
class CondTextDataset(Dataset):
    def __init__(self, seq_ds, tokenizer=None):
        t0 = time.time()
        self.seq_ds = seq_ds
        self.meta = seq_ds.meta
        self.D_cond = seq_ds.D_eff + seq_ds.D_go
        print(f"[DEBUG] Dataset ready. Build time: {time.time()-t0:.2f}s")

    def __len__(self):
        return len(self.seq_ds)

    def __getitem__(self, i: int):
        x, _, _ = self.seq_ds[i]
        cond = x.numpy().astype("float32")
        cond /= np.linalg.norm(cond) + 1e-6
        prompt, target = row_to_example(self.meta.iloc[i])
        return {"cond_vec": cond, "prompt": prompt, "target": target}


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

        prompt_lens = enc_prompt["attention_mask"].sum(dim=1).tolist()
        for i, p_len in enumerate(prompt_lens):
            labels[i, : int(p_len)] = -100  # mask system+user

        # --- label token upweighting ---
        label_weights = torch.ones_like(labels, dtype=torch.float32)
        for i, p_len in enumerate(prompt_lens):
            label_weights[i, : int(p_len)] = 0.0  # no weight for prompt
            ids = input_ids[i].tolist()
            start = int(p_len)
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
                start = int(prompt_lens[i])  # only consider assistant region
                seek_from = start
                # handle multiple think blocks just in case
                while True:
                    j = _find_subseq(ids, tok_TOPEN, seek_from)
                    if j == -1:
                        break
                    k = _find_subseq(ids, tok_TCLOSE, j + len(tok_TOPEN))
                    if k == -1:
                        # unmatched close; stop masking (or mask to end if you prefer)
                        break
                    k2 = k + len(tok_TCLOSE)
                    # mask labels and zero weight over the think region
                    labels[i, j:k2] = -100
                    label_weights[i, j:k2] = 0.0
                    seek_from = k2  # continue searching after this block

        cond_vec = torch.from_numpy(
            np.stack([b["cond_vec"] for b in batch]).astype("float32")
        )

        dt = time.time() - t0
        if dt > 0.2:
            print(
                f"[DEBUG] Collator latency: {dt:.3f}s batch_size={len(batch)} (tokenizer likely the bottleneck)"
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
            "label_weights": label_weights,
            "cond_vec": cond_vec,
        }

    return collate
