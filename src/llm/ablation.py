# src/llm/ablation_prompt_cond.py
# -*- coding: utf-8 -*-
# Prompt vs Conditioning ablation: {cond+prompt, cond only, prompt only}
#
# - cond+prompt: normal evaluation (conditioning tokens + prompt)
# - cond only  : prompt fields blanked, conditioning tokens present
# - prompt only: prompt intact, conditioning vector zeroed
#
# Non-mutating:
#   * Dataset content is never permanently modified.
#   * Model files are never touched; we only load/score.
#
# Requires: src.llm.eval.evaluate_split_batched (fixed to use D_eff+D_go+D_prot)

from __future__ import annotations
import re
import hashlib
import numpy as np
import torch
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm

from src.llm.eval import evaluate_split_batched
from src.clinvar import pick_gene

# ----------------------------
# Prompt-source columns used by your prompt builder
# ----------------------------
PROMPT_SRC_COLS = [
    # variant strings
    "Name",
    "HGVS",
    "HGVS.p",
    "HGVS.c",
    "HGVSp",
    "HGVSc",
    "ProteinChange",
    "cDNAChange",
    # gene identity
    "Gene",
    "GeneSymbol",
    "GeneSymbolSimple",
]

# (used for small debug sanity)
_HGVS_RX_P_REAL = re.compile(r"p\.(?!\[)[^\s,;:/\]]+", flags=re.IGNORECASE)
_HGVS_RX_C_REAL = re.compile(r"c\.(?!\[)[^\s,;:/\]]+", flags=re.IGNORECASE)


def _join_prompt_sources(df, cols) -> np.ndarray:
    ser = df[cols].astype(str).replace("nan", "")
    return ser.agg(" | ".join, axis=1).values


def _count_real_hgvs(arr_str: np.ndarray) -> Tuple[int, int]:
    s = np.asarray(arr_str, dtype=object)
    p = sum(1 for x in s if _HGVS_RX_P_REAL.search(x or ""))
    c = sum(1 for x in s if _HGVS_RX_C_REAL.search(x or ""))
    return p, c


def _sha10(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:10]


def _debug_dump_sources(meta_before, meta_after, tag: str, cols):
    b_join = _join_prompt_sources(meta_before, cols)
    a_join = _join_prompt_sources(meta_after, cols)
    b_p, b_c = _count_real_hgvs(b_join)
    a_p, a_c = _count_real_hgvs(a_join)
    print(f"[debug:{tag}] REAL HGVS.p: {b_p} -> {a_p} | REAL HGVS.c: {b_c} -> {a_c}")
    for i in range(min(3, len(a_join))):
        print(
            f"[debug:{tag}] i={i} sha(before)={_sha10(b_join[i])} sha(after)={_sha10(a_join[i])}"
        )
        print(f"  BEFORE: {b_join[i][:180] + ('…' if len(b_join[i]) > 180 else '')}")
        print(f"  AFTER : {a_join[i][:180] + ('…' if len(a_join[i]) > 180 else '')}")


@contextmanager
def _prompt_blank_context(ds_split, *, debug: bool = True):
    """
    Temporarily blank *only the prompt source fields* for a split.
    Fully restores original values on exit.
    """
    meta = ds_split.meta
    cols = [c for c in PROMPT_SRC_COLS if c in meta.columns]
    if not cols:
        # Nothing to edit
        yield
        return

    originals = {c: meta[c].copy(deep=True) for c in cols}
    before = meta[cols].copy(deep=True)
    try:
        for c in cols:
            meta[c] = ""
        if debug:
            _debug_dump_sources(before, meta[cols], tag="prompt_blank", cols=cols)
        yield
    finally:
        for c, ser in originals.items():
            meta[c] = ser


# ----------------------------
# Lightweight masking view
# ----------------------------
@dataclass
class _MaskCfg:
    mode: str  # 'identity' | 'zero'
    # mode='identity' → return x unchanged (cond present)
    # mode='zero'     → return zeros_like(x) (cond removed)


class _MaskedSplit:
    """
    Read-through view of a split that optionally zeros the conditioning vector.
    Returns a CPU torch.Tensor so downstream `.numpy()` continues to work.
    """

    def __init__(self, base_split, cfg: _MaskCfg):
        self.base = base_split
        self.cfg = cfg
        self.meta = getattr(base_split, "meta", None)  # forward metadata

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y1, y2 = self.base[i]
        v = x.numpy() if hasattr(x, "numpy") else np.asarray(x, dtype=np.float32)
        v = v.astype(np.float32, copy=False)
        if self.cfg.mode == "zero":
            z = np.zeros_like(v, dtype=np.float32)
        else:
            z = v  # identity
        return torch.from_numpy(np.asarray(z, dtype=np.float32)), y1, y2


def _with_cond_identity(seq_ds: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _MaskCfg(mode="identity")
    return {k: _MaskedSplit(seq_ds[k], cfg) for k in ("train", "val", "test")}


def _with_cond_zero(seq_ds: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _MaskCfg(mode="zero")
    return {k: _MaskedSplit(seq_ds[k], cfg) for k in ("train", "val", "test")}


# ----------------------------
# Main runner
# ----------------------------
def run_prompt_vs_cond_ablation(
    seq_ds: Dict[str, Any],
    *,
    out_dir: str = "artifacts/qwen3_hpo_lora",  # points to trained adapter dir
    split: str = "val",
    max_n: Optional[int] = None,
    batch_size: int = 64,
    debug_prompts: bool = True,
) -> Dict[str, dict]:
    """
    Runs three evaluations using the SAME fine-tuned model:
      1) cond+prompt  (normal; conditioning + prompt)
      2) cond only    (prompt blanked)
      3) prompt only  (conditioning zeroed)
    """
    results = {}

    # --- (1) cond+prompt ---
    print("\n[ABL] cond+prompt (baseline)")
    view_normal = _with_cond_identity(seq_ds)
    out = evaluate_split_batched(
        view_normal,
        out_dir=out_dir,
        split=split,
        max_n=max_n,
        batch_size=batch_size,
        sample_rationales=0,
    )
    results["cond+prompt"] = out

    # --- (2) cond only (blank prompts) ---
    print("\n[ABL] cond only (prompts blanked)")
    with _prompt_blank_context(seq_ds[split], debug=debug_prompts):
        view_cond_only = _with_cond_identity(seq_ds)
        out = evaluate_split_batched(
            view_cond_only,
            out_dir=out_dir,
            split=split,
            max_n=max_n,
            batch_size=batch_size,
            sample_rationales=0,
        )
    results["cond_only"] = out

    # --- (3) prompt only (conditioning removed) ---
    print("\n[ABL] prompt only (conditioning zeroed)")
    view_prompt_only = _with_cond_zero(seq_ds)
    out = evaluate_split_batched(
        view_prompt_only,
        out_dir=out_dir,
        split=split,
        max_n=max_n,
        batch_size=batch_size,
        sample_rationales=0,
    )
    results["prompt_only"] = out

    # --- Pretty print ---
    def _get(res, k):
        return float(res.get(k, float("nan")))

    print("\nAblation summary [{split}]")
    print(f"{'Mode':14} {'N':>6} {'Acc':>8} {'F1':>8} {'ROC-AUC':>10} {'PR-AUC':>10}")
    print("-" * 54)
    for name in ("cond+prompt", "cond_only", "prompt_only"):
        r = results[name]
        print(
            f"{name:14} {int(r['n']):6d} "
            f"{_get(r, 'accuracy'):8.4f} {_get(r, 'f1'):8.4f} "
            f"{_get(r, 'auc_roc'):10.4f} {_get(r, 'auc_pr'):10.4f}"
        )

    # Δ vs cond+prompt
    ref = results["cond+prompt"]
    print("\nΔ vs cond+prompt:")
    for name in ("cond_only", "prompt_only"):
        r = results[name]
        print(
            f"- {name:12}  "
            f"ΔAcc={_get(r, 'accuracy') - _get(ref, 'accuracy'):+.4f}  "
            f"ΔF1={_get(r, 'f1') - _get(ref, 'f1'):+.4f}  "
            f"ΔROC-AUC={_get(r, 'auc_roc') - _get(ref, 'auc_roc'):+.4f}  "
            f"ΔPR-AUC={_get(r, 'auc_pr') - _get(ref, 'auc_pr'):+.4f}"
        )

    return results


# Example usage:
# res = run_prompt_vs_cond_ablation(
#     seq_ds, out_dir="artifacts/qwen3_hpo_lora",
#     split="val", max_n=2000, batch_size=64, debug_prompts=True
# )
