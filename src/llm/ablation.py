# src/llm/ablation_prompt_cond.py
# -*- coding: utf-8 -*-
# Prompt vs Conditioning ablation: {cond+prompt, cond only, prompt only, zero_dna, zero_go, zero_prot}
#
# - cond+prompt: normal evaluation (conditioning tokens + prompt)
# - cond only  : prompt fields blanked, conditioning tokens present
# - prompt only: prompt intact, conditioning vector zeroed
# - zero_dna   : prompt intact, conditioning present but DNA slice zeroed
# - zero_go    : prompt intact, conditioning present but GO slice zeroed
# - zero_prot  : prompt intact, conditioning present but Protein slice zeroed
#
# Non-mutating:
#   * Dataset content is never permanently modified (prompt blanking is a context manager).
#   * Model files are never touched; we call evaluate_split_batched which loads read-only.
#
# Requires: src.llm.eval.evaluate_split_batched (fixed to use D_eff + D_go + D_prot)

from __future__ import annotations
import re
import hashlib
import numpy as np
import torch
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Dict, Any, Optional, Tuple, List

from tqdm import tqdm

from src.llm.eval import evaluate_split_batched
from src.clinvar import (
    pick_gene,
)  # noqa: F401 (kept for potential future prompt tweaks)

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
    Temporarily blank *all prompt source fields* for a split, including `Name`.
    Also sets a flag so evaluate_split_batched can force ctx="".
    Fully restores original values on exit.
    """
    meta = ds_split.meta
    cols = [c for c in PROMPT_SRC_COLS + ["Name"] if c in meta.columns]
    if not cols:
        yield
        return

    originals = {c: meta[c].copy(deep=True) for c in cols}
    before = meta[cols].copy(deep=True)
    try:
        for c in cols:
            meta[c] = ""
        # flag for eval.py
        setattr(ds_split, "_force_blank_prompts", True)
        if debug:
            _debug_dump_sources(before, meta[cols], tag="prompt_blank", cols=cols)
        yield
    finally:
        for c, ser in originals.items():
            meta[c] = ser
        if hasattr(ds_split, "_force_blank_prompts"):
            delattr(ds_split, "_force_blank_prompts")


# ----------------------------
# Conditioning zeroing / masking views
# ----------------------------
@dataclass
class _MaskCfg:
    mode: str  # 'identity' | 'zero' | 'selective'
    zero_slices: Optional[List[Tuple[int, int]]] = None
    # zero_slices is a list of [start, end) index ranges to zero when mode='selective'


class _MaskedSplit:
    """
    Read-through view of a split that optionally zeros the conditioning vector (all or slices).
    Returns a CPU torch.Tensor so downstream `.numpy()` continues to work.
    Forwards attributes to the underlying split (including D_eff/D_go/D_prot/meta).
    """

    def __init__(self, base_split, cfg: _MaskCfg):
        self.base = base_split
        self.cfg = cfg
        # Explicitly keep meta for pandas ops
        self.meta = getattr(base_split, "meta", None)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y1, y2 = self.base[i]
        v = x.numpy() if hasattr(x, "numpy") else np.asarray(x, dtype=np.float32)
        v = v.astype(np.float32, copy=False)

        if self.cfg.mode == "zero":
            z = np.zeros_like(v, dtype=np.float32)
        elif self.cfg.mode == "selective" and self.cfg.zero_slices:
            z = v.copy()
            for s, e in self.cfg.zero_slices:
                # Guard slice within bounds
                s_cl = max(0, min(s, z.shape[0]))
                e_cl = max(s_cl, min(e, z.shape[0]))
                if e_cl > s_cl:
                    z[s_cl:e_cl] = 0.0
        else:
            z = v  # identity

        return torch.from_numpy(z), y1, y2

    def __getattr__(self, name):
        # Forward everything else (e.g., D_eff/D_go/D_prot)
        return getattr(self.base, name)


def _with_cond_identity(seq_ds: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _MaskCfg(mode="identity")
    return {k: _MaskedSplit(seq_ds[k], cfg) for k in ("train", "val", "test")}


def _with_cond_zero(seq_ds: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _MaskCfg(mode="zero")
    return {k: _MaskedSplit(seq_ds[k], cfg) for k in ("train", "val", "test")}


def _with_cond_selective(
    seq_ds: Dict[str, Any], zero_slices: List[Tuple[int, int]]
) -> Dict[str, Any]:
    cfg = _MaskCfg(mode="selective", zero_slices=list(zero_slices))
    return {k: _MaskedSplit(seq_ds[k], cfg) for k in ("train", "val", "test")}


# ----------------------------
# Helpers to derive slice layout and verify masking really happened
# ----------------------------
def _slice_layout(seq_ds: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
    """
    Returns index ranges [start, end) for dna/go/prot within the conditioning vector.
    """
    D_eff = int(seq_ds["train"].D_eff)
    D_go = int(seq_ds["train"].D_go)
    D_prot = int(getattr(seq_ds["train"], "D_prot", 0) or 0)

    i = 0
    dna = (i, i + D_eff)
    i += D_eff
    go = (i, i + D_go)
    i += D_go
    prot = (i, i + D_prot) if D_prot > 0 else (i, i)
    return {"dna": dna, "go": go, "prot": prot}


def _debug_check_masking(
    ds_split, zero_slices: List[Tuple[int, int]], tag: str, n: int = 3
):
    """
    Print norms of cond vec segments before and after masking for a few items.
    Uses the underlying (unmasked) split plus a masked view.
    """
    if n <= 0:
        return

    # Build masked view for this split only
    view = _MaskedSplit(ds_split, _MaskCfg(mode="selective", zero_slices=zero_slices))

    print(f"[debug:{tag}] masking slices:", zero_slices)
    for i in range(min(n, len(ds_split))):
        x0, *_ = ds_split[i]
        v0 = x0.numpy().astype(np.float32, copy=False)
        xm, *_ = view[i]
        vm = xm.numpy().astype(np.float32, copy=False)

        # Quick sanity: shapes
        if v0.shape != vm.shape:
            print(f"[debug:{tag}] shape mismatch at i={i}: {v0.shape} vs {vm.shape}")

        # Norms / segment norms
        n0 = float(np.linalg.norm(v0))
        nm = float(np.linalg.norm(vm))
        segs = []
        for s, e in zero_slices:
            s_cl = max(0, min(s, v0.shape[0]))
            e_cl = max(s_cl, min(e, v0.shape[0]))
            segs.append(float(np.linalg.norm(vm[s_cl:e_cl])))
        print(
            f"[debug:{tag}] i={i} ||v0||={n0:.4f} ||vm||={nm:.4f} "
            f"masked_seg_norms={', '.join(f'{x:.4f}' for x in segs)}"
        )


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
    debug_masking: bool = True,
) -> Dict[str, dict]:
    """
    Runs six evaluations using the SAME training artifacts:
      1) cond+prompt  (normal; conditioning + prompt)
      2) cond only    (prompt blanked; conditioning present)
      3) prompt only  (conditioning zeroed entirely)
      4) zero_dna     (prompt intact; zero DNA slice in cond)
      5) zero_go      (prompt intact; zero GO slice in cond)
      6) zero_prot    (prompt intact; zero Protein slice in cond, if present)

    Returns a dict of mode -> metrics dict (from evaluate_split_batched).
    """
    results: Dict[str, dict] = {}

    # Compute slice layout once (based on train split metadata)
    layout = _slice_layout(seq_ds)
    dna_slice, go_slice, prot_slice = layout["dna"], layout["go"], layout["prot"]

    # --- (1) cond only (blank prompts) ---
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
            # debug_prompt_probe=True,
        )
    results["cond_only"] = out

    # --- (2) cond+prompt ---
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

    # --- (3) prompt only (conditioning removed) ---
    print("\n[ABL] prompt only (conditioning zeroed)")
    view_prompt_only = _with_cond_zero(seq_ds)
    # Quick check: show a few norms to prove zeroing took effect
    if debug_masking:
        _debug_check_masking(
            seq_ds[split],
            [(0, dna_slice[1] if prot_slice[1] == 0 else prot_slice[1])],
            tag="prompt_only_total",
            n=3,
        )
    out = evaluate_split_batched(
        view_prompt_only,
        out_dir=out_dir,
        split=split,
        max_n=max_n,
        batch_size=batch_size,
        sample_rationales=0,
    )
    results["prompt_only"] = out

    # --- (4) zero_dna (selectively zero DNA slice) ---
    print("\n[ABL] zero_dna (conditioning present; DNA slice zeroed)")
    view_zero_dna = _with_cond_selective(seq_ds, [dna_slice])
    if debug_masking:
        _debug_check_masking(seq_ds[split], [dna_slice], tag="zero_dna", n=3)
    out = evaluate_split_batched(
        view_zero_dna,
        out_dir=out_dir,
        split=split,
        max_n=max_n,
        batch_size=batch_size,
        sample_rationales=0,
    )
    results["zero_dna"] = out

    # --- (5) zero_go (selectively zero GO slice) ---
    print("\n[ABL] zero_go (conditioning present; GO slice zeroed)")
    view_zero_go = _with_cond_selective(seq_ds, [go_slice])
    if debug_masking:
        _debug_check_masking(seq_ds[split], [go_slice], tag="zero_go", n=3)
    out = evaluate_split_batched(
        view_zero_go,
        out_dir=out_dir,
        split=split,
        max_n=max_n,
        batch_size=batch_size,
        sample_rationales=0,
    )
    results["zero_go"] = out

    # --- (6) zero_prot (selectively zero Prot slice, if present) ---
    has_prot = (prot_slice[1] - prot_slice[0]) > 0
    if has_prot:
        print("\n[ABL] zero_prot (conditioning present; Protein slice zeroed)")
        view_zero_prot = _with_cond_selective(seq_ds, [prot_slice])
        if debug_masking:
            _debug_check_masking(seq_ds[split], [prot_slice], tag="zero_prot", n=3)
        out = evaluate_split_batched(
            view_zero_prot,
            out_dir=out_dir,
            split=split,
            max_n=max_n,
            batch_size=batch_size,
            sample_rationales=0,
        )
        results["zero_prot"] = out
    else:
        print("\n[ABL] zero_prot skipped (no protein features present)")

    # --- Pretty print ---
    def _get(res, k):
        try:
            return float(res.get(k, float("nan")))
        except Exception:
            return float("nan")

    print(f"\nAblation summary [{split}]")
    print(f"{'Mode':14} {'N':>6} {'Acc':>8} {'F1':>8} {'ROC-AUC':>10} {'PR-AUC':>10}")
    print("-" * 54)
    ordered = ["cond+prompt", "cond_only", "prompt_only", "zero_dna", "zero_go"]
    if has_prot:
        ordered.append("zero_prot")
    for name in ordered:
        r = results[name]
        print(
            f"{name:14} {int(r['n']):6d} "
            f"{_get(r, 'accuracy'):8.4f} {_get(r, 'f1'):8.4f} "
            f"{_get(r, 'auc_roc'):10.4f} {_get(r, 'auc_pr'):10.4f}"
        )

    # Δ vs cond+prompt
    ref = results["cond+prompt"]
    print("\nΔ vs cond+prompt:")
    for name in ordered[1:]:
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
#     split="val", max_n=2000, batch_size=64, debug_prompts=True, debug_masking=True
# )
