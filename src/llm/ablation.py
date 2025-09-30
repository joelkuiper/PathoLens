# src/llm/ablation_prompt_cond.py
# -*- coding: utf-8 -*-
# Prompt vs Conditioning ablation:
#   - cond+prompt : normal evaluation (conditioning + prompt)
#   - cond only   : prompt blanked, conditioning present
#   - prompt only : prompt intact, conditioning zeroed
#   - cond+noise  : prompt intact, conditioning + strong Gaussian noise
#   - pure-noise  : prompt intact, conditioning replaced by pure noise

from __future__ import annotations
import re
import hashlib
import numpy as np
import torch
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Dict, Any, Optional, Tuple, List, Sequence

from src.llm.config import PROMPT_TMPL
from src.llm.eval import evaluate_split_batched

# ----------------------------
# Prompt-source columns used by the prompt builder
# ----------------------------
PROMPT_SRC_COLS = [
    # VEP-derived fields
    "gene_symbol",
    "GeneSymbol",
    "mane_select",
    "transcript_id",
    "protein_id",
    "hgvsc",
    "hgvsp",
    "most_severe_consequence",
    "impact",
    "consequence_terms",
    "variant_allele",
    "amino_acids",
    "codons",
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
    """Temporarily blank *all prompt fields* (including Name)."""
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
        setattr(ds_split, "_force_blank_prompts", True)
        setattr(ds_split, "_force_blank_prompt_template", PROMPT_TMPL.format(ctx=""))
        if debug:
            _debug_dump_sources(before, meta[cols], tag="prompt_blank", cols=cols)
        yield
    finally:
        for c, ser in originals.items():
            meta[c] = ser
        if hasattr(ds_split, "_force_blank_prompts"):
            delattr(ds_split, "_force_blank_prompts")
        if hasattr(ds_split, "_force_blank_prompt_template"):
            delattr(ds_split, "_force_blank_prompt_template")


@contextmanager
def _prompt_blank_fields_context(
    ds_split,
    field_keys: Sequence[str],
    *,
    tag: str,
    debug: bool = True,
):
    """Temporarily blank selected prompt fields based on canonical keys."""

    meta = getattr(ds_split, "meta", None)
    if meta is None:
        yield False
        return

    cols_all = [c for c in PROMPT_SRC_COLS + ["Name"] if c in meta.columns]
    if not cols_all:
        yield False
        return

    low_keys = {str(k).lower() for k in field_keys}
    target_cols = [c for c in cols_all if c.lower() in low_keys]
    if not target_cols:
        yield False
        return

    before = meta[cols_all].copy(deep=True)
    try:
        for col in target_cols:
            meta[col] = ""
        if debug:
            _debug_dump_sources(before, meta[cols_all], tag=tag, cols=cols_all)
        yield True
    finally:
        for col in cols_all:
            meta[col] = before[col]


# ----------------------------
# Conditioning transforms (identity / zero / noise / permute / scale / pure-noise)
# ----------------------------
@dataclass
class _CondTransformCfg:
    mode: str  # 'identity'|'zero'|'noise'|'permute'|'scale'|'pure-noise'
    noise_std: Optional[float] = None  # absolute sigma if provided
    noise_alpha: float = 0.0  # if noise_std is None, use alpha*std(v)
    scale: float = 1.0
    perm_index: Optional[np.ndarray] = None  # length = len(split)
    zero_ranges: Optional[Tuple[Tuple[int, int], ...]] = (
        None  # slices to zero post-transform
    )


class _TransformedSplit:
    """
    Read-through view that transforms the conditioning vector per item.
    Returns a CPU torch.Tensor so downstream `.numpy()` continues to work.
    Forwards attributes to the underlying split (including meta/D_eff/D_prot).
    """

    def __init__(self, base_split, cfg: _CondTransformCfg):
        self.base = base_split
        self.cfg = cfg
        self.meta = getattr(base_split, "meta", None)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        base_item = self.base[i]
        if isinstance(base_item, tuple):
            x = base_item[0]
            extras = base_item[1:]
            pack_tuple = True
        else:
            x = base_item
            extras = ()
            pack_tuple = False

        def _to_numpy(val):
            return (
                val.numpy()
                if hasattr(val, "numpy")
                else np.asarray(val, dtype=np.float32)
            )

        # Optionally fetch a different vector (permute across dataset)
        if self.cfg.mode == "permute":
            j = int(self.cfg.perm_index[i])  # remap index
            perm_item = self.base[j]
            perm_x = perm_item[0] if isinstance(perm_item, tuple) else perm_item
            v = _to_numpy(perm_x)
        else:
            v = _to_numpy(x)

        v = v.astype(np.float32, copy=False)
        mode = self.cfg.mode

        if mode == "zero":
            z = np.zeros_like(v, dtype=np.float32)

        elif mode == "pure-noise":
            # Replace entirely by noise
            sigma = float(self.cfg.noise_std if self.cfg.noise_std is not None else 1.0)
            z = np.random.normal(0.0, sigma, size=v.shape).astype(np.float32)

        elif mode == "noise":
            # Additive noise to existing vector
            if self.cfg.noise_std is not None:
                sigma = float(self.cfg.noise_std)
            else:
                sigma = float(self.cfg.noise_alpha) * (np.std(v) + 1e-6)
            noise = np.random.normal(0.0, sigma, size=v.shape).astype(np.float32)
            z = v + noise

        elif mode == "scale":
            z = (float(self.cfg.scale) * v).astype(np.float32, copy=False)

        else:  # identity or permute (already swapped)
            z = v

        zero_ranges = self.cfg.zero_ranges or ()
        if zero_ranges:
            if not isinstance(z, np.ndarray):
                z = np.asarray(z, dtype=np.float32)
            if not z.flags.writeable or mode in ("identity", "permute"):
                z = z.copy()
            n = z.shape[0]
            for start, end in zero_ranges:
                s = max(0, min(n, int(start)))
                e = max(0, min(n, int(end)))
                if e > s:
                    z[s:e] = 0.0
            z = z.astype(np.float32, copy=False)

        out_x = torch.from_numpy(z)
        if pack_tuple:
            return (out_x, *extras)
        return out_x

    def __getattr__(self, name):
        return getattr(self.base, name)


def _block_bounds(entry) -> Optional[Tuple[int, int]]:
    """Return the [start, stop) bounds for a conditioning block."""

    if entry is None:
        return None

    if isinstance(entry, slice):
        return (int(entry.start or 0), int(entry.stop or 0))

    if isinstance(entry, dict):
        if "slice" in entry and isinstance(entry["slice"], slice):
            sl = entry["slice"]
            return (int(sl.start or 0), int(sl.stop or 0))

        if "slices" in entry and isinstance(entry["slices"], dict):
            starts: List[int] = []
            stops: List[int] = []
            for sl in entry["slices"].values():
                if isinstance(sl, slice):
                    starts.append(int(sl.start or 0))
                    stops.append(int(sl.stop or 0))
            if starts and stops:
                return (min(starts), max(stops))

    return None


def _range_len(bounds: Optional[Tuple[int, int]]) -> int:
    if not bounds:
        return 0
    start, stop = bounds
    return max(0, int(stop) - int(start))


def _with_cond_identity(seq_ds: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _CondTransformCfg(mode="identity")
    return {k: _TransformedSplit(seq_ds[k], cfg) for k in ("train", "val", "test")}


def _with_cond_zero(seq_ds: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _CondTransformCfg(mode="zero")
    return {k: _TransformedSplit(seq_ds[k], cfg) for k in ("train", "val", "test")}


def _with_cond_zero_ranges(
    seq_ds: Dict[str, Any], ranges: Sequence[Tuple[int, int]]
) -> Dict[str, Any]:
    cleaned: List[Tuple[int, int]] = []
    for start, end in ranges:
        s = int(start)
        e = int(end)
        if e > s:
            cleaned.append((s, e))
    cfg = _CondTransformCfg(
        mode="identity",
        zero_ranges=tuple(cleaned) if cleaned else None,
    )
    return {k: _TransformedSplit(seq_ds[k], cfg) for k in ("train", "val", "test")}


def _with_cond_noise(
    seq_ds: Dict[str, Any], *, noise_std: Optional[float], noise_alpha: float
) -> Dict[str, Any]:
    cfg = _CondTransformCfg(mode="noise", noise_std=noise_std, noise_alpha=noise_alpha)
    return {k: _TransformedSplit(seq_ds[k], cfg) for k in ("train", "val", "test")}


def _with_cond_pure_noise(
    seq_ds: Dict[str, Any], *, noise_std: float = 1.0
) -> Dict[str, Any]:
    cfg = _CondTransformCfg(mode="pure-noise", noise_std=float(noise_std))
    return {k: _TransformedSplit(seq_ds[k], cfg) for k in ("train", "val", "test")}


def _with_cond_permute(
    seq_ds: Dict[str, Any], *, split: str, seed: int = 13
) -> Dict[str, Any]:
    n = len(seq_ds[split])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).astype(np.int64)
    cfg = _CondTransformCfg(mode="permute", perm_index=perm)
    out = {}
    for k in ("train", "val", "test"):
        # for splits other than `split`, just identity (we only evaluate one split at a time)
        out[k] = _TransformedSplit(
            seq_ds[k], cfg if k == split else _CondTransformCfg(mode="identity")
        )
    return out


def _with_cond_scale(seq_ds: Dict[str, Any], *, scale: float) -> Dict[str, Any]:
    cfg = _CondTransformCfg(mode="scale", scale=float(scale))
    return {k: _TransformedSplit(seq_ds[k], cfg) for k in ("train", "val", "test")}


# ----------------------------
# Debug helpers
# ----------------------------
def _debug_check_noise(
    ds_split, noise_std: Optional[float], noise_alpha: float, tag: str, n: int = 3
):
    view = _TransformedSplit(
        ds_split,
        _CondTransformCfg(mode="noise", noise_std=noise_std, noise_alpha=noise_alpha),
    )
    print(f"[debug:{tag}] noise_std={noise_std} noise_alpha={noise_alpha}")
    for i in range(min(n, len(ds_split))):
        x0, *_ = ds_split[i]
        v0 = x0.numpy().astype(np.float32, copy=False)
        xm, *_ = view[i]
        vm = xm.numpy().astype(np.float32, copy=False)
        sig = np.std(v0) + 1e-8
        nse = np.std(vm - v0) + 1e-8
        snr = (sig / nse) if nse > 0 else float("inf")
        print(
            f"[debug:{tag}] i={i} ||v||_2(before)={np.linalg.norm(v0):.4f} "
            f"||v||_2(after)={np.linalg.norm(vm):.4f}  std(v)={sig:.5f} std(noise)={nse:.5f}  SNR≈{snr:.3f}"
        )


def _debug_check_scale(ds_split, scale: float, tag: str, n: int = 3):
    view = _TransformedSplit(ds_split, _CondTransformCfg(mode="scale", scale=scale))
    print(f"[debug:{tag}] scale={scale}")
    for i in range(min(n, len(ds_split))):
        x0, *_ = ds_split[i]
        v0 = x0.numpy().astype(np.float32, copy=False)
        xm, *_ = view[i]
        vm = xm.numpy().astype(np.float32, copy=False)
        print(
            f"[debug:{tag}] i={i} ||v||_2(before)={np.linalg.norm(v0):.4f} ||v||_2(after)={np.linalg.norm(vm):.4f}"
        )


# ----------------------------
# Main runner
# ----------------------------
def run_prompt_vs_cond_ablation(
    seq_ds: Dict[str, Any],
    *,
    out_dir: str = "artifacts/qwen3",
    split: str = "val",
    max_n: Optional[int] = None,
    batch_size: int = 64,
    debug_prompts: bool = True,
    debug_noise: bool = True,
    noise_std: Optional[float] = 1.5,  # strong absolute sigma by default
    noise_alpha: float = 0.0,  # ignored when noise_std is set
    do_permute: bool = True,
    scale_sweep: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
    do_pure_noise: bool = True,
) -> Dict[str, dict]:
    """
    Evaluates:
      - cond only         (prompt blanked; conditioning present)
      - cond+prompt       (baseline; conditioning + prompt)
      - cond_zero_dna     (conditioning DNA block zeroed)
      - cond_zero_prot    (conditioning protein block zeroed)
      - prompt only       (conditioning zeroed entirely)
      - prompt_no_hgvsp   (prompt intact minus HGVS.p)
      - prompt_no_hgvsc   (prompt intact minus HGVS.c)
      - prompt_no_gene    (prompt intact minus gene symbol)
      - cond+noise        (conditioning + strong Gaussian noise)
      - cond+permute      (conditioning shuffled across items; prompt intact)
      - pure-noise        (conditioning replaced by noise; prompt intact)
      - cond×scale(s)     (conditioning scaled by scalar(s); prompt intact)

    Returns: dict of mode -> metrics.
    """
    results: Dict[str, dict] = {}

    train_split = seq_ds.get("train")
    if train_split is None:
        raise ValueError("Sequence dataset missing 'train' split for ablation")
    cond_spec = getattr(train_split, "cond_spec", None)
    if cond_spec:
        dna_bounds = _block_bounds(cond_spec.get("dna"))
        go_bounds = _block_bounds(cond_spec.get("go"))
        prot_bounds = _block_bounds(cond_spec.get("protein"))
        D_eff = _range_len(dna_bounds)
        D_go = _range_len(go_bounds)
        D_prot = _range_len(prot_bounds)
        dna_range = dna_bounds if D_eff > 0 else None
        go_range = go_bounds if D_go > 0 else None
        prot_range = prot_bounds if D_prot > 0 else None
    else:
        raise ValueError("Expected cond_spec")

    # --- cond only (blank prompts) ---
    print("\n[ABL] cond only (prompts blanked)")
    with _prompt_blank_context(seq_ds[split], debug=debug_prompts):
        view_cond_only = _with_cond_identity(seq_ds)
        out = evaluate_split_batched(
            view_cond_only,
            out_dir=out_dir,
            split=split,
            max_n=max_n,
            batch_size=batch_size,
        )
    results["cond_only"] = out

    # --- cond+prompt (baseline) ---
    print("\n[ABL] cond+prompt (baseline)")
    view_normal = _with_cond_identity(seq_ds)
    out = evaluate_split_batched(
        view_normal,
        out_dir=out_dir,
        split=split,
        max_n=max_n,
        batch_size=batch_size,
    )
    results["cond+prompt"] = out

    # --- cond_zero_dna (zero only DNA features) ---
    if dna_range is not None:
        print("\n[ABL] cond_zero_dna (conditioning: zero DNA block)")
        view_zero_dna = _with_cond_zero_ranges(seq_ds, [dna_range])
        out = evaluate_split_batched(
            view_zero_dna,
            out_dir=out_dir,
            split=split,
            max_n=max_n,
            batch_size=batch_size,
        )
        results["cond_zero_dna"] = out

    # --- cond_zero_go (zero only GO features) ---
    if go_range is not None:
        go_start, go_end = go_range
        print("\n[ABL] cond_zero_go (conditioning: zero GO block)")
        view_zero_go = _with_cond_zero_ranges(seq_ds, [(go_start, go_end)])
        out = evaluate_split_batched(
            view_zero_go,
            out_dir=out_dir,
            split=split,
            max_n=max_n,
            batch_size=batch_size,
        )
        results["cond_zero_go"] = out

    # --- cond_zero_prot (zero only protein features) ---
    if prot_range is not None:
        print("\n[ABL] cond_zero_prot (conditioning: zero protein block)")
        view_zero_prot = _with_cond_zero_ranges(seq_ds, [prot_range])
        out = evaluate_split_batched(
            view_zero_prot,
            out_dir=out_dir,
            split=split,
            max_n=max_n,
            batch_size=batch_size,
        )
        results["cond_zero_prot"] = out
    else:
        print("\n[ABL] cond_zero_prot skipped (no protein block)")

    # --- prompt only (conditioning removed) ---
    print("\n[ABL] prompt only (conditioning zeroed)")
    view_prompt_only = _with_cond_zero(seq_ds)
    out = evaluate_split_batched(
        view_prompt_only,
        out_dir=out_dir,
        split=split,
        max_n=max_n,
        batch_size=batch_size,
    )
    results["prompt_only"] = out

    # --- prompt field removals ---
    prompt_field_abls = [
        ("prompt_no_hgvsp", ("hgvsp", "hgvs.p", "hgvs_p"), "remove HGVS.p from prompt"),
        ("prompt_no_hgvsc", ("hgvsc", "hgvs.c", "hgvs_c"), "remove HGVS.c from prompt"),
        (
            "prompt_no_gene",
            ("gene_symbol", "genesymbol"),
            "remove gene symbol from prompt",
        ),
    ]
    for name, keys, desc in prompt_field_abls:
        print(f"\n[ABL] {name} ({desc})")
        with _prompt_blank_fields_context(
            seq_ds[split], field_keys=keys, tag=name, debug=debug_prompts
        ) as applied:
            if not applied:
                print(f"[ABL] {name}: no matching columns; skipped.")
                continue
            view_prompt_variant = _with_cond_identity(seq_ds)
            out = evaluate_split_batched(
                view_prompt_variant,
                out_dir=out_dir,
                split=split,
                max_n=max_n,
                batch_size=batch_size,
            )
        results[name] = out

    # --- (4) cond+noise (conditioning + strong Gaussian noise) ---
    print("\n[ABL] cond+noise (conditioning + Gaussian noise)")
    if debug_noise:
        _debug_check_noise(
            seq_ds[split],
            noise_std=noise_std,
            noise_alpha=noise_alpha,
            tag="cond_noise",
            n=3,
        )
    view_noise = _with_cond_noise(seq_ds, noise_std=noise_std, noise_alpha=noise_alpha)
    out = evaluate_split_batched(
        view_noise,
        out_dir=out_dir,
        split=split,
        max_n=max_n,
        batch_size=batch_size,
    )
    results["cond+noise"] = out

    # --- (5) cond+permute (destroy alignment but keep distribution) ---
    if do_permute:
        print("\n[ABL] cond+permute (conditioning shuffled across items)")
        view_perm = _with_cond_permute(seq_ds, split=split, seed=13)
        out = evaluate_split_batched(
            view_perm,
            out_dir=out_dir,
            split=split,
            max_n=max_n,
            batch_size=batch_size,
        )
        results["cond+permute"] = out

    # --- (6) pure-noise (conditioning replaced entirely by noise) ---
    if do_pure_noise:
        print("\n[ABL] pure-noise (conditioning replaced by noise)")
        view_pn = _with_cond_pure_noise(
            seq_ds, noise_std=(noise_std if noise_std is not None else 1.0)
        )
        out = evaluate_split_batched(
            view_pn,
            out_dir=out_dir,
            split=split,
            max_n=max_n,
            batch_size=batch_size,
        )
        results["pure-noise"] = out

    # --- (7) cond×scale sweep ---
    if scale_sweep:
        for s in scale_sweep:
            tag = f"cond×scale={s:g}"
            print(f"\n[ABL] {tag}")
            _debug_check_scale(seq_ds[split], scale=s, tag=f"scale_{s}", n=3)
            view_scale = _with_cond_scale(seq_ds, scale=s)
            out = evaluate_split_batched(
                view_scale,
                out_dir=out_dir,
                split=split,
                max_n=max_n,
                batch_size=batch_size,
            )
            results[tag] = out

    # --- Pretty print ---
    def _get(res, k):
        try:
            return float(res.get(k, float("nan")))
        except Exception:
            return float("nan")

    print(f"\nAblation summary [{split}]")
    print(f"{'Mode':14} {'N':>6} {'Acc':>8} {'F1':>8} {'ROC-AUC':>10} {'PR-AUC':>10}")
    print("-" * 54)
    ordered = ["cond+prompt"]
    for key in [
        "cond_only",
        "cond_zero_dna",
        "cond_zero_prot",
        "cond_zero_go",
        "prompt_only",
        "prompt_no_hgvsp",
        "prompt_no_hgvsc",
        "prompt_no_gene",
        "cond+noise",
    ]:
        if key in results:
            ordered.append(key)
    if "cond+permute" in results:
        ordered.append("cond+permute")
    if "pure-noise" in results:
        ordered.append("pure-noise")
    for s in scale_sweep:
        tag = f"cond×scale={s:g}"
        if tag in results:
            ordered.append(tag)

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


# Example:
# res = run_prompt_vs_cond_ablation(
#     seq_ds, out_dir="artifacts/qwen3",
#     split="test", max_n=None, batch_size=64,
#     noise_std=1.5,  # strong absolute noise
#     do_permute=True,
#     scale_sweep=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
#     do_pure_noise=True,
# )
