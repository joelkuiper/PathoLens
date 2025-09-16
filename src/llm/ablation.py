# -*- coding: utf-8 -*-
# Prompt + conditioning ablations with progress bars (robust + non-mutating).

import re
import hashlib
import numpy as np
import torch
from dataclasses import dataclass
from contextlib import contextmanager
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple

from src.clinvar import pick_gene, extract_hgvsc, extract_hgvsp
from src.llm.eval import evaluate_split_batched


# ======================
# Conditioning masks
# ======================
@dataclass
class MaskCfg:
    mode: str  # 'both','dna_only','go_only','prompt_only','go_shuffled','dna_shuffled'
    D_eff: int
    D_go: int
    go_perm: Optional[np.ndarray] = None  # used for shuffled modes


class MaskedSplit:
    """
    Read-through view of a split that masks/reshapes the conditioning vector.
    Returns a CPU torch.Tensor so downstream .numpy() works.
    """

    def __init__(self, base_split, cfg: MaskCfg):
        self.base = base_split
        self.cfg = cfg
        self.meta = getattr(base_split, "meta", None)  # forward metadata

    # Expose dims so evaluate_split_batched can read D_eff/D_go
    @property
    def D_eff(self) -> int:
        return int(self.cfg.D_eff)

    @property
    def D_go(self) -> int:
        return int(self.cfg.D_go)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y1, y2 = self.base[i]
        v = x.numpy() if hasattr(x, "numpy") else np.asarray(x, dtype=np.float32)
        v = v.astype(np.float32, copy=False)

        D_eff, D_go = self.cfg.D_eff, self.cfg.D_go
        mode = self.cfg.mode

        if mode == "both":
            z = v
        elif mode == "dna_only":
            z = v.copy()
            z[D_eff : D_eff + D_go] = 0.0
        elif mode == "go_only":
            z = v.copy()
            z[:D_eff] = 0.0
        elif mode == "prompt_only":
            z = np.zeros_like(v, dtype=np.float32)
        elif mode == "go_shuffled":
            assert self.cfg.go_perm is not None, "go_perm required for go_shuffled"
            j = int(self.cfg.go_perm[i])
            xj, _, _ = self.base[j]
            w = xj.numpy() if hasattr(xj, "numpy") else np.asarray(xj, dtype=np.float32)
            w = w.astype(np.float32, copy=False)
            z = v.copy()
            z[D_eff : D_eff + D_go] = w[D_eff : D_eff + D_go]
        elif mode == "dna_shuffled":
            assert self.cfg.go_perm is not None, "perm required for dna_shuffled"
            j = int(self.cfg.go_perm[i])
            xj, *_ = self.base[j]
            w = xj.numpy() if hasattr(xj, "numpy") else np.asarray(xj, dtype=np.float32)
            w = w.astype(np.float32, copy=False)
            z = v.copy()
            z[:D_eff] = w[:D_eff]  # swap DNA effect only
        else:
            raise ValueError(f"unknown mask mode: {mode}")

        # Return CPU tensor so downstream `x.numpy()` works
        return torch.from_numpy(np.asarray(z, dtype=np.float32)), y1, y2


# ======================
# Helpers
# ======================
def _extract_dims_from_split(ds) -> Tuple[int, int]:
    """
    Robustly extract (D_eff, D_go) from a split or its wrappers.
    Tries: ds.D_eff/D_go, ds.base.D_eff/D_go, ds.cfg.D_eff/D_go, ds.base.cfg.D_eff/D_go.
    """
    candidates = [
        ds,
        getattr(ds, "base", None),
        getattr(ds, "cfg", None),
        getattr(getattr(ds, "base", None), "cfg", None),
    ]
    for obj in candidates:
        if obj is None:
            continue
        if hasattr(obj, "D_eff") and hasattr(obj, "D_go"):
            try:
                return int(getattr(obj, "D_eff")), int(getattr(obj, "D_go"))
            except Exception:
                pass
    raise AttributeError(
        "Could not determine D_eff/D_go from provided seq_ds; ensure your splits "
        "expose D_eff/D_go (or MaskedSplit.cfg holds them)."
    )


def make_ablation_seq_ds(
    seq_ds: Dict[str, Any], mode: str, seed: int = 0
) -> Dict[str, Any]:
    D_eff, D_go = _extract_dims_from_split(seq_ds["train"])
    if mode in ("go_shuffled", "dna_shuffled"):
        rng = np.random.default_rng(seed)
        view = {}
        for split in ("train", "val", "test"):
            n = len(seq_ds[split])
            perm = rng.permutation(n)
            cfg = MaskCfg(mode=mode, D_eff=D_eff, D_go=D_go, go_perm=perm)
            view[split] = MaskedSplit(seq_ds[split], cfg)
        return view
    else:
        cfg = MaskCfg(mode=mode, D_eff=D_eff, D_go=D_go)
        return {s: MaskedSplit(seq_ds[s], cfg) for s in ("train", "val", "test")}


# ======================
# Prompt-side utilities
# ======================
# These are the *actual source columns* your prompt builder reads.
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

# Regexes for *real* HGVS tokens (exclude placeholders like p.[REDACTED])
_HGVS_RX_P_REAL = re.compile(r"p\.(?!\[)[^\s,;:/\]]+", flags=re.IGNORECASE)
_HGVS_RX_C_REAL = re.compile(r"c\.(?!\[)[^\s,;:/\]]+", flags=re.IGNORECASE)


def _join_prompt_sources(df, cols) -> np.ndarray:
    ser = df[cols].astype(str).replace("nan", "")
    return ser.agg(" | ".join, axis=1).values


def _count_real_hgvs(arr_str: np.ndarray) -> Tuple[int, int]:
    # Fast-ish count of real HGVS tokens in concatenated source strings
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
        print(f"  BEFORE: {b_join[i][:180] + ('…' if len(b_join[i])>180 else '')}")
        print(f"  AFTER : {a_join[i][:180] + ('…' if len(a_join[i])>180 else '')}")


@contextmanager
def _temp_prompt_mode(ds, mode: str, seed: int = 0, debug: bool = True):
    """
    Temporarily override *prompt source fields* in ds.meta for a split,
    because evaluate_split_batched doesn't read meta['context'].
    Fully restores on exit.
    """
    meta = ds.meta
    rng = np.random.default_rng(seed)

    # Which of our source columns actually exist?
    cols = [c for c in PROMPT_SRC_COLS if c in meta.columns]
    if not cols:
        # Nothing to edit; no-op
        yield
        return

    # Save originals
    originals = {c: meta[c].copy(deep=True) for c in cols}
    meta_before = meta[cols].copy(deep=True)

    try:
        if mode == "prompt_blank":
            # Drop all leak-prone fields to empty
            for c in cols:
                meta[c] = ""

        elif mode == "prompt_shuffled":
            # Apply one permutation jointly to all prompt fields (swap whole prompts)
            idx = np.arange(len(meta))
            rng.shuffle(idx)
            for c in cols:
                meta[c] = meta[c].iloc[idx].to_list()

        elif mode == "gene_only":
            # Keep gene columns; blank variant strings
            gvals = []
            for _, row in meta.iterrows():
                gvals.append(pick_gene(row) or "")
            gvals = np.array(gvals, dtype=object)
            if "GeneSymbol" in cols:
                meta["GeneSymbol"] = gvals
            if "GeneSymbolSimple" in cols:
                meta["GeneSymbolSimple"] = gvals
            if "Gene" in cols:
                meta["Gene"] = gvals
            for c in cols:
                if c not in ("Gene", "GeneSymbol", "GeneSymbolSimple"):
                    meta[c] = ""

        elif mode == "hgvsc_only":
            # Keep c. columns; blank p. and genes.
            # If c. missing, try to extract from Name.
            c_cols = [c for c in cols if c in ("HGVS.c", "HGVSc", "cDNAChange")]
            if not c_cols and "Name" in cols:
                meta["HGVS.c"] = (
                    meta["Name"]
                    .astype(str)
                    .str.extract(_HGVS_RX_C_REAL, expand=False)
                    .fillna("")
                )
                c_cols = ["HGVS.c"]
            for c in cols:
                if c in c_cols:
                    meta[c] = meta[c].fillna("").astype(str)
                else:
                    meta[c] = ""

        elif mode == "hgvsp_only":
            # Keep p. columns; blank c. and genes.
            p_cols = [c for c in cols if c in ("HGVS.p", "HGVSp", "ProteinChange")]
            if not p_cols and "Name" in cols:
                meta["HGVS.p"] = (
                    meta["Name"]
                    .astype(str)
                    .str.extract(_HGVS_RX_P_REAL, expand=False)
                    .fillna("")
                )
                p_cols = ["HGVS.p"]
            for c in cols:
                if c in p_cols:
                    meta[c] = meta[c].fillna("").astype(str)
                else:
                    meta[c] = ""

        else:
            # Unknown mode → no-op
            pass

        if debug:
            _debug_dump_sources(meta_before, meta[cols], tag=mode, cols=cols)

        yield

    finally:
        # Restore originals
        for c, ser in originals.items():
            meta[c] = ser


# ======================
# Labels for printing
# ======================
PROMPT_MODES = [
    ("prompt_blank", "prompt blank"),
    ("prompt_shuffled", "prompt shuffled"),
    ("gene_only", "gene only"),
    ("hgvsc_only", "HGVS.c only"),
    ("hgvsp_only", "HGVS.p only"),
]

COND_MODES = [
    ("both", "dna+go"),  # baseline
    ("dna_only", "dna only"),
    ("go_only", "go only"),
    ("prompt_only", "prompt only"),
    # You can also add ('go_shuffled','go shuffled') / ('dna_shuffled','dna shuffled')
]


# ======================
# Runner
# ======================
def run_prompt_and_cond_ablation(
    seq_ds: Dict[str, Any],
    out_dir: str = "artifacts/qwen3_hpo_lora",  # keep pointing to your single model dir
    split: str = "val",
    max_n: int = 2000,
    batch_size: int = 64,
    go_shuffle_seeds=(0,),
    prompt_shuffle_seeds=(0,),
    debug_prompts: bool = True,
):
    rows = []
    print(
        f"\n=== Ablation on split: {split} (max_n={max_n}, batch_size={batch_size}) ==="
    )

    # 1) Conditioning ablations (no prompt tampering)
    for mode, label in tqdm(COND_MODES, desc="Cond modes", unit="mode"):
        view = make_ablation_seq_ds(seq_ds, mode=mode, seed=0)
        # Keep out_dir constant so adapter is found at <out_dir>/adapter
        out = evaluate_split_batched(
            view,
            out_dir=out_dir,
            split=split,
            max_n=max_n,
            batch_size=batch_size,
            sample_rationales=0,
        )
        rows.append(
            {
                "group": "cond",
                "mode": label,
                "seed": None,
                "n": out["n"],
                "acc": out["accuracy"],
                "f1": out["f1"],
                "roc": out.get("auc_roc", float("nan")),
                "pr": out.get("auc_pr", float("nan")),
            }
        )

    # # Optional: GO shuffled / DNA shuffled (uncomment if you want them)
    # for s in tqdm(go_shuffle_seeds, desc="GO shuffled seeds", unit="seed"):
    #     view = make_ablation_seq_ds(seq_ds, mode="go_shuffled", seed=int(s))
    #     out = evaluate_split_batched(
    #         view, out_dir=out_dir, split=split, max_n=max_n,
    #         batch_size=batch_size, sample_rationales=0
    #     )
    #     rows.append({"group":"cond","mode":"go shuffled","seed":int(s),
    #                  "n":out["n"],"acc":out["accuracy"],"f1":out["f1"],
    #                  "roc":out.get("auc_roc",float("nan")),
    #                  "pr":out.get("auc_pr",float("nan"))})

    # for s in tqdm(go_shuffle_seeds, desc="DNA shuffled seeds", unit="seed"):
    #     view = make_ablation_seq_ds(seq_ds, mode="dna_shuffled", seed=int(s))
    #     out = evaluate_split_batched(
    #         view, out_dir=out_dir, split=split, max_n=max_n,
    #         batch_size=batch_size, sample_rationales=0
    #     )
    #     rows.append({"group":"cond","mode":"dna shuffled","seed":int(s),
    #                  "n":out["n"],"acc":out["accuracy"],"f1":out["f1"],
    #                  "roc":out.get("auc_roc",float("nan")),
    #                  "pr":out.get("auc_pr",float("nan"))})

    # 2) Prompt ablations (rewrite actual prompt source fields on the requested split)
    for mode, label in tqdm(PROMPT_MODES, desc="Prompt modes", unit="mode"):
        if mode == "prompt_shuffled":
            for s in prompt_shuffle_seeds:
                with _temp_prompt_mode(
                    seq_ds[split], mode=mode, seed=int(s), debug=debug_prompts
                ):
                    view = make_ablation_seq_ds(seq_ds, mode="both", seed=0)
                    out = evaluate_split_batched(
                        view,
                        out_dir=out_dir,  # same model dir
                        split=split,
                        max_n=max_n,
                        batch_size=batch_size,
                        sample_rationales=0,
                    )
                rows.append(
                    {
                        "group": "prompt",
                        "mode": label,
                        "seed": int(s),
                        "n": out["n"],
                        "acc": out["accuracy"],
                        "f1": out["f1"],
                        "roc": out.get("auc_roc", float("nan")),
                        "pr": out.get("auc_pr", float("nan")),
                    }
                )
        else:
            with _temp_prompt_mode(
                seq_ds[split], mode=mode, seed=0, debug=debug_prompts
            ):
                view = make_ablation_seq_ds(seq_ds, mode="both", seed=0)
                out = evaluate_split_batched(
                    view,
                    out_dir=out_dir,  # same model dir
                    split=split,
                    max_n=max_n,
                    batch_size=batch_size,
                    sample_rationales=0,
                )
            rows.append(
                {
                    "group": "prompt",
                    "mode": label,
                    "seed": None,
                    "n": out["n"],
                    "acc": out["accuracy"],
                    "f1": out["f1"],
                    "roc": out.get("auc_roc", float("nan")),
                    "pr": out.get("auc_pr", float("nan")),
                }
            )

    # ---- Pretty print ----
    print(f"\nAblation summary [{split}]")
    print(
        f"{'Group':8} {'Mode':18} {'Seed':>5} {'N':>6} {'Acc':>8} {'F1':>8} {'ROC-AUC':>10} {'PR-AUC':>10}"
    )
    print("-" * 76)
    for r in rows:
        print(
            f"{r['group']:8} {r['mode']:18} {str(r['seed'] if r['seed'] is not None else '-'):>5} "
            f"{int(r['n']):6d} {float(r['acc']):8.4f} {float(r['f1']):8.4f} "
            f"{float(r['roc']):10.4f} {float(r['pr']):10.4f}"
        )

    # Δ vs dna+go baseline (conditioning group)
    try:
        ref = next(
            rr for rr in rows if rr["group"] == "cond" and rr["mode"] == "dna+go"
        )
        print("\nΔ vs dna+go (conditioning ablations):")
        for r in rows:
            if r["group"] != "cond" or r["mode"] == "dna+go":
                continue
            print(
                f"- {r['mode']:18} (seed={r['seed']})  "
                f"ΔAcc={r['acc']-ref['acc']:+.4f}  ΔF1={r['f1']-ref['f1']:+.4f}  "
                f"ΔROC-AUC={r['roc']-ref['roc']:+.4f}  ΔPR-AUC={r['pr']-ref['pr']:+.4f}"
            )
    except StopIteration:
        pass

    # Δ vs dna+go for prompt ablations
    try:
        ref = next(
            rr for rr in rows if rr["group"] == "cond" and rr["mode"] == "dna+go"
        )
        print("\nΔ vs dna+go (prompt ablations):")
        for r in rows:
            if r["group"] != "prompt":
                continue
            print(
                f"- {r['mode']:18} (seed={r['seed']})  "
                f"ΔAcc={r['acc']-ref['acc']:+.4f}  ΔF1={r['f1']-ref['f1']:+.4f}  "
                f"ΔROC-AUC={r['roc']-ref['roc']:+.4f}  ΔPR-AUC={r['pr']-ref['pr']:+.4f}"
            )
    except StopIteration:
        pass

    return rows


# Example:
# rows = run_prompt_and_cond_ablation(
#     seq_ds, split="val", max_n=1500, batch_size=64,
#     go_shuffle_seeds=(0,1), prompt_shuffle_seeds=(0,1), debug_prompts=True
# )
