# src/train.py (cleaned)
from __future__ import annotations
import os

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.clinvar import load_clinvar_variants
from src.dna_utils import load_fasta, process_and_cache_dna
from src.protein_utils_runner import build_protein_caches
from src.seq_dataset import SequenceTowerDataset
from util import get_device

# ---------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------

DATA_DIR: Path = Path("data")
OUT_DIR: Path = Path("artifacts")
FASTA_PATH: Path = DATA_DIR / "raw" / "GCF_000001405.40_GRCh38.p14_genomic.fna"
CLINVAR_PATH: Path = DATA_DIR / "raw" / "variant_summary.txt.gz"
CLINVAR_CACHE_MARKER: Path = OUT_DIR / "_cached_clinvar.ok"
GO_NPZ: Path = DATA_DIR / "processed" / "go_n2v_genes.npz"


# ---------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------


def split_by_gene(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Gene-disjoint split (train/val/test) with rough class balance.

    Each gene is assigned a single label = majority of its variants' labels,
    and we stratify on that label at the gene level.

    Returns: (train_df, val_df, test_df)
    """
    if "GeneSymbol" not in df.columns or "_y" not in df.columns:
        missing = [c for c in ("GeneSymbol", "_y") if c not in df.columns]
        raise ValueError(f"split_by_gene: missing required columns: {missing}")

    rng = np.random.default_rng(random_state)

    # 1) Majority class per gene
    gene2label: Dict[str, int] = {}
    for g, sub in df.groupby("GeneSymbol"):
        y = sub["_y"].astype(int)
        label = int(round(y.mean()))
        gene2label[g] = label

    genes = np.array(list(gene2label.keys()))
    labels = np.array(list(gene2label.values()))

    # 2) Shuffle once
    idx = rng.permutation(len(genes))
    genes, labels = genes[idx], labels[idx]

    # 3) Per-class allocation for rough balance
    train_genes, val_genes, test_genes = [], [], []
    for cls in (0, 1):
        cls_genes = genes[labels == cls]
        rng.shuffle(cls_genes)
        n_cls = len(cls_genes)
        n_test = int(round(test_size * n_cls))
        n_val = int(round(val_size * n_cls))
        test_genes.extend(cls_genes[:n_test])
        val_genes.extend(cls_genes[n_test : n_test + n_val])
        train_genes.extend(cls_genes[n_test + n_val :])

    # 4) Project back to rows
    test_df = df[df["GeneSymbol"].isin(test_genes)].reset_index(drop=True)
    val_df = df[df["GeneSymbol"].isin(val_genes)].reset_index(drop=True)
    train_df = df[df["GeneSymbol"].isin(train_genes)].reset_index(drop=True)

    return train_df, val_df, test_df


def prepare_clinvar_splits(
    force: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load cached ClinVar splits if available; otherwise create, save, and mark cache.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_p = OUT_DIR / "clinvar_train.feather"
    val_p = OUT_DIR / "clinvar_val.feather"
    test_p = OUT_DIR / "clinvar_test.feather"

    if CLINVAR_CACHE_MARKER.exists() and not force:
        print(f"[cache] Using existing splits in {OUT_DIR}")
        train_df = pd.read_feather(train_p)
        val_df = pd.read_feather(val_p)
        test_df = pd.read_feather(test_p)
        return train_df, val_df, test_df

    print("[prepare] Generating new train/val/test splits...")
    if not CLINVAR_PATH.exists():
        raise FileNotFoundError(
            f"ClinVar file not found at {CLINVAR_PATH}. "
            "Update CLINVAR_PATH or download the required file."
        )

    df = load_clinvar_variants(str(CLINVAR_PATH))
    train_df, val_df, test_df = split_by_gene(df, test_size=0.1, val_size=0.1)

    train_df.to_feather(train_p)
    val_df.to_feather(val_p)
    test_df.to_feather(test_p)

    # Write marker last to only signal success after files are saved
    CLINVAR_CACHE_MARKER.write_text("ok\n")

    print(f"[prepare] Saved splits to {OUT_DIR}")
    return train_df, val_df, test_df


# ---------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------


@dataclass
class RunCfg:
    out_dir: Path
    fasta_path: Path
    go_npz: Path
    dna_window: int = 512
    dna_pool: str = "mean"
    dna_max_len: int = 512
    num_workers: int = 2
    train_bs_seq: int = 1024
    epochs: int = 1
    lr: float = 3e-4
    wd: float = 5e-2
    force_caps: bool = False
    force_txt_emb: bool = False
    force_dna: bool = False


# ---------------------------------------------------------------------
# DNA caches
# ---------------------------------------------------------------------


def _bs_by_device(device: str, cpu_bs: int, cuda_bs: int) -> int:
    """Small helper: pick a batch size based on device."""
    return cuda_bs if device == "cuda" else max(1, cpu_bs)


def _dna_paths(cfg: RunCfg, split: str) -> Tuple[str, str]:
    """Return (meta_feather_path, dna_npz_path) for a split."""
    return (
        str(cfg.out_dir / f"dna_{split}.feather"),
        str(cfg.out_dir / f"dna_{split}_eff_fp16.npz"),
    )


def build_dna_caches(
    cfg: RunCfg, device: str, fasta, splits: Dict[str, pd.DataFrame]
) -> Dict[str, Tuple[pd.DataFrame, str]]:
    """
    Build (or reuse) DNA windows + embeddings for each split.

    Returns: { split: (kept_df, dna_npz_path) }
    """
    out: Dict[str, Tuple[pd.DataFrame, str]] = {}
    bs = _bs_by_device(device, cpu_bs=8, cuda_bs=32)

    for name, df in splits.items():
        meta_out, npz_out = _dna_paths(cfg, name)
        kept, npz = process_and_cache_dna(
            df,
            fasta,
            window=cfg.dna_window,
            batch_size=bs,
            pool=cfg.dna_pool,
            max_length=cfg.dna_max_len,
            out_meta=meta_out,
            out_npz=npz_out,
            device=device,
            limit=None,
        )
        out[name] = (kept, npz)

    return out


# ---------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------


def build_seq_datasets(cfg: RunCfg) -> Dict[str, SequenceTowerDataset]:
    """
    Create SequenceTowerDataset per split (train/val/test) using DNA + GO.
    Protein features can be attached later via SequenceTowerDataset if desired.
    """
    seq: Dict[str, SequenceTowerDataset] = {}
    for name in ("train", "val", "test"):
        dna_meta, dna_npz = _dna_paths(cfg, name)
        ds = SequenceTowerDataset(
            meta_feather=dna_meta,
            dna_npz=dna_npz,
            go_npz=str(cfg.go_npz),
            make_label=True,
            label_col="_y",
        )
        seq[name] = ds
    return seq


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------


def main() -> None:
    # --- setup ---
    device = get_device()
    cfg = RunCfg(
        out_dir=OUT_DIR,
        fasta_path=FASTA_PATH,
        go_npz=GO_NPZ,
        epochs=1,
        force_dna=False,
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"OUT_DIR: {cfg.out_dir.resolve()}")

    # --- 0) splits ---
    train_df, val_df, test_df = prepare_clinvar_splits(force=False)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    print(f"Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # --- 1) DNA caches ---
    if not cfg.fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found at {cfg.fasta_path}")
    fasta = load_fasta(str(cfg.fasta_path))
    _ = build_dna_caches(cfg, device, fasta, splits)

    # --- 2) Protein caches ---
    # Skips expensive VEP if vep_combined outputs already exist (guard is in runner)
    _ = build_protein_caches(
        cfg,
        device,
        splits,
        vep_cache_dir=os.path.expanduser("~/vep_cache"),
        vep_fasta_relpath="homo_sapiens/115_GRCh38/Homo_sapiens.GRCh38.dna.toplevel.fa.gz",
        image="ensemblorg/ensembl-vep",
        filter_mode="all",  # "protein_changing" / "patchable" / "all"
        chunk_size=1000,
        jobs=6,
        vep_fork=0,
        esm_model_id="facebook/esm2_t12_35M_UR50D",  # smaller ESM2; faster
        bs_cuda=16,
        bs_cpu=8,
        max_len=2048,
        pool="mean",
    )

    # --- 3) Datasets ---
    seq_ds = build_seq_datasets(cfg)

    # --- 4) LLM (Qwen QLoRA) with conditioning on [DNA ⊕ GO] ---
    from src.llm.train import run_llm_pipeline
    import src.llm.eval as llm_eval

    llm_out_dir = str(OUT_DIR / "qwen3_hpo_lora")

    llm_res = run_llm_pipeline(
        out_dir=llm_out_dir,
        seq_ds=seq_ds,  # {"train","val","test"} SequenceTowerDataset
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        epochs=1,
        max_len=256,
        per_device_bs=16,
        grad_accum=8,
        lr=3e-4,
        balanced_sampling=True,  # uses WeightedRandomSampler
    )

    llm_eval.run_all(seq_ds, out_dir=llm_out_dir)
    print("[done] LLM pipeline complete.")


if __name__ == "__main__":
    main()
