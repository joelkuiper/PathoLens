from pathlib import Path
import numpy as np
import pandas as pd
import torch
from __future__ import annotations
import os
from dataclasses import dataclass
from torch.utils.data import DataLoader

from src.dna_utils import (
    load_fasta,
    process_and_cache_dna,
)
from src.clinvar import load_clinvar_variants
from src.seq_dataset import SequenceTowerDataset, collate_seq
from src.hp import load_hpo_id2label

from util import get_device

DATA_DIR = Path("data")
OUT_DIR = Path("artifacts")
FASTA_PATH = DATA_DIR / "raw" / "GCF_000001405.40_GRCh38.p14_genomic.fna"
CLINVAR_PATH = DATA_DIR / "raw" / "variant_summary.txt.gz"
CLINVAR_CACHE_MARKER = OUT_DIR / "_cached_clinvar.ok"
GO_NPZ = DATA_DIR / "processed" / "go_n2v_genes.npz"

def split_by_gene(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split ClinVar variants into train/val/test so that *all variants
    of a given gene* go to exactly one split (no gene leakage).

    In addition to gene-level disjointness, this function also tries
    to keep the splits roughly balanced between pathogenic and benign
    classes. To do that, each gene is assigned a single "class label"
    (the majority label across its variants), and the gene-level split
    is stratified on that label.
    """
    rng = np.random.default_rng(random_state)

    # assign each gene a label = majority class of its variants
    gene2label = {}
    for g, sub in df.groupby("GeneSymbol"):
        y = (
            sub["ClinicalSignificance"]
            .fillna("")
            .str.contains("Pathogenic", case=False)
            .astype(int)
        )
        label = int(round(y.mean()))
        gene2label[g] = label

    genes = np.array(list(gene2label.keys()))
    labels = np.array(list(gene2label.values()))

    # shuffle once
    idx = rng.permutation(len(genes))
    genes, labels = genes[idx], labels[idx]

    # per-class split to keep rough balance
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

    # map back to rows
    test_df = df[df["GeneSymbol"].isin(test_genes)].reset_index(drop=True)
    val_df = df[df["GeneSymbol"].isin(val_genes)].reset_index(drop=True)
    train_df = df[df["GeneSymbol"].isin(train_genes)].reset_index(drop=True)

    return train_df, val_df, test_df


def prepare_clinvar_splits(force: bool = False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if CLINVAR_CACHE_MARKER.exists() and not force:
        print(f"[cache] Using existing splits in {OUT_DIR}")
        train_df = pd.read_feather(OUT_DIR / "clinvar_train.feather")
        val_df = pd.read_feather(OUT_DIR / "clinvar_val.feather")
        test_df = pd.read_feather(OUT_DIR / "clinvar_test.feather")
        return train_df, val_df, test_df

    print("[prepare] Generating new train/val/test splits...")
    df = load_clinvar_variants(str(CLINVAR_PATH))
    train_df, val_df, test_df = split_by_gene(df, test_size=0.1, val_size=0.1)

    train_df.to_feather(OUT_DIR / "clinvar_train.feather")
    val_df.to_feather(OUT_DIR / "clinvar_val.feather")
    test_df.to_feather(OUT_DIR / "clinvar_test.feather")

    # write marker file
    CLINVAR_CACHE_MARKER.write_text("ok\n")

    print(f"[prepare] Saved splits to {OUT_DIR}")
    return train_df, val_df, test_df


@dataclass
class RunCfg:
    out_dir: Path
    fasta_path: Path
    go_npz: str
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


def _bs_by_device(device: str, cpu_bs: int, cuda_bs: int) -> int:
    return cuda_bs if device == "cuda" else max(1, cpu_bs // 4)


def _dna_paths(cfg: RunCfg, split: str) -> tuple[str, str]:
    return (
        str(cfg.out_dir / f"dna_{split}.feather"),
        str(cfg.out_dir / f"dna_{split}_eff_fp16.npz"),
    )


def build_dna_caches(
    cfg: RunCfg, device: str, fasta, splits: dict[str, "pd.DataFrame"]
):
    """
    Returns {split: (kept_df, dna_npz_path)}
    """
    out = {}
    bs = 32 if device == "cuda" else 8
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


def build_seq_datasets(cfg: RunCfg):
    seq = {}
    for name in ("train", "val", "test"):
        dna_meta, dna_npz = _dna_paths(cfg, name)
        ds = SequenceTowerDataset(
            meta_feather=dna_meta,
            dna_npz=dna_npz,
            go_npz=cfg.go_npz,
            make_label=True,
            label_col="_y",
        )
        seq[name] = ds
    return seq


def loaders_for_seq(seq_ds: dict[str, "SequenceTowerDataset"], cfg: RunCfg):
    return {
        "train": DataLoader(
            seq_ds["train"],
            batch_size=cfg.train_bs_seq,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_seq,
        ),
        "val": DataLoader(
            seq_ds["val"],
            batch_size=cfg.train_bs_seq,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_seq,
        ),
        "test": DataLoader(
            seq_ds["test"],
            batch_size=cfg.train_bs_seq,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_seq,
        ),
    }


def main():
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

    # --- 1) splits ---
    train_df, val_df, test_df = prepare_clinvar_splits(force=False)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    print(f"Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # --- 2) DNA caches ---
    fasta = load_fasta(str(cfg.fasta_path))
    _ = build_dna_caches(cfg, device, fasta, splits)

    # --- 3) Datasets & loaders ---
    seq_ds = build_seq_datasets(cfg)
    loaders_seq = loaders_for_seq(seq_ds, cfg)

  # --- 4) LLM: Qwen QLoRA with a virtual token conditioned on [DNA ⊕ GO] ---


  from src.llm import run_llm_pipeline

  llm_out_dir = str(OUT_DIR / "qwen3_hpo_lora")
  llm_res = run_llm_pipeline(
      out_dir=llm_out_dir,
      seq_ds=seq_ds,                # {"train","val","test"} SequenceTowerDataset
  )

  print("LLM eval:", llm_res["eval"])




if __name__ == "__main__":
    main()
