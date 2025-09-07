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
from src.text_utils import process_and_cache_text, SeqTextPairDataset, collate_pair
from src.clip_trainer import SeqTextCLIP, train_epoch
from src.clip_eval import evaluate_clip_fast

from util import get_device

DATA_DIR = Path("data")
OUT_DIR = Path("artifacts")
FASTA_PATH = DATA_DIR / "raw" / "GCF_000001405.38_GRCh38.p12_genomic.fna"
CLINVAR_PATH = DATA_DIR / "raw" / "variant_summary.txt.gz"
CLINVAR_CACHE_MARKER = OUT_DIR / "_cached_clinvar.ok"
GO_NPZ = "data/processed/go_n2v_genes.npz"


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
    text_max_len: int = 128
    text_pool: str = "cls"
    train_bs_pair: int = 512
    train_bs_seq: int = 1024
    num_workers: int = 2
    epochs: int = 1
    lr: float = 3e-4
    wd: float = 5e-2
    amp_dtype = torch.bfloat16
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


def _text_paths(cfg: RunCfg, split: str) -> tuple[str, str]:
    return (
        str(cfg.out_dir / f"text_{split}.feather"),
        str(cfg.out_dir / f"text_{split}_fp16.npz"),
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


def build_text_caches(cfg: RunCfg):
    """
    Uses dna_*.feather metas to build text captions + embeddings.
    Returns {split: (text_meta_df, text_npz_path)}
    """
    out = {}
    for name in ("train", "val", "test"):
        dna_meta, _ = _dna_paths(cfg, name)
        text_meta, text_npz = _text_paths(cfg, name)
        df_txt, npz_txt = process_and_cache_text(
            dna_meta,
            out_meta=text_meta,
            out_npz=text_npz,  # <- use text_npz here
            max_length=cfg.text_max_len,
            batch_size=512,
            pool=cfg.text_pool,
            force_captions=cfg.force_caps,
            force_embeddings=cfg.force_txt_emb,
        )
        out[name] = (df_txt, npz_txt)
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


def build_pair_datasets(seq_ds: dict[str, "SequenceTowerDataset"], cfg: RunCfg):
    pair = {}
    for name in ("train", "val", "test"):
        text_meta, text_npz = _text_paths(cfg, name)
        pair[name] = SeqTextPairDataset(
            seq_dataset=seq_ds[name],
            text_meta_feather=text_meta,
            text_npz=text_npz,
            label_col="_y",
        )
    return pair


def loaders_for_pair(pair_ds: dict[str, "SeqTextPairDataset"], cfg: RunCfg):
    return {
        "train": DataLoader(
            pair_ds["train"],
            batch_size=cfg.train_bs_pair,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_pair,
        ),
        "val": DataLoader(
            pair_ds["val"],
            batch_size=cfg.train_bs_pair,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_pair,
        ),
        "test": DataLoader(
            pair_ds["test"],
            batch_size=cfg.train_bs_pair,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_pair,
        ),
    }


def main():
    # --- setup ---
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = get_device()
    cfg = RunCfg(
        out_dir=OUT_DIR,
        fasta_path=FASTA_PATH,
        go_npz=GO_NPZ,
        text_pool="mean",  # try mean for HGVS-heavy captions; "cls" also fine
        epochs=1,
        force_caps=False,
        force_txt_emb=False,
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

    # --- 3) Text caches (row-aligned to dna metas) ---
    _ = build_text_caches(cfg)

    # --- 4) Datasets & loaders ---
    seq_ds = build_seq_datasets(cfg)
    loaders_seq = loaders_for_seq(seq_ds, cfg)

    pair_ds = build_pair_datasets(seq_ds, cfg)
    loaders_pair = loaders_for_pair(pair_ds, cfg)

    print("Ready:")
    D_seq = seq_ds["train"].D_eff + seq_ds["train"].D_go
    D_txt = pair_ds["train"].D_txt
    print(f"  D_seq = {D_seq}  D_txt = {D_txt}")
    print(
        f"  N (train/val/test) = {len(seq_ds['train'])} {len(seq_ds['val'])} {len(seq_ds['test'])}"
    )

    # --- 5) Train CLIP heads ---
    model = SeqTextCLIP(
        D_seq, D_txt, d_out=512, hidden=1024, p=0.2
    )  # freeze_tau default True
    best = train_epoch(
        loaders_pair["train"],
        loaders_pair["val"],
        model,
        epochs=cfg.epochs,
        lr=cfg.lr,
        wd=cfg.wd,
        loss_mode="ce",  # explicit; switch to "supcon" if you want
    )
    torch.save(best["state"], cfg.out_dir / "clip_seqtext.pt")
    print("best val loss:", best["val_loss"])

    # --- 6) Eval (fast) ---
    val_metrics = evaluate_clip_fast(
        model, loaders_pair["val"], device=device, k_list=(1, 5, 10, 50)
    )
    print("VAL:", val_metrics)


if __name__ == "__main__":
    main()
