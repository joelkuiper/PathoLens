from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.clinvar import load_clinvar_variants
from .config import PipelineConfig


SplitDict = Dict[str, pd.DataFrame]
PathDict = Dict[str, Path]


def _split_by_gene(
    df: pd.DataFrame,
    *,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "GeneSymbol" not in df.columns or "_y" not in df.columns:
        missing = [c for c in ("GeneSymbol", "_y") if c not in df.columns]
        raise ValueError(f"ClinVar frame missing required columns: {missing}")

    rng = np.random.default_rng(random_state)

    gene2label: Dict[str, int] = {}
    for gene, sub in df.groupby("GeneSymbol"):
        y = sub["_y"].astype(int)
        gene2label[gene] = int(round(y.mean()))

    genes = np.array(list(gene2label.keys()))
    labels = np.array(list(gene2label.values()))

    order = rng.permutation(len(genes))
    genes, labels = genes[order], labels[order]

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

    test_df = df[df["GeneSymbol"].isin(test_genes)].reset_index(drop=True)
    val_df = df[df["GeneSymbol"].isin(val_genes)].reset_index(drop=True)
    train_df = df[df["GeneSymbol"].isin(train_genes)].reset_index(drop=True)
    return train_df, val_df, test_df


def prepare_clinvar_splits(
    cfg: PipelineConfig,
    *,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[SplitDict, PathDict]:
    splits_dir = cfg.paths.artifacts / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_names = ("train", "val", "test")
    split_paths: PathDict = {
        name: splits_dir / f"clinvar_{name}.feather" for name in split_names
    }

    reuse = all(p.exists() for p in split_paths.values()) and not cfg.run.force_splits
    if reuse:
        print("[splits] reusing cached ClinVar splits")
        splits: SplitDict = {
            name: pd.read_feather(path) for name, path in split_paths.items()
        }
        for name, path in split_paths.items():
            print(f"  [{name}] {path} (rows={len(splits[name])})")
        return splits, split_paths

    print("[splits] generating ClinVar splits …")
    if not cfg.paths.clinvar.exists():
        raise FileNotFoundError(
            f"ClinVar input not found at {cfg.paths.clinvar}. Update Paths.clinvar."
        )

    df = load_clinvar_variants(str(cfg.paths.clinvar))
    train_df, val_df, test_df = _split_by_gene(
        df,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    splits = {"train": train_df, "val": val_df, "test": test_df}
    for name, frame in splits.items():
        path = split_paths[name]
        frame.reset_index(drop=True).to_feather(path)
        print(f"  [write] {name}: {path} (rows={len(frame)})")

    return splits, split_paths
