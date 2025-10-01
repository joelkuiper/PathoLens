from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from src.seq_dataset import SequenceTowerDataset

from .config import ConfigError, PipelineConfig, load_pipeline_config
from .manifest import PipelineManifest


def build_sequence_datasets(
    cfg: PipelineConfig,
    manifest: PipelineManifest,
    *,
    make_label: bool = True,
    label_col: str = "_y",
) -> Dict[str, SequenceTowerDataset]:
    datasets: Dict[str, SequenceTowerDataset] = {}
    go_npz = manifest.go_npz
    for split, artifact in manifest.splits.items():
        if not artifact.meta or not artifact.dna_h5:
            raise ValueError(f"Manifest entry for '{split}' missing DNA artifacts")
        if not artifact.protein_h5:
            raise ValueError(
                f"Manifest entry for '{split}' missing protein artifacts"
            )
        frame = pd.read_feather(artifact.meta)

        ds = SequenceTowerDataset(
            meta_df=frame,
            dna_h5=artifact.dna_h5,
            go_npz=go_npz,
            make_label=make_label,
            label_col=label_col,
            protein_h5=artifact.protein_h5,
            go_normalize=cfg.go.normalize,
            go_uppercase=cfg.go.uppercase_keys,
        )
        datasets[split] = ds
        total_dim = ds.dna_dim + ds.go_dim + ds.protein_dim
        print(
            f"[dataset] {split}: rows={len(ds)} dim(dna={ds.dna_dim}, go={ds.go_dim}, "
            f"protein={ds.protein_dim}) total={total_dim}"
        )
        print(
            f"[dataset] {split}: GO_hit_rate={ds.go_hit_rate:.3f} "
            f"protein_coverage={ds.protein_coverage:.3f}"
        )
    return datasets


def load_manifest_datasets(
    config_path: str | Path,
    *,
    manifest_path: Optional[str | Path] = None,
    make_label: bool = True,
    label_col: str = "_y",
) -> Tuple[PipelineConfig, PipelineManifest, Dict[str, SequenceTowerDataset]]:
    """Load config + manifest and build datasets in a single convenience call."""

    cfg = load_pipeline_config(config_path)
    manifest_location = manifest_path or cfg.run.manifest_path
    if manifest_location is None:
        raise ConfigError(
            "Manifest path is undefined; set Run.manifest in the TOML or pass "
            "manifest_path explicitly."
        )
    manifest = PipelineManifest.load(manifest_location)
    datasets = build_sequence_datasets(
        cfg,
        manifest,
        make_label=make_label,
        label_col=label_col,
    )
    return cfg, manifest, datasets
