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
    make_label: bool = True,
    label_col: str = "_y",
) -> Dict[str, SequenceTowerDataset]:
    datasets: Dict[str, SequenceTowerDataset] = {}
    go_npz = manifest.go_npz
    for split, artifact in manifest.splits.items():
        if not artifact.meta:
            raise ValueError(f"Manifest entry for '{split}' missing metadata path")
        frame = pd.read_feather(artifact.meta)

        ds = SequenceTowerDataset(
            meta_df=frame,
            go_npz=go_npz,
            make_label=make_label,
            label_col=label_col,
            go_normalize=cfg.go.normalize,
            go_uppercase=cfg.go.uppercase_keys,
            dna_model_id=cfg.dna.model_id,
            dna_model_revision=cfg.dna.revision,
            dna_max_length=cfg.dna.max_length,
            dna_batch_size=cfg.dna.batch_size,
            protein_model_id=cfg.protein.model_id,
            protein_model_revision=cfg.protein.revision,
            protein_max_length=cfg.protein.max_length,
            protein_batch_size=cfg.protein.batch_size,
            device=cfg.run.device,
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
