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
    for split, artifact in manifest.splits.items():
        if cfg.dna.enabled:
            if not artifact.meta or not artifact.dna_h5:
                raise ValueError(f"Manifest entry for '{split}' missing DNA artifacts")
            dna_h5 = artifact.dna_h5
        else:
            dna_h5 = None
        if cfg.protein.enabled:
            if not artifact.meta or not artifact.protein_h5:
                raise ValueError(
                    f"Manifest entry for '{split}' missing protein artifacts"
                )
            protein_h5 = artifact.protein_h5
        else:
            protein_h5 = None
        if dna_h5 is None and protein_h5 is None:
            raise ValueError(
                f"Manifest entry for '{split}' does not provide any enabled modalities"
            )
        frame = pd.read_feather(artifact.meta)

        ds = SequenceTowerDataset(
            meta_df=frame,
            dna_h5=dna_h5,
            make_label=make_label,
            label_col=label_col,
            protein_h5=protein_h5,
            use_dna=cfg.dna.enabled,
            use_protein=cfg.protein.enabled,
        )
        datasets[split] = ds
        spec = getattr(ds, "cond_spec", {})
        dna_info = spec.get("dna", {})
        prot_info = spec.get("protein", {})
        dna_status = "on" if dna_info.get("enabled") else "off"
        prot_status = "on" if prot_info.get("enabled") else "off"
        print(
            f"[dataset] {split}: rows={len(ds)} dna={dna_status}"
            f"(L={dna_info.get('seq_len', 0)} D={dna_info.get('embed_dim', 0)}) "
            f"protein={prot_status}"
            f"(L={prot_info.get('seq_len', 0)} D={prot_info.get('embed_dim', 0)})"
        )
        if ds.use_protein:
            print(f"[dataset] {split}: protein_coverage={ds.protein_coverage:.3f}")
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
