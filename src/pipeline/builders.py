from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.dna_utils import load_fasta, process_and_cache_dna
from src.protein_utils import process_and_cache_protein
from vep_pipeline import run_vep_pipeline

from .config import PipelineConfig
from .manifest import SplitArtifact


def build_dna_caches(
    cfg: PipelineConfig,
    splits: Dict[str, pd.DataFrame],
    device: str,
) -> Dict[str, SplitArtifact]:
    dna_dir = cfg.paths.artifacts / "dna"
    dna_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dna] output dir: {dna_dir}")

    fasta = load_fasta(str(cfg.paths.fasta))
    artifacts: Dict[str, SplitArtifact] = {}
    try:
        for split, df in splits.items():
            meta_path = dna_dir / f"dna_{split}.feather"
            npz_path = dna_dir / f"dna_{split}_eff_fp16.npz"
            windows_path = dna_dir / f"dna_{split}_windows.feather"
            print(
                f"[dna] split={split} input_rows={len(df)} meta={meta_path} npz={npz_path}"
            )
            kept_df, _ = process_and_cache_dna(
                df,
                fasta,
                window=cfg.dna.window,
                batch_size=cfg.dna.batch_size,
                pool=cfg.dna.pool,
                max_length=cfg.dna.max_length,
                out_meta=str(meta_path),
                out_npz=str(npz_path),
                windows_meta=str(windows_path),
                device=device,
                limit=cfg.dna.limit,
                dna_model_id=cfg.dna.model_id,
                force_windows=cfg.dna.force_windows or cfg.run.force_dna_windows,
                force_embeddings=cfg.dna.force_embeddings
                or cfg.run.force_dna_embeddings,
            )
            artifacts[split] = SplitArtifact(
                dna_meta=str(meta_path),
                dna_npz=str(npz_path),
                extras={
                    "dna_rows": int(len(kept_df)),
                    "dna_input_rows": int(len(df)),
                    "dna_windows": str(windows_path),
                },
            )
    finally:
        close_fn = getattr(fasta, "close", None)
        if callable(close_fn):
            close_fn()
    return artifacts


def build_protein_caches(
    cfg: PipelineConfig,
    splits: Dict[str, pd.DataFrame],
    device: str,
    *,
    split_paths: Optional[Dict[str, Path]] = None,
    vep_paths: Optional[Dict[str, Path]] = None,
) -> Dict[str, SplitArtifact]:
    if not cfg.protein.enabled:
        print("[protein] disabled in config; skipping protein caches")
        return {}

    protein_dir = cfg.paths.artifacts / "protein"
    protein_dir.mkdir(parents=True, exist_ok=True)
    vep_root = cfg.paths.artifacts / "vep"
    vep_root.mkdir(parents=True, exist_ok=True)

    if split_paths is None:
        splits_dir = cfg.paths.artifacts / "splits"
        split_paths = {
            name: splits_dir / f"clinvar_{name}.feather" for name in splits.keys()
        }

    artifacts: Dict[str, SplitArtifact] = {}
    for split, df in splits.items():
        inp_path = split_paths.get(split)
        if inp_path is None:
            inp_path = cfg.paths.artifacts / "splits" / f"clinvar_{split}.feather"
        inp_path.parent.mkdir(parents=True, exist_ok=True)
        if not inp_path.exists():
            df.reset_index(drop=True).to_feather(inp_path)
            print(f"[protein] wrote input feather for {split}: {inp_path}")
        else:
            print(f"[protein] using input feather for {split}: {inp_path}")

        vep_dir = vep_root / split
        vep_dir.mkdir(parents=True, exist_ok=True)
        provided = Path(vep_paths[split]) if vep_paths and split in vep_paths else None
        if provided is not None:
            combined_feather = provided
            if not combined_feather.exists():
                raise FileNotFoundError(
                    f"VEP output for split '{split}' not found at {combined_feather}"
                )
            print(f"[protein] using existing VEP output for {split}: {combined_feather}")
        else:
            combined_feather = vep_dir / "vep_combined.feather"
            if combined_feather.exists() and not cfg.protein.force_vep:
                print(f"[protein] reuse VEP output for {split}: {combined_feather}")
            else:
                print(f"[protein] running VEP for {split} → {vep_dir}")
                combined_feather = run_vep_pipeline(
                    input_path=Path(inp_path),
                    host_cache_dir=Path(cfg.protein.vep_cache_dir),
                    fasta_relpath=str(cfg.protein.vep_fasta_relpath),
                    out_dir=vep_dir,
                    image=cfg.protein.image,
                    filter_mode=cfg.protein.filter_mode,
                    chunk_size=cfg.protein.chunk_size,
                    jobs=cfg.protein.jobs,
                    vep_fork=cfg.protein.vep_fork,
                )

        meta_path = protein_dir / f"protein_{split}.feather"
        npz_path = protein_dir / f"protein_{split}_eff_fp16.npz"
        kept_df, _ = process_and_cache_protein(
            combined_feather,
            out_meta=str(meta_path),
            out_npz=str(npz_path),
            device=device,
            model_id=cfg.protein.model_id,
            batch_size=cfg.protein.batch_size,
            pool=cfg.protein.pool,
            max_length=cfg.protein.max_length,
            force_embeddings=cfg.protein.force_embeddings
            or cfg.run.force_protein_embeddings,
        )
        artifacts[split] = SplitArtifact(
            protein_meta=str(meta_path),
            protein_npz=str(npz_path),
            extras={
                "protein_rows": int(len(kept_df)),
                "vep_output": str(combined_feather),
            },
        )
    return artifacts
