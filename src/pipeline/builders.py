from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np

from src.dna_utils import load_fasta, process_and_cache_dna
from src.protein_utils import process_and_cache_protein
from vep_pipeline import run_vep_pipeline

from .config import PipelineConfig
from .manifest import SplitArtifact


def _ensure_variation_key(df: pd.DataFrame, *, column: str = "variation_key") -> pd.DataFrame:
    if column in df.columns:
        return df
    if "VariationID" not in df.columns:
        raise ValueError("ClinVar metadata frame is missing 'VariationID' column")
    result = df.copy()
    result[column] = df["VariationID"].astype("string").fillna("")
    return result


def build_dna_caches(
    cfg: PipelineConfig,
    splits: Dict[str, pd.DataFrame],
    device: str,
) -> Dict[str, SplitArtifact]:
    """Build DNA embeddings and update ``splits`` in-place.

    The incoming ``splits`` mapping is mutated to hold the enriched metadata
    frames that now include ``dna_embedding_idx``/``protein_embedding_idx``.
    The returned dictionary only captures the serialized artifacts written to
    disk, which later feed into the manifest.
    """
    dna_dir = cfg.paths.artifacts / "dna"
    dna_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = cfg.paths.artifacts / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dna] output dir: {dna_dir}")

    fasta = load_fasta(str(cfg.paths.fasta))
    artifacts: Dict[str, SplitArtifact] = {}
    try:
        for split, df in splits.items():
            meta_path = meta_dir / f"clinvar_{split}.feather"
            archive_path = dna_dir / f"dna_{split}_eff_fp16.h5"
            windows_path = dna_dir / f"dna_{split}_windows.feather"
            print(
                f"[dna] split={split} input_rows={len(df)} meta={meta_path} archive={archive_path}"
            )
            kept_df, _ = process_and_cache_dna(
                df,
                fasta,
                window=cfg.dna.window,
                batch_size=cfg.dna.batch_size,
                max_length=cfg.dna.max_length,
                out_meta=str(meta_path),
                out_h5=str(archive_path),
                windows_meta=str(windows_path),
                device=device,
                limit=cfg.dna.limit,
                dna_model_id=cfg.dna.model_id,
                force_windows=cfg.dna.force_windows or cfg.run.force_dna_windows,
                force_embeddings=cfg.dna.force_embeddings
                or cfg.run.force_dna_embeddings,
            )
            kept_df = _ensure_variation_key(kept_df)
            if "dna_embedding_idx" not in kept_df.columns:
                kept_df["dna_embedding_idx"] = np.arange(
                    len(kept_df), dtype=np.int32
                )
            kept_df = kept_df.reset_index(drop=True)
            kept_df.to_feather(meta_path)

            artifacts[split] = SplitArtifact(
                meta=str(meta_path),
                dna_h5=str(archive_path),
                extras={
                    "dna_rows": int(len(kept_df)),
                    "dna_input_rows": int(len(df)),
                    "dna_windows": str(windows_path),
                },
            )
            splits[split] = kept_df
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
    vep_tables: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, SplitArtifact]:
    """Build protein embeddings and merge them into the shared ``splits``.

    This stage assumes ``build_dna_caches`` already populated ``splits`` with
    DNA indices. Each split is enriched with protein annotations before the
    consolidated frame is written back to disk.
    """
    protein_dir = cfg.paths.artifacts / "protein"
    protein_dir.mkdir(parents=True, exist_ok=True)
    vep_root = cfg.paths.artifacts / "vep"
    vep_root.mkdir(parents=True, exist_ok=True)
    meta_dir = cfg.paths.artifacts / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    if split_paths is None:
        splits_dir = cfg.paths.artifacts / "splits"
        split_paths = {
            name: splits_dir / f"clinvar_{name}.feather" for name in splits.keys()
        }

    artifacts: Dict[str, SplitArtifact] = {}
    for split, base_df in splits.items():
        inp_path = split_paths.get(split)
        if inp_path is None:
            inp_path = cfg.paths.artifacts / "splits" / f"clinvar_{split}.feather"
        inp_path.parent.mkdir(parents=True, exist_ok=True)
        if not inp_path.exists():
            base_df.reset_index(drop=True).to_feather(inp_path)
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
                    use_docker=cfg.protein.use_docker,
                    filter_mode=cfg.protein.filter_mode,
                    chunk_size=cfg.protein.chunk_size,
                    jobs=cfg.protein.jobs,
                    vep_fork=cfg.protein.vep_fork,
                )

        protein_meta_path = protein_dir / f"protein_{split}.feather"
        archive_path = protein_dir / f"protein_{split}_eff_fp16.h5"
        preloaded = vep_tables.get(split) if vep_tables else None
        if preloaded is not None:
            print(
                f"[protein] using cached VEP dataframe for {split}: rows={len(preloaded)}"
            )

        kept_df, _ = process_and_cache_protein(
            preloaded if preloaded is not None else combined_feather,
            out_meta=str(protein_meta_path),
            out_h5=str(archive_path),
            device=device,
            model_id=cfg.protein.model_id,
            batch_size=cfg.protein.batch_size,
            window=cfg.protein.window,
            max_length=cfg.protein.max_length,
            force_embeddings=cfg.protein.force_embeddings
            or cfg.run.force_protein_embeddings,
        )
        final_meta_path = meta_dir / f"clinvar_{split}.feather"
        final_meta_path.parent.mkdir(parents=True, exist_ok=True)

        base_df = _ensure_variation_key(base_df)
        prot_df = kept_df.rename(
            columns={
                "seq_wt": "protein_seq_wt",
                "seq_mt": "protein_seq_mt",
                "consequence_terms": "protein_consequence_terms",
            }
        ).copy()
        prot_df["variation_key"] = kept_df["id"].astype("string").fillna("")

        merge_cols = [
            "protein_embedding_idx",
            "protein_seq_wt",
            "protein_seq_mt",
            "protein_consequence_terms",
        ]
        available_cols = [c for c in merge_cols if c in prot_df.columns]
        drop_cols = [c for c in available_cols if c in base_df.columns]
        if drop_cols:
            base_df = base_df.drop(columns=drop_cols)
        merged = base_df.merge(
            prot_df[["variation_key", *available_cols]],
            how="left",
            on="variation_key",
        )

        if "protein_embedding_idx" in merged.columns:
            merged["protein_embedding_idx"] = merged["protein_embedding_idx"].fillna(-1).astype(
                np.int32
            )
        else:
            merged["protein_embedding_idx"] = np.full(len(merged), -1, dtype=np.int32)

        merged = merged.reset_index(drop=True)
        merged.to_feather(final_meta_path)
        base_df = merged
        print(
            f"[protein] merged features for {split}: meta={final_meta_path} rows={len(base_df)}"
        )

        covered = int((base_df["protein_embedding_idx"] >= 0).sum())
        artifacts[split] = SplitArtifact(
            meta=str(final_meta_path),
            protein_h5=str(archive_path),
            extras={
                "protein_rows": int(len(kept_df)),
                "protein_aligned": covered,
                "vep_output": str(combined_feather),
                "protein_meta_cache": str(protein_meta_path),
            },
        )
        splits[split] = base_df
    return artifacts
