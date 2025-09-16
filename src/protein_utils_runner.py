# src/protein_utils_runner.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict
import pandas as pd

from vep_pipeline import run_vep_pipeline
from src.protein_utils import (
    process_and_cache_protein,
)  # embeds and writes prot meta/npz


def _protein_paths(out_dir: Path, split: str) -> tuple[str, str]:
    meta = str(out_dir / f"protein_{split}.feather")
    npz = str(out_dir / f"protein_{split}_eff_fp16.npz")
    return meta, npz


def _split_input_path(out_dir: Path, split: str) -> str:
    return str(out_dir / f"clinvar_{split}.feather")


def _vep_out_dir(out_dir: Path, split: str) -> Path:
    return out_dir / f"out_vep_{split}"


def _write_split(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_feather(path)


def _run_vep_pipeline(
    input_feather: str | Path,
    host_cache_dir: str | Path,
    fasta_relpath: str,
    out_dir: str | Path,
    *,
    image: str = "ensemblorg/ensembl-vep",
    filter_mode: str = "protein_changing",
    chunk_size: int = 1000,
    jobs: int = 6,
    vep_fork: int = 0,
) -> Path:
    """In-process VEP run with a cache guard."""
    out_dir = Path(out_dir)
    combined_parquet = out_dir / "vep_combined.parquet"
    if combined_parquet.exists():
        print(f"[vep_pipeline] Skipping, found existing {combined_parquet}")
        return combined_parquet

    return run_vep_pipeline(
        Path(input_feather),
        Path(host_cache_dir),
        fasta_relpath,
        out_dir,
        image=image,
        filter_mode=filter_mode,
        chunk_size=chunk_size,
        jobs=jobs,
        vep_fork=vep_fork,
    )


def build_protein_caches(
    out_dir: Path,
    device: str,
    splits: Dict[str, pd.DataFrame],
    *,
    vep_cache_dir: str,
    vep_fasta_relpath: str,
    image: str = "ensemblorg/ensembl-vep",
    filter_mode: str = "protein_changing",  # "patchable" / "all"
    chunk_size: int = 1000,
    jobs: int = 6,
    vep_fork: int = 0,
    esm_model_id: str = "facebook/esm2_t33_650M_UR50D",
    bs_cuda: int = 16,
    bs_cpu: int = 4,
    max_len: int = 2048,
    pool: str = "mean",
    force_embeddings: bool = False,
) -> Dict[str, Dict[str, str]]:
    """
    Build (or reuse) protein artifacts for each split and return a manifest:
      { split: { "meta": <protein_meta_feather>, "npz": <protein_npz> } }

    Pipeline per split:
      - write <out_dir>/clinvar_<split>.feather
      - run VEP into <out_dir>/out_vep_<split> (guarded if already done)
      - embed WT/MT effect vectors and write:
          <out_dir>/protein_<split>.feather
          <out_dir>/protein_<split>_eff_fp16.npz ('prot_eff')
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Dict[str, str]] = {}
    bs = bs_cuda if device == "cuda" else bs_cpu

    for split, df in splits.items():
        # Input for VEP
        inp = _split_input_path(out_dir, split)
        _write_split(df, inp)

        # VEP out dir
        vep_out = _vep_out_dir(out_dir, split)
        vep_out.mkdir(parents=True, exist_ok=True)

        # Run (or reuse) VEP
        combined_parquet = _run_vep_pipeline(
            input_feather=inp,
            host_cache_dir=vep_cache_dir,
            fasta_relpath=vep_fasta_relpath,
            out_dir=str(vep_out),
            image=image,
            filter_mode=filter_mode,
            chunk_size=chunk_size,
            jobs=jobs,
            vep_fork=vep_fork,
        )

        # Embed + save protein effect vectors
        meta_out, npz_out = _protein_paths(out_dir, split)
        _kept_df, npz_path = process_and_cache_protein(
            combined_parquet,
            out_meta=meta_out,
            out_npz=npz_out,
            device=device,
            model_id=esm_model_id,
            batch_size=bs,
            max_length=max_len,
            pool=pool,
            force_embeddings=force_embeddings,
        )

        manifest[split] = {"meta": meta_out, "npz": npz_path}

    # Human-readable summary
    print("[PROTEIN manifest]")
    for k, v in manifest.items():
        print(f"  {k}: meta={v['meta']}  npz={v['npz']}")
    return manifest
