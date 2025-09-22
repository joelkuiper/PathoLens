from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from util import get_device

from src.clinvar import load_clinvar_variants
from src.pipeline import PipelineManifest, SplitArtifact, load_pipeline_config
from src.pipeline.builders import build_dna_caches, build_protein_caches
from src.pipeline.clinvar import _split_by_gene, prepare_clinvar_splits
from src.pipeline.datasets import build_sequence_datasets
from vep_pipeline import ensure_vep_annotations


def _merge_artifacts(
    dna_artifacts: Dict[str, SplitArtifact],
    protein_artifacts: Dict[str, SplitArtifact],
) -> Dict[str, SplitArtifact]:
    combined: Dict[str, SplitArtifact] = {k: v for k, v in dna_artifacts.items()}
    for split, part in protein_artifacts.items():
        if split not in combined:
            combined[split] = part
            continue
        existing = combined[split]
        if part.protein_meta:
            existing.protein_meta = part.protein_meta
        if part.protein_npz:
            existing.protein_npz = part.protein_npz
        existing.extras.update(part.extras)
    return combined


def _filter_splits_by_consequence(
    splits: Dict[str, pd.DataFrame],
    vep_tables: Dict[str, pd.DataFrame],
    consequence: str,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    print(
        "[filter] applying filter: most_severe_consequence =="
        f" '{consequence}'"
    )

    filtered_splits: Dict[str, pd.DataFrame] = {}
    filtered_tables: Dict[str, pd.DataFrame] = {
        split: table for split, table in vep_tables.items()
    }

    for split, df in splits.items():
        if "most_severe_consequence" not in df.columns:
            raise ValueError(
                "ClinVar split is missing 'most_severe_consequence'; run VEP first"
            )

        col = df["most_severe_consequence"].astype("string")
        mask = col.fillna("").eq(consequence)
        filtered_df = df[mask].reset_index(drop=True)
        dropped = len(df) - len(filtered_df)
        print(
            f"  [filter] {split}: kept {len(filtered_df):,}/{len(df):,} rows "
            f"(dropped {dropped:,})"
        )
        filtered_splits[split] = filtered_df

        table = vep_tables.get(split)
        if table is None:
            raise ValueError(
                f"Missing VEP annotations for split '{split}' while applying filter"
            )
        if "most_severe_consequence" not in table.columns:
            print(
                f"  [filter] warning: VEP table for {split} lacks "
                "'most_severe_consequence'; skipping table filter"
            )
            filtered_tables[split] = table
            continue
        table_mask = table["most_severe_consequence"].astype("string").fillna("").eq(
            consequence
        )
        filtered_tables[split] = table[table_mask].reset_index(drop=True)

    return filtered_splits, filtered_tables


def _sanitize_consequence(value: str) -> str:
    slug = value.strip().lower()
    slug = re.sub(r"[^a-z0-9._-]+", "-", slug)
    slug = slug.strip("-")
    return slug or "value"


def _prepare_filtered_clinvar(
    cfg,
    consequence: str,
) -> Tuple[
    Dict[str, pd.DataFrame],
    Dict[str, Path],
    Dict[str, Path],
    Dict[str, pd.DataFrame],
]:
    slug = _sanitize_consequence(consequence)
    filter_root = cfg.paths.artifacts / "filters" / f"most_severe_consequence={slug}"
    splits_dir = filter_root / "splits"
    vep_dir = filter_root / "vep"
    splits_dir.mkdir(parents=True, exist_ok=True)
    vep_dir.mkdir(parents=True, exist_ok=True)

    split_names = ("train", "val", "test")
    split_paths: Dict[str, Path] = {
        name: splits_dir / f"clinvar_{name}.feather" for name in split_names
    }
    vep_paths: Dict[str, Path] = {
        name: (vep_dir / name / "vep_combined.feather") for name in split_names
    }

    reuse = (
        not cfg.run.force_splits
        and not getattr(cfg.protein, "force_vep", False)
        and all(path.exists() for path in split_paths.values())
        and all(path.exists() for path in vep_paths.values())
    )
    if reuse:
        print(
            "[filter] reusing cached filtered ClinVar splits for",
            f"'{consequence}'",
        )
        splits = {name: pd.read_feather(path) for name, path in split_paths.items()}
        vep_tables = {
            name: pd.read_feather(path) for name, path in vep_paths.items()
        }
        return splits, split_paths, vep_paths, vep_tables

    if not cfg.paths.clinvar.exists():
        raise FileNotFoundError(
            f"ClinVar input not found at {cfg.paths.clinvar}. Update Paths.clinvar."
        )

    print(
        "[filter] generating filtered ClinVar splits for",
        f"most_severe_consequence='{consequence}'",
    )
    full_df = load_clinvar_variants(str(cfg.paths.clinvar))

    full_split_name = f"filter_{slug}_full"
    full_dir = filter_root / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    full_split_path = full_dir / "clinvar_full.feather"
    base_splits = {full_split_name: full_df}
    base_paths = {full_split_name: full_split_path}

    annotated_splits, _, full_tables = ensure_vep_annotations(
        cfg, base_splits, base_paths, return_tables=True
    )

    filtered_splits, filtered_tables = _filter_splits_by_consequence(
        annotated_splits, full_tables, consequence
    )
    filtered_df = filtered_splits[full_split_name]
    filtered_table = filtered_tables[full_split_name]

    if filtered_df.empty:
        raise ValueError(
            "ClinVar dataset contains no rows after applying consequence filter "
            f"'{consequence}'"
        )

    train_df, val_df, test_df = _split_by_gene(filtered_df)
    splits: Dict[str, pd.DataFrame] = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }

    filtered_table = filtered_table.copy()
    filtered_table["_vid_str"] = filtered_table["VariationID"].astype(str)
    vep_tables: Dict[str, pd.DataFrame] = {}
    for name, frame in splits.items():
        path = split_paths[name]
        frame.to_feather(path)
        print(f"  [write] {name}: {path} (rows={len(frame)})")

        vids = frame["VariationID"].astype(str)
        table = filtered_table[filtered_table["_vid_str"].isin(vids)].reset_index(
            drop=True
        )
        table = table.drop(columns=["_vid_str"], errors="ignore")
        table_path = vep_paths[name]
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_feather(table_path)
        print(
            f"  [filter] wrote VEP table for {name}: {table_path} (rows={len(table)})"
        )
        vep_tables[name] = table

    return splits, split_paths, vep_paths, vep_tables


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train the PathoLens pipeline")
    ap.add_argument("--config", required=True, help="Path to pipeline TOML config")
    ap.add_argument("--device", help="Override device string (cpu/cuda/...)")
    ap.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip the LLM fine-tuning stage",
    )
    ap.add_argument(
        "--filter-by",
        help=(
            "Filter ClinVar variants to the provided most_severe_consequence "
            "value before building caches"
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_pipeline_config(args.config)

    device = args.device or cfg.run.device or get_device()
    print(f"[config] loaded from {Path(args.config).resolve()}")
    print(f"[config] artifacts dir: {cfg.paths.artifacts}")
    print(f"[config] device: {device}")

    # ----- ClinVar splits + VEP annotations -----
    if args.filter_by:
        splits, split_paths, vep_paths, vep_tables = _prepare_filtered_clinvar(
            cfg, args.filter_by
        )
    else:
        splits, split_paths = prepare_clinvar_splits(cfg)
        splits, vep_paths, vep_tables = ensure_vep_annotations(
            cfg, splits, split_paths, return_tables=True
        )

    # ----- DNA caches -----
    dna_artifacts = build_dna_caches(cfg, splits, device)

    # ----- Protein caches -----
    protein_artifacts = build_protein_caches(
        cfg,
        splits,
        device,
        split_paths=split_paths,
        vep_paths=vep_paths,
        vep_tables=vep_tables,
    )

    # ----- Manifest -----
    manifest_splits = _merge_artifacts(dna_artifacts, protein_artifacts)
    manifest_extras = {"config": str(Path(args.config).resolve())}
    if args.filter_by:
        manifest_extras["filter_by"] = args.filter_by
    manifest = PipelineManifest(
        go_npz=str(cfg.go.npz),
        splits=manifest_splits,
        extras=manifest_extras,
    )
    if cfg.run.write_manifest:
        manifest_path = cfg.run.manifest_path or (
            cfg.paths.artifacts / "pipeline_manifest.json"
        )
        manifest.dump(manifest_path)
    else:
        print("[manifest] skipping write (Run.write_manifest=false)")

    # ----- Datasets -----
    seq_datasets = build_sequence_datasets(cfg, manifest)

    if args.skip_train or not cfg.llm.enabled:
        reason = "--skip-train" if args.skip_train else "LLM.enabled=false"
        print(f"[train] skipping LLM stage ({reason})")
        return

    # ----- LLM stage -----
    from src.llm.train import run_llm_pipeline
    import src.llm.eval as llm_eval

    llm_cfg = cfg.llm
    out_dir = llm_cfg.out_dir
    print(f"[llm] output dir: {out_dir}")
    run_llm_pipeline(llm_cfg, seq_datasets)
    print("[done] training complete")

    print("[llm] running evaluation")
    llm_eval.run_all(seq_datasets, out_dir=str(out_dir), model_id=llm_cfg.model_id)

    print("[llm] running ablation test")
    import src.llm.ablation as ablation

    ablation.run_prompt_vs_cond_ablation(
        seq_datasets,
        out_dir=str(out_dir),
        split="val",
        max_n=5_000,
        batch_size=128,
    )


if __name__ == "__main__":
    main()
