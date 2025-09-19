from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from util import get_device

from src.pipeline import PipelineManifest, SplitArtifact, load_pipeline_config
from src.pipeline.builders import build_dna_caches, build_protein_caches
from src.pipeline.clinvar import prepare_clinvar_splits
from vep_pipeline import ensure_vep_annotations
from src.pipeline.datasets import build_sequence_datasets


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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train the PathoLens pipeline")
    ap.add_argument("--config", required=True, help="Path to pipeline TOML config")
    ap.add_argument("--device", help="Override device string (cpu/cuda/...)")
    ap.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip the LLM fine-tuning stage",
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
    manifest = PipelineManifest(
        go_npz=str(cfg.go.npz),
        splits=manifest_splits,
        extras={"config": str(Path(args.config).resolve())},
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
