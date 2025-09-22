#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VEP-in-Docker pipeline for ClinVar-like DataFrames
--------------------------------------------------

Input DF must have (at minimum):
  VariationID, ChromosomeAccession, PositionVCF, ReferenceAlleleVCF, AlternateAlleleVCF
Optional but helpful:
  GeneSymbol, hgvsc, hgvsp  (for filtering)

It will:
  * write chunk VCFs
  * run VEP in Docker (JSON + ProteinSeqs FASTAs)
  * parse & join JSON+FASTA into one tidy DataFrame
  * write combined Feather

Requirements on the host:
  * Docker available
  * You already have a VEP cache + FASTA under <host_cache_dir> (mounted to /opt/vep/.vep)
    e.g. /your/path/vep_cache/homo_sapiens/115_GRCh38/...
  * ProteinSeqs.pm is present in <host_cache_dir>/Plugins  (script can ensure/install)
  * Python: pandas, pyarrow, biopython

Example:
  python vep_pipeline.py \
      --input clinvar_valid.feather \
      --host_cache_dir /path/to/vep_cache \
      --fasta Homo_sapiens.GRCh38.dna.toplevel.fa.gz \
      --out_dir out_vep \
      --filter patchable \
      --chunk_size 1000 \
      --jobs 6 \
      --vep_fork 2
"""

import argparse
import os
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
from Bio import SeqIO
from multiprocessing.pool import ThreadPool  # IO-bound (docker), threads are fine

from src.pipeline.config import _norm_path, PipelineConfig, ConfigError


SplitDict = Dict[str, pd.DataFrame]
PathDict = Dict[str, Path]


# ---------------------------
# FASTA helpers for ProteinSeqs FASTAs
# ---------------------------


def strip_protein_version(x: Optional[str]) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x)
    m = re.match(r"^(ENSP\d+)", s)
    return m.group(1) if m else s


def only_pdot(h: Optional[str]) -> Optional[str]:
    if h is None or (isinstance(h, float) and pd.isna(h)):
        return None
    m = re.search(r"(p\.[^ \t|]+)", str(h))
    return m.group(1) if m else None


def _strip_hgvs_prefix(value: Optional[str]) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value)
    return s.split(":", 1)[1] if ":" in s else s


def _join_terms(values: Optional[list]) -> Optional[str]:
    if not isinstance(values, (list, tuple)):
        if values is None or values == "" or pd.isna(values):
            return None
        return str(values)
    cleaned = [str(v) for v in values if v not in (None, "") and not pd.isna(v)]
    return ",".join(cleaned) if cleaned else None


def parse_fasta_indexed(path: Path) -> Dict[str, str]:
    seqs = {}
    for rec in SeqIO.parse(str(path), "fasta"):
        seqs[rec.id] = str(rec.seq)
    return seqs


def load_proteinseqs_fastas(
    wt_path: Path, mut_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (wt_df keyed by protein_id_norm, mt_df keyed by (protein_id_norm, hgvsp_norm))."""
    # WT headers look like: >ENSP00000268712
    wt_map = {}
    for rec in SeqIO.parse(str(wt_path), "fasta"):
        pid_norm = strip_protein_version(rec.id)  # safe even if versionless
        if pid_norm and pid_norm not in wt_map:  # keep first
            wt_map[pid_norm] = str(rec.seq)
    wt_df = pd.DataFrame(
        [{"protein_id_norm": k, "seq_wt": v} for k, v in wt_map.items()]
    )

    # MUT headers look like: >ENSP00000268712.2:p.Arg2132=
    mt_rows = []
    for rec in SeqIO.parse(str(mut_path), "fasta"):
        header = rec.id  # includes ':' piece
        left = header.split(":", 1)[0]
        pid_norm = strip_protein_version(left)
        pdot = only_pdot(header)
        mt_rows.append(
            {"protein_id_norm": pid_norm, "hgvsp_norm": pdot, "seq_mt": str(rec.seq)}
        )
    mt_df = pd.DataFrame(mt_rows)
    return wt_df, mt_df


# ---------------------------
# VCF writer
# ---------------------------

VCF_COLS = [
    "VariationID",
    "ChromosomeAccession",
    "PositionVCF",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF",
]


def write_vcf(df: pd.DataFrame, out_vcf: Path) -> None:
    # ensure sorted within the chunk as well
    df = df.sort_values(["ChromosomeAccession", "PositionVCF"])
    with out_vcf.open("w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for _, r in df.iterrows():
            f.write(
                f"{r['ChromosomeAccession']}\t{int(r['PositionVCF'])}\t{str(r['VariationID'])}\t"
                f"{str(r['ReferenceAlleleVCF']).upper()}\t{str(r['AlternateAlleleVCF']).upper()}\t.\t.\t.\n"
            )


# ---------------------------
# VEP Docker runner
# ---------------------------


def ensure_proteinseqs(host_cache_dir: Path, image: str) -> None:
    """Ensure ProteinSeqs.pm exists under cache Plugins by invoking INSTALL.pl inside the container."""
    plugin_path = host_cache_dir / "Plugins" / "ProteinSeqs.pm"
    if plugin_path.exists():
        return
    # run as host user to avoid permission mismatches
    uid = os.getuid()
    gid = os.getgid()

    cmd = [
        "docker",
        "run",
        "--user",
        f"{uid}:{gid}",
        "-it",
        "-v",
        f"{str(host_cache_dir)}:/opt/vep/.vep",
        image,
        "bash",
        "-lc",
        "set -euo pipefail; "
        "cd /opt/vep/src/ensembl-vep; "
        "mkdir -p /opt/vep/.vep/Plugins; "
        "perl INSTALL.pl -a p --PLUGINS ProteinSeqs --PLUGINSDIR /opt/vep/.vep/Plugins",
    ]
    print("[install plugin]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if not plugin_path.exists():
        raise RuntimeError("Failed to install ProteinSeqs.pm into cache Plugins")


def run_vep_docker(
    vcf_path: Path,
    out_prefix: Path,
    host_cache_dir: Path,
    fasta_relpath: str,
    image: str,
    vep_fork: int = 2,
    pick_order: str = "mane_select,canonical,appris,ccds,rank,tsl,length",
) -> Tuple[Path, Path, Path]:
    """Run VEP (JSON + ProteinSeqs FASTAs) in Docker; returns (json, wt.fa, mut.fa) paths."""
    # Ensure absolute paths for Docker mounts
    host_cache_dir = host_cache_dir.resolve()
    vcf_path = vcf_path.resolve()
    out_dir = out_prefix.parent.resolve()

    out_json = (out_prefix.with_suffix(".json")).resolve()
    out_wt = (out_prefix.with_suffix(".wt.fa")).resolve()
    out_mt = (out_prefix.with_suffix(".mut.fa")).resolve()

    # Inside-container paths
    work_mount = vcf_path.parent
    out_mount = out_dir
    fasta_in_container = f"/opt/vep/.vep/{fasta_relpath}"

    # run as host user to avoid permission mismatches
    uid = os.getuid()
    gid = os.getgid()

    cmd = [
        "docker",
        "run",
        "--user",
        f"{uid}:{gid}",
        "--rm",
        "-v",
        f"{str(host_cache_dir)}:/opt/vep/.vep",
        "-v",
        f"{str(work_mount)}:/work",
        "-v",
        f"{str(out_mount)}:/out",
        image,
        "bash",
        "-lc",
        "set -euo pipefail; "
        "vep "
        f"--format vcf "
        f"--input_file /work/{vcf_path.name} "
        f"--output_file /out/{out_json.name} "
        "--species homo_sapiens --assembly GRCh38 "
        "--cache --offline --dir_cache /opt/vep/.vep "
        f"--fasta {fasta_in_container} "
        "--symbol --protein --uniprot --hgvs --mane "
        f"--pick --pick_order {pick_order} "
        "--dir_plugins /opt/vep/.vep/Plugins "
        f"--plugin ProteinSeqs,/out/{out_wt.name},/out/{out_mt.name} "
        "--json --no_stats --force_overwrite "
        f"--fork {vep_fork}",
    ]
    print("[vep]", " ".join(cmd[:12]), "...")
    subprocess.run(cmd, check=True)
    return out_json, out_wt, out_mt


# ---------------------------
# JSON parser + joiner
# ---------------------------


def parse_vep_json(json_path: Path) -> pd.DataFrame:
    rows = []
    with json_path.open() as f:
        for line in f:
            rec = json.loads(line)
            vid = rec.get("id")
            base = {
                "id": str(vid),
                "most_severe_consequence": rec.get("most_severe_consequence"),
                "assembly_name": rec.get("assembly_name"),
                "allele_string": rec.get("allele_string"),
            }

            tlist = rec.get("transcript_consequences") or []
            # with --pick there should be a single chosen transcript; still loop defensively
            if tlist:
                t = tlist[0]
                row = dict(base)
                row.update(
                    {
                        "gene_symbol": t.get("gene_symbol"),
                        "gene_symbol_source": t.get("gene_symbol_source"),
                        "gene_id": t.get("gene_id"),
                        "hgnc_id": t.get("hgnc_id"),
                        "transcript_id": t.get("transcript_id"),
                        "mane_select": t.get("mane_select"),
                        "mane_plus_clinical": t.get("mane_plus_clinical"),
                        "canonical": t.get("canonical"),
                        "strand": t.get("strand"),
                        "variant_allele": t.get("variant_allele"),
                        "impact": t.get("impact"),
                        "consequence_terms": _join_terms(t.get("consequence_terms")),
                        "hgvsc": _strip_hgvs_prefix(t.get("hgvsc")),
                        "hgvsp": _strip_hgvs_prefix(t.get("hgvsp")),
                        "amino_acids": t.get("amino_acids"),
                        "codons": t.get("codons"),
                        "protein_id": t.get("protein_id"),
                        "protein_start": t.get("protein_start"),
                        "protein_end": t.get("protein_end"),
                        "protein_id_source": t.get("protein_id_source"),
                        "protein_length": t.get("protein_length"),
                        "protein_id_version": t.get("protein_id_version"),
                        "swissprot": _join_terms(t.get("swissprot")),
                        "uniprot_isoform": _join_terms(t.get("uniprot_isoform")),
                        "uniparc": _join_terms(t.get("uniparc")),
                        "cdna_start": t.get("cdna_start"),
                        "cdna_end": t.get("cdna_end"),
                        "cds_start": t.get("cds_start"),
                        "cds_end": t.get("cds_end"),
                    }
                )
                rows.append(row)
            else:  # keep record for alignment
                rows.append(base)
    return pd.DataFrame(rows)


def join_json_and_fastas(
    df_json: pd.DataFrame, wt_path: Path, mt_path: Path
) -> pd.DataFrame:
    wt_df, mt_df = load_proteinseqs_fastas(wt_path, mt_path)

    ann = df_json.copy()
    ann["protein_id_norm"] = ann["protein_id"].map(strip_protein_version)
    ann["hgvsp_norm"] = ann["hgvsp"].map(only_pdot)

    out = ann.merge(wt_df, on="protein_id_norm", how="left").merge(
        mt_df, on=["protein_id_norm", "hgvsp_norm"], how="left"
    )
    return out


# ---------------------------
# Chunk runner
# ---------------------------


def run_chunk(args) -> pd.DataFrame:
    (chunk_id, df_chunk, work_dir, host_cache_dir, fasta_relpath, image, vep_fork) = (
        args
    )
    out_dir = work_dir / f"chunk_{chunk_id:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    vcf_path = out_dir / "input.vcf"
    write_vcf(df_chunk, vcf_path)
    json_path, wt_path, mt_path = run_vep_docker(
        vcf_path,
        out_dir / "vep_out",
        host_cache_dir,
        fasta_relpath,
        image,
        vep_fork=vep_fork,
    )
    df_json = parse_vep_json(json_path)
    joined = join_json_and_fastas(df_json, wt_path, mt_path)
    # keep original ID/coords if present
    joined = df_chunk.assign(id=df_chunk["VariationID"].astype(str)).merge(
        joined, on="id", how="left", suffixes=("", "_vep")
    )
    return joined


def run_vep_pipeline(
    input_path: Path,
    host_cache_dir: Path,
    fasta_relpath: str,
    out_dir: Path,
    *,
    image: str = "ensemblorg/ensembl-vep",
    chunk_size: int = 1000,
    jobs: int = 4,
    vep_fork: int = 2,
    limit: int = 0,
) -> Path:
    """
    Run the full VEP workflow in-process.
    Returns the path to vep_combined.feather.
    """
    input_path = _norm_path(input_path)
    host_cache_dir = _norm_path(host_cache_dir)
    out_dir = _norm_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load DF (infer by suffix)
    if input_path.suffix.lower() in {".feather", ".ft"}:
        df = pd.read_feather(input_path)
    elif input_path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(input_path)
    else:
        df = (
            pd.read_csv(input_path)
            if input_path.suffix.lower().endswith("csv")
            else pd.read_table(input_path)
        )

    df["PositionVCF"] = df["PositionVCF"].astype(int)
    df = df.sort_values(["ChromosomeAccession", "PositionVCF"]).reset_index(drop=True)
    print("[sort] globally sorted by ChromosomeAccession,PositionVCF")

    # Required columns
    missing = [c for c in VCF_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    if limit and limit > 0:
        df = df.iloc[:limit].copy()

    ensure_proteinseqs(host_cache_dir, image)

    # Shard
    chunks: list[pd.DataFrame] = [
        df.iloc[i : i + chunk_size].copy() for i in range(0, len(df), chunk_size)
    ]
    print(
        f"[plan] chunks: {len(chunks)}  chunk_size: {chunk_size}  jobs: {jobs}  vep_fork: {vep_fork}"
    )

    work_dir = out_dir / "work"
    work_dir.mkdir(exist_ok=True, parents=True)

    job_args = []
    for idx, cdf in enumerate(chunks):
        need = set(VCF_COLS + ["GeneSymbol", "hgvsc", "hgvsp", "ClinicalSignificance"])
        cdf = cdf[[col for col in cdf.columns if col in need]].copy()
        job_args.append(
            (idx, cdf, work_dir, host_cache_dir, fasta_relpath, image, vep_fork)
        )

    with ThreadPool(processes=jobs) as pool:
        dfs = list(pool.imap_unordered(run_chunk, job_args))

    combined = pd.concat(dfs, ignore_index=True)
    out_feather = out_dir / "vep_combined.feather"
    combined.to_feather(out_feather)
    print(f"[done] wrote {out_feather}")

    has_wt = combined["seq_wt"].notna().sum() if "seq_wt" in combined else 0
    has_mt = combined["seq_mt"].notna().sum() if "seq_mt" in combined else 0
    print(f"[summary] rows: {len(combined):,}  WT seq: {has_wt:,}  MT seq: {has_mt:,}")

    return out_feather


def _pct_nonempty(series: Optional[pd.Series]) -> float:
    if series is None:
        return 0.0
    if len(series) == 0:
        return 0.0
    s = series.astype("string")
    mask = s.fillna("").str.strip().ne("")
    return float(mask.mean() * 100.0)


def _format_consequence_counts(
    series: Optional[pd.Series], *, max_terms: int = 5
) -> str:
    if series is None or len(series) == 0:
        return ""
    s = series.astype("string").fillna("").str.strip()
    if s.empty:
        return ""
    pretty = s.replace("", "missing").str.replace("_", " ")
    counts = pretty.value_counts().head(max_terms)
    if counts.empty:
        return ""
    return ", ".join(f"{name}: {count}" for name, count in counts.items())


def _log_annotation_report(split: str, df: pd.DataFrame) -> None:
    total = len(df)
    if total == 0:
        print(f"  [vep-report] {split}: no annotations")
        return

    pct_hgvsc = _pct_nonempty(df.get("hgvsc"))
    pct_hgvsp = _pct_nonempty(df.get("hgvsp"))
    pct_mane = _pct_nonempty(df.get("mane_select"))
    pct_wt_missing = 100.0 - _pct_nonempty(df.get("seq_wt"))
    pct_mt_missing = 100.0 - _pct_nonempty(df.get("seq_mt"))
    top_cons = _format_consequence_counts(df.get("most_severe_consequence"))

    print(
        f"  [vep-report] {split}: rows={total:,} "
        f"hgvsc={pct_hgvsc:.1f}% hgvsp={pct_hgvsp:.1f}% "
        f"MANE_Select={pct_mane:.1f}% seq_wt_missing={pct_wt_missing:.1f}% "
        f"seq_mt_missing={pct_mt_missing:.1f}%"
    )
    if top_cons:
        print(f"               top consequences: {top_cons}")


def ensure_vep_annotations(
    cfg: PipelineConfig,
    splits: SplitDict,
    split_paths: PathDict,
    *,
    return_tables: bool = False,
) -> Tuple[SplitDict, PathDict] | Tuple[SplitDict, PathDict, Dict[str, pd.DataFrame]]:
    """Run VEP per split (with caching) and merge annotations back into splits.

    When ``return_tables`` is ``True`` the third element in the returned tuple
    contains the raw combined VEP tables keyed by split name. Callers that only
    expect two return values can omit the flag and retain the previous
    (splits, vep_outputs) behavior.
    """

    protein_cfg = cfg.protein
    cache_dir = protein_cfg.vep_cache_dir
    fasta_rel = protein_cfg.vep_fasta_relpath
    if cache_dir is None or not fasta_rel:
        raise ConfigError(
            "VEP cache/fasta paths are required. Set Protein.vep_cache_dir and "
            "Protein.vep_fasta_relpath in the config."
        )

    vep_root = cfg.paths.artifacts / "vep"
    vep_root.mkdir(parents=True, exist_ok=True)

    updated: SplitDict = {}
    vep_outputs: Dict[str, Path] = {}
    vep_tables: Dict[str, pd.DataFrame] = {}

    desired_cols = [
        "VariationID",
        "hgvsc",
        "hgvsp",
        "most_severe_consequence",
        "impact",
        "consequence_terms",
        "gene_symbol",
        "gene_symbol_source",
        "gene_id",
        "hgnc_id",
        "transcript_id",
        "mane_select",
        "mane_plus_clinical",
        "canonical",
        "strand",
        "variant_allele",
        "protein_id",
        "protein_start",
        "protein_end",
        "amino_acids",
        "codons",
        "cdna_start",
        "cdna_end",
        "cds_start",
        "cds_end",
        "assembly_name",
        "allele_string",
        "swissprot",
        "uniprot_isoform",
        "uniparc",
    ]

    str_cols = [
        "hgvsc",
        "hgvsp",
        "most_severe_consequence",
        "impact",
        "consequence_terms",
        "gene_symbol",
        "gene_symbol_source",
        "gene_id",
        "hgnc_id",
        "transcript_id",
        "mane_select",
        "mane_plus_clinical",
        "canonical",
        "strand",
        "variant_allele",
        "protein_id",
        "amino_acids",
        "codons",
        "assembly_name",
        "allele_string",
        "swissprot",
        "uniprot_isoform",
        "uniparc",
    ]

    int_cols = [
        "protein_start",
        "protein_end",
        "cdna_start",
        "cdna_end",
        "cds_start",
        "cds_end",
    ]

    for split, df in splits.items():
        inp_path = split_paths.get(split)
        if inp_path is None:
            raise ConfigError(f"Missing cached split path for '{split}'")
        if not inp_path.exists():
            inp_path.parent.mkdir(parents=True, exist_ok=True)
            df.reset_index(drop=True).to_feather(inp_path)

        vep_dir = vep_root / split
        vep_dir.mkdir(parents=True, exist_ok=True)
        combined_path = vep_dir / "vep_combined.feather"

        force = bool(getattr(protein_cfg, "force_vep", False))
        if combined_path.exists() and not force:
            print(f"[vep] reuse cached annotations for {split}: {combined_path}")
        else:
            print(f"[vep] running annotations for {split} → {vep_dir}")
            combined_path = run_vep_pipeline(
                input_path=inp_path,
                host_cache_dir=Path(cache_dir),
                fasta_relpath=str(fasta_rel),
                out_dir=vep_dir,
                image=protein_cfg.image,
                chunk_size=protein_cfg.chunk_size,
                jobs=protein_cfg.jobs,
                vep_fork=protein_cfg.vep_fork,
            )

        combined_path = Path(combined_path)
        vep_outputs[split] = combined_path

        vep_df = pd.read_feather(combined_path)
        if "VariationID" not in vep_df.columns:
            if "id" in vep_df.columns:
                vep_df["VariationID"] = vep_df["id"].astype(str)
            else:
                raise RuntimeError(
                    f"VEP output for split '{split}' missing VariationID column"
                )

        _log_annotation_report(split, vep_df)

        ann_cols = [c for c in desired_cols if c in vep_df.columns]
        ann = vep_df[ann_cols].drop_duplicates(subset=["VariationID"]).copy()
        ann["_vid_str"] = ann["VariationID"].astype(str)
        merge_cols = [c for c in ann.columns if c not in {"VariationID", "_vid_str"}]

        merged = df.copy()
        drop_cols = [c for c in merge_cols if c in merged.columns]
        if drop_cols:
            merged = merged.drop(columns=drop_cols)
        merged["_vid_str"] = merged["VariationID"].astype(str)
        merged = merged.merge(
            ann.drop(columns=["VariationID"]), on="_vid_str", how="left"
        )
        merged = merged.drop(columns=["_vid_str"])

        for col in str_cols:
            if col in merged.columns:
                merged[col] = merged[col].astype("string")

        for col in int_cols:
            if col in merged.columns:
                merged[col] = merged[col].astype("Int64")

        merged = merged.reset_index(drop=True)
        merged.to_feather(inp_path)
        updated[split] = merged
        vep_tables[split] = vep_df

    if return_tables:
        return updated, vep_outputs, vep_tables
    return updated, vep_outputs


def main():

    ap = argparse.ArgumentParser(
        description="Run VEP-in-Docker on a ClinVar-like DataFrame"
    )
    ap.add_argument(
        "--input", required=True, help="Path to input DF (feather/parquet/csv)"
    )
    ap.add_argument(
        "--host_cache_dir", required=True, help="Host path to VEP cache root"
    )
    ap.add_argument(
        "--fasta", required=True, help="FASTA path relative to host_cache_dir"
    )
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument(
        "--image", default="ensemblorg/ensembl-vep", help="Docker image for VEP"
    )
    ap.add_argument(
        "--filter",
        default="all",
        choices=["all", "hgvsp_present", "patchable", "protein_changing"],
        help="Filter on HGVSp before VEP",
    )
    ap.add_argument("--chunk_size", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=4, help="Parallel Docker jobs")
    ap.add_argument("--vep_fork", type=int, default=2, help="--fork inside container")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N rows")
    args = ap.parse_args()

    run_vep_pipeline(
        _norm_path(args.input),
        _norm_path(args.host_cache_dir),
        args.fasta,
        _norm_path(args.out_dir),
        image=args.image,
        chunk_size=args.chunk_size,
        jobs=args.jobs,
        vep_fork=args.vep_fork,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
