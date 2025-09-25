"""Prompt context builders backed by VEP annotations."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import pandas as pd


def _get_value(row: Any, key: str) -> Any:
    if isinstance(row, Mapping):
        return row.get(key, None)
    getter = getattr(row, "get", None)
    if callable(getter):
        return getter(key, None)
    try:
        return row[key]
    except Exception:
        return None


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):  # handles pandas.NA/NaN
            return ""
    except Exception:
        pass
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return ""
    return s


def _coalesce(row: Any, keys: Iterable[str]) -> str:
    for key in keys:
        val = _get_value(row, key)
        text = _clean_str(val)
        if text:
            return text
    return ""


def _format_terms(raw: str) -> str:
    if not raw:
        return ""
    parts = [p.strip() for p in str(raw).split(",") if p and p.strip()]
    if not parts:
        return ""
    pretty = []
    seen = set()
    for part in parts:
        nice = part.replace("_", " ")
        low = nice.lower()
        if low not in seen:
            pretty.append(nice)
            seen.add(low)
    return ", ".join(pretty)


def build_ctx_from_row(row) -> str:
    """Build the textual context for a ClinVar row using VEP-derived fields."""
    lines: list[str] = []

    gene = _coalesce(row, ["gene_symbol", "GeneSymbol"])
    if gene:
        lines.append(f"Gene: {gene}")

    # transcript = _coalesce(row, ["mane_select", "transcript_id"])
    # if transcript:
    #     lines.append(f"Transcript: {transcript}")

    # protein = _coalesce(row, ["protein_id"])
    # if protein:
    #     lines.append(f"Protein: {protein}")

    hgvsc = _coalesce(row, ["hgvsc"])
    if hgvsc:
        lines.append(f"HGVS.c: {hgvsc}")

    hgvsp = _coalesce(row, ["hgvsp"])
    if hgvsp:
        lines.append(f"HGVS.p: {hgvsp}")

    most = _clean_str(_get_value(row, "most_severe_consequence")).replace("_", " ")
    lines.append(f"Most severe consequence: {most}")

    # impact_raw = _clean_str(_get_value(row, "impact"))
    # impact = impact_raw.title() if impact_raw else ""
    # if most and impact:
    #     lines.append(f"Most severe consequence: {most} (impact: {impact})")
    # elif most:
    #     lines.append(f"Most severe consequence: {most}")
    # elif impact:
    #     lines.append(f"Impact: {impact}")

    terms_fmt = _format_terms(_coalesce(row, ["consequence_terms"]))
    if terms_fmt and terms_fmt.lower() != most.lower():
        lines.append(f"Consequence terms: {terms_fmt}")

    # variant = _coalesce(row, ["variant_allele"])
    # if variant:
    #     lines.append(f"Variant allele: {variant}")

    # amino_acids = _coalesce(row, ["amino_acids"])
    # if amino_acids:
    #     lines.append(f"Amino acids: {amino_acids}")

    # codons = _coalesce(row, ["codons"])
    # if codons:
    #     lines.append(f"Codons: {codons}")

    return "\n".join(lines)
