import pandas as pd
import re

CLINVAR_COLS = [
    "VariationID",
    "Name",
    "GeneSymbol",
    "ClinicalSignificance",
    "ReviewStatus",
    "Assembly",
    "Chromosome",
    "ChromosomeAccession",
    "PositionVCF",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF",
    "Start",
    "Stop",
    "ReferenceAllele",
    "AlternateAllele",
    "PhenotypeIDS",
    "PhenotypeList",
]

POS = {"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"}
NEG = {"benign", "likely benign", "benign/likely benign"}


_GENE_OK = re.compile(r"^[A-Za-z0-9\-]{2,15}$")


def clinsig_to_binary(cs: str | None) -> int | None:
    if not isinstance(cs, str):
        return None
    s = cs.strip().lower()
    if s in POS:
        return 1
    if s in NEG:
        return 0
    return None


def extract_gene_from_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    m = re.search(r"\(([A-Za-z0-9\-]+)\)\s*:", name)
    return m.group(1) if m else ""


def sanitize_gene_symbol(gs: str) -> str:
    if not isinstance(gs, str):
        return ""
    s = gs.strip()
    if not s:
        return ""
    lower = s.lower()
    if any(
        k in lower for k in ("covers", "genes", "region", "none of which", "dosage")
    ):
        return ""
    tok = re.split(r"[;,\s|/]+", s)[0]
    return tok if _GENE_OK.match(tok or "") else ""


def pick_gene(row) -> str:
    g = extract_gene_from_name(row.get("Name", ""))
    if _GENE_OK.match(g or ""):
        return g
    return sanitize_gene_symbol(row.get("GeneSymbol", ""))


def clean_label(cs: str | None) -> int | None:
    if not isinstance(cs, str):
        return None
    cs = cs.strip().lower()
    if cs in POS:
        return 1
    if cs in NEG:
        return 0
    return None


def load_clinvar_variants(path) -> pd.DataFrame:
    df = pd.read_table(
        path, compression="gzip" if path.endswith(".gz") else None, sep="\t"
    )
    df = df[CLINVAR_COLS]

    # Focus on GRCh38 + high-confidence
    df = df[df["Assembly"] == "GRCh38"]
    df = df[
        df["ReviewStatus"] == "criteria provided, multiple submitters, no conflicts"
    ]

    # Require VCF-style fields
    df = df.dropna(
        subset=[
            "ChromosomeAccession",
            "PositionVCF",
            "ReferenceAlleleVCF",
            "AlternateAlleleVCF",
        ]
    )

    # Safeguard against PositionVCF = -1
    df = df[df["PositionVCF"] > 0].dropna(
        subset=["ReferenceAlleleVCF", "AlternateAlleleVCF"]
    )

    # Filter to pathogenic/non-pathogenic, add _y
    df["_y"] = df["ClinicalSignificance"].map(clean_label)
    df = df.dropna(subset=["_y"]).copy()
    df["_y"] = df["_y"].astype(int)

    # Precompute gene symbol (ClinVar-derived)
    df["GeneSymbol"] = df.apply(pick_gene, axis=1)

    # Normalize dtypes for Arrow/Feather
    for col in [
        "Chromosome",
        "ChromosomeAccession",
        "GeneSymbol",
        "ClinicalSignificance",
        "ReviewStatus",
        "PhenotypeIDS",
        "PhenotypeList",
    ]:
        df[col] = df[col].astype("string")

    return df.reset_index(drop=True)
