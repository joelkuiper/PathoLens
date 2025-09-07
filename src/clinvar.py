import pandas as pd

CLINVAR_COLS = [
    "VariationID",
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

    # Filter to pathogenic/non-pathogenic, add _y
    df["_y"] = df["ClinicalSignificance"].map(clean_label)
    df = df.dropna(subset=["_y"]).copy()
    df["_y"] = df["_y"].astype(int)

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
