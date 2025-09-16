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


def clinsig_to_binary(cs: str | None) -> int | None:
    if not isinstance(cs, str):
        return None
    s = cs.strip().lower()
    if s in POS:
        return 1
    if s in NEG:
        return 0
    return None


# ============================================================
# ClinVar Name/GeneSymbol parsers (ONLY these two fields)
# ============================================================

_AA3 = r"[A-Z][a-z]{2}"
_AA1 = r"[A-Z*]"
_NUM = r"\d+"
_GENE_OK = re.compile(r"^[A-Za-z0-9\-]{2,15}$")


def extract_hgvsc(name: str) -> str:
    if not isinstance(name, str):
        return ""
    m = re.search(r":\s*(c\.[^)\s;]+)", name)
    return m.group(1) if m else ""


def extract_hgvsp(name: str) -> str:
    if not isinstance(name, str):
        return ""
    m = re.search(r"\((p\.[^)]+)\)", name)
    return m.group(1) if m else ""


def classify_protein_change(hgvsp: str) -> str:
    if not hgvsp:
        return ""
    s = hgvsp
    if "fs" in s:
        return "frameshift"
    if re.search(r"(Ter|\*)$", s):
        return "nonsense"
    # synonymous: p.(=) or AA==AA
    if (
        s.endswith("(=")
        or re.search(r"p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}$", s)
        and s[-3:] == s[-6:-3]
        or re.search(r"p\.[A-Z]\d+[A-Z]$", s)
        and s[-1:] == s[-2:-1]
    ):
        return "synonymous"
    if re.search(r"(del|dup|ins)", s):
        return "inframe_indel"
    if re.search(rf"p\.{_AA3}{_NUM}{_AA3}$", s) or re.search(
        rf"p\.{_AA1}{_NUM}{_AA1}$", s
    ):
        return "missense"
    return "other"


def parse_hgvsc_features(hgvsc: str) -> dict:
    out = {
        "region": "unknown",
        "splice": False,
        "kind": "other",
        "indel_len": 0,
        "frameshift_guess": None,
        "snv_change": "",
    }
    if not isinstance(hgvsc, str) or not hgvsc.startswith("c."):
        return out
    s = hgvsc
    out["region"] = (
        "utr5" if s.startswith("c.-") else ("utr3" if s.startswith("c.*") else "cds")
    )
    if re.search(r"[+-]\d+", s) or "IVS" in s:
        out["splice"] = True
    if "delins" in s:
        out["kind"] = "delins"
    elif "del" in s:
        out["kind"] = "del"
    elif "dup" in s:
        out["kind"] = "dup"
    elif ">" in s:
        out["kind"] = "snv"
    elif "ins" in s:
        out["kind"] = "ins"
    mchg = re.search(r"c\.[^A-Za-z0-9]*\d+[+-]?\d*([ACGT])>([ACGT])", s)
    if mchg:
        out["snv_change"] = f"{mchg.group(1)}>{mchg.group(2)}"
    if out["kind"] in {"del", "dup", "ins", "delins"}:
        del_len = 0
        mspan = re.search(r"c\.(\d+)(?:_(\d+))?", s)
        if "del" in s and mspan:
            a = int(mspan.group(1))
            b = int(mspan.group(2) or a)
            del_len = max(1, b - a + 1)
        ins_len = 0
        mins = re.search(r"ins([ACGT]+)", s)
        if mins:
            ins_len = len(mins.group(1))
        if out["kind"] == "dup":
            ins_len = del_len
        delta = ins_len - del_len
        out["indel_len"] = abs(delta) if delta != 0 else max(del_len, ins_len)
        if out["region"] == "cds" and not out["splice"]:
            out["frameshift_guess"] = (delta % 3 != 0) if (ins_len or del_len) else None
    return out


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


def build_context(row) -> str:
    """Build a simple textual context from precomputed columns."""
    lines = []
    gene = row.get("GeneSymbol", "") or ""
    hgvsc = row.get("hgvsc", "")
    hgvsp = row.get("hgvsp", "")
    pcls = row.get("protein_change", "")
    region = row.get("cfeat_region", "unknown")
    splice = bool(row.get("cfeat_splice", False))
    kind = row.get("cfeat_kind", "other")
    indel_len = int(row.get("cfeat_indel_len", 0) or 0)
    fs = row.get("cfeat_frameshift_guess", None)
    snv_change = row.get("cfeat_snv_change", "")

    if gene:
        lines.append(f"Gene: {gene}")
    if hgvsc:
        lines.append(f"HGVS.c: {hgvsc}")
    if hgvsp:
        lines.append(f"HGVS.p: {hgvsp}")
    cons = pcls or ("splice" if splice else (kind if kind != "other" else ""))
    if cons:
        lines.append(f"Consequence: {cons}")
    if region != "unknown":
        lines.append(f"Region: {region}")
    if splice:
        lines.append("Splice-site: yes")
    if indel_len > 0:
        lines.append(f"Indel length: {indel_len}")
    if fs is not None:
        lines.append(f"Frameshift (heuristic): {'yes' if fs else 'no'}")
    if snv_change:
        lines.append(f"SNV change: {snv_change}")
    return "\n".join(lines) if lines else "Variant context provided."


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

    # Precompute gene + variant features
    df["GeneSymbol"] = df.apply(pick_gene, axis=1)
    df["hgvsc"] = df["Name"].map(extract_hgvsc)
    df["hgvsp"] = df["Name"].map(extract_hgvsp)
    df["protein_change"] = df["hgvsp"].map(classify_protein_change)
    cfeat = df["hgvsc"].map(parse_hgvsc_features).apply(pd.Series)
    cfeat = cfeat.add_prefix("cfeat_")
    df = pd.concat([df, cfeat], axis=1)
    df["context"] = df.apply(build_context, axis=1)

    # Normalize dtypes for Arrow/Feather
    for col in [
        "Chromosome",
        "ChromosomeAccession",
        "GeneSymbol",
        "ClinicalSignificance",
        "ReviewStatus",
        "PhenotypeIDS",
        "PhenotypeList",
        "hgvsc",
        "hgvsp",
        "protein_change",
        "cfeat_region",
        "cfeat_kind",
        "cfeat_snv_change",
        "context",
    ]:
        df[col] = df[col].astype("string")
    df["cfeat_splice"] = df["cfeat_splice"].astype("boolean")
    df["cfeat_indel_len"] = df["cfeat_indel_len"].astype("Int64")
    df["cfeat_frameshift_guess"] = df["cfeat_frameshift_guess"].astype("boolean")

    return df.reset_index(drop=True)
