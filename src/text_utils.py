from __future__ import annotations
from typing import List, Tuple, Optional, Iterable, Dict

import os
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset
from util import get_device

# ---------------------------
# Encoder
# ---------------------------


def build_text_encoder(
    # model_id: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    model_id: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    device: Optional[str] = None,
    max_length: int = 128,
):
    """
    Load a BERT-ish biomedical text encoder (frozen) for caption embeddings.
    Returns (tokenizer, model, device, max_length).
    """
    device = device or get_device()
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": tok.unk_token or "[PAD]"})
    tok.padding_side = "right"
    tok.model_max_length = max_length

    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    enc = AutoModel.from_pretrained(model_id, torch_dtype=torch_dtype)
    if enc.get_input_embeddings().num_embeddings < len(tok):
        enc.resize_token_embeddings(len(tok))
    enc = enc.to(device).eval()
    return tok, enc, device, max_length


# ---------------------------
# Caption building
# ---------------------------


def _norm_str(x) -> str:
    if not isinstance(x, str):
        return ""
    return x.strip()


# ---------------------------
# Batch encode text
# ---------------------------
AA3 = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
    "Ter": "*",
    "Sec": "U",
    "Pyl": "O",
}


def _p3_to_p1(p: str | None) -> str | None:
    if not p:
        return None
    m = re.match(r"p\.\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|\*)\)?", str(p))
    if not m:
        return None
    a1 = AA3.get(m.group(1))
    pos = m.group(2)
    a3_raw = m.group(3)
    a3 = AA3.get(a3_raw, a3_raw if len(a3_raw) == 1 else None)
    if a1 is None or a3 is None:
        return None
    return f"p.{a1}{pos}{a3}"


def _as_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    # Many ClinVar fields can be "nan" as string after feather/csv; normalize to empty
    return "" if s.lower() == "nan" else s


def first_nonempty(row, keys) -> str:
    """Return first non-empty string among given keys from a Series/dict."""
    get = (
        row.get
        if hasattr(row, "get")
        else (lambda k, d=None: row[k] if k in row else d)
    )
    for k in keys:
        v = get(k, None)
        s = _as_str(v)
        if s:
            return s
    return ""


def _normalize_row_dict(row) -> Dict[str, object]:
    """Case/sep-insensitive view of a row (Series/dict)."""
    if hasattr(row, "to_dict"):
        d = row.to_dict()
    else:
        d = dict(row)
    norm = {}
    for k, v in d.items():
        kk = str(k).lower().replace("-", "_")
        norm[kk] = v
    return norm


_HGVS_PATTERNS = [
    re.compile(r"\bc\.[A-Za-z0-9_\-\+\*\>\=\?\(\)]+"),  # c.-like
    re.compile(r"\bp\.\(?[A-Z][a-z]{2}\d+(?:[A-Z][a-z]{2}|\*)\)?"),  # p.(Arg117His)
    re.compile(r"\bp\.[A-Z]\d+(?:[A-Z\*])"),  # p.R117H
]


def _find_hgvs_anywhere(row) -> Tuple[str, str]:
    """
    Scan likely columns then all strings; return (p_token, c_token) like ('p.R117H','c.350G>A').
    Empty strings if not found.
    """
    norm = _normalize_row_dict(row)
    p_tok, c_tok = "", ""

    # pass 1: obvious fields first
    likely = [
        (
            "hgvs_p",
            "hgvsp",
            "hgvsp_short",
            "protein_change",
            "protein_change_short",
            "p.",
        ),
        ("hgvs_c", "hgvsc", "c.", "cdna_change", "hgvscv"),
        ("name", "submittedhgvs"),
    ]
    for key_group in likely:
        for k in list(norm.keys()):
            if any(kg in k for kg in key_group):
                s = _as_str(norm[k])
                if not s:
                    continue
                if not p_tok:
                    for pat in _HGVS_PATTERNS[1:]:
                        m = pat.search(s)
                        if m:
                            p_tok = _p3_to_p1(m.group(0)) or m.group(0)
                            break
                if not c_tok:
                    m = _HGVS_PATTERNS[0].search(s)
                    if m:
                        c_tok = m.group(0)
                if p_tok and c_tok:
                    return p_tok, c_tok

    # pass 2: brute scan all string-ish values
    for v in norm.values():
        s = _as_str(v)
        if not s:
            continue
        if not p_tok:
            for pat in _HGVS_PATTERNS[1:]:
                m = pat.search(s)
                if m:
                    p_tok = _p3_to_p1(m.group(0)) or m.group(0)
                    break
        if not c_tok:
            m = _HGVS_PATTERNS[0].search(s)
            if m:
                c_tok = m.group(0)
        if p_tok and c_tok:
            break

    return p_tok or "", c_tok or ""


def _pick_first(norm: Dict[str, object], keys) -> str:
    for k in keys:
        for kk, v in norm.items():
            if k == kk or kk.endswith(f"_{k}") or kk.startswith(k):
                s = _as_str(v)
                if s:
                    return s
    return ""


def _build_hgvs_g(row) -> str:
    """
    Try many aliases to build 'g.chr{chrom}:{pos}{ref}>{alt}'.
    Returns '' if we can’t find enough pieces.
    """
    norm = _normalize_row_dict(row)
    chrom_keys = ["chrom", "chr", "chromosome", "grch38chromosome", "grch37chromosome"]
    pos_keys = ["pos", "position", "start", "grch38start", "grch37start"]
    ref_keys = ["ref", "reference", "referenceallele", "ref_allele", "reference_allele"]
    alt_keys = ["alt", "alternate", "alternateallele", "alt_allele", "alternate_allele"]

    c = _pick_first(norm, chrom_keys)
    p = _pick_first(norm, pos_keys)
    r = _pick_first(norm, ref_keys)
    a = _pick_first(norm, alt_keys)

    if c and p and r and a:
        c = str(c).replace("chr", "").replace("CHR", "")
        return f"g.chr{c}:{p}{r}>{a}"
    return ""


def build_variant_caption(row) -> str:
    """
    Row-wise (Series/dict). Prefers HGVS if present; else synthesizes HGVS-g from VCF-like fields.
    """
    if isinstance(row, pd.DataFrame):
        raise TypeError(
            "build_variant_caption expects a row (Series/dict), got DataFrame."
        )

    g = first_nonempty(row, ["GeneSymbol", "gene", "GENE", "genesymbol"])

    p_tok, c_tok = _find_hgvs_anywhere(row)
    g_tok = "" if (p_tok or c_tok) else _build_hgvs_g(row)

    cons = first_nonempty(
        row, ["Consequence", "MolecularConsequence", "ConsequenceType", "consequence"]
    ).replace("_", " ")
    signif = first_nonempty(
        row,
        [
            "ClinicalSignificance",
            "Significance",
            "clinicalsignificance",
            "significance",
        ],
    ).lower()
    disease = first_nonempty(row, ["PhenotypeList", "Disease", "disease"]).split("|")[0]
    domain = first_nonempty(row, ["ProteinDomain", "Domain", "domain"])
    zyg = first_nonempty(row, ["Zygosity", "zygosity"])

    bits = []
    if g:
        bits.append(g)
    if p_tok:
        bits.append(p_tok)
    if c_tok:
        bits.append(c_tok)
    if not (p_tok or c_tok) and g_tok:
        bits.append(g_tok)  # <- HGVS-g

    # if cons:
    #     bits.append(f"consequence: {cons}")
    # if domain:
    #     bits.append(f"domain: {domain}")
    # if zyg:
    #     bits.append(f"zygosity: {zyg}")
    # if signif:
    #     bits.append(f"significance: {signif}")
    # if disease:
    #     bits.append(f"disease: {disease}")
    return " ".join(bits)


def batch_iter(xs: Iterable, bs: int):
    buf = []
    for x in xs:
        buf.append(x)
        if len(buf) == bs:
            yield buf
            buf = []
    if buf:
        yield buf


@torch.inference_mode()
def encode_text_batch(
    texts: List[str],
    model,
    tokenizer,
    device: str,
    max_length: int,
    pool: str = "cls",
) -> torch.Tensor:
    """
    Encode texts and return pooled embeddings.
    For BERT-family, CLS pooling is standard; mean also works.
    """
    toks = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    toks = {k: v.to(device) for k, v in toks.items()}
    out = model(**toks).last_hidden_state  # [B,T,D]
    if pool == "mean":
        attn = toks["attention_mask"].unsqueeze(-1)
        z = (out * attn).sum(1) / attn.sum(1).clamp_min(1e-6)
    else:
        z = out[:, 0, :]  # CLS
    return z.to(torch.float32)


def embed_texts_in_batches(
    texts: List[str],
    tok,
    enc,
    device: str,
    *,
    batch_size: int = 256,
    max_length: int = 128,
    pool: str = "cls",
    desc: str = "Encoding text",
) -> np.ndarray:
    if not isinstance(texts, list):
        texts = list(texts)
    chunks = []
    total = (len(texts) + batch_size - 1) // batch_size
    for chunk in tqdm(batch_iter(texts, batch_size), total=total, desc=desc):
        z = (
            encode_text_batch(chunk, enc, tok, device, max_length=max_length, pool=pool)
            .cpu()
            .numpy()
            .astype("float32")
        )
        chunks.append(z)
    D = enc.get_input_embeddings().embedding_dim
    return (
        np.concatenate(chunks, axis=0) if chunks else np.empty((0, D), dtype="float32")
    )


# ---------------------------
# I/O helpers
# ---------------------------


def save_text_arrays(out_npz: str, txt_embs: np.ndarray) -> None:
    """
    Save fp16 embeddings as uncompressed NPZ so we can mmap later.
    """
    np.savez(out_npz, txt=txt_embs.astype("float16", copy=False))


def write_text_meta_feather(df_with_caps: pd.DataFrame, out_meta: str) -> None:
    """
    Writes a Feather with the subset used + 'emb_row' mapping (0..N-1 if needed)
    and a 'caption' column.
    - If df already has 'emb_row', we keep it as-is (alignment with dna npz).
    """
    import pyarrow.feather as feather

    df = df_with_caps.copy()
    if "emb_row" not in df.columns:
        df["emb_row"] = np.arange(len(df), dtype=np.int32)
    feather.write_feather(df, out_meta)


def load_text_npz(npz_path: str):
    """
    Loads text NPZ with mmap.
    """
    return np.load(npz_path, mmap_mode="r")["txt"]


# ---------------------------
# Orchestrator (with caching)
# ---------------------------


def process_and_cache_text(
    meta_feather_in: str,
    *,
    out_meta: str,
    out_npz: str,
    model_id: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    device: Optional[str] = None,
    max_length: int = 128,
    batch_size: int = 256,
    force_captions: bool = False,
    force_embeddings: bool = False,
    pool: str = "cls",
) -> tuple[pd.DataFrame, str]:
    os.makedirs(os.path.dirname(out_meta), exist_ok=True)

    # 1) load meta
    df = pd.read_feather(meta_feather_in)

    # 2) captions (cache to feather)
    build_caps = True
    if os.path.exists(out_meta) and not force_captions:
        try:
            df_cached = pd.read_feather(out_meta)
            if "caption" in df_cached.columns and len(df_cached) == len(df):
                df = df_cached
                build_caps = False
                print(f"[cache] using captions: {out_meta}")
        except Exception:
            pass

    if build_caps:
        # IMPORTANT: apply row-wise, not on the whole DataFrame
        caps = df.apply(build_variant_caption, axis=1)

        # quick sanity: detect Series-stringification artifacts
        if caps.dtype == "object":
            s0 = str(caps.iloc[0])
            if "Name:" in s0 or re.match(r"^\s*0\s+", s0):
                raise RuntimeError(
                    "Caption building failed: looks like a whole Series got stringified into a caption."
                )

        df = df.copy()
        df["caption"] = caps.astype("string")
        write_text_meta_feather(df, out_meta)
        print(f"[cache] wrote captions: {out_meta} (N={len(df)})")

    # 3) embeddings (cache to npz)
    need_embed = True
    if os.path.exists(out_npz) and not force_embeddings:
        try:
            z = np.load(out_npz, mmap_mode="r")
            if "txt" in z.files and z["txt"].shape[0] == len(df):
                need_embed = False
                print(f"[cache] using text embeddings: {out_npz}")
        except Exception:
            need_embed = True

    if need_embed:
        tok, enc, device, ml = build_text_encoder(model_id, device, max_length)
        txt_embs = embed_texts_in_batches(
            df["caption"].tolist(),
            tok,
            enc,
            device,
            batch_size=batch_size,
            max_length=ml,
            pool=pool,
            desc="Encoding captions",
        )
        save_text_arrays(out_npz, txt_embs)
        print(
            f"[cache] wrote text embeddings: {out_npz} (N={len(df)}, D={txt_embs.shape[1]})"
        )

    return df, out_npz


# ---------------------------
# Datasets
# ---------------------------


class TextEmbDataset(Dataset):
    """
    Streams text embeddings from NPZ (key 'txt') and optional labels.
    """

    def __init__(
        self, meta_feather: str, txt_npz: str, label_col: Optional[str] = None
    ):
        self.meta = pd.read_feather(meta_feather)
        z = np.load(txt_npz, mmap_mode="r")
        if "txt" not in z.files:
            raise KeyError(f"{txt_npz} missing 'txt'")
        self.txt = z["txt"]  # (N, D_txt), memmap
        if self.txt.shape[0] != len(self.meta):
            raise RuntimeError(
                f"text rows {self.txt.shape[0]} != meta rows {len(self.meta)}"
            )
        self.label_col = label_col
        self.D_txt = int(self.txt.shape[1])

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(np.asarray(self.txt[idx], dtype="float32"))
        y = None
        if self.label_col is not None and self.label_col in self.meta.columns:
            y = torch.tensor(int(self.meta.iloc[idx][self.label_col]), dtype=torch.long)
        aux = {"idx": idx}
        return x, y, aux


class SeqTextPairDataset(Dataset):
    """
    Pairs sequence tower input (already cached elsewhere) with text embeddings by 'emb_row'.
    You pass:
      - seq_meta_feather: has 'emb_row', 'GeneSymbol', labels, etc.
      - seq_dna_npz: NPZ with 'dna_eff' (uncompressed)
      - go_npz: GO embeddings NPZ ('emb','genes')  -- same as your sequence dataset used
      - text_meta_feather: should be aligned subset of seq_meta (same N after any filtering)
      - text_npz: NPZ with 'txt' (uncompressed)

    It assumes 'emb_row' in text_meta is identical to seq_meta (typical if both built from the same meta subset).
    """

    def __init__(
        self,
        seq_dataset,  # your existing SequenceTowerDataset instance (already mmap'd dna+go)
        text_meta_feather: str,
        text_npz: str,
        label_col: Optional[str] = "_y",
    ):
        self.seq_ds = seq_dataset
        self.text_meta = pd.read_feather(text_meta_feather)
        z = np.load(text_npz, mmap_mode="r")
        if "txt" not in z.files:
            raise KeyError(f"{text_npz} missing 'txt'")
        self.txt = z["txt"]
        if len(self.text_meta) != len(self.seq_ds):
            raise RuntimeError(
                f"text N={len(self.text_meta)} != seq N={len(self.seq_ds)}"
            )
        self.label_col = label_col
        self.D_txt = int(self.txt.shape[1])

        # fast sanity: emb_row vectors match length; deeper checks optional
        if "emb_row" in self.text_meta.columns:
            if not np.all(self.text_meta["emb_row"].to_numpy() == self.seq_ds.emb_rows):
                print(
                    "[warn] emb_row mismatch between text and seq meta; assuming same order."
                )

    def __len__(self) -> int:
        return len(self.seq_ds)

    def __getitem__(self, idx: int):
        x_seq, y, aux = self.seq_ds[idx]  # (D_eff+D_go)
        x_txt = torch.from_numpy(np.asarray(self.txt[idx], dtype="float32"))  # (D_txt,)
        # labels: prefer cleaned label if present in text_meta; fall back to seq_ds y
        if self.label_col and self.label_col in self.text_meta.columns:
            y = torch.tensor(
                int(self.text_meta.iloc[idx][self.label_col]), dtype=torch.long
            )
        return x_seq, x_txt, y, aux


def collate_pair(batch):
    xs, xt, ys, aux = zip(*batch)
    Xs = torch.stack(xs, dim=0)
    Xt = torch.stack(xt, dim=0)
    y = None if ys[0] is None else torch.stack(ys, dim=0)
    return Xs, Xt, y, list(aux)
