# src/protein_utils.py
from __future__ import annotations
from typing import Iterable, Tuple, Optional, List
import os, re, json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

# =========================
# AA sanitization
# =========================
# keep 20 standard + B/Z/U/J/X/* as X; strip gaps
_AA_VALID = set("ACDEFGHIKLMNPQRSTVWY")


def sanitize_aa(seq: str) -> str:
    if not isinstance(seq, str):
        return ""
    s = seq.replace("-", "").upper()
    return "".join(c if c in _AA_VALID else "X" for c in s)


# =========================
# Encoder
# =========================
def build_protein_encoder(
    model_id: str = "facebook/esm2_t33_650M_UR50D",
    device: Optional[str] = None,
):
    """
    Works with HF ESM models. Uses mean pooling by default.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=False
    )
    if tok.pad_token is None:
        # common ESM pad
        tok.add_special_tokens({"pad_token": tok.eos_token or "<pad>"})
    tok.padding_side = "right"

    dtype = torch.bfloat16 if (device == "cuda") else torch.float32
    enc = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype)
    if enc.get_input_embeddings().num_embeddings < len(tok):
        enc.resize_token_embeddings(len(tok))
    enc = enc.to(device).eval()
    return tok, enc, device


@torch.inference_mode()
def encode_batch(
    seqs: List[str],
    model,
    tokenizer,
    device: str,
    max_length: int = 4096,
    pool: str = "mean",
) -> torch.Tensor:
    toks = tokenizer(
        seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    toks = {k: v.to(device) for k, v in toks.items()}
    out = model(**toks).last_hidden_state  # [B,T,D]

    # pool over non-pad tokens
    attn = toks["attention_mask"].unsqueeze(-1).to(out.dtype)  # [B,T,1]
    if pool == "cls":
        z = out[:, 0, :]
    else:
        z = (out * attn).sum(1) / attn.sum(1).clamp_min(1e-6)
    return z.to(torch.float32)


def batch_iter(xs: Iterable, bs: int):
    buf = []
    for x in xs:
        buf.append(x)
        if len(buf) == bs:
            yield buf
            buf = []
    if buf:
        yield buf


def embed_sequences_in_batches(
    seqs: List[str],
    tok,
    enc,
    device: str,
    batch_size: int = 8,
    max_length: int = 4096,
    pool: str = "mean",
    desc: str = "Encoding protein",
) -> np.ndarray:
    if not isinstance(seqs, list):
        seqs = list(seqs)
    chunks = []
    total = (len(seqs) + batch_size - 1) // batch_size
    for chunk in tqdm(batch_iter(seqs, batch_size), total=total, desc=desc):
        z = encode_batch(chunk, enc, tok, device, max_length=max_length, pool=pool)
        chunks.append(z.cpu().numpy().astype("float32"))
    D = enc.get_input_embeddings().embedding_dim
    return (
        np.concatenate(chunks, axis=0) if chunks else np.empty((0, D), dtype="float32")
    )


# =========================
# I/O + VEP integration
# =========================
def _read_any_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()
    if suf in {".feather", ".ft"}:
        return pd.read_feather(p)
    if suf in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if suf in {".csv"}:
        return pd.read_csv(p)
    # fallback tsv
    return pd.read_table(p)


def load_vep_sequences(vep_combined_path: str | Path) -> pd.DataFrame:
    """
    Expect columns from your vep_pipeline join:
      id (VariationID as str), protein_id, hgvsp, consequence_terms, seq_wt, seq_mt
    Returns rows with both WT and MT present, de-duplicated per (id, protein_id, hgvsp).
    """
    df = _read_any_table(vep_combined_path)
    # normalize id
    if "id" not in df.columns and "VariationID" in df.columns:
        df["id"] = df["VariationID"].astype(str)
    else:
        df["id"] = df["id"].astype(str)

    # sanitize sequences
    for col in ("seq_wt", "seq_mt"):
        if col in df.columns:
            df[col] = df[col].map(sanitize_aa)

    # keep rows that actually have both
    keep = (
        df["seq_wt"].notna()
        & df["seq_mt"].notna()
        & (df["seq_wt"] != "")
        & (df["seq_mt"] != "")
    )
    df = df.loc[keep].copy()

    # de-dup (id, protein, p-dot)
    keys = ["id", "protein_id", "hgvsp"]
    keys = [k for k in keys if k in df.columns]
    if keys:
        df = (
            df.sort_values(keys)
            .drop_duplicates(keys, keep="first")
            .reset_index(drop=True)
        )

    return df[
        ["id", "protein_id", "hgvsp", "consequence_terms", "seq_wt", "seq_mt"]
    ].copy()


# =========================
# Post-processing & cache
# =========================
def compute_effect_vectors(wt: np.ndarray, mt: np.ndarray) -> np.ndarray:
    eff = (mt - wt).astype("float32")
    norms = np.linalg.norm(eff, axis=1, keepdims=True) + 1e-8
    return (eff / norms).astype("float32")


def save_protein_arrays(
    out_npz: str, prot_wt=None, prot_mt=None, prot_eff=None
) -> None:
    np.savez(
        out_npz,
        **(
            {}
            if prot_wt is None
            else {"prot_wt": prot_wt.astype("float16", copy=False)}
        ),
        **(
            {}
            if prot_mt is None
            else {"prot_mt": prot_mt.astype("float16", copy=False)}
        ),
        **(
            {}
            if prot_eff is None
            else {"prot_eff": prot_eff.astype("float16", copy=False)}
        ),
    )


def write_protein_meta(meta_df: pd.DataFrame, out_feather: str) -> None:
    import pyarrow.feather as feather

    feather.write_feather(meta_df, out_feather)


# =========================
# Orchestrator
# =========================
def process_and_cache_protein(
    vep_combined_path: str | Path,
    *,
    out_meta: str,
    out_npz: str,
    device: Optional[str] = None,
    model_id: str = "facebook/esm2_t33_650M_UR50D",
    batch_size: int = 8,
    pool: str = "mean",
    max_length: int = 2048,  # safe default; ESM2 supports up to 4k on bigger models
    force_embeddings: bool = False,
) -> tuple[pd.DataFrame, str]:
    """
    Loads VEP-joined WT/MT sequences, embeds with ESM2, saves NPZ(feats) + meta(feather).
    Returns (kept_meta_df, out_npz_path).
    """
    os.makedirs(os.path.dirname(out_meta), exist_ok=True)

    # Load and keep usable rows
    pairs = load_vep_sequences(vep_combined_path)
    if len(pairs) == 0:
        raise RuntimeError("No WT/MT sequences found in VEP combined file.")
    kept = pairs.reset_index(drop=True).copy()
    kept["emb_row"] = np.arange(len(kept), dtype=np.int32)

    need_embed = True
    if os.path.exists(out_npz) and not force_embeddings:
        npz = np.load(out_npz)
        if "prot_eff" in npz.files and npz["prot_eff"].shape[0] == len(kept):
            need_embed = False
            print(f"[cache] using protein embeddings: {out_npz}")

    if need_embed:
        tok, enc, device = build_protein_encoder(model_id=model_id, device=device)
        wt_emb = embed_sequences_in_batches(
            kept["seq_wt"].tolist(),
            tok,
            enc,
            device,
            batch_size=batch_size,
            max_length=max_length,
            pool=pool,
            desc="ESM WT",
        )
        mt_emb = embed_sequences_in_batches(
            kept["seq_mt"].tolist(),
            tok,
            enc,
            device,
            batch_size=batch_size,
            max_length=max_length,
            pool=pool,
            desc="ESM MT",
        )
        eff = compute_effect_vectors(wt_emb, mt_emb)
        save_protein_arrays(out_npz, prot_eff=eff)  # keep only effect to save space
        print(f"[cache] wrote protein embeddings: {out_npz} (N={len(kept)})")

    write_protein_meta(kept, out_meta)
    print(f"[meta] wrote protein meta: {out_meta} (N={len(kept)})")
    return kept, out_npz
