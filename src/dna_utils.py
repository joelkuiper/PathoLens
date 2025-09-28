# src/dna_utils.py
from __future__ import annotations
from typing import Iterable, Tuple, Optional, List

import os
import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from util import get_device

# ----- constants for the “toy but stable” setup -----
IUPAC_VALID = set("ACGTN")


def sanitize(seq: str) -> str:
    s = seq.upper()
    return "".join(c if c in IUPAC_VALID else "N" for c in s)


# ---------------------------
# Encoder
# ---------------------------
def build_dna_encoder(
    model_id: str = "InstaDeepAI/nucleotide-transformer-500m-1000g",
    max_length: int = 512,
    device: Optional[str] = None,
):
    device = device or get_device()
    tok = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=False
    )
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": tok.unk_token or tok.eos_token or "[PAD]"})
    tok.padding_side = "right"
    tok.model_max_length = max_length

    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    enc = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch_dtype
    )
    if enc.get_input_embeddings().num_embeddings < len(tok):
        enc.resize_token_embeddings(len(tok))
    # cap to position embeddings if present
    pmax = getattr(enc.config, "max_position_embeddings", None)
    if pmax is not None:
        tok.model_max_length = min(int(tok.model_max_length), int(pmax))
    enc = enc.to(device).eval()
    return tok, enc, device


# ---------------------------
# FASTA / windowing
# ---------------------------
def load_fasta(path: str) -> Fasta:
    return Fasta(path)


def fetch_dna_window(
    fasta: Fasta, row: pd.Series, window: int
) -> Tuple[str, str, int, int]:
    chrom = str(row["ChromosomeAccession"])
    pos = int(row["PositionVCF"])  # 1-based
    ref = str(row["ReferenceAlleleVCF"])
    alt = str(row["AlternateAlleleVCF"])

    win_start = max(1, pos - window)
    win_end = pos + len(ref) - 1 + window

    seq_ref = fasta[chrom][win_start - 1 : win_end].seq.upper()
    rel = pos - win_start
    if seq_ref[rel : rel + len(ref)] != ref.upper():
        raise ValueError("REF mismatch")

    seq_alt = seq_ref[:rel] + alt.upper() + seq_ref[rel + len(ref) :]
    return sanitize(seq_ref), sanitize(seq_alt), win_start, win_end


def collect_variant_windows(
    df: pd.DataFrame, fasta: Fasta, window: int
) -> Tuple[pd.DataFrame, List[str], List[str], List[int], List[int]]:
    ref_seqs, alt_seqs, starts, ends, keep_idx = [], [], [], [], []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Fetching DNA windows"):
        try:
            rseq, aseq, win_start, win_end = fetch_dna_window(
                fasta, row, window=window
            )
        except Exception:
            continue
        ref_seqs.append(rseq)
        alt_seqs.append(aseq)
        starts.append(int(win_start))
        ends.append(int(win_end))
        keep_idx.append(i)
    if not keep_idx:
        raise RuntimeError("No valid variants matched FASTA/REF.")
    kept = df.loc[keep_idx].reset_index(drop=True)
    return kept, ref_seqs, alt_seqs, starts, ends


# ---------------------------
# Encoding
# ---------------------------
@torch.inference_mode()
def encode_sequence_batch(
    seqs: List[str],
    model,
    tokenizer,
    device: str,
    max_length: int,
    pool: str = "mean",
) -> torch.Tensor:
    """
    Encode sequences and return pooled embeddings.

    pool:
      - "mean": masked mean pooling
      - "cls": take embedding of token 0 (only if you're sure it exists/works)
    """
    toks = tokenizer(
        seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    toks = {k: v.to(device) for k, v in toks.items()}
    out = model(**toks).last_hidden_state  # [B,T,D]

    if pool == "cls":
        z = out[:, 0, :]
    else:  # mean pooling
        attn = toks["attention_mask"].unsqueeze(-1)  # [B,T,1]
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
    max_length: int = 512,
    batch_size: int = 16,
    pool: str = "mean",
    desc: str = "Encoding",
) -> np.ndarray:
    if not isinstance(seqs, list):
        seqs = list(seqs)
    chunks = []
    total = (len(seqs) + batch_size - 1) // batch_size
    for chunk in tqdm(batch_iter(seqs, batch_size), total=total, desc=f"{desc}"):
        z = (
            encode_sequence_batch(
                chunk, enc, tok, device, max_length=max_length, pool=pool
            )
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
# Post-processing & I/O
# ---------------------------
def compute_effect_vectors(ref_embs: np.ndarray, alt_embs: np.ndarray) -> np.ndarray:
    eff = (alt_embs - ref_embs).astype("float32")
    norms = np.linalg.norm(eff, axis=1, keepdims=True) + 1e-8
    return (eff / norms).astype("float32")


def save_dna_arrays(out_npz: str, dna_ref, dna_alt, dna_eff) -> None:
    # effect-only + fp16 on disk
    np.savez(out_npz, dna_eff=dna_eff.astype("float16", copy=False))


def write_windows_feather(
    kept_df: pd.DataFrame,
    ref_seqs,
    alt_seqs,
    start_positions,
    end_positions,
    out_path: str,
    *,
    window: int,
    fasta_path: str,
):
    import pyarrow.feather as feather

    df = kept_df.copy()
    df["dna_ref_seq"] = list(ref_seqs)
    df["dna_alt_seq"] = list(alt_seqs)
    if start_positions:
        df["dna_window_start"] = list(start_positions)
    if end_positions:
        df["dna_window_end"] = list(end_positions)
    df["dna_window_size"] = int(window)
    df["dna_fasta_path"] = str(fasta_path)
    feather.write_feather(df, out_path)


def load_windows_feather(
    path: str, window_bp: int
) -> tuple[
    pd.DataFrame,
    list[str],
    list[str],
    list[int],
    list[int],
    int,
    str,
]:
    df = pd.read_feather(path)
    starts = df["dna_window_start"].tolist() if "dna_window_start" in df else []
    ends = df["dna_window_end"].tolist() if "dna_window_end" in df else []
    window_size = int(
        df.get("dna_window_size", pd.Series([window_bp], dtype="int64")).iloc[0]
    )
    fasta = str(df.get("dna_fasta_path", pd.Series([""], dtype="string")).iloc[0])
    return (
        df,
        df["dna_ref_seq"].tolist(),
        df["dna_alt_seq"].tolist(),
        starts,
        ends,
        window_size,
        fasta,
    )


# ---------------------------
# Orchestrator (with caching)
# ---------------------------
def process_and_cache_dna(
    df: pd.DataFrame,
    fasta: Fasta,
    *,
    window: int,
    batch_size: int = 16,
    pool: str = "cls",
    max_length: int,
    out_meta: str = "out/clinvar_dna.feather",
    out_npz: str = "out/dna_embeddings.npz",
    windows_meta: str | None = None,
    device: Optional[str] = None,
    limit: Optional[int] = None,
    dna_model_id: str = "InstaDeepAI/nucleotide-transformer-500m-1000g",
    force_windows: bool = False,
    force_embeddings: bool = False,
) -> tuple[pd.DataFrame, str]:
    os.makedirs(os.path.dirname(out_meta), exist_ok=True)
    if limit:
        df = df.iloc[:limit].copy()

    # A) windows cache
    if windows_meta is None:
        stem = os.path.splitext(os.path.basename(out_meta))[0]
        windows_meta = os.path.join(
            os.path.dirname(out_meta), stem.replace(".feather", "") + "_windows.feather"
        )
    if os.path.exists(windows_meta) and not force_windows:
        (
            kept_df,
            ref_seqs,
            alt_seqs,
            start_positions,
            end_positions,
            window,
            fasta_path,
        ) = load_windows_feather(windows_meta, window)
        print(
            f"[dna][windows] reuse {windows_meta} rows={len(kept_df)} force={force_windows}"
        )
    else:
        (
            kept_df,
            ref_seqs,
            alt_seqs,
            start_positions,
            end_positions,
        ) = collect_variant_windows(df, fasta, window=window)
        fasta_path = getattr(fasta, "filename", "") or ""
        write_windows_feather(
            kept_df,
            ref_seqs,
            alt_seqs,
            start_positions,
            end_positions,
            windows_meta,
            window=window,
            fasta_path=fasta_path,
        )
        print(f"[dna][windows] wrote {windows_meta} rows={len(kept_df)}")
    if len(kept_df) == 0:
        raise RuntimeError("No valid variants after window collection.")

    # B) embeddings cache
    need_embed = True
    if os.path.exists(out_npz) and not force_embeddings:
        npz = np.load(out_npz)
        if "dna_eff" in npz.files:
            need_embed = npz["dna_eff"].shape[0] != len(kept_df)
        else:
            need_embed = True
        if not need_embed:
            print(
                f"[dna][embeddings] reuse {out_npz} rows={len(kept_df)} force={force_embeddings}"
            )

    if need_embed:
        tok, enc, device = build_dna_encoder(dna_model_id, max_length, device)
        ref_embs = embed_sequences_in_batches(
            ref_seqs,
            tok,
            enc,
            device,
            batch_size=batch_size,
            pool=pool,
            max_length=max_length,
            desc="Encoding ref",
        )
        alt_embs = embed_sequences_in_batches(
            alt_seqs,
            tok,
            enc,
            device,
            batch_size=batch_size,
            pool=pool,
            max_length=max_length,
            desc="Encoding alt",
        )
        eff_embs = compute_effect_vectors(ref_embs, alt_embs)
        save_dna_arrays(out_npz, ref_embs, alt_embs, eff_embs)
        print(f"[dna][embeddings] wrote {out_npz} rows={len(kept_df)}")

    # C) final meta aligned to arrays
    npz = np.load(out_npz)
    if "dna_eff" not in npz.files:
        raise RuntimeError(f"{out_npz} missing 'dna_eff'")
    N = npz["dna_eff"].shape[0]
    if N != len(kept_df):
        raise RuntimeError(
            f"Alignment mismatch: arrays N={N} vs kept_df N={len(kept_df)}"
        )
    kept_df = kept_df.copy()
    kept_df["dna_embedding_idx"] = np.arange(N, dtype=np.int32)
    kept_df["dna_ref_seq"] = list(ref_seqs)
    kept_df["dna_alt_seq"] = list(alt_seqs)
    if start_positions:
        kept_df["dna_window_start"] = list(start_positions)
    if end_positions:
        kept_df["dna_window_end"] = list(end_positions)
    kept_df["dna_window_size"] = int(window)
    kept_df["dna_fasta_path"] = str(fasta_path)
    print(f"[dna][meta] prepared frame rows={N}")
    return kept_df, out_npz
