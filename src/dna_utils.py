# src/dna_utils.py
from __future__ import annotations
from typing import Iterable, Tuple, Optional, List, Dict, Any, Sequence

import os
from pathlib import Path

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


def _center_crop_sequence(seq: str, center: int, target_len: int) -> tuple[str, int, int]:
    """Crop ``seq`` to ``target_len`` characters centred on ``center``.

    Returns ``(cropped_seq, new_center, offset)`` where ``offset`` is the number
    of characters removed from the start of the original sequence.
    """

    if target_len <= 0:
        raise ValueError("target_len must be positive")

    seq = seq or ""
    L = len(seq)
    if L == 0:
        return "", 0, 0

    center = max(0, min(int(center), L - 1))
    if L <= target_len:
        return seq, center, 0

    half = target_len // 2
    start = center - half
    if start < 0:
        start = 0
    end = start + target_len
    if end > L:
        end = L
        start = max(0, end - target_len)

    cropped = seq[start:end]
    new_center = center - start
    new_center = max(0, min(new_center, len(cropped) - 1))
    return cropped, new_center, start


# ---------------------------
# Encoder
# ---------------------------
def build_dna_encoder(
    model_id: str = "InstaDeepAI/nucleotide-transformer-500m-1000g",
    max_length: int = 512,
    device: Optional[str] = None,
    revision: Optional[str] = None,
):
    device = device or get_device()
    load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if revision:
        load_kwargs["revision"] = revision

    def _load_tokenizer(*, force_download: bool = False):
        tok_kwargs = dict(load_kwargs)
        if force_download:
            tok_kwargs["force_download"] = True
        return AutoTokenizer.from_pretrained(
            model_id,
            use_fast=False,
            **tok_kwargs,
        )

    tok = _load_tokenizer()
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": tok.unk_token or tok.eos_token or "[PAD]"})
    tok.padding_side = "right"
    tok.model_max_length = max_length

    dev_str = device.type if isinstance(device, torch.device) else str(device)
    dev_str = dev_str.lower()
    prefer_dtype = torch.bfloat16 if dev_str.startswith("cuda") else torch.float32

    def _load_model(*, force_download: bool = False):
        model_kwargs = dict(load_kwargs)
        if force_download:
            model_kwargs["force_download"] = True
        try:
            return AutoModel.from_pretrained(
                model_id,
                dtype=prefer_dtype,
                **model_kwargs,
            )
        except TypeError:
            # transformers<4.45 fallback where ``dtype`` arg unsupported
            model_kwargs.pop("dtype", None)
            model_kwargs["torch_dtype"] = prefer_dtype
            return AutoModel.from_pretrained(
                model_id,
                **model_kwargs,
            )

    try:
        enc = _load_model()
    except RuntimeError as exc:
        if revision is None and "size mismatch" in str(exc):
            tok = _load_tokenizer(force_download=True)
            enc = _load_model(force_download=True)
        else:
            raise

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
) -> Tuple[str, str, int, int, int]:
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
    return sanitize(seq_ref), sanitize(seq_alt), win_start, win_end, int(rel)


def collect_variant_windows(
    df: pd.DataFrame, fasta: Fasta, window: int
) -> Tuple[pd.DataFrame, List[str], List[str], List[int], List[int], List[int]]:
    ref_seqs, alt_seqs, starts, ends, edits, keep_idx = [], [], [], [], [], []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Fetching DNA windows"):
        try:
            rseq, aseq, win_start, win_end, rel = fetch_dna_window(
                fasta, row, window=window
            )
        except Exception:
            continue
        ref_seqs.append(rseq)
        alt_seqs.append(aseq)
        starts.append(int(win_start))
        ends.append(int(win_end))
        edits.append(int(rel))
        keep_idx.append(i)
    if not keep_idx:
        raise RuntimeError("No valid variants matched FASTA/REF.")
    kept = df.loc[keep_idx].reset_index(drop=True)
    return kept, ref_seqs, alt_seqs, starts, ends, edits


# ---------------------------
# Encoding
# ---------------------------
@torch.inference_mode()
def encode_sequence_tokens(
    seqs: List[str],
    model,
    tokenizer,
    device: str,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-token embeddings and attention masks for ``seqs``."""

    toks = tokenizer(
        seqs,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    toks = {k: v.to(device) for k, v in toks.items()}
    out = model(**toks).last_hidden_state  # [B, T, D]
    attn = toks["attention_mask"]  # [B, T]

    # drop the first token (special CLS/BOS) to align with raw sequence positions
    out = out[:, 1:, :]
    attn = attn[:, 1:]
    return out.to(torch.float32), attn.to(torch.float32)


def batch_iter(xs: Iterable, bs: int):
    buf = []
    for x in xs:
        buf.append(x)
        if len(buf) == bs:
            yield buf
            buf = []
    if buf:
        yield buf


class DNAConditioningStreamer:
    """Helper that lazily encodes DNA conditioning tensors on demand."""

    def __init__(
        self,
        *,
        tokenizer,
        model,
        device,
        seq_len: int,
        embed_dim: int,
        batch_size: int,
        records: Dict[int, Dict[str, Any]],
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.seq_len = int(seq_len)
        self.embed_dim = int(embed_dim)
        self.batch_size = max(1, int(batch_size))
        self.records = records
        self._base_pos = np.arange(self.seq_len, dtype="float32")

    def encode(self, indices: Sequence[int]) -> Dict[int, Dict[str, np.ndarray]]:
        if self.seq_len <= 0 or self.embed_dim <= 0:
            return {}

        valid = [int(i) for i in indices if int(i) in self.records]
        if not valid:
            return {}

        outputs: Dict[int, Dict[str, np.ndarray]] = {}
        max_len = self.seq_len + 1

        for start in range(0, len(valid), self.batch_size):
            chunk = valid[start : start + self.batch_size]
            chunk_records = [self.records[i] for i in chunk]

            ref_batch = [rec["ref_seq"] for rec in chunk_records]
            alt_batch = [rec["alt_seq"] for rec in chunk_records]
            edit_batch = [int(rec["edit_idx"]) for rec in chunk_records]

            ref_hidden, ref_attn = encode_sequence_tokens(
                ref_batch,
                self.model,
                self.tokenizer,
                self.device,
                max_length=max_len,
            )
            alt_hidden, alt_attn = encode_sequence_tokens(
                alt_batch,
                self.model,
                self.tokenizer,
                self.device,
                max_length=max_len,
            )

            ref_hidden = ref_hidden[:, : self.seq_len, :].to(torch.float32).cpu().numpy()
            alt_hidden = alt_hidden[:, : self.seq_len, :].to(torch.float32).cpu().numpy()
            ref_attn = ref_attn[:, : self.seq_len].to(torch.float32).cpu().numpy()
            alt_attn = alt_attn[:, : self.seq_len].to(torch.float32).cpu().numpy()
            gap = np.logical_xor(ref_attn > 0, alt_attn > 0).astype("float32")

            for j, arr_idx in enumerate(chunk):
                edit_pos = min(max(int(edit_batch[j]), 0), self.seq_len - 1)
                edit_vec = np.zeros((self.seq_len,), dtype="float32")
                if self.seq_len > 0:
                    edit_vec[edit_pos] = 1.0

                splice_mask = chunk_records[j].get("splice_mask")
                if splice_mask is None:
                    splice_arr = np.zeros((self.seq_len,), dtype="float32")
                else:
                    splice_arr = np.zeros((self.seq_len,), dtype="float32")
                    src = np.asarray(splice_mask, dtype="float32").reshape(-1)
                    take = min(len(src), self.seq_len)
                    if take > 0:
                        splice_arr[:take] = src[:take]

                outputs[arr_idx] = {
                    "ref_tokens": ref_hidden[j],
                    "alt_tokens": alt_hidden[j],
                    "ref_mask": ref_attn[j],
                    "alt_mask": alt_attn[j],
                    "edit_mask": edit_vec,
                    "gap_mask": gap[j],
                    "pos": (self._base_pos - float(edit_pos))[:, None],
                    "splice_mask": splice_arr,
                }

        return outputs

    def empty(self) -> Dict[str, np.ndarray]:
        zeros_2d = np.zeros((self.seq_len, self.embed_dim), dtype="float32")
        zeros_1d = np.zeros((self.seq_len,), dtype="float32")
        return {
            "ref_tokens": zeros_2d,
            "alt_tokens": zeros_2d,
            "ref_mask": zeros_1d,
            "alt_mask": zeros_1d,
            "edit_mask": zeros_1d,
            "gap_mask": zeros_1d,
            "pos": np.zeros((self.seq_len, 1), dtype="float32"),
            "splice_mask": zeros_1d,
        }


def encode_dna_conditioning(
    meta: pd.DataFrame,
    *,
    model_id: str = "InstaDeepAI/nucleotide-transformer-500m-1000g",
    revision: Optional[str] = None,
    max_length: int = 384,
    batch_size: int = 16,
    device: Optional[str] = None,
) -> DNAConditioningStreamer:
    """Prepare a streamer that lazily encodes DNA conditioning tensors."""

    required = {
        "dna_ref_seq",
        "dna_alt_seq",
        "dna_edit_idx",
        "dna_seq_len",
        "dna_embedding_idx",
    }
    missing = [col for col in required if col not in meta.columns]
    if missing:
        raise KeyError(
            "Metadata frame missing DNA sequence columns: "
            + ", ".join(sorted(missing))
        )

    extra_cols: list[str] = []
    if "dna_splice_mask" in meta.columns:
        extra_cols.append("dna_splice_mask")

    uniq = (
        meta[meta["dna_embedding_idx"] >= 0][list(required) + extra_cols]
        .drop_duplicates("dna_embedding_idx")
        .sort_values("dna_embedding_idx")
    )

    if uniq.empty:
        tok, enc, run_device = build_dna_encoder(
            model_id=model_id,
            max_length=max_length,
            device=device,
            revision=revision,
        )
        enc.eval()
        return DNAConditioningStreamer(
            tokenizer=tok,
            model=enc,
            device=run_device,
            seq_len=0,
            embed_dim=0,
            batch_size=batch_size,
            records={},
        )

    seq_len = int(uniq["dna_seq_len"].iloc[0])
    tok, enc, run_device = build_dna_encoder(
        model_id=model_id,
        max_length=max_length,
        device=device,
        revision=revision,
    )
    seq_len = max(1, min(seq_len, int(tok.model_max_length) - 1))
    embed_dim = int(enc.get_input_embeddings().embedding_dim)

    records: Dict[int, Dict[str, Any]] = {}
    has_splice = "dna_splice_mask" in uniq.columns

    for row in uniq.itertuples(index=False):
        arr_idx = int(row.dna_embedding_idx)
        rec: Dict[str, Any] = {
            "ref_seq": str(row.dna_ref_seq),
            "alt_seq": str(row.dna_alt_seq),
            "edit_idx": int(row.dna_edit_idx),
        }
        if has_splice:
            mask = getattr(row, "dna_splice_mask")
            if isinstance(mask, (list, tuple, np.ndarray)):
                arr = np.asarray(mask, dtype="float32")
                rec["splice_mask"] = arr[:seq_len]
        records[arr_idx] = rec

    return DNAConditioningStreamer(
        tokenizer=tok,
        model=enc,
        device=run_device,
        seq_len=seq_len,
        embed_dim=embed_dim,
        batch_size=batch_size,
        records=records,
    )


# ---------------------------
# Post-processing & I/O
# ---------------------------
def write_windows_feather(
    kept_df: pd.DataFrame,
    ref_seqs,
    alt_seqs,
    start_positions,
    end_positions,
    edit_positions,
    out_path: str,
    *,
    seq_len: int,
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
    if edit_positions:
        df["dna_edit_idx"] = list(edit_positions)
    df["dna_seq_len"] = int(seq_len)
    df["dna_window_size"] = int(window)
    df["dna_fasta_path"] = str(fasta_path)
    feather.write_feather(df, out_path)


def load_windows_feather(
    path: str,
    window_bp: int,
    *,
    expected_seq_len: int,
) -> tuple[
    pd.DataFrame,
    list[str],
    list[str],
    list[int],
    list[int],
    list[int],
    int,
    int,
    str,
]:
    df = pd.read_feather(path)
    starts = df["dna_window_start"].tolist() if "dna_window_start" in df else []
    ends = df["dna_window_end"].tolist() if "dna_window_end" in df else []
    edits = df["dna_edit_idx"].astype("int64").tolist() if "dna_edit_idx" in df else []
    if "dna_window_size" not in df.columns:
        raise KeyError(
            "DNA windows metadata missing 'dna_window_size'; regenerate the cache."
        )
    window_size = int(df["dna_window_size"].iloc[0])
    if "dna_seq_len" not in df.columns:
        raise KeyError(
            "DNA windows metadata missing 'dna_seq_len'; regenerate the cache."
        )
    seq_len = int(df["dna_seq_len"].iloc[0])
    if seq_len != int(expected_seq_len):
        raise ValueError(
            "DNA windows metadata seq_len mismatch: "
            f"found {seq_len}, expected {expected_seq_len}."
        )
    fasta = str(df.get("dna_fasta_path", pd.Series([""], dtype="string")).iloc[0])
    return (
        df,
        df["dna_ref_seq"].tolist(),
        df["dna_alt_seq"].tolist(),
        starts,
        ends,
        edits,
        seq_len,
        window_size,
        fasta,
    )


# ---------------------------
# Orchestrator (metadata only)
# ---------------------------
def process_dna_sequences(
    df: pd.DataFrame,
    fasta: Fasta,
    *,
    window: int,
    max_length: int,
    out_meta: str = "out/clinvar_dna.feather",
    windows_meta: str | None = None,
    limit: Optional[int] = None,
    force_windows: bool = False,
) -> pd.DataFrame:
    """
    Prepare DNA reference/alternate windows and accompanying metadata.

    This replaces the old caching pipeline by returning a dataframe containing
    the sequences required for downstream embedding.
    """

    if limit:
        df = df.iloc[:limit].copy()

    if windows_meta is None:
        stem = os.path.splitext(os.path.basename(out_meta))[0]
        windows_meta = os.path.join(
            os.path.dirname(out_meta), stem.replace(".feather", "") + "_windows.feather"
        )

    target_seq_len = max(1, min(int(max_length) - 1, 2 * int(window) + 1))
    fasta_path = getattr(fasta, "filename", "") or ""
    reuse_windows = False
    if os.path.exists(windows_meta) and not force_windows:
        try:
            (
                kept_df,
                ref_seqs,
                alt_seqs,
                start_positions,
                end_positions,
                edit_positions,
                seq_len_meta,
                window_meta,
                fasta_path,
            ) = load_windows_feather(
                windows_meta, window, expected_seq_len=target_seq_len
            )
            if int(seq_len_meta) == int(target_seq_len):
                too_long = any(
                    len(str(seq)) > target_seq_len for seq in ref_seqs[: min(8, len(ref_seqs))]
                )
                if too_long:
                    print(
                        "[dna][windows] cached sequences exceed target length -> recompute",
                        f"target={target_seq_len}",
                    )
                else:
                    reuse_windows = True
                    window = int(window_meta)
                    print(
                        f"[dna][windows] reuse {windows_meta} rows={len(kept_df)} force={force_windows}"
                    )
            else:
                print(
                    "[dna][windows] cached seq_len mismatch -> recompute",
                    f"cached={seq_len_meta} desired={target_seq_len}",
                )
        except Exception:
            reuse_windows = False

    if not reuse_windows:
        (
            kept_df,
            ref_raw,
            alt_raw,
            start_positions,
            end_positions,
            edit_raw,
        ) = collect_variant_windows(df, fasta, window=window)
        if len(kept_df) == 0:
            raise RuntimeError("No valid variants after window collection.")

        ref_seqs, alt_seqs = [], []
        edit_positions: list[int] = []
        start_adj: list[int] = []
        end_adj: list[int] = []
        for rseq, aseq, start, end, rel in zip(
            ref_raw, alt_raw, start_positions, end_positions, edit_raw
        ):
            cropped_ref, edit_idx, offset = _center_crop_sequence(
                rseq, rel, target_seq_len
            )
            cropped_alt, _, _ = _center_crop_sequence(aseq, rel, target_seq_len)
            ref_seqs.append(cropped_ref)
            alt_seqs.append(cropped_alt)
            edit_positions.append(int(edit_idx))
            if start_positions:
                new_start = int(start) + int(offset)
                start_adj.append(new_start)
                span = max(len(cropped_ref), 1)
                end_adj.append(new_start + span - 1)
            else:
                start_adj.append(int(start))
                end_adj.append(int(end))

        start_positions = start_adj
        end_positions = end_adj
        os.makedirs(os.path.dirname(windows_meta), exist_ok=True)
        write_windows_feather(
            kept_df,
            ref_seqs,
            alt_seqs,
            start_positions,
            end_positions,
            edit_positions,
            windows_meta,
            seq_len=target_seq_len,
            window=window,
            fasta_path=fasta_path,
        )
        print(f"[dna][windows] wrote {windows_meta} rows={len(kept_df)}")
    else:
        if len(kept_df) == 0:
            raise RuntimeError("No valid variants after window collection.")
        if not edit_positions:
            edit_positions = [
                min(target_seq_len // 2, max(len(seq) - 1, 0)) for seq in ref_seqs
            ]

    kept_df = kept_df.copy()
    kept_df["dna_embedding_idx"] = np.arange(len(kept_df), dtype=np.int32)
    kept_df["dna_ref_seq"] = list(ref_seqs)
    kept_df["dna_alt_seq"] = list(alt_seqs)
    if start_positions:
        kept_df["dna_window_start"] = list(start_positions)
    if end_positions:
        kept_df["dna_window_end"] = list(end_positions)
    kept_df["dna_window_size"] = int(window)
    kept_df["dna_fasta_path"] = str(fasta_path)
    kept_df["dna_edit_idx"] = list(int(x) for x in edit_positions)
    kept_df["dna_seq_len"] = int(target_seq_len)
    print(f"[dna][meta] prepared frame rows={len(kept_df)}")
    return kept_df
