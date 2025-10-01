# src/dna_utils.py
from __future__ import annotations
from typing import Iterable, Tuple, Optional, List, Dict, Mapping

import math

import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from util import get_device

from src.sequence_cache import (
    SequenceArchiveLayout,
    ensure_h5_path,
    initialise_sequence_archive,
    open_sequence_archive,
)

# Layout describing DNA HDF5 dataset names
DNA_ARCHIVE_LAYOUT = SequenceArchiveLayout(
    wt_tokens="dna_ref_tokens",
    mt_tokens="dna_alt_tokens",
    wt_mask="dna_ref_mask",
    mt_mask="dna_alt_mask",
    edit_mask="dna_edit_mask",
    gap_mask="dna_gap_mask",
    pos="dna_pos",
    extra_masks=("dna_splice_mask",),
)

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


# ---------------------------
# HDF5 archive helpers
# ---------------------------

_DNA_REQUIRED_KEYS = {
    "dna_ref_tokens",
    "dna_alt_tokens",
    "dna_ref_mask",
    "dna_alt_mask",
    "dna_edit_mask",
    "dna_gap_mask",
    "dna_pos",
    "dna_splice_mask",
}


def load_dna_archive(
    dna_h5: str,
) -> tuple[h5py.File, Dict[str, h5py.Dataset], int, int]:
    """Open ``dna_h5`` and validate the expected datasets are present."""

    dna_path = Path(dna_h5)
    if dna_path.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError(
            f"DNA archive must be an HDF5 file (.h5/.hdf5); got '{dna_path}'"
        )
    try:
        handle = h5py.File(dna_path, "r")
    except Exception as exc:  # pragma: no cover - file errors are environmental
        raise RuntimeError(f"Failed to open DNA archive '{dna_h5}': {exc}") from exc

    keys = set(handle.keys())
    missing = _DNA_REQUIRED_KEYS - keys
    if missing:
        handle.close()
        raise KeyError(
            f"DNA archive missing keys: {sorted(missing)} | found={sorted(keys)}"
        )

    seq_len_attr = handle.attrs.get("seq_len")
    if seq_len_attr is None:
        handle.close()
        raise KeyError("DNA archive missing 'seq_len' attribute")

    datasets: Dict[str, h5py.Dataset] = {
        "ref_tokens": handle["dna_ref_tokens"],
        "alt_tokens": handle["dna_alt_tokens"],
        "ref_mask": handle["dna_ref_mask"],
        "alt_mask": handle["dna_alt_mask"],
        "edit_mask": handle["dna_edit_mask"],
        "gap_mask": handle["dna_gap_mask"],
        "pos": handle["dna_pos"],
        "splice_mask": handle["dna_splice_mask"],
    }

    seq_len = int(datasets["ref_tokens"].shape[1])
    seq_len_attr = int(seq_len_attr)
    if seq_len != seq_len_attr:
        handle.close()
        raise ValueError(
            "DNA archive seq_len mismatch: "
            f"stored={seq_len_attr} actual={seq_len}"
        )

    embed_dim = int(datasets["ref_tokens"].shape[2])

    return handle, datasets, seq_len, embed_dim


def validate_dna_indices(indices: np.ndarray, total_rows: int) -> None:
    """Ensure ``indices`` reference valid rows in the DNA archive."""

    if indices.size == 0:
        return
    dna_min = int(indices.min(initial=0))
    dna_max = int(indices.max(initial=-1))
    if dna_min < 0 or dna_max >= total_rows:
        raise RuntimeError(
            "dna_embedding_idx out of range "
            f"(min={dna_min}, max={dna_max}, N={total_rows})"
        )


def compute_dna_slices(
    seq_len: int, embed_dim: int, offset: int
) -> tuple[Optional[Dict[str, slice]], slice, int]:
    """Return the layout slices for DNA conditioning vectors."""

    block_start = offset
    slices: Optional[Dict[str, slice]] = None
    if seq_len > 0 and embed_dim > 0:
        L = seq_len
        E = embed_dim
        slices = {}
        slices["ref_tokens"] = slice(offset, offset + L * E)
        offset += L * E
        slices["alt_tokens"] = slice(offset, offset + L * E)
        offset += L * E
        for key in ("ref_mask", "alt_mask", "edit_mask", "gap_mask", "splice_mask", "pos"):
            slices[key] = slice(offset, offset + L)
            offset += L
    block_stop = offset
    return slices, slice(block_start, block_stop), offset


def write_dna_features(
    dest: np.ndarray,
    slices: Optional[Mapping[str, slice]],
    idx: int,
    datasets: Mapping[str, h5py.Dataset],
) -> None:
    """Populate ``dest`` with DNA features for ``idx`` if slices are defined."""

    if slices is None:
        return
    dest[slices["ref_tokens"]] = datasets["ref_tokens"][idx].astype("float32").reshape(-1)
    dest[slices["alt_tokens"]] = datasets["alt_tokens"][idx].astype("float32").reshape(-1)
    dest[slices["ref_mask"]] = datasets["ref_mask"][idx].astype("float32")
    dest[slices["alt_mask"]] = datasets["alt_mask"][idx].astype("float32")
    dest[slices["edit_mask"]] = datasets["edit_mask"][idx].astype("float32")
    dest[slices["gap_mask"]] = datasets["gap_mask"][idx].astype("float32")
    dest[slices["splice_mask"]] = datasets["splice_mask"][idx].astype("float32")
    dest[slices["pos"]] = datasets["pos"][idx].astype("float32").reshape(-1)


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
# Orchestrator (with caching)
# ---------------------------
def process_and_cache_dna(
    df: pd.DataFrame,
    fasta: Fasta,
    *,
    window: int,
    batch_size: int = 16,
    max_length: int,
    out_meta: str = "out/clinvar_dna.feather",
    out_h5: str = "out/dna_embeddings.h5",
    windows_meta: str | None = None,
    device: Optional[str] = None,
    limit: Optional[int] = None,
    dna_model_id: str = "InstaDeepAI/nucleotide-transformer-500m-1000g",
    force_windows: bool = False,
    force_embeddings: bool = False,
) -> tuple[pd.DataFrame, str]:
    os.makedirs(os.path.dirname(out_meta), exist_ok=True)
    out_h5_path = ensure_h5_path(out_h5, kind="DNA")
    if limit:
        df = df.iloc[:limit].copy()

    # A) windows cache
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

    # B) embeddings cache
    need_embed = True
    required = {
        "dna_ref_tokens",
        "dna_alt_tokens",
        "dna_ref_mask",
        "dna_alt_mask",
        "dna_edit_mask",
        "dna_gap_mask",
        "dna_pos",
        "dna_splice_mask",
    }
    if os.path.exists(out_h5_path) and not force_embeddings:
        try:
            with h5py.File(out_h5_path, "r") as store:
                keys = set(store.keys())
                if required.issubset(keys):
                    same_rows = store["dna_ref_tokens"].shape[0] == len(kept_df)
                    stored_seq_len = int(store.attrs.get("seq_len", -1))
                    same_len = stored_seq_len == target_seq_len
                    completed = int(store.attrs.get("complete", 0)) == 1
                    need_embed = not (same_rows and same_len and completed)
                else:
                    need_embed = True
            if not need_embed:
                print(
                    f"[dna][embeddings] reuse {out_h5_path} rows={len(kept_df)} force={force_embeddings}"
                )
        except Exception:
            need_embed = True

    if need_embed:
        tok, enc, device = build_dna_encoder(dna_model_id, max_length, device)
        seq_len = max(1, min(int(target_seq_len), int(tok.model_max_length) - 1))
        if seq_len <= 0:
            raise RuntimeError("Tokenizer max length too small to build sequence tensors")
        D = enc.get_input_embeddings().embedding_dim
        N = len(ref_seqs)
        ref_seqs = [str(s) for s in ref_seqs]
        alt_seqs = [str(s) for s in alt_seqs]
        edit_positions = [int(e) for e in edit_positions]
        edit_indices = [min(max(int(idx), 0), seq_len - 1) for idx in edit_positions]
        base_pos = np.arange(seq_len, dtype="float16")
        batch_rows = max(1, int(batch_size))
        initialise_sequence_archive(
            out_h5_path,
            layout=DNA_ARCHIVE_LAYOUT,
            n_rows=N,
            seq_len=seq_len,
            embed_dim=D,
            chunk_rows=batch_rows,
            kind="DNA",
        )

        with open_sequence_archive(
            out_h5_path, layout=DNA_ARCHIVE_LAYOUT, kind="DNA"
        ) as archive:
            ref_tokens_ds = archive.wt_tokens
            alt_tokens_ds = archive.mt_tokens
            ref_mask_ds = archive.wt_mask
            alt_mask_ds = archive.mt_mask
            edit_mask_ds = archive.edit_mask
            gap_mask_ds = archive.gap_mask
            pos_ds = archive.pos
            splice_mask_ds = archive.extra_masks.get("dna_splice_mask")

            total_batches = max(1, math.ceil(N / max(int(batch_size), 1)))
            for chunk in tqdm(
                batch_iter(list(range(N)), batch_size),
                total=total_batches,
                desc="[dna][embeddings] batches",
                unit="batch",
            ):
                idxs = list(chunk)
                idxs_arr = np.array(idxs, dtype=np.int64)
                ref_batch = [ref_seqs[i] for i in idxs_arr]
                alt_batch = [alt_seqs[i] for i in idxs_arr]

                ref_hidden, ref_attn = encode_sequence_tokens(
                    ref_batch, enc, tok, device, max_length=seq_len + 1
                )
                alt_hidden, alt_attn = encode_sequence_tokens(
                    alt_batch, enc, tok, device, max_length=seq_len + 1
                )

                ref_hidden = (
                    ref_hidden[:, :seq_len, :]
                    .to(dtype=torch.float16)
                    .cpu()
                    .contiguous()
                    .numpy()
                )
                alt_hidden = (
                    alt_hidden[:, :seq_len, :]
                    .to(dtype=torch.float16)
                    .cpu()
                    .contiguous()
                    .numpy()
                )
                ref_attn = ref_attn[:, :seq_len]
                alt_attn = alt_attn[:, :seq_len]
                ref_mask_batch = (
                    ref_attn.to(dtype=torch.uint8).cpu().contiguous().numpy()
                )
                alt_mask_batch = (
                    alt_attn.to(dtype=torch.uint8).cpu().contiguous().numpy()
                )
                gap_batch = (
                    torch.logical_xor(ref_attn.bool(), alt_attn.bool())
                    .to(dtype=torch.uint8)
                    .cpu()
                    .contiguous()
                    .numpy()
                )

                bsz = len(idxs_arr)
                edit_batch = np.zeros((bsz, seq_len), dtype=np.uint8)
                pos_batch = np.zeros((bsz, seq_len, 1), dtype=np.float16)
                splice_batch = np.zeros((bsz, seq_len), dtype=np.uint8)
                for j, row_idx in enumerate(idxs_arr):
                    idx = edit_indices[row_idx]
                    edit_batch[j, idx] = 1
                    pos_batch[j, :, 0] = base_pos - np.float16(idx)

                if bsz and np.all(np.diff(idxs_arr) == 1):
                    start = int(idxs_arr[0])
                    end = start + bsz
                    ref_tokens_ds[start:end, :, :] = ref_hidden
                    alt_tokens_ds[start:end, :, :] = alt_hidden
                    ref_mask_ds[start:end, :] = ref_mask_batch
                    alt_mask_ds[start:end, :] = alt_mask_batch
                    edit_mask_ds[start:end, :] = edit_batch
                    gap_mask_ds[start:end, :] = gap_batch
                    pos_ds[start:end, :, :] = pos_batch
                    if splice_mask_ds is not None:
                        splice_mask_ds[start:end, :] = splice_batch
                else:
                    ref_tokens_ds[idxs_arr, :, :] = ref_hidden
                    alt_tokens_ds[idxs_arr, :, :] = alt_hidden
                    ref_mask_ds[idxs_arr, :] = ref_mask_batch
                    alt_mask_ds[idxs_arr, :] = alt_mask_batch
                    edit_mask_ds[idxs_arr, :] = edit_batch
                    gap_mask_ds[idxs_arr, :] = gap_batch
                    pos_ds[idxs_arr, :, :] = pos_batch
                    if splice_mask_ds is not None:
                        splice_mask_ds[idxs_arr, :] = splice_batch

                archive.flush()

            archive.store.attrs["complete"] = 1
            archive.flush()
        print(f"[dna][embeddings] wrote {out_h5_path} rows={len(kept_df)}")

    # C) final meta aligned to arrays
    with h5py.File(out_h5_path, "r") as store:
        missing = required - set(store.keys())
        if missing:
            raise RuntimeError(f"{out_h5_path} missing keys: {sorted(missing)}")
        N = store["dna_ref_tokens"].shape[0]
        seq_len_stored = int(store.attrs.get("seq_len", -1))
        completed = int(store.attrs.get("complete", 0))
    if N != len(kept_df):
        raise RuntimeError(
            f"Alignment mismatch: arrays N={N} vs kept_df N={len(kept_df)}"
        )
    if seq_len_stored != target_seq_len:
        raise RuntimeError(
            f"DNA archive seq_len mismatch: stored={seq_len_stored} expected={target_seq_len}"
        )
    if completed != 1:
        raise RuntimeError(
            f"DNA archive {out_h5_path} is marked incomplete; regenerate the embeddings."
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
    kept_df["dna_edit_idx"] = list(int(x) for x in edit_positions)
    kept_df["dna_seq_len"] = int(target_seq_len)
    print(f"[dna][meta] prepared frame rows={N}")
    return kept_df, str(out_h5_path)
