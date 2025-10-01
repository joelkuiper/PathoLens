# src/protein_utils.py
from __future__ import annotations
from typing import Tuple, Optional, List, Dict, Mapping
from difflib import SequenceMatcher
import os
from pathlib import Path

import math
import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from src.sequence_cache import (
    SequenceArchiveLayout,
    ensure_h5_path,
    initialise_sequence_archive,
    open_sequence_archive,
)

# Layout describing protein HDF5 dataset names
PROTEIN_ARCHIVE_LAYOUT = SequenceArchiveLayout(
    wt_tokens="prot_wt_tokens",
    mt_tokens="prot_mt_tokens",
    wt_mask="prot_wt_mask",
    mt_mask="prot_mt_mask",
    edit_mask="prot_edit_mask",
    gap_mask="prot_gap_mask",
    pos="prot_pos",
    extra_masks=("prot_frameshift_mask",),
)

# ============================================================
# Amino-acid sanitization
# ============================================================

# keep 20 standard; map everything else (incl. B/Z/U/J/* and gaps) to 'X'
_AA_VALID = set("ACDEFGHIKLMNPQRSTVWY")


def sanitize_aa(seq: str) -> str:
    if not isinstance(seq, str):
        return ""
    s = seq.replace("-", "").upper()
    return "".join(c if c in _AA_VALID else "X" for c in s)


def _center_crop_sequence(seq: str, center: int, target_len: int) -> tuple[str, int]:
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    seq = seq or ""
    L = len(seq)
    if L == 0:
        return "", 0
    center = max(0, min(int(center), L - 1))
    if L <= target_len:
        return seq, center
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
    return cropped, new_center


def _infer_alignment_indices(wt: str, mt: str) -> tuple[int, int]:
    wt = wt or ""
    mt = mt or ""

    len_wt = len(wt)
    len_mt = len(mt)

    if len_wt == 0 or len_mt == 0:
        return 0, 0

    # Fast path for simple substitutions (same length)
    if len_wt == len_mt:
        for idx, (a, b) in enumerate(zip(wt, mt)):
            if a != b:
                return idx, idx
        center = len_wt // 2
        return center, center

    length_diff = len_wt - len_mt

    # Fast path for single-residue indels (length diff of 1)
    if abs(length_diff) == 1:
        i = j = 0
        skipped = False
        mismatch_wt: Optional[int] = None
        mismatch_mt: Optional[int] = None

        while i < len_wt and j < len_mt:
            if wt[i] == mt[j]:
                i += 1
                j += 1
                continue

            if skipped:
                mismatch_wt = None
                break

            skipped = True
            mismatch_wt = i
            mismatch_mt = j
            if length_diff > 0:
                i += 1  # wt contains the extra residue
            else:
                j += 1  # mt contains the extra residue

        if skipped and mismatch_wt is not None:
            return mismatch_wt, mismatch_mt if mismatch_mt is not None else mismatch_wt

        if skipped:
            # mismatch encountered but unresolved (multiple edits)
            pass
        else:
            # Pure insertion/deletion at sequence end
            if length_diff > 0:
                return len_mt, len_mt
            else:
                return len_wt, len_wt

    matcher = SequenceMatcher(None, wt, mt, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            return max(0, i1), max(0, j1)

    center_wt = len_wt // 2
    center_mt = min(center_wt, len_mt - 1)
    return center_wt, center_mt


# ---------------------------
# HDF5 archive helpers
# ---------------------------

_PROTEIN_REQUIRED_KEYS = {
    "prot_wt_tokens",
    "prot_mt_tokens",
    "prot_wt_mask",
    "prot_mt_mask",
    "prot_edit_mask",
    "prot_gap_mask",
    "prot_pos",
    "prot_frameshift_mask",
}


def load_protein_archive(
    protein_h5: str,
) -> tuple[h5py.File, Dict[str, h5py.Dataset], int, int]:
    """Open ``protein_h5`` and validate the expected datasets are present."""

    prot_path = Path(protein_h5)
    if prot_path.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError(
            f"Protein archive must be an HDF5 file (.h5/.hdf5); got '{prot_path}'"
        )
    try:
        handle = h5py.File(protein_h5, "r")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to open protein archive '{protein_h5}': {exc}"
        ) from exc

    keys = set(handle.keys())
    missing = _PROTEIN_REQUIRED_KEYS - keys
    if missing:
        handle.close()
        raise KeyError(
            "Protein archive missing keys: "
            f"{sorted(missing)} | found={sorted(keys)}"
        )

    seq_len_attr = handle.attrs.get("seq_len")
    if seq_len_attr is None:
        handle.close()
        raise KeyError("Protein archive missing 'seq_len' attribute")

    datasets: Dict[str, h5py.Dataset] = {
        "wt_tokens": handle["prot_wt_tokens"],
        "mt_tokens": handle["prot_mt_tokens"],
        "wt_mask": handle["prot_wt_mask"],
        "mt_mask": handle["prot_mt_mask"],
        "edit_mask": handle["prot_edit_mask"],
        "gap_mask": handle["prot_gap_mask"],
        "pos": handle["prot_pos"],
        "frameshift_mask": handle["prot_frameshift_mask"],
    }

    seq_len = int(datasets["wt_tokens"].shape[1])
    seq_len_attr = int(seq_len_attr)
    if seq_len != seq_len_attr:
        handle.close()
        raise ValueError(
            "Protein archive seq_len mismatch: "
            f"stored={seq_len_attr} actual={seq_len}"
        )

    embed_dim = int(datasets["wt_tokens"].shape[2])

    return handle, datasets, seq_len, embed_dim


def validate_protein_indices(indices: np.ndarray, total_rows: int) -> float:
    """Ensure ``indices`` reference valid rows in the protein archive."""

    if (indices < -1).any():
        raise RuntimeError("protein_embedding_idx must be -1 or non-negative integers")

    if total_rows == 0 and (indices >= 0).any():
        raise RuntimeError(
            "protein_embedding_idx refers to embeddings but protein tokens array is empty"
        )

    prot_max = int(indices.max(initial=-1))
    if prot_max >= total_rows:
        raise RuntimeError(
            "protein_embedding_idx out of range "
            f"(max={prot_max}, N={total_rows})"
        )

    covered = int((indices >= 0).sum())
    return covered / max(1, len(indices))


def compute_protein_slices(
    seq_len: int, embed_dim: int, offset: int
) -> tuple[Optional[Dict[str, slice]], slice, int]:
    """Return the layout slices for protein conditioning vectors."""

    block_start = offset
    slices: Optional[Dict[str, slice]] = None
    if seq_len > 0 and embed_dim > 0:
        L = seq_len
        E = embed_dim
        slices = {}
        slices["wt_tokens"] = slice(offset, offset + L * E)
        offset += L * E
        slices["mt_tokens"] = slice(offset, offset + L * E)
        offset += L * E
        for key in ("wt_mask", "mt_mask", "edit_mask", "gap_mask", "frameshift_mask", "pos"):
            slices[key] = slice(offset, offset + L)
            offset += L
    block_stop = offset
    return slices, slice(block_start, block_stop), offset


def write_protein_features(
    dest: np.ndarray,
    slices: Optional[Mapping[str, slice]],
    idx: int,
    datasets: Mapping[str, h5py.Dataset],
) -> None:
    """Populate ``dest`` with protein features for ``idx`` if available."""

    if slices is None or idx < 0:
        return
    dest[slices["wt_tokens"]] = datasets["wt_tokens"][idx].astype("float32").reshape(-1)
    dest[slices["mt_tokens"]] = datasets["mt_tokens"][idx].astype("float32").reshape(-1)
    dest[slices["wt_mask"]] = datasets["wt_mask"][idx].astype("float32")
    dest[slices["mt_mask"]] = datasets["mt_mask"][idx].astype("float32")
    dest[slices["edit_mask"]] = datasets["edit_mask"][idx].astype("float32")
    dest[slices["gap_mask"]] = datasets["gap_mask"][idx].astype("float32")
    dest[slices["frameshift_mask"]] = datasets["frameshift_mask"][idx].astype("float32")
    dest[slices["pos"]] = datasets["pos"][idx].astype("float32").reshape(-1)


# ============================================================
# Encoder construction (HF ESM2)
# ============================================================


def _device_is_cuda(device: str | torch.device) -> bool:
    return (isinstance(device, str) and device.startswith("cuda")) or (
        isinstance(device, torch.device) and device.type == "cuda"
    )


def _device_is_mps(device: str | torch.device) -> bool:
    return (isinstance(device, str) and device.startswith("mps")) or (
        isinstance(device, torch.device) and device.type == "mps"
    )


def build_protein_encoder(
    model_id: str = "facebook/esm2_t33_650M_UR50D",
    device: Optional[str] = None,
):
    """
    Create tokenizer + model on the requested device.
    - CUDA: bf16 weights (fast & memory-friendly with AMP)
    - MPS:  fp16 if supported, otherwise fp32
    - CPU:  fp32
    """
    device = device or (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    tok = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=False
    )
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": tok.eos_token or "<pad>"})
    tok.padding_side = "right"

    # Choose a sensible torch_dtype per backend
    if _device_is_cuda(device):
        prefer_dtype = torch.bfloat16
    elif _device_is_mps(device):
        # MPS fp16 can reduce memory a lot; if load fails we fall back to fp32
        prefer_dtype = torch.float16
    else:
        prefer_dtype = torch.float32

    try:
        enc = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=prefer_dtype
        )
    except Exception:
        # Fallback to fp32 if half/bf16 unsupported
        enc = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float32
        )

    if enc.get_input_embeddings().num_embeddings < len(tok):
        enc.resize_token_embeddings(len(tok))
    enc = enc.to(device).eval()

    # Light perf hint for matmuls (no-op on some backends)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    return tok, enc, device


@torch.inference_mode()
def encode_sequence_tokens(
    seqs: List[str],
    model,
    tokenizer,
    device: str | torch.device,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    toks = tokenizer(
        seqs,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    toks = {k: v.to(device) for k, v in toks.items()}
    out = model(**toks).last_hidden_state
    attn = toks["attention_mask"]
    out = out[:, 1:, :]
    attn = attn[:, 1:]
    return out.to(torch.float32), attn.to(torch.float32)
# ============================================================
# Embedding (unique dedupe + interleaved schedule + memmap)
# ============================================================


def _build_dedup(seqs: List[str]) -> Tuple[List[str], Dict[int, int]]:
    """
    Deduplicate sequences (preserve first occurrence).
    Returns:
      uniq_list: unique sequences in stable order
      row2uniq:  map row_idx -> uniq_idx
    """
    uniq_map: Dict[str, int] = {}
    row2uniq: Dict[int, int] = {}
    uniq_list: List[str] = []
    for i, s in enumerate(seqs):
        j = uniq_map.get(s)
        if j is None:
            j = len(uniq_list)
            uniq_map[s] = j
            uniq_list.append(s)
        row2uniq[i] = j
    return uniq_list, row2uniq


def _length_interleave_order(lengths: List[int], bin_size: int) -> List[int]:
    """
    Sort by length, then within windows interleave [long, short, long, short, ...].
    This keeps batch Lmax relatively stable across the run and avoids the
    "fast 60% then crawl" pattern.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    bins = [order[i : i + bin_size] for i in range(0, len(order), bin_size)]

    def interleave_minmax(idxs: List[int]) -> List[int]:
        i, j, out = 0, len(idxs) - 1, []
        while i <= j:
            out.append(idxs[j])
            j -= 1  # long
            if i <= j:
                out.append(idxs[i])
                i += 1  # short
        return out

    schedule: List[int] = []
    for idxs in bins:
        schedule.extend(interleave_minmax(idxs))
    return schedule


def _dynamic_batches(
    schedule: List[int],
    uniq_list: List[str],
    target_tokens: Optional[int],
    max_bs: int,
):
    """
    Yield batches as lists of uniq indices.
    If target_tokens is set, choose batch size so max_len * batch_size ≲ target_tokens.
    Otherwise, use fixed batch chunks of size max_bs.
    """
    if not schedule:
        return
    if not target_tokens:
        for i in range(0, len(schedule), max_bs):
            yield schedule[i : i + max_bs]
        return

    i = 0
    while i < len(schedule):
        Lmax = 0
        j = i
        while j < len(schedule):
            Lmax = max(Lmax, len(uniq_list[schedule[j]]))
            bs = j - i + 1
            if (Lmax * bs) > target_tokens and bs > 1:
                break
            j += 1
        yield schedule[i:j]
        i = j


def _empty_backend_cache_if_possible(device):
    # Avoid fragmentation on GPU backends
    if _device_is_cuda(device):
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    elif _device_is_mps(device):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


@torch.inference_mode()
def embed_unique_to_memmap(
    seqs: List[str],
    tok,
    enc,
    device: str | torch.device,
    out_path: str | Path,
    *,
    max_length: int = 4096,
    pool: str = "mean",
    batch_size: int = 32,
    interleave_window_factor: int = 8,
    target_tokens: Optional[int] = None,  # auto-picked on MPS if None
) -> Tuple[str, List[str], Dict[int, int]]:
    """
    Encode UNIQUE sequences to a disk memmap [U, D].
    - Interleaves long/short within windows to avoid end-of-run crawl.
    - Token-budgeted dynamic batching (max_len * batch_size ≲ target_tokens).
    - Automatic backoff: split batches on OOM.
    - **MPS OOM fallback**: if a single-item batch still OOMs on MPS, move the model
      to CPU once and finish the remaining pending work on CPU (no crash).
    """
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    uniq_list, row2uniq = _build_dedup(seqs)
    U = len(uniq_list)
    if U == 0:
        np.memmap(out_path, mode="w+", dtype="float32", shape=(0, 1)).flush()
        return out_path, uniq_list, row2uniq

    # Model dim
    D = enc.get_input_embeddings().embedding_dim
    mem = np.memmap(out_path, mode="w+", dtype="float32", shape=(U, D))

    # Build schedule (interleaved)
    lengths = [len(s) for s in uniq_list]
    window = max(batch_size * interleave_window_factor, batch_size)
    schedule = _length_interleave_order(lengths, bin_size=window)

    # Safe default token budget on MPS
    if target_tokens is None and _device_is_mps(device):
        target_tokens = 48_000  # conservative; reduces late spikes

    # AMP only on CUDA (MPS AMP isn’t reliable for mem)
    use_autocast = _device_is_cuda(device)
    amp_dtype = (
        torch.bfloat16
        if next(enc.parameters()).dtype == torch.bfloat16
        else torch.float16
    )

    # Prepare batches
    batches = list(_dynamic_batches(schedule, uniq_list, target_tokens, batch_size))
    pbar = tqdm(range(len(batches)), total=len(batches), desc="Embedding (uniq→memmap)")

    # Utility: run a batch on a given device (cuda/mps/cpu)
    def run_batch(cur_idxs: List[int], run_device: str | torch.device):
        # free cache before moving
        if _device_is_mps(run_device) or _device_is_cuda(run_device):
            _empty_backend_cache_if_possible(run_device)
        batch_seqs = [uniq_list[i] for i in cur_idxs]
        toks = tok(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        # move inputs
        toks = {k: v.to(run_device) for k, v in toks.items()}

        if _device_is_cuda(run_device) and use_autocast:
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                out = enc.to(run_device)(**toks).last_hidden_state
        else:
            out = enc.to(run_device)(**toks).last_hidden_state

        attn = toks["attention_mask"].unsqueeze(-1).to(out.dtype)
        if pool == "cls":
            z = out[:, 0, :]
        else:
            z = (out * attn).sum(1) / attn.sum(1).clamp_min(1e-6)
        z_np = z.detach().to(torch.float32).cpu().numpy()
        for k, u in enumerate(cur_idxs):
            mem[u, :] = z_np[k]
        # free cache after
        _empty_backend_cache_if_possible(run_device)

    # Walk batches with backoff and CPU fallback on MPS OOM
    mps_fell_back_to_cpu = False
    for bi in pbar:
        idxs = batches[bi]
        pending: List[List[int]] = [idxs]
        while pending:
            cur = pending.pop(0)
            try:
                run_batch(cur, device)
            except RuntimeError as e:
                msg = str(e).lower()
                oomish = (
                    ("out of memory" in msg)
                    or ("mps backend out of memory" in msg)
                    or ("cuda out of memory" in msg)
                )
                if oomish and len(cur) > 1:
                    # split and retry both halves
                    mid = len(cur) // 2
                    pending.insert(0, cur[mid:])
                    pending.insert(0, cur[:mid])
                    _empty_backend_cache_if_possible(device)
                    continue
                # single item still OOM → if MPS, move model once to CPU and finish there
                if oomish and _device_is_mps(device):
                    if not mps_fell_back_to_cpu:
                        print(
                            "[embed] MPS OOM on single item; falling back to CPU for remaining work."
                        )
                        enc.to("cpu")
                        mps_fell_back_to_cpu = True
                    run_batch(cur, "cpu")
                else:
                    raise

    mem.flush()
    return out_path, uniq_list, row2uniq


# ============================================================
# I/O + VEP integration
# ============================================================


def _read_any_table(path_or_df: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(path_or_df, pd.DataFrame):
        return path_or_df.copy()

    p = Path(path_or_df)
    suf = p.suffix.lower()
    if suf in {".feather", ".ft"}:
        return pd.read_feather(p)
    if suf in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if suf in {".csv"}:
        return pd.read_csv(p)
    # fallback TSV
    return pd.read_table(p)


def load_vep_sequences(vep_combined: str | Path | pd.DataFrame) -> pd.DataFrame:
    """
    Expect columns (from your VEP join):
      id (VariationID as str), protein_id, hgvsp, consequence_terms, seq_wt, seq_mt
    - Sanitizes sequences
    - Keeps only rows with both WT/MT present
    - De-duplicates per (id, protein_id, hgvsp)
    """
    df = _read_any_table(vep_combined)

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
    keys = [k for k in ["id", "protein_id", "hgvsp"] if k in df.columns]
    if keys:
        df = (
            df.sort_values(keys)
            .drop_duplicates(keys, keep="first")
            .reset_index(drop=True)
        )

    return df[
        ["id", "protein_id", "hgvsp", "consequence_terms", "seq_wt", "seq_mt"]
    ].copy()


# ============================================================
# Post-processing & cache
# ============================================================


def write_protein_meta(meta_df: pd.DataFrame, out_feather: str) -> None:
    import pyarrow.feather as feather

    feather.write_feather(meta_df, out_feather)


# ============================================================
# Orchestrator used by pipeline builders
# ============================================================


def process_and_cache_protein(
    vep_combined: str | Path | pd.DataFrame,
    *,
    out_meta: str,
    out_h5: str,
    device: Optional[str] = None,
    model_id: str = "facebook/esm2_t33_650M_UR50D",
    batch_size: int = 8,
    window: int = 127,
    max_length: int = 2048,
    force_embeddings: bool = False,
) -> tuple[pd.DataFrame, str]:
    """
    Loads VEP-joined WT/MT sequences, embeds with ESM2 (unique-only with interleaved scheduling),
    saves an embedding archive (HDF5) + Feather (meta). Returns (kept_meta_df, out_h5_path).

    Memory behavior:
      - Embeddings for UNIQUE sequences are written to disk memmaps (no big RAM spikes).
      - Batch sizes adapt to token budget and back off on OOM (esp. on MPS).
    """
    os.makedirs(os.path.dirname(out_meta), exist_ok=True)
    out_h5_path = ensure_h5_path(out_h5, kind="Protein")

    if os.path.exists(out_h5_path):
        try:
            size = os.path.getsize(out_h5_path)
        except OSError:
            size = -1
        print(
            f"[protein][cache] existing archive at {out_h5_path} size={size if size >= 0 else 'unknown'}"
        )
    else:
        print(f"[protein][cache] no existing archive at {out_h5_path}")

    # Load and keep usable rows
    pairs = load_vep_sequences(vep_combined)
    if len(pairs) == 0:
        raise RuntimeError("No WT/MT sequences found in VEP combined file.")
    kept = pairs.reset_index(drop=True).copy()
    print(f"[protein][cache] loaded rows={len(kept)} from VEP combined input")
    kept["protein_embedding_idx"] = np.arange(len(kept), dtype=np.int32)

    wt_sanitized = [sanitize_aa(str(s)) for s in kept["seq_wt"]]
    mt_sanitized = [sanitize_aa(str(s)) for s in kept["seq_mt"]]
    kept["seq_wt"] = wt_sanitized
    kept["seq_mt"] = mt_sanitized

    target_seq_len = max(1, min(int(window), int(max_length) - 1))
    print(
        f"[protein][cache] preparing windows: target_seq_len={target_seq_len} window_param={window} max_length={max_length}"
    )

    # Short-circuit if cache matches (keeps re-runs instant)
    required = {
        "prot_wt_tokens",
        "prot_mt_tokens",
        "prot_wt_mask",
        "prot_mt_mask",
        "prot_edit_mask",
        "prot_gap_mask",
        "prot_pos",
        "prot_frameshift_mask",
    }
    need_embed = True
    if os.path.exists(out_h5_path) and not force_embeddings:
        try:
            with h5py.File(out_h5_path, "r") as store:
                keys = set(store.keys())
                if required.issubset(keys):
                    same_rows = store["prot_wt_tokens"].shape[0] == len(kept)
                    stored_seq_len = int(store.attrs.get("seq_len", -1))
                    same_len = stored_seq_len == target_seq_len
                    completed = int(store.attrs.get("complete", 0)) == 1
                    need_embed = not (same_rows and same_len and completed)
                else:
                    need_embed = True
            if not need_embed:
                print(
                    f"[protein][embeddings] reuse {out_h5_path} rows={len(kept)} force={force_embeddings}"
                )
        except Exception:
            need_embed = True

    encoder_bundle: Optional[Tuple] = None
    archive_prepared = False
    if need_embed:
        tok, enc, device = build_protein_encoder(model_id=model_id, device=device)
        seq_len = max(1, min(int(target_seq_len), int(max_length) - 1))
        D = enc.get_input_embeddings().embedding_dim
        N = len(kept)
        batch_rows = max(1, int(batch_size))
        print(
            f"[protein][embeddings] initialise archive path={out_h5_path} "
            f"rows={N} seq_len={seq_len} embed_dim={D}"
        )
        initialise_sequence_archive(
            out_h5_path,
            layout=PROTEIN_ARCHIVE_LAYOUT,
            n_rows=N,
            seq_len=seq_len,
            embed_dim=D,
            chunk_rows=batch_rows,
            kind="Protein",
        )
        archive_prepared = True
        encoder_bundle = (tok, enc, device, seq_len, D, N, batch_rows)

    # window trimming now that archive (if needed) exists on disk
    trimmed_wt: List[str] = []
    trimmed_mt: List[str] = []
    edit_positions: List[int] = []
    frameshift_flags: List[bool] = []
    for wt_clean, mt_clean in tqdm(
        zip(wt_sanitized, mt_sanitized),
        total=len(kept),
        desc="[protein][windows]",
        unit="seq",
    ):
        idx_wt_raw, idx_mt_raw = _infer_alignment_indices(wt_clean, mt_clean)
        wt_trim, idx_wt = _center_crop_sequence(wt_clean, idx_wt_raw, target_seq_len)
        mt_trim, _ = _center_crop_sequence(mt_clean, idx_mt_raw, target_seq_len)
        trimmed_wt.append(wt_trim)
        trimmed_mt.append(mt_trim)
        edit_positions.append(int(idx_wt))
        frameshift_flags.append((len(mt_clean) - len(wt_clean)) % 3 != 0)

    kept["seq_wt"] = trimmed_wt
    kept["seq_mt"] = trimmed_mt
    kept["protein_edit_idx"] = [int(x) for x in edit_positions]
    kept["protein_seq_len"] = int(target_seq_len)
    kept["protein_frameshift"] = [bool(x) for x in frameshift_flags]

    if need_embed and encoder_bundle is not None:
        tok, enc, device, seq_len, D, N, batch_rows = encoder_bundle
        indices = list(range(N))
        wt_seqs = [str(s) for s in kept["seq_wt"].tolist()]
        mt_seqs = [str(s) for s in kept["seq_mt"].tolist()]
        edit_indices = [min(max(int(edit_positions[i]), 0), seq_len - 1) for i in range(N)]
        base_pos = np.arange(seq_len, dtype=np.float16)

        if not archive_prepared:
            initialise_sequence_archive(
                out_h5_path,
                layout=PROTEIN_ARCHIVE_LAYOUT,
                n_rows=N,
                seq_len=seq_len,
                embed_dim=D,
                chunk_rows=batch_rows,
                kind="Protein",
            )
        print(
            f"[protein][embeddings] start rows={N} seq_len={seq_len} embed_dim={D} batch_size={batch_rows} device={device}"
        )

        with open_sequence_archive(
            out_h5_path, layout=PROTEIN_ARCHIVE_LAYOUT, kind="Protein"
        ) as archive:
            wt_tokens_ds = archive.wt_tokens
            mt_tokens_ds = archive.mt_tokens
            wt_mask_ds = archive.wt_mask
            mt_mask_ds = archive.mt_mask
            edit_mask_ds = archive.edit_mask
            gap_mask_ds = archive.gap_mask
            pos_ds = archive.pos
            frameshift_ds = archive.extra_masks.get("prot_frameshift_mask")

            total_batches = max(1, math.ceil(N / max(int(batch_size), 1)))
            for chunk in tqdm(
                batch_iter(indices, batch_size),
                total=total_batches,
                desc="[protein][embeddings] batches",
                unit="batch",
            ):
                idxs = list(chunk)
                idxs_arr = np.array(idxs, dtype=np.int64)
                bsz = len(idxs_arr)
                wt_batch = [wt_seqs[i] for i in idxs_arr]
                mt_batch = [mt_seqs[i] for i in idxs_arr]

                combined = wt_batch + mt_batch
                hidden, attn = encode_sequence_tokens(
                    combined, enc, tok, device, max_length=seq_len + 1
                )

                hidden = hidden[:, :seq_len, :]
                attn = attn[:, :seq_len]
                wt_hidden = (
                    hidden[:bsz, :, :]
                    .to(dtype=torch.float16)
                    .cpu()
                    .contiguous()
                    .numpy()
                )
                mt_hidden = (
                    hidden[bsz:, :, :]
                    .to(dtype=torch.float16)
                    .cpu()
                    .contiguous()
                    .numpy()
                )
                wt_attn = attn[:bsz]
                mt_attn = attn[bsz:]
                wt_mask_batch = (
                    wt_attn.to(dtype=torch.uint8).cpu().contiguous().numpy()
                )
                mt_mask_batch = (
                    mt_attn.to(dtype=torch.uint8).cpu().contiguous().numpy()
                )
                gap_batch = (
                    torch.logical_xor(wt_attn.bool(), mt_attn.bool())
                    .to(dtype=torch.uint8)
                    .cpu()
                    .contiguous()
                    .numpy()
                )

                edit_batch = np.zeros((bsz, seq_len), dtype=np.uint8)
                pos_batch = np.zeros((bsz, seq_len, 1), dtype=np.float16)
                frameshift_batch = np.zeros((bsz, seq_len), dtype=np.uint8)
                for j, row_idx in enumerate(idxs_arr):
                    idx = edit_indices[row_idx]
                    edit_batch[j, idx] = 1
                    pos_batch[j, :, 0] = base_pos - np.float16(idx)
                    if frameshift_flags[row_idx]:
                        frameshift_batch[j, idx:] = 1

                if bsz and np.all(np.diff(idxs_arr) == 1):
                    start = int(idxs_arr[0])
                    end = start + bsz
                    wt_tokens_ds[start:end, :, :] = wt_hidden
                    mt_tokens_ds[start:end, :, :] = mt_hidden
                    wt_mask_ds[start:end, :] = wt_mask_batch
                    mt_mask_ds[start:end, :] = mt_mask_batch
                    edit_mask_ds[start:end, :] = edit_batch
                    gap_mask_ds[start:end, :] = gap_batch
                    pos_ds[start:end, :, :] = pos_batch
                    if frameshift_ds is not None:
                        frameshift_ds[start:end, :] = frameshift_batch
                else:
                    wt_tokens_ds[idxs_arr, :, :] = wt_hidden
                    mt_tokens_ds[idxs_arr, :, :] = mt_hidden
                    wt_mask_ds[idxs_arr, :] = wt_mask_batch
                    mt_mask_ds[idxs_arr, :] = mt_mask_batch
                    edit_mask_ds[idxs_arr, :] = edit_batch
                    gap_mask_ds[idxs_arr, :] = gap_batch
                    pos_ds[idxs_arr, :, :] = pos_batch
                    if frameshift_ds is not None:
                        frameshift_ds[idxs_arr, :] = frameshift_batch

                archive.flush()

            archive.store.attrs["complete"] = 1
            archive.flush()
        print(f"[protein][embeddings] wrote {out_h5_path} rows={N}")

    with h5py.File(out_h5_path, "r") as store:
        missing = required - set(store.keys())
        if missing:
            raise RuntimeError(f"{out_h5_path} missing keys: {sorted(missing)}")
        seq_len_stored = int(store.attrs.get("seq_len", -1))
        completed = int(store.attrs.get("complete", 0))
        if seq_len_stored != target_seq_len:
            raise RuntimeError(
                "Protein archive seq_len mismatch: "
                f"stored={seq_len_stored} expected={target_seq_len}"
            )
        N = store["prot_wt_tokens"].shape[0]
    if completed != 1:
        raise RuntimeError(
            f"Protein archive {out_h5_path} is marked incomplete; regenerate the embeddings."
        )
    if N != len(kept):
        raise RuntimeError(
            f"Alignment mismatch: arrays N={N} vs kept N={len(kept)}"
        )

    # Always (re)write meta so it matches whatever we embedded
    write_protein_meta(kept, out_meta)
    print(f"[protein][meta] wrote {out_meta} rows={len(kept)}")
    return kept, str(out_h5_path)
def batch_iter(xs: List[int], bs: int):
    buf: List[int] = []
    for x in xs:
        buf.append(x)
        if len(buf) == bs:
            yield list(buf)
            buf = []
    if buf:
        yield list(buf)


