# src/protein_utils.py
from __future__ import annotations
from typing import Tuple, Optional, List, Dict, Mapping, Sequence
from difflib import SequenceMatcher
import os
from pathlib import Path

import math
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

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
            f"Protein archive missing keys: {sorted(missing)} | found={sorted(keys)}"
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
            f"Protein archive seq_len mismatch: stored={seq_len_attr} actual={seq_len}"
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
            f"protein_embedding_idx out of range (max={prot_max}, N={total_rows})"
        )

    covered = int((indices >= 0).sum())
    return covered / max(1, len(indices))


def read_protein_features(
    datasets: Mapping[str, h5py.Dataset], idx: int
) -> Dict[str, torch.Tensor]:
    """Materialise protein feature tensors for ``idx`` from ``datasets``."""

    if idx < 0:
        raise ValueError("Protein feature index must be non-negative")

    features: Dict[str, torch.Tensor] = {}
    for name, ds in datasets.items():
        if ds.shape[0] <= idx:
            raise IndexError(f"Protein dataset '{name}' index out of range: {idx}")
        arr = np.asarray(ds[idx])
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        tensor = torch.from_numpy(arr)
        features[name] = tensor
    return features


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


def _probe_embedding_dim(model: ESMC) -> int:
    """Run a one-off encode/logits pass to determine the embedding width."""

    tokenizer = getattr(model, "sequence_tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "mask_token", None) is None:
        replacement: str | None = None
        special_tokens = getattr(tokenizer, "special_tokens_map", None)
        if isinstance(special_tokens, Mapping):
            replacement = special_tokens.get("mask_token") or special_tokens.get("mask")
        if not isinstance(replacement, str):
            replacement = ""
        try:
            setattr(tokenizer, "mask_token", replacement)
        except Exception:
            pass

    probe = ESMProtein(sequence="ACDEFGHIKLMNPQRSTVWY")

    try:
        encoded = model.encode(probe)
        logits = model.logits(
            encoded, LogitsConfig(sequence=True, return_embeddings=True)
        )
    except Exception as exc:  # pragma: no cover - defensive logging for unexpected SDK issues
        raise RuntimeError(
            "Failed to run probe encode to infer embedding dimension"
        ) from exc

    embeddings = getattr(logits, "embeddings", None)
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim < 3:
        raise RuntimeError(
            "ESMC logits output did not provide token embeddings for probe sequence"
        )

    return int(embeddings.shape[-1])


def build_protein_encoder(
    model_id: str = "esmc_600m",
    device: Optional[str] = None,
):
    """
    Create an ESMC model on the requested device.
    - CUDA: preferred when available
    - MPS:  fall back to CPU (ESMC currently targets CUDA/CPU)
    - CPU:  always available
    """

    device = device or (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    if _device_is_mps(device):
        device = "cpu"

    model = ESMC.from_pretrained(model_id)
    model = model.to(device).eval()

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    embed_dim = _probe_embedding_dim(model)
    return model, device, embed_dim


def _pad_to_length(t: torch.Tensor, length: int, value: float = 0.0) -> torch.Tensor:
    if t.dim() == 3:
        if t.size(1) == length:
            return t
        if t.size(1) > length:
            return t[:, :length, :]
        pad = length - t.size(1)
        return F.pad(t, (0, 0, 0, pad), value=value)
    if t.dim() == 2:
        if t.size(1) == length:
            return t
        if t.size(1) > length:
            return t[:, :length]
        pad = length - t.size(1)
        return F.pad(t, (0, pad), value=value)
    if t.dim() == 1:
        if t.size(0) == length:
            return t
        if t.size(0) > length:
            return t[:length]
        pad = length - t.size(0)
        return F.pad(t, (0, pad), value=value)
    raise RuntimeError(f"Cannot pad tensor with rank {t.dim()}")


def _extract_sequence_mask(
    encoded,
    logits,
    seqs: List[str],
    device: str | torch.device,
) -> torch.Tensor:
    mask = None

    token_data = getattr(encoded, "token_data", None)
    if token_data is not None:
        mask = getattr(token_data, "mask", None)

    if mask is None:
        for attr in (
            "mask",
            "sequence_mask",
            "sequence_token_mask",
            "attention_mask",
        ):
            mask = getattr(encoded, attr, None)
            if mask is not None:
                break

    if mask is None and logits is not None:
        for attr in (
            "mask",
            "sequence_mask",
            "sequence_token_mask",
            "attention_mask",
        ):
            mask = getattr(logits, attr, None)
            if mask is not None:
                break

    if mask is not None:
        mask = mask.to(device=device)
        return mask

    lengths = [len(s or "") for s in seqs]
    batch = len(lengths)
    if batch == 0:
        return torch.zeros((0, 0), dtype=torch.float32, device=device)

    emb = getattr(logits, "embeddings", None)
    if emb is not None and isinstance(emb, torch.Tensor):
        L = emb.size(1)
    else:
        tokens = getattr(encoded, "sequence_tokens", None)
        if isinstance(tokens, torch.Tensor):
            L = tokens.size(1)
        else:
            L = max(lengths) if lengths else 0

    mask_tensor = torch.zeros((batch, L), dtype=torch.float32, device=device)
    for i, length in enumerate(lengths):
        keep = max(0, min(length, L))
        if keep:
            mask_tensor[i, :keep] = 1.0
    return mask_tensor


@torch.inference_mode()
def encode_sequence_tokens(
    seqs: Sequence[str | ESMProtein],
    model: ESMC,
    device: str | torch.device,
    max_length: int,
    *,
    logits_config: Optional[LogitsConfig] = None,
    embed_dim: Optional[int] = None,
    seq_strings: Optional[Sequence[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_items = list(seqs)

    if not seq_items:
        if embed_dim is None:
            raise ValueError("embed_dim must be provided when seqs is empty")
        return (
            torch.empty((0, max_length, embed_dim), dtype=torch.float32, device=device),
            torch.empty((0, max_length), dtype=torch.float32, device=device),
        )

    logits_config = logits_config or LogitsConfig(
        sequence=True, return_embeddings=True
    )

    if seq_strings is not None:
        seq_strings_list = [str(s or "") for s in seq_strings]
    else:
        seq_strings_list = []
        for item in seq_items:
            if isinstance(item, ESMProtein):
                seq_strings_list.append(str(getattr(item, "sequence", "") or ""))
            else:
                seq_strings_list.append(str(item or ""))

    proteins: List[ESMProtein] = []
    for item, seq_str in zip(seq_items, seq_strings_list):
        if isinstance(item, ESMProtein):
            proteins.append(item)
        else:
            proteins.append(ESMProtein(sequence=seq_str))

    try:
        encoded = model.encode(proteins)
        logits = model.logits(encoded, logits_config)
        embeddings = logits.embeddings
        mask = _extract_sequence_mask(encoded, logits, seq_strings_list, device)
        if embeddings.ndim == 2:
            embeddings = embeddings.unsqueeze(0)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        embeddings = _pad_to_length(embeddings, max_length)
        mask = _pad_to_length(mask, max_length)
        return embeddings.to(torch.float32), mask.to(torch.float32)
    except Exception:
        all_embeddings: List[torch.Tensor] = []
        all_masks: List[torch.Tensor] = []
        for protein, seq_str in zip(proteins, seq_strings_list):
            encoded_single = model.encode(protein)
            logits_single = model.logits(encoded_single, logits_config)
            emb = logits_single.embeddings
            mask_single = _extract_sequence_mask(
                encoded_single, logits_single, [seq_str], device
            )
            if emb.ndim == 2:
                emb = emb.unsqueeze(0)
            if mask_single.ndim == 1:
                mask_single = mask_single.unsqueeze(0)
            emb = _pad_to_length(emb, max_length)
            mask_single = _pad_to_length(mask_single, max_length)
            all_embeddings.append(emb)
            all_masks.append(mask_single)
        hidden = torch.cat(all_embeddings, dim=0)
        masks = torch.cat(all_masks, dim=0)
        return hidden.to(torch.float32), masks.to(torch.float32)


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
    model_id: str = "esmc_600m",
    batch_size: int = 8,
    window: int = 127,
    max_length: int = 2048,
    force_embeddings: bool = False,
    archive_compression: Optional[str] = "lzf",
    archive_compression_opts: Optional[int] = None,
) -> tuple[pd.DataFrame, str]:
    """
    Loads VEP-joined WT/MT sequences, embeds with ESMC (unique-only with interleaved scheduling),
    saves an embedding archive (HDF5) + Feather (meta). Returns (kept_meta_df, out_h5_path).

    Memory behavior:
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

    target_seq_len = max(1, min(int(window), int(max_length)))
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
        model, resolved_device, embed_dim = build_protein_encoder(
            model_id=model_id, device=device
        )
        device = resolved_device
        seq_len = max(1, min(int(target_seq_len), int(max_length)))
        D = embed_dim
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
            compression=archive_compression,
            compression_opts=archive_compression_opts,
        )
        archive_prepared = True
        logits_config = LogitsConfig(sequence=True, return_embeddings=True)
        encoder_bundle = (model, device, seq_len, D, N, batch_rows, logits_config)
        with h5py.File(out_h5_path, "r") as store:
            compression_desc = store.attrs.get("compression", "unknown")
        print(
            f"[protein][embeddings] initialise archive path={out_h5_path} rows={N} seq_len={seq_len} "
            f"embed_dim={D} compression={compression_desc}"
        )

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
        model, device, seq_len, D, N, batch_rows, logits_config = encoder_bundle
        indices = list(range(N))
        wt_seqs = [str(s) for s in kept["seq_wt"].tolist()]
        mt_seqs = [str(s) for s in kept["seq_mt"].tolist()]
        edit_indices = np.clip(
            np.array(edit_positions, dtype=np.int64), 0, seq_len - 1
        ).astype(np.int64, copy=False)
        frameshift_flags = np.array(frameshift_flags, dtype=bool)
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
                compression=archive_compression,
                compression_opts=archive_compression_opts,
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
            frameshift_positions = (
                np.arange(seq_len, dtype=np.int64) if frameshift_ds is not None else None
            )

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
                    combined,
                    model,
                    device,
                    seq_len,
                    logits_config=logits_config,
                    embed_dim=D,
                    seq_strings=combined,
                )
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
                wt_mask_batch = wt_attn.to(dtype=torch.uint8).cpu().contiguous().numpy()
                mt_mask_batch = mt_attn.to(dtype=torch.uint8).cpu().contiguous().numpy()
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
                if bsz:
                    batch_edit_indices = edit_indices[idxs_arr].astype(np.int64, copy=False)
                    edit_batch[np.arange(bsz), batch_edit_indices] = 1
                    batch_offsets = batch_edit_indices.astype(np.float16, copy=False)[:, None]
                    pos_batch[:, :, 0] = base_pos[None, :] - batch_offsets

                    if frameshift_ds is not None:
                        frameshift_active = frameshift_flags[idxs_arr]
                        if np.any(frameshift_active):
                            frameshift_batch[frameshift_active] = (
                                frameshift_positions[None, :]
                                >= batch_edit_indices[frameshift_active, None]
                            ).astype(np.uint8)

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
        raise RuntimeError(f"Alignment mismatch: arrays N={N} vs kept N={len(kept)}")

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
