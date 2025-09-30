# src/protein_utils.py
from __future__ import annotations
from typing import Tuple, Optional, List, Dict, Any, Sequence
from difflib import SequenceMatcher
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

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
    revision: Optional[str] = None,
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

    def _load_model(
        target_dtype: torch.dtype, *, force_download: bool = False
    ) -> torch.nn.Module:
        model_kwargs = dict(load_kwargs)
        if force_download:
            model_kwargs["force_download"] = True
        try:
            return AutoModel.from_pretrained(
                model_id,
                dtype=target_dtype,
                **model_kwargs,
            )
        except TypeError:
            model_kwargs.pop("dtype", None)
            model_kwargs["torch_dtype"] = target_dtype
            return AutoModel.from_pretrained(
                model_id,
                **model_kwargs,
            )

    try:
        enc = _load_model(prefer_dtype)
    except RuntimeError as exc:
        if revision is None and "size mismatch" in str(exc):
            tok = _load_tokenizer(force_download=True)
            enc = _load_model(prefer_dtype, force_download=True)
        elif prefer_dtype != torch.float32:
            enc = _load_model(torch.float32)
        else:
            raise
    except Exception:
        if prefer_dtype != torch.float32:
            enc = _load_model(torch.float32)
        else:
            raise

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


class ProteinConditioningStreamer:
    """Helper that lazily encodes protein conditioning tensors on demand."""

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

            wt_batch = [rec["wt_seq"] for rec in chunk_records]
            mt_batch = [rec["mt_seq"] for rec in chunk_records]
            edit_batch = [int(rec["edit_idx"]) for rec in chunk_records]
            frame_batch = [bool(rec["frameshift"]) for rec in chunk_records]

            combined = wt_batch + mt_batch
            hidden, attn = encode_sequence_tokens(
                combined,
                self.model,
                self.tokenizer,
                self.device,
                max_length=max_len,
            )

            hidden = hidden[:, : self.seq_len, :].to(torch.float32).cpu().numpy()
            attn = attn[:, : self.seq_len].to(torch.float32).cpu().numpy()
            half = len(chunk)
            wt_hidden = hidden[:half]
            mt_hidden = hidden[half:]
            wt_attn = attn[:half]
            mt_attn = attn[half:]
            gap = np.logical_xor(wt_attn > 0, mt_attn > 0).astype("float32")

            for j, arr_idx in enumerate(chunk):
                edit_pos = min(max(int(edit_batch[j]), 0), self.seq_len - 1)
                edit_vec = np.zeros((self.seq_len,), dtype="float32")
                if self.seq_len > 0:
                    edit_vec[edit_pos] = 1.0

                frame_mask = np.zeros((self.seq_len,), dtype="float32")
                if frame_batch[j] and self.seq_len > 0:
                    frame_mask[edit_pos:] = 1.0

                outputs[arr_idx] = {
                    "wt_tokens": wt_hidden[j],
                    "mt_tokens": mt_hidden[j],
                    "wt_mask": wt_attn[j],
                    "mt_mask": mt_attn[j],
                    "edit_mask": edit_vec,
                    "gap_mask": gap[j],
                    "pos": (self._base_pos - float(edit_pos))[:, None],
                    "frameshift_mask": frame_mask,
                }

        return outputs

    def empty(self) -> Dict[str, np.ndarray]:
        zeros_2d = np.zeros((self.seq_len, self.embed_dim), dtype="float32")
        zeros_1d = np.zeros((self.seq_len,), dtype="float32")
        return {
            "wt_tokens": zeros_2d,
            "mt_tokens": zeros_2d,
            "wt_mask": zeros_1d,
            "mt_mask": zeros_1d,
            "edit_mask": zeros_1d,
            "gap_mask": zeros_1d,
            "pos": np.zeros((self.seq_len, 1), dtype="float32"),
            "frameshift_mask": zeros_1d,
        }


def encode_protein_conditioning(
    meta: pd.DataFrame,
    *,
    model_id: str = "facebook/esm2_t12_35M_UR50D",
    revision: Optional[str] = None,
    max_length: int = 2048,
    batch_size: int = 8,
    device: Optional[str] = None,
) -> ProteinConditioningStreamer:
    """Prepare a streamer that lazily encodes protein conditioning tensors."""

    required = {
        "protein_seq_wt",
        "protein_seq_mt",
        "protein_edit_idx",
        "protein_seq_len",
        "protein_frameshift",
        "protein_embedding_idx",
    }
    missing = [col for col in required if col not in meta.columns]
    if missing:
        raise KeyError(
            "Metadata frame missing protein sequence columns: "
            + ", ".join(sorted(missing))
        )

    uniq = (
        meta[meta["protein_embedding_idx"] >= 0][list(required)]
        .drop_duplicates("protein_embedding_idx")
        .sort_values("protein_embedding_idx")
    )

    if uniq.empty:
        tok, enc, run_device = build_protein_encoder(
            model_id=model_id,
            device=device,
            revision=revision,
        )
        enc.eval()
        return ProteinConditioningStreamer(
            tokenizer=tok,
            model=enc,
            device=run_device,
            seq_len=0,
            embed_dim=0,
            batch_size=batch_size,
            records={},
        )

    seq_len = int(uniq["protein_seq_len"].iloc[0])
    tok, enc, run_device = build_protein_encoder(
        model_id=model_id,
        device=device,
        revision=revision,
    )
    seq_len = max(1, min(seq_len, int(max_length) - 1))
    embed_dim = int(enc.get_input_embeddings().embedding_dim)

    records: Dict[int, Dict[str, Any]] = {}

    for row in uniq.itertuples(index=False):
        arr_idx = int(row.protein_embedding_idx)
        records[arr_idx] = {
            "wt_seq": str(row.protein_seq_wt),
            "mt_seq": str(row.protein_seq_mt),
            "edit_idx": int(row.protein_edit_idx),
            "frameshift": bool(row.protein_frameshift),
        }

    return ProteinConditioningStreamer(
        tokenizer=tok,
        model=enc,
        device=run_device,
        seq_len=seq_len,
        embed_dim=embed_dim,
        batch_size=batch_size,
        records=records,
    )
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
    """Return the combined VEP annotations with sanitised WT/MT sequences.

    The helper keeps the full annotation table so downstream consumers can
    persist additional VEP-derived fields alongside the conditioning metadata.
    """

    df = _read_any_table(vep_combined).copy()

    # Normalise ID so callers can rely on a stable join key regardless of the
    # VEP writer version (some emit ``id`` while older caches used
    # ``VariationID``).
    if "id" in df.columns:
        df["id"] = df["id"].astype("string")
    elif "VariationID" in df.columns:
        df["id"] = df["VariationID"].astype("string")
    else:  # pragma: no cover - defensive guard for unexpected schemas
        raise KeyError("VEP annotations missing 'id' column")

    # Sanitise the raw sequences (remove gaps, map non-canonical residues to X)
    for col in ("seq_wt", "seq_mt"):
        if col in df.columns:
            df[col] = df[col].map(sanitize_aa)

    # Keep only rows that carry both WT and MT sequences – the conditioning
    # streamer requires both sides to be present.
    if {"seq_wt", "seq_mt"}.issubset(df.columns):
        keep = (
            df["seq_wt"].astype("string").str.len().gt(0)
            & df["seq_mt"].astype("string").str.len().gt(0)
        )
        df = df.loc[keep].copy()

    # De-duplicate by (id, protein_id, hgvsp) to keep a stable transcript
    # choice while still allowing callers to reason about every retained VEP
    # column.
    keys = [k for k in ["id", "protein_id", "hgvsp"] if k in df.columns]
    if keys:
        df = (
            df.sort_values(keys)
            .drop_duplicates(keys, keep="first")
            .reset_index(drop=True)
        )

    return df


# ============================================================
# Post-processing & cache
# ============================================================


def write_protein_meta(meta_df: pd.DataFrame, out_feather: str) -> None:
    import pyarrow.feather as feather

    feather.write_feather(meta_df, out_feather)


# ============================================================
# Orchestrator used by pipeline builders
# ============================================================


def process_protein_sequences(
    vep_combined: str | Path | pd.DataFrame,
    *,
    out_meta: str,
    window: int = 127,
    max_length: int = 2048,
) -> pd.DataFrame:
    """Prepare protein WT/MT windows and associated metadata for streaming."""

    os.makedirs(os.path.dirname(out_meta), exist_ok=True)

    # Load and keep usable rows
    pairs = load_vep_sequences(vep_combined)
    if len(pairs) == 0:
        raise RuntimeError("No WT/MT sequences found in VEP combined file.")
    kept = pairs.reset_index(drop=True).copy()
    print(f"[protein] loaded rows={len(kept)} from VEP combined input")

    base_cols = list(kept.columns)
    kept["protein_embedding_idx"] = np.arange(len(kept), dtype=np.int32)

    wt_sanitized = kept.get("seq_wt", pd.Series(dtype="string")).astype("string")
    mt_sanitized = kept.get("seq_mt", pd.Series(dtype="string")).astype("string")

    if wt_sanitized.empty or mt_sanitized.empty:
        raise RuntimeError("VEP annotations missing WT/MT sequences after sanitisation")

    wt_sanitized = [sanitize_aa(str(s)) for s in wt_sanitized]
    mt_sanitized = [sanitize_aa(str(s)) for s in mt_sanitized]

    target_seq_len = max(1, min(int(window), int(max_length) - 1))
    # Trim sequences to the target window length for downstream encoding
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

    kept["protein_seq_wt"] = trimmed_wt
    kept["protein_seq_mt"] = trimmed_mt
    kept["protein_edit_idx"] = [int(x) for x in edit_positions]
    kept["protein_seq_len"] = int(target_seq_len)
    kept["protein_frameshift"] = [bool(x) for x in frameshift_flags]

    # Preserve the raw (sanitised) WT/MT sequences and the rest of the VEP
    # annotations by prefixing them with ``vep_`` so they can live alongside the
    # conditioning features in the merged dataframe.
    rename_map: Dict[str, str] = {}
    for col in base_cols:
        if col == "id":
            rename_map[col] = "vep_id"
        else:
            rename_map[col] = f"vep_{col}"

    kept = kept.rename(columns=rename_map)
    kept["variation_key"] = kept.get("vep_id", pd.Series(dtype="string")).astype("string").fillna("")

    # Maintain commonly-used aliases for downstream code while still exposing the
    # full VEP block.
    if "vep_protein_id" in kept.columns and "protein_id" not in kept.columns:
        kept["protein_id"] = kept["vep_protein_id"].astype("string")
    if "vep_consequence_terms" in kept.columns:
        kept["protein_consequence_terms"] = kept["vep_consequence_terms"]

    # Ensure string columns retain a consistent dtype when written to feather.
    for col in ["protein_seq_wt", "protein_seq_mt"]:
        kept[col] = pd.Series(kept[col], dtype="string").fillna("")

    kept["protein_embedding_idx"] = kept["protein_embedding_idx"].astype(np.int32)
    kept["protein_edit_idx"] = kept["protein_edit_idx"].astype(np.int32)
    kept["protein_seq_len"] = kept["protein_seq_len"].astype(np.int32)
    kept["protein_frameshift"] = kept["protein_frameshift"].astype(bool)

    # Always (re)write meta so it matches whatever we embedded
    write_protein_meta(kept, out_meta)
    print(f"[protein][meta] wrote {out_meta} rows={len(kept)}")
    return kept
