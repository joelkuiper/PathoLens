# src/protein_utils.py
from __future__ import annotations
from typing import Tuple, Optional, List, Dict
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

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


def compute_effect_vectors(wt: np.ndarray, mt: np.ndarray) -> np.ndarray:
    eff = (mt - wt).astype("float32")
    norms = np.linalg.norm(eff, axis=1, keepdims=True) + 1e-8
    return (eff / norms).astype("float32")


def save_protein_arrays(out_npz: str, prot_eff=None) -> None:
    """
    Save as .npz; we store only the effect vectors to keep disk small.
    """
    np.savez(
        out_npz,
        **(
            {}
            if prot_eff is None
            else {"prot_eff": prot_eff.astype("float16", copy=False)}
        ),
    )


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
    out_npz: str,
    device: Optional[str] = None,
    model_id: str = "facebook/esm2_t33_650M_UR50D",
    batch_size: int = 8,
    pool: str = "mean",
    max_length: int = 2048,
    force_embeddings: bool = False,
) -> tuple[pd.DataFrame, str]:
    """
    Loads VEP-joined WT/MT sequences, embeds with ESM2 (unique-only with interleaved scheduling),
    saves NPZ (effect vectors) + Feather (meta). Returns (kept_meta_df, out_npz_path).

    Memory behavior:
      - Embeddings for UNIQUE sequences are written to disk memmaps (no big RAM spikes).
      - Batch sizes adapt to token budget and back off on OOM (esp. on MPS).
    """
    os.makedirs(os.path.dirname(out_meta), exist_ok=True)

    # Load and keep usable rows
    pairs = load_vep_sequences(vep_combined)
    if len(pairs) == 0:
        raise RuntimeError("No WT/MT sequences found in VEP combined file.")
    kept = pairs.reset_index(drop=True).copy()
    kept["protein_embedding_idx"] = np.arange(len(kept), dtype=np.int32)

    # Short-circuit if cache matches (keeps re-runs instant)
    need_embed = True
    if os.path.exists(out_npz) and not force_embeddings:
        try:
            npz = np.load(out_npz)
            if "prot_eff" in npz.files and npz["prot_eff"].shape[0] == len(kept):
                need_embed = False
                print(
                    f"[protein][embeddings] reuse {out_npz} rows={len(kept)} force={force_embeddings}"
                )
        except Exception:
            need_embed = True

    if need_embed:
        # Build encoder
        tok, enc, device = build_protein_encoder(model_id=model_id, device=device)

        # Encode UNIQUE WT sequences to memmap
        wt_mm_path, _, wt_row2uniq = embed_unique_to_memmap(
            kept["seq_wt"].tolist(),
            tok,
            enc,
            device,
            out_path=str(Path(out_npz).with_suffix(".wt.memmap")),
            batch_size=batch_size,
            max_length=max_length,
            pool=pool,
            # Let embed_unique_to_memmap pick a safe token budget on MPS; keep None here.
            target_tokens=None,
        )

        # Encode UNIQUE MT sequences to memmap
        mt_mm_path, _, mt_row2uniq = embed_unique_to_memmap(
            kept["seq_mt"].tolist(),
            tok,
            enc,
            device,
            out_path=str(Path(out_npz).with_suffix(".mt.memmap")),
            batch_size=batch_size,
            max_length=max_length,
            pool=pool,
            target_tokens=None,
        )

        # Load uniques back (still memory-light; each memmap is [U, D])
        wt_unique = np.memmap(wt_mm_path, mode="r", dtype="float32")
        mt_unique = np.memmap(mt_mm_path, mode="r", dtype="float32")

        # Infer D from model; infer U from mapping (max index + 1)
        D = enc.get_input_embeddings().embedding_dim
        U_wt = (max(wt_row2uniq.values()) + 1) if wt_row2uniq else 0
        U_mt = (max(mt_row2uniq.values()) + 1) if mt_row2uniq else 0
        wt_unique = wt_unique.reshape(U_wt, D)
        mt_unique = mt_unique.reshape(U_mt, D)

        # Gather to row order (RAM proportional to N x D just once here)
        N = len(kept)
        wt_emb = wt_unique[[wt_row2uniq[i] for i in range(N)]]
        mt_emb = mt_unique[[mt_row2uniq[i] for i in range(N)]]

        # Compute effect vectors and save
        eff = compute_effect_vectors(wt_emb, mt_emb)
        save_protein_arrays(out_npz, prot_eff=eff)
        print(
            f"[protein][embeddings] wrote {out_npz} rows={N} uniques_wt={len(wt_row2uniq)}"
        )

    # Always (re)write meta so it matches whatever we embedded
    write_protein_meta(kept, out_meta)
    print(f"[protein][meta] wrote {out_meta} rows={len(kept)}")
    return kept, out_npz
