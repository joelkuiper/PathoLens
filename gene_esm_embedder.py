# gene_esm_embedder.py
# Build gene-level protein embeddings from a UniProt DataFrame via ESM2.
# Returns: (gene2vec: Dict[str, np.ndarray], blocks: Dict[str, np.ndarray])

from __future__ import annotations
import os
import hashlib
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
from tqdm import tqdm

try:
    # HF ESM2
    from transformers import AutoTokenizer, AutoModel
except Exception as e:
    raise RuntimeError(
        "transformers is required for ESM2 embeddings. "
        "pip install transformers accelerate"
    ) from e


# ----------------------------
# Small utilities
# ----------------------------
def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def _sanitize_seq(seq: str) -> str:
    """
    Keep only valid amino-acid letters that ESM can handle; map the rest to 'X'.
    Uppercase, strip whitespace. Remove internal '*' if any (rare in UniProt).
    """
    if not isinstance(seq, str):
        return ""
    s = "".join(ch for ch in seq.upper() if not ch.isspace())
    s = s.replace("*", "")  # strip stop markers if present
    allowed = set(list("ACDEFGHIKLMNPQRSTVWY") + ["U", "O", "X"])
    return "".join(ch if ch in allowed else "X" for ch in s)


def _pick_primary_symbol(gene_names: str) -> Optional[str]:
    """
    UniProt 'Gene Names' often contains space-separated synonyms.
    We take the first token as the primary symbol.
    """
    if not isinstance(gene_names, str) or not gene_names.strip():
        return None
    return gene_names.strip().split()[0]


# ----------------------------
# Build symbol → sequence map
# ----------------------------
def build_symbol_to_sequence(
    uniprot_df,
    *,
    symbol_col: str = "Gene Names",
    seq_col: str = "Sequence",
    reviewed_col: str = "Reviewed",
) -> Dict[str, str]:
    """
    Construct a dict {SYMBOL -> canonical protein sequence}, preferring reviewed entries;
    if multiple reviewed or none, take the longest sequence.
    """
    if symbol_col not in uniprot_df.columns or seq_col not in uniprot_df.columns:
        raise ValueError(
            f"DataFrame must contain '{symbol_col}' and '{seq_col}' columns."
        )

    has_reviewed = reviewed_col in uniprot_df.columns
    items: Dict[str, List[Tuple[bool, str]]] = {}

    for _, row in uniprot_df.iterrows():
        sym = _pick_primary_symbol(row.get(symbol_col, ""))
        if not sym:
            continue
        seq = _sanitize_seq(row.get(seq_col, ""))
        if not seq:
            continue
        is_rev = False
        if has_reviewed:
            # UniProt tsv typically has 'reviewed' / 'unreviewed'
            val = str(row.get(reviewed_col, "")).lower()
            is_rev = ("reviewed" in val) and ("unreviewed" not in val)
        items.setdefault(sym, []).append((is_rev, seq))

    symbol2seq: Dict[str, str] = {}
    for sym, candidates in items.items():
        # prefer reviewed; tie-breaker by length
        reviewed = [seq for rev, seq in candidates if rev]
        pool = reviewed if reviewed else [seq for _, seq in candidates]
        best = max(pool, key=lambda s: len(s))
        symbol2seq[sym] = best

    return symbol2seq


# ----------------------------
# ESM embedder
# ----------------------------
@dataclass
class ESMConfig:
    model_id: str = "facebook/esm2_t6_8M_UR50D"  # tiny & fast
    device: Optional[str] = None  # "cuda" / "cpu" / None->auto
    dtype: Optional[str] = "float16"  # "float16" | "bfloat16" | None
    batch_size: int = 16
    # For long proteins, average over sliding windows (safe default)
    chunking: bool = True


class GeneESMEmbedder:
    def __init__(self, cfg: ESMConfig):
        self.cfg = cfg
        dev = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id, do_lower_case=False
        )
        self.model = AutoModel.from_pretrained(cfg.model_id)
        self.model.eval().to(self.device)
        # determine dtype
        if cfg.dtype in ("float16", "fp16") and self.device.type == "cuda":
            self.autocast_dtype = torch.float16
        elif cfg.dtype in ("bfloat16", "bf16") and self.device.type != "cpu":
            self.autocast_dtype = torch.bfloat16
        else:
            self.autocast_dtype = None
        # max length (including specials)
        # HF ESM2 exposes config.max_position_embeddings (typically 1022)
        self.max_len = getattr(self.model.config, "max_position_embeddings", 1022)
        # We'll create windows of size (max_len - 2) to account for BOS/EOS
        self.window_len = max(1, self.max_len - 2)

    def _seq_windows(self, seq: str) -> List[str]:
        if (not self.cfg.chunking) or len(seq) <= self.window_len:
            return [seq]
        # simple non-overlapping windows; you can change to overlapping if you prefer
        wins = []
        for i in range(0, len(seq), self.window_len):
            wins.append(seq[i : i + self.window_len])
        return wins

    def _embed_batch(self, sequences: List[str]) -> np.ndarray:
        """
        Mean-pool last hidden state over tokens (excluding pads/specials).
        Returns (B, D).
        """
        if not sequences:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)

        with torch.no_grad():
            # autocast for speed if allowed
            ctx = (
                torch.cuda.amp.autocast(dtype=self.autocast_dtype)
                if (self.autocast_dtype is not None)
                else torch.no_grad()
            )
            with ctx:
                toks = self.tokenizer(
                    sequences,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    add_special_tokens=True,
                    max_length=self.max_len,
                )
                input_ids = toks["input_ids"].to(self.device)
                attn = toks["attention_mask"].to(self.device)
                out = self.model(input_ids=input_ids, attention_mask=attn)
                # last hidden state: (B, L, D)
                hs = out.last_hidden_state
                # exclude special tokens if tokenizer provides IDs; else rely on attn mask
                # (CLS/BOS, EOS are included in attention_mask; we can just mean-pool masked tokens)
                masked = hs * attn.unsqueeze(-1)
                sums = masked.sum(dim=1)
                lens = attn.sum(dim=1).clamp(min=1).unsqueeze(-1)
                mean = (sums / lens).detach().cpu().float().numpy()
                return mean

    def embed_symbol_to_vec(self, symbol2seq: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Embed each symbol's protein sequence. For long sequences, average window embeddings.
        """
        # Process in batches of sequences (each seq may further split into windows)
        symbols = list(symbol2seq.keys())
        vecs: Dict[str, np.ndarray] = {}

        for i in tqdm(
            range(0, len(symbols), self.cfg.batch_size),
            desc="[ESM] genes",
            unit="batch",
        ):
            batch_syms = symbols[i : i + self.cfg.batch_size]
            # prepare windows for all sequences in this mini-batch
            win_list: List[str] = []
            owner: List[int] = []  # map window -> index in batch_syms
            for k, sym in enumerate(batch_syms):
                seq = symbol2seq[sym]
                wins = self._seq_windows(seq)
                win_list.extend(wins)
                owner.extend([k] * len(wins))

            # embed all windows together (still batched by self.cfg.batch_size via tokenizer padding)
            emb_windows = self._embed_batch(win_list)  # shape (sum_wins, D)
            D = (
                emb_windows.shape[1]
                if emb_windows.size
                else self.model.config.hidden_size
            )
            # reduce windows → per-sequence by mean
            for k, sym in enumerate(batch_syms):
                idx = [j for j, own in enumerate(owner) if own == k]
                if not idx:
                    vecs[sym] = np.zeros((D,), dtype=np.float32)
                else:
                    vecs[sym] = emb_windows[idx].mean(axis=0).astype(np.float32)

        return vecs


# ----------------------------
# Cache (optional)
# ----------------------------
def _cache_key(symbol2seq: Dict[str, str], model_id: str) -> str:
    # hash on model + a compact digest of sequences (symbol order independent)
    # WARNING: changing symbol->sequence alters the key; good for invalidation.
    items = [f"{k}:{len(v)}:{_sha(v[:64])}" for k, v in sorted(symbol2seq.items())]
    raw = model_id + "|" + "|".join(items[:5000])  # cap to keep key bounded
    return _sha(raw)


def save_gene2vec_npz(path: str, gene2vec: Dict[str, np.ndarray], model_id: str):
    genes = np.array(list(gene2vec.keys()), dtype=object)
    mat = np.vstack([gene2vec[g] for g in genes]).astype(np.float32)
    np.savez_compressed(path, genes=genes, emb=mat, model_id=model_id)


def load_gene2vec_npz(path: str) -> Tuple[Dict[str, np.ndarray], str]:
    z = np.load(path, allow_pickle=True)
    genes = z["genes"]
    emb = z["emb"]
    model_id = str(z.get("model_id", ""))
    d = {str(g): emb[i] for i, g in enumerate(genes)}
    return d, model_id


# ----------------------------
# Public API
# ----------------------------
def build_gene_esm_embeddings(
    uniprot_df,
    *,
    model_id: str = "facebook/esm2_t6_8M_UR50D",
    device: Optional[str] = None,
    dtype: Optional[str] = "float16",
    batch_size: int = 16,
    cache_dir: Optional[str] = "artifacts/esm_gene_cache",
) -> Dict[str, np.ndarray]:
    """
    Main entry: from UniProt DataFrame → {gene_symbol: ESM vector}.
    Uses on-disk cache keyed by (model, sequences), if cache_dir is provided.
    """
    os.makedirs(cache_dir, exist_ok=True) if cache_dir else None

    symbol2seq = build_symbol_to_sequence(uniprot_df)
    key = _cache_key(symbol2seq, model_id)
    cache_path = (
        os.path.join(cache_dir, f"gene2vec_{model_id.split('/')[-1]}_{key}.npz")
        if cache_dir
        else None
    )

    if cache_path and os.path.exists(cache_path):
        gene2vec, cached_model = load_gene2vec_npz(cache_path)
        # simple sanity: if model_id changed, ignore cache
        if cached_model == model_id:
            print(f"[cache] loaded gene2vec from {cache_path} ({len(gene2vec)} genes)")
            return gene2vec

    # compute fresh
    cfg = ESMConfig(
        model_id=model_id,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        chunking=True,
    )
    embedder = GeneESMEmbedder(cfg)
    gene2vec = embedder.embed_symbol_to_vec(symbol2seq)

    if cache_path:
        save_gene2vec_npz(cache_path, gene2vec, model_id)
        print(f"[cache] saved gene2vec to {cache_path} ({len(gene2vec)} genes)")

    return gene2vec


def _pick_gene_from_row(row) -> Optional[str]:
    # Light fallback; replace with your src.clinvar.pick_gene if you prefer
    for k in ("GeneSymbol", "GeneSymbolSimple", "Gene", "gene", "GENE"):
        val = row.get(k, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def block_from_split(
    split, gene2vec: Dict[str, np.ndarray], d: Optional[int] = None
) -> np.ndarray:
    """
    Assemble an (N, D) matrix aligned to a split (train/val/test) using gene2vec.
    Missing genes → zeros.
    """
    N = len(split)
    # infer D from any vector in dict, or use provided d
    if d is None:
        d = next((len(v) for v in gene2vec.values()), 0)
    out = np.zeros((N, d), dtype=np.float32)
    meta = getattr(split, "meta", None)

    for i in range(N):
        g = None
        if meta is not None:
            row = meta.iloc[i]
            g = _pick_gene_from_row(row)
        if g is None and hasattr(split, "get_gene"):
            try:
                g = split.get_gene(i)  # optional custom accessor
            except Exception:
                g = None
        vec = gene2vec.get(g)
        if vec is not None:
            out[i] = vec
    return out


def make_gene_esm_blocks(
    uniprot_df,
    seq_ds: Dict[str, any],
    *,
    model_id: str = "facebook/esm2_t6_8M_UR50D",
    device: Optional[str] = None,
    dtype: Optional[str] = "float16",
    batch_size: int = 16,
    cache_dir: Optional[str] = "artifacts/esm_gene_cache",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Convenience: compute gene2vec and build aligned blocks for all splits.
    Returns (gene2vec, {'train': Xtr, 'val': Xv, 'test': Xt})
    """
    gene2vec = build_gene_esm_embeddings(
        uniprot_df,
        model_id=model_id,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        cache_dir=cache_dir,
    )
    # infer D
    D = next((len(v) for v in gene2vec.values()), 0)
    blocks = {
        k: block_from_split(seq_ds[k], gene2vec, d=D) for k in ("train", "val", "test")
    }

    # quick coverage report
    def _coverage(split):
        meta = getattr(seq_ds[split], "meta", None)
        if meta is None:
            return (0, 0.0)
        hits = 0
        for i in range(len(seq_ds[split])):
            g = _pick_gene_from_row(meta.iloc[i])
            if g in gene2vec:
                hits += 1
        return hits, (hits / max(1, len(seq_ds[split])))

    for split in ("train", "val", "test"):
        h, frac = _coverage(split)
        print(f"[ESM] coverage {split}: {h}/{len(seq_ds[split])} = {frac:.3f}")

    return gene2vec, blocks
