from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.dna_utils import DNAConditioningStreamer, encode_dna_conditioning
from src.go.gene_go_embeddings import GeneGOEmbeddings
from src.protein_utils import (
    ProteinConditioningStreamer,
    encode_protein_conditioning,
)


def _normalize_device(device: Optional[Union[str, torch.device]]) -> str:
    if device is None:
        return ""
    if isinstance(device, torch.device):
        return device.type
    dev = str(device).strip()
    return "" if dev.lower() == "auto" else dev


def _select_conditioning_dtype(
    requested: Optional[Union[str, np.dtype]],
    device: Optional[Union[str, torch.device]],
) -> np.dtype:
    if requested is not None and str(requested).lower() not in {"auto", "default"}:
        return np.dtype(requested)

    normalized = _normalize_device(device)
    specified = [normalized] if normalized else []

    def _is_accel(dev: str) -> bool:
        return dev.startswith("cuda") or dev.startswith("mps")

    if any(_is_accel(dev) for dev in specified):
        return np.dtype("float16")
    if specified and all(dev == "cpu" for dev in specified):
        return np.dtype("float32")
    if torch.cuda.is_available():
        return np.dtype("float16")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return np.dtype("float16")
    return np.dtype("float32")


class SequenceTowerDataset(Dataset):
    """Dataset that streams DNA/GO/Protein conditioning features."""

    def __init__(
        self,
        *,
        meta_df: Optional[pd.DataFrame] = None,
        meta_feather: Optional[str] = None,
        go_npz: Optional[str] = None,
        make_label: bool = True,
        label_col: str = "_y",
        go_normalize: bool = True,
        go_uppercase: bool = True,
        schema: Optional[Mapping[str, str]] = None,
        dna_model_id: str = "InstaDeepAI/nucleotide-transformer-500m-1000g",
        dna_model_revision: Optional[str] = None,
        dna_max_length: int = 384,
        dna_batch_size: int = 16,
        protein_model_id: str = "facebook/esm2_t12_35M_UR50D",
        protein_model_revision: Optional[str] = None,
        protein_max_length: int = 2048,
        protein_batch_size: int = 8,
        device: Optional[Union[str, torch.device]] = None,
        conditioning_dtype: Optional[Union[str, np.dtype]] = "auto",
    ) -> None:
        if meta_df is None:
            if not meta_feather:
                raise ValueError(
                    "SequenceTowerDataset requires either meta_df or meta_feather path"
                )
            meta_df = pd.read_feather(meta_feather)
        if schema is not None:
            missing = [col for col in schema if col not in meta_df.columns]
            if missing:
                raise ValueError(
                    "Metadata frame is missing columns expected by schema: "
                    + ", ".join(sorted(missing))
                )
            mismatched: list[str] = []
            for col, expected in schema.items():
                actual = str(meta_df[col].dtype)
                if actual != expected:
                    mismatched.append(f"{col}: expected {expected}, found {actual}")
            if mismatched:
                raise ValueError(
                    "Metadata frame columns have mismatched dtypes: "
                    + "; ".join(mismatched)
                )

        self.meta = meta_df.reset_index(drop=True)
        if "dna_embedding_idx" not in self.meta.columns:
            raise KeyError("Metadata frame missing 'dna_embedding_idx' column")
        if "protein_embedding_idx" not in self.meta.columns:
            raise KeyError("Metadata frame missing 'protein_embedding_idx' column")

        dna_idx_series = self.meta["dna_embedding_idx"]
        if dna_idx_series.isna().any():
            raise ValueError("dna_embedding_idx column contains missing values")
        self._dna_idx = dna_idx_series.astype("int64").to_numpy()

        protein_idx_series = self.meta["protein_embedding_idx"].fillna(-1)
        self._prot_idx = protein_idx_series.astype("int64").to_numpy()

        # Pre-compute uppercase gene names once for GO lookups
        gene_series = self.meta.get(
            "GeneSymbol",
            pd.Series(index=self.meta.index, data="", dtype="string"),
        )
        gene_series = gene_series.astype("string").fillna("")
        self._gene_values = gene_series.tolist()
        self._gene_upper = gene_series.str.upper().tolist()
        self._gene_lookup = list(self._gene_upper)

        self.cond_dtype = _select_conditioning_dtype(conditioning_dtype, device)
        self._cond_dtype_str = str(self.cond_dtype)

        # ---- DNA embeddings ----
        dna_stream = encode_dna_conditioning(
            self.meta,
            model_id=dna_model_id,
            revision=dna_model_revision,
            max_length=dna_max_length,
            batch_size=dna_batch_size,
            device=device,
        )
        if not isinstance(dna_stream, DNAConditioningStreamer):
            raise TypeError(
                "encode_dna_conditioning must return DNAConditioningStreamer for streaming"
            )
        self._dna_stream = dna_stream
        self.dna_seq_len = int(getattr(dna_stream, "seq_len", 0))
        self.dna_embed_dim = int(getattr(dna_stream, "embed_dim", 0))

        # ---- GO embeddings ----
        self.go = None
        self.D_go = 0
        self.go_dim = 0
        self._go_block: Optional[np.ndarray] = None
        self.go_hit_rate = 0.0
        if go_npz:
            try:
                go_embed = GeneGOEmbeddings(
                    go_npz, normalize=go_normalize, uppercase_keys=go_uppercase
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                raise RuntimeError(
                    f"Failed to load GO embeddings from '{go_npz}': {exc}"
                ) from exc

            self.go = go_embed
            self.D_go = int(go_embed.dim)
            self.go_dim = self.D_go
            if self.D_go > 0:
                gene_lookup = (
                    self._gene_upper
                    if getattr(go_embed, "uppercase", True)
                    else self._gene_values
                )
                self._gene_lookup = list(gene_lookup)
                go_batch = go_embed.batch(self._gene_lookup)
                self._go_block = np.asarray(go_batch, dtype=self.cond_dtype)
                go_hits = sum(1 for g in self._gene_lookup if go_embed.has(g))
                total_genes = len(self._gene_lookup)
                self.go_hit_rate = go_hits / max(1, total_genes)
            else:
                self._go_block = None
        else:
            self._go_block = None

        # ---- labels ----
        self.make_label = bool(make_label)
        self.label_col = str(label_col)
        self._labels = None
        if self.make_label:
            if self.label_col not in self.meta.columns:
                raise KeyError(f"Label column '{self.label_col}' not found in meta.")
            label_series = pd.to_numeric(
                self.meta[self.label_col], errors="raise"
            )
            if label_series.isna().any():
                raise ValueError(
                    f"Label column '{self.label_col}' contains missing values"
                )
            self._labels = label_series.astype("int64").to_numpy()

        # ---- protein embeddings ----
        protein_stream = encode_protein_conditioning(
            self.meta,
            model_id=protein_model_id,
            revision=protein_model_revision,
            max_length=protein_max_length,
            batch_size=protein_batch_size,
            device=device,
        )
        if not isinstance(protein_stream, ProteinConditioningStreamer):
            raise TypeError(
                "encode_protein_conditioning must return ProteinConditioningStreamer"
            )
        self._protein_stream = protein_stream
        self.prot_seq_len = int(getattr(protein_stream, "seq_len", 0))
        self.prot_embed_dim = int(getattr(protein_stream, "embed_dim", 0))

        self.D_eff = 0
        self.dna_dim = 0
        self.D_prot = 0
        self.protein_dim = 0

        self._go_slice: Optional[slice] = None
        self._dna_block_slice: Optional[slice] = None
        self._prot_block_slice: Optional[slice] = None

        # Build global slices for conditioning vector layout
        self.cond_slices: dict[str, Optional[dict[str, slice]]] = {}
        offset = 0

        dna_block_start = offset
        dna_slices: Optional[dict[str, slice]] = None
        if self.dna_seq_len > 0 and self.dna_embed_dim > 0:
            L = self.dna_seq_len
            E = self.dna_embed_dim
            dna_slices = {}
            dna_slices["ref_tokens"] = slice(offset, offset + L * E)
            offset += L * E
            dna_slices["alt_tokens"] = slice(offset, offset + L * E)
            offset += L * E
            for key in ("ref_mask", "alt_mask", "edit_mask", "gap_mask", "splice_mask", "pos"):
                length = L if key != "pos" else L
                dna_slices[key] = slice(offset, offset + length)
                offset += length
        dna_block_stop = offset
        self.cond_slices["dna"] = dna_slices
        self.D_eff = dna_block_stop - dna_block_start
        self.dna_dim = self.D_eff

        self._dna_block_slice = (
            slice(dna_block_start, dna_block_stop) if self.D_eff > 0 else None
        )

        go_slice = slice(offset, offset + self.D_go)
        self.cond_slices["go"] = go_slice
        offset = go_slice.stop
        self._go_slice = go_slice if self.D_go > 0 else None

        prot_block_start = offset
        prot_slices: Optional[dict[str, slice]] = None
        if self.prot_seq_len > 0 and self.prot_embed_dim > 0:
            Lp = self.prot_seq_len
            Ep = self.prot_embed_dim
            prot_slices = {}
            prot_slices["wt_tokens"] = slice(offset, offset + Lp * Ep)
            offset += Lp * Ep
            prot_slices["mt_tokens"] = slice(offset, offset + Lp * Ep)
            offset += Lp * Ep
            for key in ("wt_mask", "mt_mask", "edit_mask", "gap_mask", "frameshift_mask", "pos"):
                prot_slices[key] = slice(offset, offset + Lp)
                offset += Lp
        prot_block_stop = offset
        self.cond_slices["protein"] = prot_slices
        self.D_prot = prot_block_stop - prot_block_start
        self.protein_dim = self.D_prot
        self.total_dim = offset

        self._prot_block_slice = (
            slice(prot_block_start, prot_block_stop) if self.D_prot > 0 else None
        )

        dna_extra = []
        if dna_slices and "splice_mask" in dna_slices:
            dna_extra.append("splice_mask")
        prot_extra = []
        if prot_slices and "frameshift_mask" in prot_slices:
            prot_extra.append("frameshift_mask")

        dna_block_slice = slice(dna_block_start, dna_block_stop)
        prot_block_slice = slice(prot_block_start, prot_block_stop)

        self.cond_spec: dict[str, Any] = {
            "dna": {
                "seq_len": self.dna_seq_len,
                "embed_dim": self.dna_embed_dim,
                "slice": dna_block_slice if self.D_eff > 0 else None,
                "slices": dna_slices,
                "extra_masks": dna_extra,
            },
            "protein": {
                "seq_len": self.prot_seq_len,
                "embed_dim": self.prot_embed_dim,
                "slice": prot_block_slice if self.D_prot > 0 else None,
                "slices": prot_slices,
                "extra_masks": prot_extra,
            },
            "go": {"slice": go_slice if self.D_go > 0 else None, "dim": self.D_go},
            "total_dim": self.total_dim,
            "dtype": self._cond_dtype_str,
        }

        # Protein coverage statistic (fraction of rows with embeddings)
        covered = int((self._prot_idx >= 0).sum())
        self.protein_coverage = covered / max(1, len(self.meta))

    # -----------------
    # Dataset protocol
    # -----------------

    @staticmethod
    def _fill_feature_block(
        target: np.ndarray,
        slices: Optional[Mapping[str, slice]],
        features: Optional[Mapping[str, Any]],
        dtype: np.dtype,
    ) -> None:
        if not slices or not features:
            return
        for key, sl in slices.items():
            if sl is None:
                continue
            value = features.get(key)
            if value is None:
                continue
            start = 0 if sl.start is None else int(sl.start)
            stop = start if sl.stop is None else int(sl.stop)
            if stop <= start:
                continue
            arr = np.asarray(value, dtype=dtype).reshape(-1)
            needed = stop - start
            if arr.size < needed:
                if arr.size == 0:
                    continue
                padded = np.zeros(needed, dtype=dtype)
                padded[: arr.size] = arr
                arr = padded
            elif arr.size > needed:
                arr = arr[:needed]
            target[start:stop] = arr

    def _encode_features(self, idx: int, streamer) -> Optional[Dict[str, np.ndarray]]:
        if streamer is None:
            return None
        if idx < 0:
            return streamer.empty() if hasattr(streamer, "empty") else None
        encoded = streamer.encode([idx])
        feat = encoded.get(idx)
        if feat is None and hasattr(streamer, "empty"):
            return streamer.empty()
        return feat

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_np = np.zeros(self.total_dim, dtype=self.cond_dtype)

        if self._dna_block_slice is not None:
            dna_idx = int(self._dna_idx[idx])
            feats = self._encode_features(dna_idx, self._dna_stream)
            self._fill_feature_block(
                x_np,
                self.cond_slices.get("dna"),
                feats,
                self.cond_dtype,
            )

        if self._go_slice is not None and self._go_block is not None:
            x_np[self._go_slice] = self._go_block[idx]

        if self._prot_block_slice is not None:
            prot_idx = int(self._prot_idx[idx])
            feats = self._encode_features(prot_idx, self._protein_stream)
            self._fill_feature_block(
                x_np,
                self.cond_slices.get("protein"),
                feats,
                self.cond_dtype,
            )

        x = torch.from_numpy(x_np)

        y = None
        if self.make_label and self._labels is not None:
            y = torch.tensor(int(self._labels[idx]), dtype=torch.long)

        return x, y

    # -----------------
    # Resource handling
    # -----------------
    def close(self) -> None:  # pragma: no cover - retained for API compatibility
        return None

    def __del__(self):  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass
