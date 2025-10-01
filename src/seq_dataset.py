from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.dna_utils import (
    compute_dna_slices,
    load_dna_archive,
    validate_dna_indices,
    write_dna_features,
)
from src.protein_utils import (
    compute_protein_slices,
    load_protein_archive,
    validate_protein_indices,
    write_protein_features,
)


class SequenceTowerDataset(Dataset):
    """Dataset that concatenates DNA and protein embeddings."""

    def __init__(
        self,
        *,
        meta_df: Optional[pd.DataFrame] = None,
        meta_feather: Optional[str] = None,
        dna_h5: Optional[str] = None,
        make_label: bool = True,
        label_col: str = "_y",
        protein_h5: Optional[str] = None,
        schema: Optional[Mapping[str, str]] = None,
        use_dna: bool = True,
        use_protein: bool = True,
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

        self.use_dna = bool(use_dna)
        self.use_protein = bool(use_protein)
        if not self.use_dna:
            dna_h5 = None
        if not self.use_protein:
            protein_h5 = None
        if not self.use_dna and not self.use_protein:
            raise ValueError("SequenceTowerDataset requires at least one modality enabled")

        # ---- DNA embeddings ----
        self._dna_archive_handle = None
        self._dna_datasets: Mapping[str, Any] | Dict[str, Any] = {}
        self.dna_seq_len = 0
        self.dna_embed_dim = 0
        self.D_eff = 0
        self.dna_dim = 0
        self._dna_idx: Optional[np.ndarray] = None

        if self.use_dna:
            if "dna_embedding_idx" not in self.meta.columns:
                raise KeyError("Metadata frame missing 'dna_embedding_idx' column")
            dna_idx_series = self.meta["dna_embedding_idx"]
            if dna_idx_series.isna().any():
                raise ValueError("dna_embedding_idx column contains missing values")
            self._dna_idx = dna_idx_series.to_numpy(dtype=np.int64)

            if dna_h5 is None:
                raise ValueError("DNA modality enabled but dna_h5 path is None")
            (
                self._dna_archive_handle,
                self._dna_datasets,
                self.dna_seq_len,
                self.dna_embed_dim,
            ) = load_dna_archive(dna_h5)
            N_full = int(self._dna_datasets["ref_tokens"].shape[0])
            validate_dna_indices(self._dna_idx, N_full)

        # ---- labels ----
        self.make_label = bool(make_label)
        self.label_col = str(label_col)
        if self.make_label and self.label_col not in self.meta.columns:
            raise KeyError(f"Label column '{self.label_col}' not found in meta.")
        self._labels = None
        if self.make_label:
            label_series = pd.to_numeric(
                self.meta[self.label_col], errors="raise",
            )
            if label_series.isna().any():
                raise ValueError(
                    f"Label column '{self.label_col}' contains missing values"
                )
            self._labels = label_series.astype("int64").to_numpy()

        # ---- protein embeddings ----
        self._prot_archive_handle = None
        self._prot_datasets: Mapping[str, Any] | Dict[str, Any] = {}
        self.prot_seq_len = 0
        self.prot_embed_dim = 0
        self.D_prot = 0
        self.protein_dim = 0
        self._prot_idx: Optional[np.ndarray] = None
        self.protein_coverage = 0.0

        if self.use_protein:
            if "protein_embedding_idx" not in self.meta.columns:
                raise KeyError("Metadata frame missing 'protein_embedding_idx' column")
            protein_idx_series = self.meta["protein_embedding_idx"]
            if protein_idx_series.isna().any():
                raise ValueError("protein_embedding_idx column contains missing values")

            if protein_h5 is None:
                raise ValueError("Protein modality enabled but protein_h5 path is None")
            (
                self._prot_archive_handle,
                self._prot_datasets,
                self.prot_seq_len,
                self.prot_embed_dim,
            ) = load_protein_archive(protein_h5)
            self._prot_idx = protein_idx_series.to_numpy(dtype=np.int64)
            prot_rows = int(self._prot_datasets["wt_tokens"].shape[0])
            self.protein_coverage = validate_protein_indices(self._prot_idx, prot_rows)

        # Build global slices for conditioning vector layout
        self.cond_slices: dict[str, Optional[dict[str, slice] | slice]] = {}
        offset = 0

        dna_slices: Optional[dict[str, slice]] = None
        dna_block_slice = slice(offset, offset)
        if self.use_dna and self.dna_seq_len > 0 and self.dna_embed_dim > 0:
            dna_slices, dna_block_slice, offset = compute_dna_slices(
                self.dna_seq_len, self.dna_embed_dim, offset
            )
            self.D_eff = dna_block_slice.stop - dna_block_slice.start
            self.dna_dim = self.D_eff
        else:
            self.D_eff = 0
            self.dna_dim = 0
        self.cond_slices["dna"] = dna_slices

        prot_slices: Optional[dict[str, slice]] = None
        prot_block_slice = slice(offset, offset)
        if self.use_protein and self.prot_seq_len > 0 and self.prot_embed_dim > 0:
            prot_slices, prot_block_slice, offset = compute_protein_slices(
                self.prot_seq_len, self.prot_embed_dim, offset
            )
            self.D_prot = prot_block_slice.stop - prot_block_slice.start
            self.protein_dim = self.D_prot
        else:
            self.D_prot = 0
            self.protein_dim = 0
        self.cond_slices["protein"] = prot_slices
        self.total_dim = offset

        dna_extra = []
        if dna_slices and "splice_mask" in dna_slices:
            dna_extra.append("splice_mask")
        prot_extra = []
        if prot_slices and "frameshift_mask" in prot_slices:
            prot_extra.append("frameshift_mask")

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
            "total_dim": self.total_dim,
        }

    # -----------------
    # Dataset protocol
    # -----------------
    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_np = np.zeros(self.total_dim, dtype="float32")

        dna_slice = self.cond_slices.get("dna")
        if self.use_dna and dna_slice is not None and self._dna_idx is not None:
            dna_idx = int(self._dna_idx[idx])
            write_dna_features(x_np, dna_slice, dna_idx, self._dna_datasets)

        prot_slice = self.cond_slices.get("protein")
        if (
            self.use_protein
            and prot_slice is not None
            and self.D_prot > 0
            and self._prot_idx is not None
        ):
            prot_idx = int(self._prot_idx[idx])
            write_protein_features(x_np, prot_slice, prot_idx, self._prot_datasets)

        x = torch.from_numpy(x_np)

        y = None
        if self.make_label and self._labels is not None:
            y = torch.tensor(int(self._labels[idx]), dtype=torch.long)

        return x, y

    # -----------------
    # Resource handling
    # -----------------
    def close(self) -> None:
        for attr in ("_dna_archive_handle", "_prot_archive_handle"):
            handle = getattr(self, attr, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    def __del__(self):  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass
