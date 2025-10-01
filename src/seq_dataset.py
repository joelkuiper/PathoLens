from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.go.gene_go_embeddings import GeneGOEmbeddings
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
    """Dataset that concatenates DNA, GO, and protein embeddings."""

    def __init__(
        self,
        *,
        meta_df: Optional[pd.DataFrame] = None,
        meta_feather: Optional[str] = None,
        dna_h5: str,
        go_npz: Optional[str] = None,
        make_label: bool = True,
        label_col: str = "_y",
        protein_h5: str,
        go_normalize: bool = True,
        go_uppercase: bool = True,
        schema: Optional[Mapping[str, str]] = None,
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
        self._dna_idx = dna_idx_series.to_numpy(dtype=np.int64)

        # Pre-compute uppercase gene names once for GO lookups
        gene_series = self.meta.get(
            "GeneSymbol",
            pd.Series(index=self.meta.index, data="", dtype="string"),
        )
        gene_series = gene_series.astype("string").fillna("")
        self._gene_values = gene_series.tolist()
        self._gene_upper = gene_series.str.upper().tolist()
        self._gene_lookup = list(self._gene_upper)

        # ---- DNA embeddings ----
        (
            self._dna_archive_handle,
            self._dna_datasets,
            self.dna_seq_len,
            self.dna_embed_dim,
        ) = load_dna_archive(dna_h5)
        N_full = int(self._dna_datasets["ref_tokens"].shape[0])
        self.D_eff = 0
        self.dna_dim = 0
        validate_dna_indices(self._dna_idx, N_full)

        # ---- GO embeddings ----
        self.go = None
        self.D_go = 0
        self.go_dim = 0
        self._go_vectors: Optional[np.ndarray] = None
        self.go_hit_rate = 0.0
        if go_npz:
            try:
                go_embed = GeneGOEmbeddings(
                    go_npz, normalize=go_normalize, uppercase_keys=go_uppercase
                )
            except Exception as exc:
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
                self._go_vectors = np.asarray(
                    go_embed.batch(self._gene_lookup), dtype="float32"
                )
                go_hits = sum(1 for g in self._gene_lookup if go_embed.has(g))
                total_genes = len(self._gene_lookup)
                self.go_hit_rate = go_hits / max(1, total_genes)

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
        protein_idx_series = self.meta["protein_embedding_idx"]
        if protein_idx_series.isna().any():
            raise ValueError("protein_embedding_idx column contains missing values")

        (
            self._prot_archive_handle,
            self._prot_datasets,
            self.prot_seq_len,
            self.prot_embed_dim,
        ) = load_protein_archive(protein_h5)
        self.D_prot = 0
        self.protein_dim = 0
        self._prot_idx = protein_idx_series.to_numpy(dtype=np.int64)
        prot_rows = int(self._prot_datasets["wt_tokens"].shape[0])
        self.protein_coverage = validate_protein_indices(self._prot_idx, prot_rows)

        # Build global slices for conditioning vector layout
        self.cond_slices: dict[str, Optional[dict[str, slice] | slice]] = {}
        offset = 0

        dna_slices, dna_block_slice, offset = compute_dna_slices(
            self.dna_seq_len, self.dna_embed_dim, offset
        )
        self.cond_slices["dna"] = dna_slices
        self.D_eff = dna_block_slice.stop - dna_block_slice.start
        self.dna_dim = self.D_eff

        go_slice = slice(offset, offset + self.D_go)
        self.cond_slices["go"] = go_slice
        offset = go_slice.stop

        prot_slices, prot_block_slice, offset = compute_protein_slices(
            self.prot_seq_len, self.prot_embed_dim, offset
        )
        self.cond_slices["protein"] = prot_slices
        self.D_prot = prot_block_slice.stop - prot_block_slice.start
        self.protein_dim = self.D_prot
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
            "go": {"slice": go_slice if self.D_go > 0 else None, "dim": self.D_go},
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
        if dna_slice is not None:
            dna_idx = int(self._dna_idx[idx])
            write_dna_features(x_np, dna_slice, dna_idx, self._dna_datasets)

        go_slice = self.cond_slices.get("go")
        if go_slice is not None and self.D_go > 0:
            if self._go_vectors is not None:
                x_np[go_slice] = self._go_vectors[idx].astype("float32")

        prot_slice = self.cond_slices.get("protein")
        if prot_slice is not None and self.D_prot > 0:
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
