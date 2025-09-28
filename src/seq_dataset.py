from __future__ import annotations

from typing import Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.go.gene_go_embeddings import GeneGOEmbeddings


class SequenceTowerDataset(Dataset):
    """Dataset that concatenates DNA, GO, and protein embeddings."""

    def __init__(
        self,
        *,
        meta_df: Optional[pd.DataFrame] = None,
        meta_feather: Optional[str] = None,
        dna_npz: str,
        go_npz: Optional[str] = None,
        eff_key: str = "dna_eff",
        make_label: bool = True,
        label_col: str = "_y",
        protein_npz: str,
        protein_eff_key: str = "prot_eff",
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
        try:
            dna_archive = np.load(dna_npz, mmap_mode="r")
        except Exception as exc:  # pragma: no cover - file errors are environmental
            raise RuntimeError(f"Failed to open dna_npz '{dna_npz}': {exc}") from exc
        if eff_key not in dna_archive.files:
            raise KeyError(f"'{eff_key}' not present in {dna_npz}. Keys: {list(dna_archive.files)}")
        self.dna_eff = dna_archive[eff_key]
        if self.dna_eff.ndim != 2:
            raise RuntimeError(f"dna_eff expected 2D array, got {self.dna_eff.ndim}D")
        self.D_eff = int(self.dna_eff.shape[1])
        self.dna_dim = self.D_eff
        N_full = int(self.dna_eff.shape[0])
        dna_min = int(self._dna_idx.min(initial=0))
        dna_max = int(self._dna_idx.max(initial=-1))
        if dna_min < 0 or dna_max >= N_full:
            raise RuntimeError(
                "dna_embedding_idx out of range "
                f"(min={dna_min}, max={dna_max}, N={N_full})"
            )

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

        try:
            prot_archive = np.load(protein_npz, mmap_mode="r")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open protein_npz '{protein_npz}': {exc}"
            ) from exc
        if protein_eff_key not in prot_archive.files:
            raise KeyError(
                f"'{protein_eff_key}' not present in {protein_npz}. Keys: {list(prot_archive.files)}"
            )
        self.prot_eff = prot_archive[protein_eff_key]
        if self.prot_eff.ndim != 2:
            raise RuntimeError(
                f"protein eff expected 2D array, got {self.prot_eff.ndim}D"
            )
        self.D_prot = int(self.prot_eff.shape[1])
        self.protein_dim = self.D_prot
        self._zero_prot = np.zeros((self.D_prot,), dtype="float32")
        self._prot_idx = protein_idx_series.to_numpy(dtype=np.int64)
        if (self._prot_idx < -1).any():
            raise RuntimeError(
                "protein_embedding_idx must be -1 or non-negative integers"
            )
        prot_rows = int(self.prot_eff.shape[0])
        if prot_rows == 0 and (self._prot_idx >= 0).any():
            raise RuntimeError(
                "protein_embedding_idx refers to embeddings but prot_eff is empty"
            )
        prot_max = int(self._prot_idx.max(initial=-1))
        if prot_max >= prot_rows:
            raise RuntimeError(
                "protein_embedding_idx out of range "
                f"(max={prot_max}, N={prot_rows})"
            )
        covered = int((self._prot_idx >= 0).sum())
        self.protein_coverage = covered / max(1, len(self.meta))
        self.total_dim = self.D_eff + self.D_go + self.D_prot

    # -----------------
    # Dataset protocol
    # -----------------
    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        dna_idx = int(self._dna_idx[idx])
        dna_vec = np.asarray(self.dna_eff[dna_idx], dtype="float32", order="C")

        go_vec = None
        if self._go_vectors is not None and self.D_go > 0:
            go_vec = np.asarray(self._go_vectors[idx], dtype="float32", order="C")

        prot_idx = int(self._prot_idx[idx])
        if prot_idx >= 0:
            prot_vec = np.asarray(self.prot_eff[prot_idx], dtype="float32", order="C")
        else:
            prot_vec = self._zero_prot

        x_np = np.empty(self.total_dim, dtype="float32")
        offset = self.D_eff
        x_np[:offset] = dna_vec
        if go_vec is not None and self.D_go > 0:
            next_offset = offset + self.D_go
            x_np[offset:next_offset] = go_vec
        else:
            next_offset = offset
        x_np[next_offset:] = prot_vec
        x = torch.from_numpy(x_np)

        y = None
        if self.make_label and self._labels is not None:
            y = torch.tensor(int(self._labels[idx]), dtype=torch.long)

        return x, y
