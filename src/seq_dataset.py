from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from pathlib import Path

import h5py
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
        self._dna_archive_handle = None
        dna_path = Path(dna_h5)
        if dna_path.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError(
                f"DNA archive must be an HDF5 file (.h5/.hdf5); got '{dna_path}'"
            )
        try:
            dna_archive = h5py.File(dna_path, "r")
        except Exception as exc:  # pragma: no cover - file errors are environmental
            raise RuntimeError(
                f"Failed to open DNA archive '{dna_h5}': {exc}"
            ) from exc
        self._dna_archive_handle = dna_archive
        dna_required = {
            "dna_ref_tokens",
            "dna_alt_tokens",
            "dna_ref_mask",
            "dna_alt_mask",
            "dna_edit_mask",
            "dna_gap_mask",
            "dna_pos",
            "dna_splice_mask",
        }
        dna_keys = set(dna_archive.keys())
        missing = dna_required - dna_keys
        if missing:
            raise KeyError(
                f"DNA archive missing keys: {sorted(missing)} | found={sorted(dna_keys)}"
            )
        seq_len_attr = dna_archive.attrs.get("seq_len")
        if seq_len_attr is None:
            raise KeyError("DNA archive missing 'seq_len' attribute")
        dna_seq_len_meta = int(seq_len_attr)
        self._dna_ref_tokens = dna_archive["dna_ref_tokens"]
        self._dna_alt_tokens = dna_archive["dna_alt_tokens"]
        self._dna_ref_mask = dna_archive["dna_ref_mask"]
        self._dna_alt_mask = dna_archive["dna_alt_mask"]
        self._dna_edit_mask = dna_archive["dna_edit_mask"]
        self._dna_gap_mask = dna_archive["dna_gap_mask"]
        self._dna_pos = dna_archive["dna_pos"]
        self._dna_splice_mask = dna_archive["dna_splice_mask"]

        self.dna_seq_len = int(self._dna_ref_tokens.shape[1])
        if self.dna_seq_len != dna_seq_len_meta:
            raise ValueError(
                "DNA archive seq_len mismatch: "
                f"stored={dna_seq_len_meta} actual={self.dna_seq_len}"
            )
        self.dna_embed_dim = int(self._dna_ref_tokens.shape[2])
        N_full = int(self._dna_ref_tokens.shape[0])
        self.D_eff = 0
        self.dna_dim = 0
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

        self._prot_archive_handle = None
        prot_path = Path(protein_h5)
        if prot_path.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError(
                f"Protein archive must be an HDF5 file (.h5/.hdf5); got '{prot_path}'"
            )
        try:
            prot_archive = h5py.File(protein_h5, "r")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open protein archive '{protein_h5}': {exc}"
            ) from exc
        self._prot_archive_handle = prot_archive
        prot_required = {
            "prot_wt_tokens",
            "prot_mt_tokens",
            "prot_wt_mask",
            "prot_mt_mask",
            "prot_edit_mask",
            "prot_gap_mask",
            "prot_pos",
            "prot_frameshift_mask",
        }
        prot_keys = set(prot_archive.keys())
        missing_prot = prot_required - prot_keys
        if missing_prot:
            raise KeyError(
                f"Protein archive missing keys: {sorted(missing_prot)} | found={sorted(prot_keys)}"
            )
        prot_seq_attr = prot_archive.attrs.get("seq_len")
        if prot_seq_attr is None:
            raise KeyError("Protein archive missing 'seq_len' attribute")
        prot_seq_len_meta = int(prot_seq_attr)
        self._prot_wt_tokens = prot_archive["prot_wt_tokens"]
        self._prot_mt_tokens = prot_archive["prot_mt_tokens"]
        self._prot_wt_mask = prot_archive["prot_wt_mask"]
        self._prot_mt_mask = prot_archive["prot_mt_mask"]
        self._prot_edit_mask = prot_archive["prot_edit_mask"]
        self._prot_gap_mask = prot_archive["prot_gap_mask"]
        self._prot_pos = prot_archive["prot_pos"]
        self._prot_frameshift_mask = prot_archive["prot_frameshift_mask"]

        self.prot_seq_len = int(self._prot_wt_tokens.shape[1])
        if self.prot_seq_len != prot_seq_len_meta:
            raise ValueError(
                "Protein archive seq_len mismatch: "
                f"stored={prot_seq_len_meta} actual={self.prot_seq_len}"
            )
        self.prot_embed_dim = int(self._prot_wt_tokens.shape[2])
        self.D_prot = 0
        self.protein_dim = 0
        self._prot_idx = protein_idx_series.to_numpy(dtype=np.int64)
        if (self._prot_idx < -1).any():
            raise RuntimeError(
                "protein_embedding_idx must be -1 or non-negative integers"
            )
        prot_rows = int(self._prot_wt_tokens.shape[0])
        if prot_rows == 0 and (self._prot_idx >= 0).any():
            raise RuntimeError(
                "protein_embedding_idx refers to embeddings but protein tokens array is empty"
            )
        prot_max = int(self._prot_idx.max(initial=-1))
        if prot_max >= prot_rows:
            raise RuntimeError(
                "protein_embedding_idx out of range "
                f"(max={prot_max}, N={prot_rows})"
            )
        covered = int((self._prot_idx >= 0).sum())
        self.protein_coverage = covered / max(1, len(self.meta))

        # Build global slices for conditioning vector layout
        self.cond_slices: dict[str, Optional[dict[str, slice] | slice]] = {}
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
                dna_slices[key] = slice(offset, offset + L)
                offset += L
        dna_block_stop = offset
        self.cond_slices["dna"] = dna_slices
        self.D_eff = dna_block_stop - dna_block_start
        self.dna_dim = self.D_eff

        go_slice = slice(offset, offset + self.D_go)
        self.cond_slices["go"] = go_slice
        offset = go_slice.stop

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
            x_np[dna_slice["ref_tokens"]] = self._dna_ref_tokens[dna_idx].astype(
                "float32"
            ).reshape(-1)
            x_np[dna_slice["alt_tokens"]] = self._dna_alt_tokens[dna_idx].astype(
                "float32"
            ).reshape(-1)
            x_np[dna_slice["ref_mask"]] = self._dna_ref_mask[dna_idx].astype(
                "float32"
            )
            x_np[dna_slice["alt_mask"]] = self._dna_alt_mask[dna_idx].astype(
                "float32"
            )
            x_np[dna_slice["edit_mask"]] = self._dna_edit_mask[dna_idx].astype(
                "float32"
            )
            x_np[dna_slice["gap_mask"]] = self._dna_gap_mask[dna_idx].astype(
                "float32"
            )
            x_np[dna_slice["splice_mask"]] = self._dna_splice_mask[dna_idx].astype(
                "float32"
            )
            x_np[dna_slice["pos"]] = self._dna_pos[dna_idx].astype("float32").reshape(-1)

        go_slice = self.cond_slices.get("go")
        if go_slice is not None and self.D_go > 0:
            if self._go_vectors is not None:
                x_np[go_slice] = self._go_vectors[idx].astype("float32")

        prot_slice = self.cond_slices.get("protein")
        if prot_slice is not None and self.D_prot > 0:
            prot_idx = int(self._prot_idx[idx])
            if prot_idx >= 0:
                x_np[prot_slice["wt_tokens"]] = self._prot_wt_tokens[prot_idx].astype(
                    "float32"
                ).reshape(-1)
                x_np[prot_slice["mt_tokens"]] = self._prot_mt_tokens[prot_idx].astype(
                    "float32"
                ).reshape(-1)
                x_np[prot_slice["wt_mask"]] = self._prot_wt_mask[prot_idx].astype(
                    "float32"
                )
                x_np[prot_slice["mt_mask"]] = self._prot_mt_mask[prot_idx].astype(
                    "float32"
                )
                x_np[prot_slice["edit_mask"]] = self._prot_edit_mask[prot_idx].astype(
                    "float32"
                )
                x_np[prot_slice["gap_mask"]] = self._prot_gap_mask[prot_idx].astype(
                    "float32"
                )
                x_np[prot_slice["frameshift_mask"]] = self._prot_frameshift_mask[
                    prot_idx
                ].astype("float32")
                x_np[prot_slice["pos"]] = self._prot_pos[prot_idx].astype("float32").reshape(-1)

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
