from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.dna_utils import load_dna_archive, read_dna_features, validate_dna_indices
from src.protein_utils import (
    load_protein_archive,
    read_protein_features,
    validate_protein_indices,
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
            raise ValueError(
                "SequenceTowerDataset requires at least one modality enabled"
            )

        # ---- DNA embeddings ----
        self._dna_archive_handle = None
        self._dna_datasets: Mapping[str, Any] | Dict[str, Any] = {}
        self.dna_seq_len = 0
        self.dna_embed_dim = 0
        self._dna_idx: Optional[np.ndarray] = None
        self.dna_extra_masks: list[str] = []

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
            self.dna_extra_masks = self._detect_extra_masks(
                self._dna_datasets,
                base_keys={
                    "ref_tokens",
                    "alt_tokens",
                    "ref_mask",
                    "alt_mask",
                    "edit_mask",
                    "gap_mask",
                    "pos",
                },
            )

        # ---- labels ----
        self.make_label = bool(make_label)
        self.label_col = str(label_col)
        if self.make_label and self.label_col not in self.meta.columns:
            raise KeyError(f"Label column '{self.label_col}' not found in meta.")
        self._labels = None
        if self.make_label:
            label_series = pd.to_numeric(
                self.meta[self.label_col],
                errors="raise",
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
        self._prot_idx: Optional[np.ndarray] = None
        self.protein_coverage = 0.0
        self.protein_extra_masks: list[str] = []

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
            self.protein_extra_masks = self._detect_extra_masks(
                self._prot_datasets,
                base_keys={
                    "wt_tokens",
                    "mt_tokens",
                    "wt_mask",
                    "mt_mask",
                    "edit_mask",
                    "gap_mask",
                    "pos",
                },
            )

        self.cond_spec: dict[str, Any] = {
            "version": 2,
            "dna": self._make_modality_spec(
                enabled=self.use_dna,
                seq_len=self.dna_seq_len,
                embed_dim=self.dna_embed_dim,
                datasets=self._dna_datasets,
                token_keys=("ref_tokens", "alt_tokens"),
                mask_keys=("ref_mask", "alt_mask"),
                edit_key="edit_mask",
                gap_key="gap_mask",
                pos_key="pos",
                extra_masks=self.dna_extra_masks,
            ),
            "protein": self._make_modality_spec(
                enabled=self.use_protein,
                seq_len=self.prot_seq_len,
                embed_dim=self.prot_embed_dim,
                datasets=self._prot_datasets,
                token_keys=("wt_tokens", "mt_tokens"),
                mask_keys=("wt_mask", "mt_mask"),
                edit_key="edit_mask",
                gap_key="gap_mask",
                pos_key="pos",
                extra_masks=self.protein_extra_masks,
            ),
        }
        enabled_modalities = [
            name for name, info in self.cond_spec.items() if isinstance(info, dict) and info.get("enabled")
        ]
        self.cond_spec["modalities"] = [m for m in enabled_modalities if m in {"dna", "protein"}]

    # -----------------
    # Dataset protocol
    # -----------------
    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        conditioning: Dict[str, Dict[str, torch.Tensor]] = {}

        if self.use_dna and self._dna_idx is not None:
            dna_idx = int(self._dna_idx[idx])
            conditioning["dna"] = read_dna_features(self._dna_datasets, dna_idx)

        if self.use_protein and self._prot_idx is not None:
            prot_idx = int(self._prot_idx[idx])
            conditioning["protein"] = read_protein_features(
                self._prot_datasets, prot_idx
            )

        if not conditioning:
            raise RuntimeError("No conditioning modalities available for this sample")

        y = None
        if self.make_label and self._labels is not None:
            y = torch.tensor(int(self._labels[idx]), dtype=torch.long)

        return conditioning, y

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

    @staticmethod
    def _detect_extra_masks(
        datasets: Mapping[str, Any], *, base_keys: set[str]
    ) -> list[str]:
        extras: list[str] = []
        for name in datasets.keys():
            if name in base_keys:
                continue
            if name.endswith("_mask"):
                extras.append(str(name))
        return sorted(extras)

    @staticmethod
    def _make_modality_spec(
        *,
        enabled: bool,
        seq_len: int,
        embed_dim: int,
        datasets: Mapping[str, Any],
        token_keys: tuple[str, str],
        mask_keys: tuple[str, str],
        edit_key: Optional[str],
        gap_key: Optional[str],
        pos_key: Optional[str],
        extra_masks: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        present = bool(enabled and seq_len > 0 and embed_dim > 0 and datasets)

        def _present(key: Optional[str]) -> Optional[str]:
            if key is None:
                return None
            return key if key in datasets else None

        feature_shapes = {
            str(name): [int(dim) for dim in datasets[name].shape[1:]]
            for name in datasets
        }
        feature_dtypes = {str(name): str(datasets[name].dtype) for name in datasets}

        token_present = all(key in datasets for key in token_keys)
        mask_present = all(key in datasets for key in mask_keys)

        spec: Dict[str, Any] = {
            "enabled": present,
            "seq_len": int(seq_len),
            "embed_dim": int(embed_dim),
            "feature_shapes": feature_shapes,
            "feature_dtypes": feature_dtypes,
            "token_keys": {
                "wt": token_keys[0],
                "mt": token_keys[1],
            }
            if token_present
            else None,
            "mask_keys": {
                "wt": mask_keys[0],
                "mt": mask_keys[1],
            }
            if mask_present
            else None,
            "edit_key": _present(edit_key),
            "gap_key": _present(gap_key),
            "pos_key": _present(pos_key),
        }

        if extra_masks:
            spec["extra_mask_keys"] = [
                key for key in extra_masks if key in datasets
            ]
        else:
            spec["extra_mask_keys"] = []

        return spec
