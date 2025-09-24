# src/seq_dataset.py
from __future__ import annotations
from typing import Optional, Tuple, Dict
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _fmt_path(p: Optional[str], label: str) -> str:
    if not p:
        return f"{label}: <none>"
    try:
        ab = os.path.abspath(p)
        ex = os.path.exists(p)
        return f"{label}: {p}  ({'exists' if ex else 'MISSING'})"
    except Exception:
        return f"{label}: {p}"


class SequenceTowerDataset(Dataset):
    def __init__(
        self,
        meta_feather: str,
        dna_npz: str,
        *,
        eff_key: str = "dna_eff",
        make_label: bool = True,
        label_col: str = "_y",
        # --- protein inputs ---
        protein_meta_feather: str,
        protein_npz: str,
        protein_eff_key: str = "prot_eff",
    ):
        # --------- Report the inputs up front ---------
        print("\n[SequenceTowerDataset] constructing...")
        print(_fmt_path(meta_feather, "meta_feather"))
        print(_fmt_path(dna_npz, "dna_npz"))
        print(_fmt_path(protein_meta_feather, "protein_meta_feather"))
        print(_fmt_path(protein_npz, "protein_npz"))

        # ---- main meta (DNA-aligned) ----
        self.meta = pd.read_feather(meta_feather)
        if "emb_row" not in self.meta.columns:
            raise ValueError("meta feather must contain 'emb_row'")
        self.emb_rows = self.meta["emb_row"].to_numpy(dtype=np.int64)

        # Basic meta debug
        n_rows = len(self.meta)
        ex_cols = [
            c for c in ("VariationID", "GeneSymbol", "_y") if c in self.meta.columns
        ]
        print(f"[meta] rows={n_rows}  cols={len(self.meta.columns)}  has={ex_cols}")
        if "VariationID" in self.meta.columns:
            # show a tiny peek of IDs to verify types/shape
            sample_ids = self.meta["VariationID"].astype(str).head(3).tolist() + (
                ["..."] if n_rows > 3 else []
            )
            print(f"[meta] VariationID sample: {sample_ids}")
        if "GeneSymbol" in self.meta.columns:
            sample_genes = self.meta["GeneSymbol"].astype(str).head(3).tolist() + (
                ["..."] if n_rows > 3 else []
            )
            print(f"[meta] GeneSymbol sample: {sample_genes}")

        # ---- DNA arrays (memmap) ----
        try:
            self._npz = np.load(dna_npz, mmap_mode="r")
        except Exception as e:
            raise RuntimeError(f"Failed to open dna_npz '{dna_npz}': {e}") from e

        if eff_key not in self._npz.files:
            raise KeyError(
                f"'{eff_key}' not in {dna_npz}. Keys: {list(self._npz.files)}"
            )
        self.dna_eff = self._npz[eff_key]  # (N_full, D_eff)

        N_full = int(self.dna_eff.shape[0])
        D_eff = int(self.dna_eff.shape[1]) if self.dna_eff.ndim == 2 else None
        print(
            f"[DNA] key='{eff_key}'  shape={tuple(self.dna_eff.shape)}  dtype={self.dna_eff.dtype}"
        )
        if self.dna_eff.ndim != 2:
            raise RuntimeError(f"dna_eff expected 2D array, got {self.dna_eff.ndim}D")
        # emb_row range check
        if (
            int(self.emb_rows.max(initial=-1)) >= N_full
            or int(self.emb_rows.min(initial=0)) < 0
        ):
            raise RuntimeError(
                f"emb_row indices out of range [0, {N_full}) "
                f"(min={self.emb_rows.min()}, max={self.emb_rows.max()})"
            )

        self.D_eff = D_eff
        self.dna_dim = D_eff

        # ---- labels ----
        self.make_label = make_label
        self.label_col = label_col
        if self.make_label and self.label_col not in self.meta.columns:
            raise KeyError(f"Label column '{self.label_col}' not found in meta.")


        # Load protein meta
        try:
            pmeta = pd.read_feather(protein_meta_feather)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read protein meta '{protein_meta_feather}': {e}"
            ) from e

        if "id" not in pmeta.columns:
            raise ValueError(
                "protein meta feather must contain 'id' (VariationID as str)"
            )
        if "emb_row" not in pmeta.columns:
            raise ValueError("protein meta feather must contain 'emb_row'")

        pmeta = pmeta.copy()
        pmeta["id"] = pmeta["id"].astype(str)

        # Choose DNA ids source
        if "VariationID" in self.meta.columns:
            ids = self.meta["VariationID"].astype(str)
            id_source = "meta[VariationID]"
        else:
            ids = self.meta["id"].astype(str)
            id_source = "meta[id]"
        print(f"[PROT] aligning by {id_source} ↔ protein_meta[id]")

        # Build id -> protein emb_row map
        id2prow = dict(zip(pmeta["id"].astype(str), pmeta["emb_row"].astype(int)))
        prow = np.full(len(self.meta), -1, dtype=np.int64)
        missing: list[str] = []
        for i, vid in enumerate(ids):
            r = id2prow.get(vid, -1)
            prow[i] = r
            if r < 0 and len(missing) < 3:
                missing.append(vid)

        if np.any(prow < 0):
            total_missing = int(np.sum(prow < 0))
            example = ", ".join(missing) if missing else "?"
            raise RuntimeError(
                "Protein embeddings missing for "
                f"{total_missing}/{len(self.meta)} samples. "
                f"Example IDs: {example}"
            )
        self._prot_rows = prow

        # load protein arrays (memmap)
        try:
            pz = np.load(protein_npz, mmap_mode="r")
        except Exception as e:
            raise RuntimeError(
                f"Failed to open protein_npz '{protein_npz}': {e}"
            ) from e

        if protein_eff_key not in pz.files:
            raise KeyError(
                f"'{protein_eff_key}' not in {protein_npz}. Keys: {list(pz.files)}"
            )
        self.prot_eff = pz[protein_eff_key]
        if self.prot_eff.ndim != 2:
            raise RuntimeError(
                f"protein eff expected 2D array, got {self.prot_eff.ndim}D"
            )
        self.D_prot = int(self.prot_eff.shape[1])
        self.protein_dim = self.D_prot

        cov = int((self._prot_rows >= 0).sum())
        self.protein_coverage = cov / max(1, len(self.meta))
        cov_pct = 100.0 * self.protein_coverage
        print(
            f"[PROT] coverage: {cov}/{len(self.meta)} ({cov_pct:.1f}%)  "
            f"shape={tuple(self.prot_eff.shape)} dtype={self.prot_eff.dtype}"
        )

        # Final layout summary
        total_D = self.D_eff + self.D_prot
        self.protein_dim = self.D_prot
        self.dna_dim = self.D_eff
        print(
            f"[SequenceTowerDataset] feature layout: "
            f"dna={self.D_eff}  prot={self.D_prot}  total={total_D}\n"
        )

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        row = self.meta.iloc[idx]
        r = int(self.emb_rows[idx])
        gene = str(row.get("GeneSymbol", ""))

        # DNA eff (always present)
        eff_np = np.asarray(self.dna_eff[r], dtype="float32", order="C")  # (D_eff,)
        parts = [eff_np]

        # Protein eff (always present)
        if self._prot_rows is None:
            raise RuntimeError("Protein row indices not initialized")
        pr = int(self._prot_rows[idx])
        if pr < 0:
            raise RuntimeError(f"Missing protein embedding for idx={idx}")
        prot_np = np.asarray(self.prot_eff[pr], dtype="float32", order="C")
        parts.append(prot_np)

        x_np = np.concatenate(parts, axis=0).astype("float32", copy=False)
        x = torch.from_numpy(x_np)

        y = None
        if self.make_label:
            try:
                y = torch.tensor(int(row[self.label_col]), dtype=torch.long)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read label '{self.label_col}' at idx={idx}: {e}"
                ) from e

        # keep varid numeric if possible, else -1
        varid = -1
        if "VariationID" in row.index:
            try:
                varid = int(row["VariationID"])
            except Exception:
                varid = -1

        aux = {
            "gene": gene,
            "varid": varid,
            "row": r,
        }
        return x, y, aux
