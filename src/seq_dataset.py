from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.go.gene_go_embeddings import GeneGOEmbeddings


def _fmt_path(path: Optional[str], label: str) -> str:
    if not path:
        return f"{label}: <none>"
    try:
        abspath = os.path.abspath(path)
        exists = os.path.exists(path)
        return f"{label}: {path}  ({'exists' if exists else 'MISSING'})"
    except Exception:
        return f"{label}: {path}"


class SequenceTowerDataset(Dataset):
    """Dataset that concatenates DNA, GO, and optional protein embeddings."""

    def __init__(
        self,
        meta_feather: str,
        dna_npz: str,
        go_npz: Optional[str] = None,
        *,
        eff_key: str = "dna_eff",
        make_label: bool = True,
        label_col: str = "_y",
        protein_meta_feather: Optional[str] = None,
        protein_npz: Optional[str] = None,
        protein_eff_key: str = "prot_eff",
        go_normalize: bool = True,
        go_uppercase: bool = True,
    ) -> None:
        print("\n[SequenceTowerDataset] constructing…")
        print(_fmt_path(meta_feather, "meta_feather"))
        print(_fmt_path(dna_npz, "dna_npz"))
        if go_npz:
            print(_fmt_path(go_npz, "go_npz"))
        if protein_meta_feather or protein_npz:
            print(_fmt_path(protein_meta_feather, "protein_meta_feather"))
            print(_fmt_path(protein_npz, "protein_npz"))

        # ---- metadata ----
        self.meta = pd.read_feather(meta_feather)
        if "emb_row" not in self.meta.columns:
            raise ValueError("meta feather must contain 'emb_row'")
        self.emb_rows = self.meta["emb_row"].to_numpy(dtype=np.int64)
        n_rows = len(self.meta)
        ex_cols = [
            c for c in ("VariationID", "GeneSymbol", label_col) if c in self.meta.columns
        ]
        print(f"[meta] rows={n_rows}  cols={len(self.meta.columns)}  has={ex_cols}")
        if "VariationID" in self.meta.columns:
            sample = self.meta["VariationID"].astype(str).head(3).tolist()
            if n_rows > 3:
                sample.append("…")
            print(f"[meta] VariationID sample: {sample}")
        if "GeneSymbol" in self.meta.columns:
            sample = self.meta["GeneSymbol"].astype(str).head(3).tolist()
            if n_rows > 3:
                sample.append("…")
            print(f"[meta] GeneSymbol sample: {sample}")

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
        print(
            f"[DNA] key='{eff_key}' shape={tuple(self.dna_eff.shape)} dtype={self.dna_eff.dtype}"
        )

        N_full = int(self.dna_eff.shape[0])
        if (
            int(self.emb_rows.min(initial=0)) < 0
            or int(self.emb_rows.max(initial=-1)) >= N_full
        ):
            raise RuntimeError(
                "emb_row indices out of range "
                f"(min={self.emb_rows.min()}, max={self.emb_rows.max()}, N={N_full})"
            )

        # ---- GO embeddings ----
        self.go = None
        self.D_go = 0
        self.go_dim = 0
        if go_npz:
            try:
                self.go = GeneGOEmbeddings(
                    go_npz, normalize=go_normalize, uppercase_keys=go_uppercase
                )
                self.D_go = int(self.go.dim)
                self.go_dim = self.D_go
                print(f"[GO] dim={self.D_go}  entries={len(self.go.name2idx)}")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load GO embeddings from '{go_npz}': {exc}"
                ) from exc
        else:
            print("[GO] embeddings not provided; will emit zeros")

        genes_series = self.meta.get("GeneSymbol", pd.Series(["" for _ in range(n_rows)]))
        genes = genes_series.astype(str).str.upper().tolist()
        if self.go is not None and self.D_go > 0:
            go_hits = sum(1 for g in genes if self.go.has(g))
            self.go_hit_rate = go_hits / max(1, len(genes))
            print(f"[GO] hit-rate: {self.go_hit_rate:.3f} ({go_hits}/{len(genes)})")
        else:
            self.go_hit_rate = 0.0
            print("[GO] hit-rate: 0.000 (no GO embeddings)")

        # ---- labels ----
        self.make_label = bool(make_label)
        self.label_col = str(label_col)
        if self.make_label and self.label_col not in self.meta.columns:
            raise KeyError(f"Label column '{self.label_col}' not found in meta.")

        # ---- protein embeddings (optional) ----
        self.prot_eff = None
        self._prot_rows = None
        self.D_prot = 0
        self.protein_dim = 0
        self.protein_coverage = 0.0
        if protein_meta_feather and protein_npz:
            try:
                pmeta = pd.read_feather(protein_meta_feather)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to read protein meta '{protein_meta_feather}': {exc}"
                ) from exc
            if "id" not in pmeta.columns or "emb_row" not in pmeta.columns:
                raise ValueError(
                    "protein meta feather must contain 'id' and 'emb_row' columns"
                )
            pmeta = pmeta.copy()
            pmeta["id"] = pmeta["id"].astype(str)

            if "VariationID" in self.meta.columns:
                ids = self.meta["VariationID"].astype(str)
                id_source = "meta[VariationID]"
            elif "id" in self.meta.columns:
                ids = self.meta["id"].astype(str)
                id_source = "meta[id]"
            else:
                raise ValueError(
                    "metadata must contain either 'VariationID' or 'id' for protein alignment"
                )
            print(f"[PROT] aligning by {id_source} ↔ protein_meta[id]")

            id2prow: Dict[str, int] = dict(
                zip(pmeta["id"].astype(str), pmeta["emb_row"].astype(int))
            )
            prow = np.full(len(self.meta), -1, dtype=np.int64)
            for i, vid in enumerate(ids):
                prow[i] = id2prow.get(vid, -1)
            self._prot_rows = prow

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
            covered = int((self._prot_rows >= 0).sum())
            self.protein_coverage = covered / max(1, len(self.meta))
            cov_pct = 100.0 * self.protein_coverage
            print(
                f"[PROT] coverage: {covered}/{len(self.meta)} ({cov_pct:.1f}%)  "
                f"shape={tuple(self.prot_eff.shape)} dtype={self.prot_eff.dtype}"
            )
        else:
            print("[PROT] embeddings not provided; will emit zeros")

        total_dim = self.D_eff + self.D_go + self.D_prot
        print(
            f"[SequenceTowerDataset] feature layout: "
            f"dna={self.D_eff}  go={self.D_go}  prot={self.D_prot}  total={total_dim}\n"
        )

    # -----------------
    # Dataset protocol
    # -----------------
    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, object]]:
        row = self.meta.iloc[idx]
        emb_row = int(self.emb_rows[idx])
        gene = str(row.get("GeneSymbol", ""))

        dna_vec = np.asarray(self.dna_eff[emb_row], dtype="float32", order="C")
        parts = [dna_vec]

        if self.D_go > 0 and self.go is not None:
            go_vec = self.go.get(gene)
            go_np = np.asarray(go_vec, dtype="float32", order="C")
        else:
            go_np = np.zeros((self.D_go,), dtype="float32")
        parts.append(go_np)

        prot_np = None
        if self.prot_eff is not None and self._prot_rows is not None:
            prow = int(self._prot_rows[idx])
            if prow >= 0:
                prot_np = np.asarray(self.prot_eff[prow], dtype="float32", order="C")
            else:
                prot_np = np.zeros((self.D_prot,), dtype="float32")
        if prot_np is None:
            prot_np = np.zeros((self.D_prot,), dtype="float32")
        parts.append(prot_np)

        x_np = np.concatenate(parts, axis=0).astype("float32", copy=False)
        x = torch.from_numpy(x_np)

        y = None
        if self.make_label:
            try:
                y = torch.tensor(int(row[self.label_col]), dtype=torch.long)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to read label '{self.label_col}' at idx={idx}: {exc}"
                ) from exc

        varid = -1
        if "VariationID" in row.index:
            try:
                varid = int(row["VariationID"])
            except Exception:
                varid = -1

        aux = {"gene": gene, "varid": varid, "row": emb_row}
        return x, y, aux
