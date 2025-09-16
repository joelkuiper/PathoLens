# src/seq_dataset.py
from __future__ import annotations
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.go_n2v_utils import GeneGOEmbeddings


class SequenceTowerDataset(Dataset):
    def __init__(
        self,
        meta_feather: str,
        dna_npz: str,
        go_npz: str,
        *,
        eff_key: str = "dna_eff",
        make_label: bool = True,
        label_col: str = "_y",
        # --- NEW: protein inputs (optional) ---
        protein_meta_feather: Optional[str] = None,
        protein_npz: Optional[str] = None,
        protein_eff_key: str = "prot_eff",
    ):
        # ---- main meta (DNA-aligned) ----
        self.meta = pd.read_feather(meta_feather)
        if "emb_row" not in self.meta.columns:
            raise ValueError("meta feather must contain 'emb_row'")
        self.emb_rows = self.meta["emb_row"].to_numpy(dtype=np.int64)

        # ---- DNA arrays (memmap) ----
        self._npz = np.load(dna_npz, mmap_mode="r")
        if eff_key not in self._npz.files:
            raise KeyError(f"'{eff_key}' not in {dna_npz}. Keys: {self._npz.files}")
        self.dna_eff = self._npz[eff_key]  # (N_full, D_eff)
        N_full = self.dna_eff.shape[0]
        if int(self.emb_rows.max()) >= N_full or int(self.emb_rows.min()) < 0:
            raise RuntimeError(f"emb_row indices out of range [0,{N_full})")

        # ---- GO embeddings ----
        self.go = GeneGOEmbeddings(go_npz, normalize=True, uppercase_keys=True)

        self.D_eff = int(self.dna_eff.shape[1])
        self.D_go = int(self.go.dim)

        # ---- labels ----
        self.make_label = make_label
        self.label_col = label_col

        # quick GO hit-rate
        genes = self.meta["GeneSymbol"].astype(str).str.upper().tolist()
        self.go_hits = sum(g in self.go.name2idx for g in genes) / max(1, len(genes))
        print(
            f"[GO] hit-rate: {self.go_hits:.3f} "
            f"(found {int(self.go_hits * len(genes))}/{len(genes)})"
        )

        # ---- OPTIONAL: protein embeddings ----
        self.prot_eff = None
        self.D_prot = 0
        self._prot_rows = None  # per-sample index into protein array, -1 if missing

        if protein_meta_feather is not None and protein_npz is not None:
            pmeta = pd.read_feather(protein_meta_feather)
            # normalize id types
            if "id" not in pmeta.columns:
                raise ValueError(
                    "protein meta feather must contain 'id' (VariationID as str)"
                )
            pmeta = pmeta.copy()
            pmeta["id"] = pmeta["id"].astype(str)
            if "VariationID" in self.meta.columns:
                ids = self.meta["VariationID"].astype(str)
            else:
                # fallback: if your meta also uses 'id'
                ids = self.meta["id"].astype(str)

            # map id -> protein emb_row
            if "emb_row" not in pmeta.columns:
                raise ValueError("protein meta feather must contain 'emb_row'")
            id2prow = dict(zip(pmeta["id"], pmeta["emb_row"]))
            prow = np.full(len(self.meta), -1, dtype=np.int64)
            for i, vid in enumerate(ids):
                r = id2prow.get(vid, -1)
                prow[i] = r
            self._prot_rows = prow

            # load protein arrays (memmap)
            pz = np.load(protein_npz, mmap_mode="r")
            if protein_eff_key not in pz.files:
                raise KeyError(
                    f"'{protein_eff_key}' not in {protein_npz}. Keys: {pz.files}"
                )
            self.prot_eff = pz[protein_eff_key]
            self.D_prot = int(self.prot_eff.shape[1])

            # coverage stat
            cov = int((self._prot_rows >= 0).sum())
            print(
                f"[PROT] coverage: {cov}/{len(self.meta)} "
                f"({100.0 * cov / len(self.meta):.1f}%)  D_prot={self.D_prot}"
            )

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        row = self.meta.iloc[idx]
        r = int(self.emb_rows[idx])
        gene = str(row.get("GeneSymbol", ""))

        eff_np = np.asarray(self.dna_eff[r], dtype="float32", order="C")  # (D_eff,)
        go_np = self.go.get(gene)  # (D_go,)

        parts = [eff_np, go_np]

        # protein eff
        if self.prot_eff is not None and self._prot_rows is not None:
            pr = int(self._prot_rows[idx])
            if pr >= 0:
                prot_np = np.asarray(self.prot_eff[pr], dtype="float32", order="C")
            else:
                prot_np = np.zeros((self.D_prot,), dtype="float32")
            parts.append(prot_np)

        x_np = np.concatenate(parts, axis=0).astype("float32", copy=False)
        x = torch.from_numpy(x_np)

        y = None
        if self.make_label:
            y = torch.tensor(int(row[self.label_col]), dtype=torch.long)

        aux = {
            "gene": gene,
            "varid": int(row.get("VariationID", -1)) if "VariationID" in row else -1,
            "row": r,
        }
        return x, y, aux


def collate_seq(batch):
    xs, ys, aux = zip(*batch)
    X = torch.stack(xs, dim=0)
    y = None if ys[0] is None else torch.stack(ys, dim=0)
    return X, y, list(aux)
