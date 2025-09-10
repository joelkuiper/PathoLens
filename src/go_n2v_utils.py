import numpy as np
import torch
from typing import Dict, Iterable, Optional


class GeneGOEmbeddings:
    """
    Loader for GO Node2Vec gene embeddings saved as:
      np.savez_compressed("..._genes.npz", emb=(N,D) float32, genes=(N,) object[str])
    """

    def __init__(
        self, npz_path: str, normalize: bool = True, uppercase_keys: bool = True
    ):
        z = np.load(npz_path, allow_pickle=True)
        self.emb: np.ndarray = z["emb"].astype("float32")  # (N, D)
        self.genes: np.ndarray = z["genes"].astype(object)  # (N,)
        self.dim: int = self.emb.shape[1]
        # Build lookup map
        keys = [str(g) for g in self.genes.tolist()]
        if uppercase_keys:
            keys = [k.upper() for k in keys]
        self.name2idx: Dict[str, int] = {k: i for i, k in enumerate(keys)}
        self.uppercase = uppercase_keys

        if normalize:
            # L2-normalize rows (optional but often helpful before CLIP concat)
            norms = np.linalg.norm(self.emb, axis=1, keepdims=True) + 1e-8
            self.emb = self.emb / norms

    def has(self, gene: str) -> bool:
        key = gene.upper() if self.uppercase else gene
        return key in self.name2idx

    def get(self, gene: str, default: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return (D,) vector for 'gene'. If not found, returns 'default' or zeros.
        """
        key = gene.upper() if self.uppercase else gene
        if key in self.name2idx:
            return self.emb[self.name2idx[key]]
        if default is not None:
            return default
        return np.zeros((self.dim,), dtype="float32")

    def batch(self, genes: Iterable[str], missing_as_zeros: bool = True) -> np.ndarray:
        """
        Return (B,D) array for a list of genes, with zeros for missing (if requested).
        """
        out = np.zeros((len(list(genes)), self.dim), dtype="float32")
        for i, g in enumerate(genes):
            if self.has(g):
                out[i] = self.get(g)
            elif not missing_as_zeros:
                raise KeyError(f"Gene not found: {g}")
        return out

    def to_torch(self, arr: np.ndarray, device: Optional[str] = None) -> torch.Tensor:
        """
        Convert numpy (B,D) or (D,) to torch on desired device (defaults to CUDA if available).
        """
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        t = torch.from_numpy(arr).to(device)
        return t
