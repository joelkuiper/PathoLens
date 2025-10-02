from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Tuple, Optional

import h5py

try:
    from h5py import h5z
except Exception:  # pragma: no cover - optional dependency branch
    h5z = None  # type: ignore[assignment]

_H5_EXTS = {".h5", ".hdf5"}


def ensure_h5_path(path: str | Path, *, kind: str) -> Path:
    """Validate ``path`` has an HDF5 extension and return it as :class:`Path`."""

    p = Path(path)
    if p.suffix.lower() not in _H5_EXTS:
        raise ValueError(
            f"{kind} embeddings must be stored as an HDF5 file (.h5/.hdf5); got '{p}'."
        )
    return p


@dataclass(frozen=True)
class SequenceArchiveLayout:
    wt_tokens: str
    mt_tokens: str
    wt_mask: str
    mt_mask: str
    edit_mask: str
    gap_mask: str
    pos: str
    extra_masks: Tuple[str, ...] = ()
    pos_channels: int = 1


def _normalise_compression(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    norm = str(name).strip().lower()
    if norm in {"", "none", "false", "0"}:
        return None
    if norm in {"gzip", "gz", "deflate"}:
        return "gzip"
    return norm


def _compression_supported(name: Optional[str]) -> bool:
    if name is None:
        return True
    if h5z is None:
        return name is None
    if name == "gzip":
        return bool(h5z.filter_avail(h5z.FILTER_DEFLATE))
    if name == "lzf":
        return bool(h5z.filter_avail(h5z.FILTER_LZF))
    # For unknown filters, optimistically assume availability and let HDF5 error if not.
    return True


def initialise_sequence_archive(
    out_path: str | Path,
    *,
    layout: SequenceArchiveLayout,
    n_rows: int,
    seq_len: int,
    embed_dim: int,
    chunk_rows: int,
    kind: str,
    compression: Optional[str] = "gzip",
    compression_opts: Optional[int] = 4,
    shuffle: bool = True,
) -> Path:
    """Create (or truncate) an archive with the requested datasets and metadata."""

    path = ensure_h5_path(out_path, kind=kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    chunk = max(1, min(int(chunk_rows), int(n_rows))) if n_rows else 1

    requested_compression = _normalise_compression(compression)
    effective_compression = (
        requested_compression if _compression_supported(requested_compression) else None
    )
    compression_kwargs: Dict[str, object] = {}
    if effective_compression is not None:
        compression_kwargs["compression"] = effective_compression
        if effective_compression == "gzip":
            level = 4 if compression_opts is None else max(0, int(compression_opts))
            compression_kwargs["compression_opts"] = level
    elif requested_compression is not None and requested_compression != effective_compression:
        print(
            f"[sequence_cache] compression '{requested_compression}' unavailable -> falling back to uncompressed"
        )

    with h5py.File(path, "w") as h5:

        def create_dataset(name: str, shape: Tuple[int, ...], dtype: str, *, fillvalue):
            ds = h5.create_dataset(
                name,
                shape=shape,
                dtype=dtype,
                shuffle=bool(shuffle and compression_kwargs.get("compression")),
                chunks=(chunk,) + shape[1:],
                fillvalue=fillvalue,
                **compression_kwargs,
            )
            return ds

        create_dataset(
            layout.wt_tokens,
            (n_rows, seq_len, embed_dim),
            "float16",
            fillvalue=0.0,
        )
        create_dataset(
            layout.mt_tokens,
            (n_rows, seq_len, embed_dim),
            "float16",
            fillvalue=0.0,
        )
        create_dataset(layout.wt_mask, (n_rows, seq_len), "uint8", fillvalue=0)
        create_dataset(layout.mt_mask, (n_rows, seq_len), "uint8", fillvalue=0)
        create_dataset(layout.edit_mask, (n_rows, seq_len), "uint8", fillvalue=0)
        create_dataset(layout.gap_mask, (n_rows, seq_len), "uint8", fillvalue=0)
        create_dataset(
            layout.pos,
            (n_rows, seq_len, layout.pos_channels),
            "float16",
            fillvalue=0.0,
        )
        for name in layout.extra_masks:
            create_dataset(name, (n_rows, seq_len), "uint8", fillvalue=0)

        h5.attrs["rows"] = int(n_rows)
        h5.attrs["seq_len"] = int(seq_len)
        h5.attrs["embed_dim"] = int(embed_dim)
        h5.attrs["complete"] = 0
        comp_attr = compression_kwargs.get("compression")
        if comp_attr == "gzip":
            level = compression_kwargs.get("compression_opts")
            h5.attrs["compression"] = f"gzip:{level}" if level is not None else "gzip"
        elif comp_attr:
            h5.attrs["compression"] = str(comp_attr)
        else:
            h5.attrs["compression"] = "none"
        h5.flush()

    return path


@dataclass
class SequenceArchiveHandles:
    store: h5py.File
    wt_tokens: h5py.Dataset
    mt_tokens: h5py.Dataset
    wt_mask: h5py.Dataset
    mt_mask: h5py.Dataset
    edit_mask: h5py.Dataset
    gap_mask: h5py.Dataset
    pos: h5py.Dataset
    extra_masks: Dict[str, h5py.Dataset]

    def flush(self) -> None:
        self.store.flush()


@contextmanager
def open_sequence_archive(
    out_path: str | Path,
    *,
    layout: SequenceArchiveLayout,
    kind: str,
    mode: str = "r+",
) -> Iterator[SequenceArchiveHandles]:
    """Open an existing archive for updates."""

    path = ensure_h5_path(out_path, kind=kind)
    store = h5py.File(path, mode)
    try:
        extra = {name: store[name] for name in layout.extra_masks if name in store}
        yield SequenceArchiveHandles(
            store=store,
            wt_tokens=store[layout.wt_tokens],
            mt_tokens=store[layout.mt_tokens],
            wt_mask=store[layout.wt_mask],
            mt_mask=store[layout.mt_mask],
            edit_mask=store[layout.edit_mask],
            gap_mask=store[layout.gap_mask],
            pos=store[layout.pos],
            extra_masks=extra,
        )
    finally:
        store.close()
