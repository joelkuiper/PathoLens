"""Typed pipeline configuration loaded from TOML files.

The dataclasses defined here model the pipeline schema and provide runtime
validation. User-facing configuration should happen in the TOML files loaded via
``load_pipeline_config`` (see ``configs/pipeline.example.toml`` for the
authoritative reference); the defaults below exist only as fallbacks when
individual keys are omitted. Avoid editing these Python defaults to change the
pipeline behaviour.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    import tomli as tomllib  # type: ignore[no-redef]


class ConfigError(RuntimeError):
    """Raised when pipeline configuration is invalid."""


def _norm_path(value: Any, base: Optional[Path] = None) -> Path:
    """Return a normalised Path with user/env/relative handling."""

    if isinstance(value, Path):
        p = value
    else:
        if value is None:
            raise ConfigError("Path value cannot be None")
        p = Path(os.path.expandvars(str(value)))
    if base is not None and not p.is_absolute():
        p = Path(base) / p
    p = Path(os.path.expandvars(str(p))).expanduser()
    try:
        return p.resolve()
    except FileNotFoundError:  # allow non-existent outputs
        return p.expanduser().absolute()


@dataclass
class PathsConfig:
    clinvar: Path
    fasta: Path
    go: Path
    artifacts: Path

    def __post_init__(self) -> None:
        for attr in ("clinvar", "fasta", "go", "artifacts"):
            val = getattr(self, attr)
            if not isinstance(val, Path):
                setattr(self, attr, Path(str(val)))


@dataclass
class DNAConfig:
    model_id: str = "InstaDeepAI/nucleotide-transformer-500m-1000g"
    window: int = 512
    max_length: int = 512
    pool: str = "mean"
    batch_size: int = 16
    limit: Optional[int] = None
    force_windows: bool = False
    force_embeddings: bool = False


@dataclass
class ProteinConfig:
    enabled: bool = True
    model_id: str = "facebook/esm2_t12_35M_UR50D"
    batch_size: int = 8
    max_length: int = 2048
    pool: str = "mean"
    vep_cache_dir: Optional[Path] = None
    vep_fasta_relpath: Optional[str] = None
    image: str = "ensemblorg/ensembl-vep"
    filter_mode: str = "protein_changing"
    chunk_size: int = 1000
    jobs: int = 6
    vep_fork: int = 0
    force_embeddings: bool = False
    force_vep: bool = False


@dataclass
class GOConfig:
    npz: Path
    normalize: bool = True
    uppercase_keys: bool = True


@dataclass
class TrainConfig:
    epochs: int = 1
    batch_size: int = 64
    lr: float = 3e-4
    device: Optional[str] = None


@dataclass
class LLMRunConfig:
    enabled: bool = True
    out_dir: Path = Path("artifacts/llm")
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    epochs: int = 1
    max_len: int = 256
    per_device_bs: int = 16
    grad_accum: int = 8
    lr: float = 3e-4
    seed: int = 42
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = False
    pin_memory: bool = True
    group_by_length: bool = False
    drop_last: bool = True
    balanced_sampling: bool = True
    n_cond_tokens: int = 8
    prompt_dropout_prob: float = 0.3


@dataclass
class RunConfig:
    device: Optional[str] = None
    write_manifest: bool = True
    manifest_path: Optional[Path] = None
    force_splits: bool = False
    force_dna_windows: bool = False
    force_dna_embeddings: bool = False
    force_protein_embeddings: bool = False


@dataclass
class PipelineConfig:
    paths: PathsConfig
    dna: DNAConfig = field(default_factory=DNAConfig)
    protein: ProteinConfig = field(default_factory=ProteinConfig)
    go: GOConfig = field(
        default_factory=lambda: GOConfig(Path("data/processed/go_n2v_genes.npz"))
    )
    train: TrainConfig = field(default_factory=TrainConfig)
    llm: LLMRunConfig = field(default_factory=LLMRunConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def __post_init__(self) -> None:
        if self.protein.enabled:
            missing: list[str] = []
            if self.protein.vep_cache_dir is None:
                missing.append("Protein.vep_cache_dir")
            if not self.protein.vep_fasta_relpath:
                missing.append("Protein.vep_fasta_relpath")
            if missing:
                raise ConfigError(
                    "Protein processing enabled but missing required fields: "
                    + ", ".join(missing)
                )
        if self.run.manifest_path is None:
            default_manifest = self.paths.artifacts / "pipeline_manifest.json"
            self.run.manifest_path = default_manifest
        elif not isinstance(self.run.manifest_path, Path):
            self.run.manifest_path = Path(str(self.run.manifest_path))
        if isinstance(self.run.device, str) and self.run.device.lower() == "auto":
            self.run.device = None


def _section(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    return data.get(key, {}) if isinstance(data, dict) else {}


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load and validate the pipeline configuration from TOML."""

    cfg_path = Path(path).expanduser().resolve()
    with cfg_path.open("rb") as f:
        raw = tomllib.load(f)

    base = cfg_path.parent
    paths_raw = _section(raw, "Paths")
    required = {"clinvar", "fasta", "go", "artifacts"}
    missing = sorted(required - set(paths_raw))
    if missing:
        raise ConfigError(f"Paths section missing required fields: {missing}")

    paths = PathsConfig(
        clinvar=_norm_path(paths_raw["clinvar"], base),
        fasta=_norm_path(paths_raw["fasta"], base),
        go=_norm_path(paths_raw["go"], base),
        artifacts=_norm_path(paths_raw["artifacts"], base),
    )

    dna = DNAConfig(**_section(raw, "DNA"))

    protein_raw = _section(raw, "Protein")
    protein = ProteinConfig(
        **{
            **{k: v for k, v in protein_raw.items() if k not in {"vep_cache_dir"}},
            "vep_cache_dir": (
                _norm_path(protein_raw["vep_cache_dir"], base)
                if protein_raw.get("vep_cache_dir") is not None
                else None
            ),
        }
    )
    if protein.enabled and protein.vep_cache_dir is not None:
        # ensure normalised
        protein.vep_cache_dir = _norm_path(protein.vep_cache_dir)

    go_raw = _section(raw, "GO")
    go_npz_val = go_raw.get("npz", paths_raw.get("go"))
    go = GOConfig(
        npz=_norm_path(go_npz_val, base) if go_npz_val is not None else paths.go,
        normalize=go_raw.get("normalize", True),
        uppercase_keys=go_raw.get("uppercase_keys", True),
    )

    train = TrainConfig(**_section(raw, "Train"))

    llm_raw = _section(raw, "LLM")
    llm = LLMRunConfig(
        **{
            **{k: v for k, v in llm_raw.items() if k not in {"out_dir"}},
            "out_dir": (
                _norm_path(llm_raw["out_dir"], base)
                if llm_raw.get("out_dir") is not None
                else paths.artifacts / "llm"
            ),
        }
    )

    run_raw = _section(raw, "Run")
    run_manifest = run_raw.get("manifest") or run_raw.get("manifest_path")
    run = RunConfig(
        **{
            **{
                k: v
                for k, v in run_raw.items()
                if k not in {"manifest", "manifest_path"}
            },
            "manifest_path": (
                _norm_path(run_manifest, base) if run_manifest is not None else None
            ),
        }
    )

    cfg = PipelineConfig(
        paths=paths,
        dna=dna,
        protein=protein,
        go=go,
        train=train,
        llm=llm,
        run=run,
    )
    return cfg
