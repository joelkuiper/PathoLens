from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class SplitArtifact:
    meta: Optional[str] = None
    dna_h5: Optional[str] = None
    protein_h5: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.meta:
            data["meta"] = self.meta
        if self.dna_h5:
            data["dna_h5"] = self.dna_h5
        if self.protein_h5:
            data["protein_h5"] = self.protein_h5
        if self.extras:
            data["extras"] = self.extras
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SplitArtifact":
        extras = data.get("extras", {})
        if extras is None or not isinstance(extras, dict):
            extras = {}
        meta_path = data.get("meta")
        dna_path = data.get("dna_h5")
        protein_path = data.get("protein_h5")
        return cls(
            meta=str(meta_path) if meta_path else None,
            dna_h5=str(dna_path) if dna_path else None,
            protein_h5=str(protein_path) if protein_path else None,
            extras=dict(extras),
        )


@dataclass
class PipelineManifest:
    splits: Dict[str, SplitArtifact]
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().replace(microsecond=0).isoformat()
    )
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "created_at": self.created_at,
            "splits": {k: v.to_dict() for k, v in self.splits.items()},
            "extras": self.extras,
        }
        return data

    def dump(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        with p.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        print(f"[manifest] wrote {p}")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "PipelineManifest":
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        splits_dict = {
            name: SplitArtifact.from_dict(block)
            for name, block in data.get("splits", {}).items()
        }
        extras = data.get("extras", {})
        if extras is None or not isinstance(extras, dict):
            extras = {}
        created_at = data.get("created_at") or datetime.utcnow().isoformat()
        return cls(
            splits=splits_dict,
            created_at=created_at,
            extras=extras,
        )
