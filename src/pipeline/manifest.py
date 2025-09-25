from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class SplitArtifact:
    dna_meta: Optional[str] = None
    dna_npz: Optional[str] = None
    protein_meta: Optional[str] = None
    protein_npz: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.dna_meta or self.dna_npz:
            data["dna"] = {"meta": self.dna_meta, "npz": self.dna_npz}
        if self.protein_meta or self.protein_npz:
            data["protein"] = {
                "meta": self.protein_meta,
                "npz": self.protein_npz,
            }
        if self.extras:
            data["extras"] = self.extras
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SplitArtifact":
        dna = data.get("dna", {})
        prot = data.get("protein", {})
        extras = data.get("extras", {})
        return cls(
            dna_meta=dna.get("meta"),
            dna_npz=dna.get("npz"),
            protein_meta=prot.get("meta"),
            protein_npz=prot.get("npz"),
            extras=dict(extras) if isinstance(extras, dict) else {},
        )


@dataclass
class PipelineManifest:
    go_npz: Optional[str]
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
        if self.go_npz:
            data["go"] = {"npz": self.go_npz}
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
        go_block = data.get("go", {}) or {}
        go_npz = go_block.get("npz")
        splits_dict = {
            name: SplitArtifact.from_dict(block)
            for name, block in data.get("splits", {}).items()
        }
        extras = data.get("extras", {})
        if not go_npz and isinstance(extras, dict):
            go_npz = (
                extras.get("legacy", {})
                .get("go", {})
                .get("npz")
            )
        created_at = data.get("created_at") or datetime.utcnow().isoformat()
        if not go_npz:
            raise ValueError(f"Manifest {p} missing GO embedding reference")
        return cls(
            go_npz=str(go_npz),
            splits=splits_dict,
            created_at=created_at,
            extras=extras,
        )
