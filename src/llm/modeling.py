# src/llm/modeling.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import COND_START_TOKEN, COND_END_TOKEN
from .sequence_encoder import SequenceCNNEncoder, SequenceInputs


def _ensure_special_tokens(tokenizer: AutoTokenizer) -> int:
    """Ensure custom delimiter tokens exist on the tokenizer."""

    missing: list[str] = []
    for token in (COND_START_TOKEN, COND_END_TOKEN):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id or token_id is None:
            missing.append(token)

    if not missing:
        return 0

    added = tokenizer.add_special_tokens({"additional_special_tokens": missing})
    return added


class CondProjector(nn.Module):
    """Map cached sequence embeddings into virtual tokens for the LLM."""

    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"

    def __init__(
        self,
        spec: Dict[str, Any],
        d_out: int,
        k: int = 8,
        n_classes: int = 2,
        conv_dim: int = 256,
        conv_layers: int = 3,
        p_drop: float = 0.10,
    ):
        super().__init__()
        self.spec = dict(spec)
        self.d_in = int(spec.get("total_dim", 0))
        self.d_out = int(d_out)
        self.k_requested = int(k)
        self.conv_dim = int(conv_dim)
        self.conv_layers = int(conv_layers)
        self.n_classes = int(n_classes)
        self.p_drop = float(p_drop)

        if self.d_in <= 0:
            raise ValueError("CondProjector requires positive input dimension")

        self.k_dna = 0
        self.k_prot = 0
        self._assign_tokens(self.k_requested)

        dna_spec = self.spec.get("dna", {})
        prot_spec = self.spec.get("protein", {})

        self.dna_encoder: Optional[SequenceCNNEncoder]
        if dna_spec.get("slices") is not None and self.k_dna > 0:
            extra = dna_spec.get("extra_masks", []) or []
            scalar = 3 + len(extra)  # pos + edit + gap + extras
            self.dna_encoder = SequenceCNNEncoder(
                embed_dim=int(dna_spec.get("embed_dim", 0)),
                conv_dim=self.conv_dim,
                conv_layers=self.conv_layers,
                k_tokens=self.k_dna,
                out_dim=self.d_out,
                scalar_features=scalar,
            )
        else:
            self.dna_encoder = None

        if prot_spec.get("slices") is not None and self.k_prot > 0:
            extra = prot_spec.get("extra_masks", []) or []
            scalar = 3 + len(extra)  # pos + edit + gap + extras
            self.protein_encoder = SequenceCNNEncoder(
                embed_dim=int(prot_spec.get("embed_dim", 0)),
                conv_dim=self.conv_dim,
                conv_layers=self.conv_layers,
                k_tokens=self.k_prot,
                out_dim=self.d_out,
                scalar_features=scalar,
            )
        else:
            self.protein_encoder = None

        self.k = self.k_dna + self.k_prot
        if self.k <= 0:
            raise ValueError("CondProjector must emit at least one virtual token")

        self.dropout = nn.Dropout(self.p_drop)
        self.aux_head = nn.Linear(self.d_out, self.n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 2 or x.size(-1) != self.d_in:
            raise ValueError(
                f"CondProjector expected input shape (B, {self.d_in}), got {tuple(x.shape)}"
            )

        compute_dtype = self.aux_head.weight.dtype
        device = next(self.parameters()).device
        x = x.to(device=device, dtype=compute_dtype)

        tokens: list[torch.Tensor] = []

        if self.dna_encoder is not None:
            dna_inputs = self._build_sequence_inputs(
                x, self.spec["dna"], modality="dna"
            )
            dna_tokens = self.dna_encoder(dna_inputs)
            tokens.append(dna_tokens)

        if self.protein_encoder is not None:
            prot_inputs = self._build_sequence_inputs(
                x, self.spec["protein"], modality="protein"
            )
            prot_tokens = self.protein_encoder(prot_inputs)
            tokens.append(prot_tokens)

        if not tokens:
            raise RuntimeError("No conditioning modalities available for CondProjector")

        out_tokens = torch.cat(tokens, dim=1)
        out_tokens = self.dropout(out_tokens)
        out_tokens = out_tokens.to(self.aux_head.weight.dtype)

        pooled = out_tokens.mean(dim=1)
        aux_logits = self.aux_head(pooled)
        return out_tokens, aux_logits

    def _build_sequence_inputs(
        self, cond_vec: torch.Tensor, spec_section: Dict[str, Any], *, modality: str
    ) -> SequenceInputs:
        slices = spec_section.get("slices")
        if slices is None:
            raise RuntimeError(f"Spec for {modality} missing slices")
        B = cond_vec.size(0)
        L = int(spec_section.get("seq_len", 0))
        D = int(spec_section.get("embed_dim", 0))
        if L == 0 or D == 0:
            raise RuntimeError(f"Invalid sequence dimensions for {modality}")

        if modality == "protein":
            wt_key, mt_key = "wt_tokens", "mt_tokens"
            mask_keys = ("wt_mask", "mt_mask")
        else:
            wt_key, mt_key = "ref_tokens", "alt_tokens"
            mask_keys = ("ref_mask", "alt_mask")

        wt = cond_vec[:, slices[wt_key]].view(B, L, D)
        mt = cond_vec[:, slices[mt_key]].view(B, L, D)
        mask_a = cond_vec[:, slices[mask_keys[0]]].view(B, L)
        mask_b = cond_vec[:, slices[mask_keys[1]]].view(B, L)
        mask = torch.clamp(mask_a + mask_b, 0.0, 1.0)
        edit_mask = cond_vec[:, slices["edit_mask"]].view(B, L, 1)
        gap_mask = cond_vec[:, slices["gap_mask"]].view(B, L, 1)
        pos_slice = slices.get("pos")
        pos = (
            cond_vec[:, pos_slice].view(B, L, 1)
            if pos_slice is not None
            else torch.zeros_like(edit_mask)
        )
        extra_masks: list[torch.Tensor] = []
        for key in spec_section.get("extra_masks", []) or []:
            sl = slices.get(key)
            if sl is not None:
                extra_masks.append(cond_vec[:, sl].view(B, L, 1))

        return SequenceInputs(
            wt=wt,
            mt=mt,
            mask=mask,
            edit_mask=edit_mask,
            gap_mask=gap_mask,
            extra_masks=extra_masks if extra_masks else None,
            pos=pos,
        )

    def _assign_tokens(self, k_total: int) -> None:
        if k_total <= 0:
            raise ValueError("Number of conditioning tokens must be positive")
        remaining = k_total
        has_dna = self.spec.get("dna", {}).get("slices") is not None
        has_prot = self.spec.get("protein", {}).get("slices") is not None

        self.k_dna = 0
        self.k_prot = 0
        if remaining > 0:
            if has_dna and has_prot:
                self.k_dna = max(1, remaining // 2)
                self.k_prot = remaining - self.k_dna
                if self.k_prot == 0:
                    self.k_prot = 1
                    self.k_dna = remaining - self.k_prot
            elif has_dna:
                self.k_dna = remaining
            elif has_prot:
                self.k_prot = remaining

        if self.k_dna + self.k_prot < k_total:
            leftover = k_total - (self.k_dna + self.k_prot)
            if self.k_dna > 0:
                self.k_dna += leftover
            elif self.k_prot > 0:
                self.k_prot += leftover

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _slice_to_tuple(slc: Optional[slice]) -> Optional[tuple[int, int]]:
        if slc is None:
            return None
        return (int(slc.start), int(slc.stop))

    @classmethod
    def _serialise_spec(cls, spec: Dict[str, Any]) -> Dict[str, Any]:
        def _map(
            section: Optional[Dict[str, slice]],
        ) -> Optional[Dict[str, tuple[int, int]]]:
            if section is None:
                return None
            return {k: cls._slice_to_tuple(v) for k, v in section.items()}

        return {
            "total_dim": int(spec.get("total_dim", 0)),
            "dna": {
                "seq_len": int(spec.get("dna", {}).get("seq_len", 0)),
                "embed_dim": int(spec.get("dna", {}).get("embed_dim", 0)),
                "slices": _map(spec.get("dna", {}).get("slices")),
                "extra_masks": list(spec.get("dna", {}).get("extra_masks", []) or []),
            },
            "protein": {
                "seq_len": int(spec.get("protein", {}).get("seq_len", 0)),
                "embed_dim": int(spec.get("protein", {}).get("embed_dim", 0)),
                "slices": _map(spec.get("protein", {}).get("slices")),
                "extra_masks": list(
                    spec.get("protein", {}).get("extra_masks", []) or []
                ),
            },
        }

    @staticmethod
    def _tuple_to_slice(t: Optional[tuple[int, int]]) -> Optional[slice]:
        if t is None:
            return None
        return slice(int(t[0]), int(t[1]))

    @classmethod
    def _deserialise_spec(cls, cfg: Dict[str, Any]) -> Dict[str, Any]:
        spec = {
            "total_dim": int(cfg.get("total_dim", 0)),
            "dna": {
                "seq_len": int(cfg.get("dna", {}).get("seq_len", 0)),
                "embed_dim": int(cfg.get("dna", {}).get("embed_dim", 0)),
                "slices": None,
                "extra_masks": list(cfg.get("dna", {}).get("extra_masks", []) or []),
            },
            "protein": {
                "seq_len": int(cfg.get("protein", {}).get("seq_len", 0)),
                "embed_dim": int(cfg.get("protein", {}).get("embed_dim", 0)),
                "slices": None,
                "extra_masks": list(
                    cfg.get("protein", {}).get("extra_masks", []) or []
                ),
            },
        }
        dna_slices_cfg = cfg.get("dna", {}).get("slices")
        if dna_slices_cfg is not None:
            spec["dna"]["slices"] = {
                k: cls._tuple_to_slice(tuple(v) if v is not None else None)
                for k, v in dna_slices_cfg.items()
            }
        protein_slices_cfg = cfg.get("protein", {}).get("slices")
        if protein_slices_cfg is not None:
            spec["protein"]["slices"] = {
                k: cls._tuple_to_slice(tuple(v) if v is not None else None)
                for k, v in protein_slices_cfg.items()
            }
        return spec

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        return {
            "spec": self._serialise_spec(self.spec),
            "d_out": self.d_out,
            "k": self.k_requested,
            "n_classes": self.n_classes,
            "p_drop": self.p_drop,
            "conv_dim": self.conv_dim,
            "conv_layers": self.conv_layers,
            "d_in": self.d_in,
            "model_cls": self.__class__.__name__,
            "module": self.__class__.__module__,
        }

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        config_path = save_dir / self.CONFIG_NAME
        weights_path = save_dir / self.WEIGHTS_NAME

        with config_path.open("w", encoding="utf-8") as f:
            json.dump(self.get_config(), f, indent=2, sort_keys=True)

        torch.save(self.state_dict(), weights_path)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CondProjector":
        cfg = dict(config)
        cfg.pop("model_cls", None)
        cfg.pop("module", None)
        spec_cfg = cfg.pop("spec")
        cfg.pop("d_in", None)
        spec = cls._deserialise_spec(spec_cfg)
        return cls(spec=spec, **cfg)

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str | Path,
        *,
        map_location: str | torch.device | None = "cpu",
        strict: bool = True,
    ) -> "CondProjector":
        load_dir = Path(load_directory)
        config_path = load_dir / cls.CONFIG_NAME
        weights_path = load_dir / cls.WEIGHTS_NAME

        if not config_path.is_file():
            raise FileNotFoundError(f"Projector config not found: {config_path}")
        if not weights_path.is_file():
            raise FileNotFoundError(f"Projector weights not found: {weights_path}")

        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        model_cls = config.get("model_cls")
        if model_cls is not None and model_cls != cls.__name__:
            raise ValueError(
                f"Saved projector expects class '{model_cls}', cannot load with '{cls.__name__}'."
            )

        projector = cls.from_config(config)
        state = torch.load(weights_path, map_location=map_location)
        projector.load_state_dict(state, strict=strict)
        return projector


def build_qwen_with_lora(cfg, cond_spec: Dict[str, Any]):
    # Prefer bf16 compute if available; otherwise fallback to fp16 for speed/compat
    compute_dtype = torch.bfloat16
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            compute_dtype = torch.float16
    except Exception:
        compute_dtype = torch.float16

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print("[DEBUG] Loading tokenizer:", cfg.model_id)
    tok = AutoTokenizer.from_pretrained(
        cfg.model_id, use_fast=True, trust_remote_code=True
    )
    added_tokens = _ensure_special_tokens(tok)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    print("[DEBUG] Loading 4-bit base model:", cfg.model_id)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        trust_remote_code=True,
        quantization_config=quant_cfg,
        device_map="auto",
    )
    if added_tokens:
        base.resize_token_embeddings(len(tok))

    try:
        base.config.pretraining_tp = 1
    except Exception:
        pass

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    base = prepare_model_for_kbit_training(base)
    base = get_peft_model(base, lora)

    hidden = base.config.hidden_size
    n_cond_tokens = getattr(cfg, "n_cond_tokens", 8)
    conv_dim = getattr(cfg, "conv_dim", 256)
    conv_layers = getattr(cfg, "conv_layers", 3)
    base_param = next(base.parameters())
    projector = CondProjector(
        spec=cond_spec,
        d_out=hidden,
        k=n_cond_tokens,
        conv_dim=conv_dim,
        conv_layers=conv_layers,
    ).to(device=base_param.device, dtype=base_param.dtype)
    # Attach projector to the model so Trainer optimizes it
    base.add_module("cond_projector", projector)
    print("[DEBUG] Model ready. Hidden:", hidden, "Cond tokens:", n_cond_tokens)
    return tok, base, projector


def enable_fast_generate(model, tokenizer):
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    model.config.use_cache = True
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
