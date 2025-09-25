# src/llm/modeling.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import COND_START_TOKEN, COND_END_TOKEN


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


class ViewTokens(nn.Module):
    def __init__(self, k: int, d_out: int):
        super().__init__()
        self.k, self.d_out = k, d_out

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: (B, k*d_out) -> (B, k, d_out)
        B = y.size(0)
        return y.view(B, self.k, self.d_out)


class CondProjector(nn.Module):
    """
    (B, d_in) -> (B, k, d_out) virtual tokens + (B, n_classes) aux logits
    Minimal head: LN -> Linear -> GELU -> Dropout -> Linear, then reshape.
    """

    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"

    def __init__(
        self,
        d_in: int,
        d_out: int,
        k: int = 8,
        n_classes: int = 2,
        p_drop: float = 0.10,
    ):
        super().__init__()
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.k = int(k)
        self.n_classes = int(n_classes)
        self.p_drop = float(p_drop)

        # (B, d_in) -> (B, k*d_out)
        self.backbone = nn.Sequential(
            nn.Identity(),
            nn.Linear(self.d_in, self.d_out * self.k),
            nn.GELU(),
            nn.Dropout(self.p_drop),
            nn.Linear(self.d_out * self.k, self.d_out * self.k),
        )

        # reshape to tokens
        self.view = ViewTokens(self.k, self.d_out)

        # auxiliary classification head on the flattened block
        self.aux_head = nn.Linear(self.d_out * self.k, self.n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, d_in)
        y = self.backbone(x)  # (B, k*d_out)
        tokens = self.view(y)  # (B, k, d_out)
        aux_logits = self.aux_head(y)  # (B, n_classes)
        return tokens, aux_logits

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        return {
            "d_in": self.d_in,
            "d_out": self.d_out,
            "k": self.k,
            "n_classes": self.n_classes,
            "p_drop": self.p_drop,
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
        return cls(**cfg)

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


def build_qwen_with_lora(cfg, D_cond: int):
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
    n_cond_tokens = getattr(cfg, "n_cond_tokens", 8)  # default to 8 as requested
    base_param = next(base.parameters())
    projector = CondProjector(D_cond, hidden, k=n_cond_tokens).to(
        device=base_param.device, dtype=base_param.dtype
    )
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
