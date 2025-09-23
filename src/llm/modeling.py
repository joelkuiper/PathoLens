# src/llm/modeling.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from typing import Tuple

# src/llm/modeling.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn


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

    def __init__(
        self,
        d_in: int,
        d_out: int,
        k: int = 8,
        n_classes: int = 2,
        p_drop: float = 0.10,
    ):
        super().__init__()
        self.k = k
        self.d_out = d_out

        # (B, d_in) -> (B, k*d_out)
        self.backbone = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_out * k),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_out * k, d_out * k),
        )

        # reshape to tokens
        self.view = ViewTokens(k, d_out)

        # auxiliary classification head on the flattened block
        self.aux_head = nn.Linear(d_out * k, n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, d_in)
        y = self.backbone(x)  # (B, k*d_out)
        tokens = self.view(y)  # (B, k, d_out)
        aux_logits = self.aux_head(y)  # (B, n_classes)
        return tokens, aux_logits


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
    projector = CondProjector(D_cond, hidden, k=n_cond_tokens).to(
        next(base.parameters()).device
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
