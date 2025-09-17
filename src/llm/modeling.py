# src/llm/modeling.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class CondProjector(nn.Module):
    """
    Projects a conditioning vector (e.g., concatenated dna/prot/go features)
    into k learnable "conditioning tokens" in the model's hidden space.
    """

    def __init__(self, d_in: int, d_out: int, k: int = 8):
        super().__init__()
        self.k = k
        # simple one-layer projector with GELU nonlinearity
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out * k, bias=True),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, d_in] -> returns [B, k, d_out]
        """
        B = x.size(0)
        y = self.net(x)
        return y.view(B, self.k, -1)


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
