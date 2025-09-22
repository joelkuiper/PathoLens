# src/llm/modeling.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import torch
import torch.nn as nn


class CondProjector(nn.Module):
    """
    Map conditioning vector -> k virtual tokens in hidden space
    and an auxiliary classification head.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        k: int = 8,
        n_classes: int = 2,
        p_drop: float = 0.10,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.k = k
        self.h = d_out
        self.eps = eps

        # Pre-normalize the conditioning vector
        self.norm_in = nn.LayerNorm(d_in)

        # 2-layer MLP projector
        self.fc1 = nn.Linear(d_in, d_out * k, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(d_out * k, d_out * k, bias=True)

        # Per-token affine after RMS centering (broadcast across hidden dim)
        self.gamma = nn.Parameter(torch.ones(k))  # multiplicative
        self.beta = nn.Parameter(torch.zeros(k))  # additive
        self.global_gain = nn.Parameter(torch.tensor(1.0))  # overall scalar

        # Aux head stays on the flattened token block
        self.aux_head = nn.Linear(d_out * k, n_classes)

        self._init_weights()

    def _init_weights(self):
        # Kaiming for the first layer
        nn.init.kaiming_uniform_(self.fc1.weight, a=0.0, nonlinearity="gelu")
        nn.init.zeros_(self.fc1.bias)

        # Small init on the second layer so outputs start near 0 (stable)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        # Aux head starts small too
        nn.init.zeros_(self.aux_head.weight)
        nn.init.zeros_(self.aux_head.bias)

        # gamma=1, beta=0, global_gain=1 already set

    def _rms_center(self, T: torch.Tensor) -> torch.Tensor:
        # T: [B, k, H] -> zero-mean + unit-RMS per token
        T = T - T.mean(dim=-1, keepdim=True)
        rms = torch.sqrt(T.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return T / rms

    def forward(self, x: torch.Tensor):
        # x: [B, d_in]
        B = x.size(0)
        x = self.norm_in(x)

        y = self.fc2(self.drop(self.act(self.fc1(x))))  # [B, k*H]
        aux_logits = self.aux_head(y)  # [B, n_classes]

        T = y.view(B, self.k, self.h)  # [B, k, H]
        T = self._rms_center(T)  # per-token normalize

        # Per-token affine + global gain (no external stats required)
        T = T * self.gamma.view(1, self.k, 1) + self.beta.view(1, self.k, 1)
        T = self.global_gain * T

        return T, aux_logits


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
