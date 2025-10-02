# src/llm/load.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .modeling import CondProjector, _ensure_special_tokens


def _select_compute_dtype():
    # Prefer bf16 on capable GPUs, fall back to fp16
    dt = torch.bfloat16
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            dt = torch.float16
    except Exception:
        dt = torch.float16
    return dt


def load_finetuned_model(
    model_id: str,
    adapter_dir: str,
    projector_path: str,
):
    """
    Load tokenizer, 4-bit base model + LoRA adapter, and the serialized CondProjector.
    The projector is expected to have been saved via ``CondProjector.save_pretrained``
    and resides in ``projector_path``.
    """
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"[LOAD] Adapter directory not found: {adapter_dir}")
    projector_dir = Path(projector_path)
    if not projector_dir.is_dir():
        raise FileNotFoundError(
            f"[LOAD] Projector directory not found: {projector_dir}"
        )

    # ----- Tokenizer -----
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    added_tokens = _ensure_special_tokens(tok)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # ----- Base (4-bit) -----
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=_select_compute_dtype(),
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=quant,
        device_map="auto",
    )

    if added_tokens:
        base.resize_token_embeddings(len(tok))

    # ----- LoRA adapter -----
    print("[LOAD] Loading adapter via PeftModel.from_pretrained:", adapter_dir)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    # ----- Load serialized projector -----
    projector = CondProjector.from_pretrained(projector_dir, map_location="cpu")

    hidden = int(model.config.hidden_size)
    if projector.d_out != hidden:
        raise ValueError(
            "[LOAD] Projector hidden dim mismatch:\n"
            f"        Saved d_out={projector.d_out}\n"
            f"        Base model hidden_size={hidden}"
        )

    emb = model.get_input_embeddings().weight
    projector = projector.to(device=emb.device, dtype=emb.dtype)
    model.add_module("cond_projector", projector)

    print(
        f"[LOAD] Projector loaded. hidden={hidden} k={projector.k} "
        f"dtype={next(projector.parameters()).dtype} device={next(projector.parameters()).device}"
    )

    # Align LoRA param dtype/device with the embedding table just in case
    for n, p in model.named_parameters():
        if "lora_" in n and (p.dtype != emb.dtype or p.device != emb.device):
            p.data = p.data.to(dtype=emb.dtype, device=emb.device)

    model.config.use_cache = True
    model.eval()
    return tok, model
