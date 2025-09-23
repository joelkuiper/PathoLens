# src/llm/load.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from .modeling import CondProjector


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
    D_cond: int,
    adapter_dir: str,
    projector_path: str,
):
    """
    Load tokenizer, 4-bit base model + LoRA adapter, and the new minimal CondProjector.

    Assumptions for the projector checkpoint (no backwards compatibility):
      - 'backbone.1.weight' exists (Linear: [k*d_out, D_cond])
      - d_out == model.config.hidden_size
      - k = (out_features / hidden_size)
    """
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"[LOAD] Adapter directory not found: {adapter_dir}")
    if not os.path.exists(projector_path):
        raise FileNotFoundError(f"[LOAD] Projector weights not found: {projector_path}")

    # ----- Tokenizer -----
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
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

    # ----- LoRA adapter -----
    print("[LOAD] Loading adapter via PeftModel.from_pretrained:", adapter_dir)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    # ----- Inspect projector checkpoint  -----
    state = torch.load(projector_path, map_location="cpu")

    key_w = (
        "backbone.1.weight"  # LN(0) -> Linear(1) -> GELU(2) -> Dropout(3) -> Linear(4)
    )
    if key_w not in state:
        raise KeyError(
            "[LOAD] Expected projector checkpoint with 'backbone.1.weight' "
            "(Linear from D_cond -> k*d_out)."
        )

    W = state[key_w]  # shape: [k*d_out, D_cond_saved]
    if W.ndim != 2:
        raise ValueError(f"[LOAD] {key_w} has ndim={W.ndim}, expected 2.")
    out_features, D_cond_saved = int(W.shape[0]), int(W.shape[1])

    hidden = int(model.config.hidden_size)
    if out_features % hidden != 0:
        raise ValueError(
            "[LOAD] backbone.1.out_features not divisible by model hidden size:\n"
            f"        out_features={out_features}, hidden_size={hidden}\n"
            "        Cannot infer k (n_cond_tokens)."
        )
    k_saved = out_features // hidden

    if D_cond != D_cond_saved:
        raise ValueError(
            "[LOAD] D_cond mismatch:\n"
            f"        Provided D_cond={D_cond}\n"
            f"        Saved projector expects D_cond={D_cond_saved}\n"
            "        Pass the correct D_cond (e.g., D_eff + D_prot)."
        )

    print(
        f"[LOAD] Projector check OK: hidden={hidden} k={k_saved} D_cond={D_cond_saved}"
    )

    # ----- Rebuild projector and load weights strictly -----
    emb = model.get_input_embeddings().weight
    projector = CondProjector(d_in=D_cond_saved, d_out=hidden, k=k_saved).to(
        device=emb.device, dtype=emb.dtype
    )
    model.add_module("cond_projector", projector)

    missing, unexpected = model.cond_projector.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"[LOAD] Projector state mismatch. Missing={missing} Unexpected={unexpected}"
        )
    print(
        f"[LOAD] Projector loaded. dtype={next(projector.parameters()).dtype} "
        f"device={next(projector.parameters()).device}"
    )

    # Align LoRA param dtype/device with the embedding table just in case
    for n, p in model.named_parameters():
        if "lora_" in n and (p.dtype != emb.dtype or p.device != emb.device):
            p.data = p.data.to(dtype=emb.dtype, device=emb.device)

    model.config.use_cache = True
    model.eval()
    return tok, model
