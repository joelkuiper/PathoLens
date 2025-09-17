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
    model_id: str, D_cond: int, adapter_dir: str, projector_path: str
):
    """
    Load tokenizer, 4-bit base model + LoRA adapter, and a validated CondProjector.

    Strong checks:
      - projector_path must exist
      - projector's Linear weight shape must match (hidden*k, D_cond)
      - k inferred from saved weights; used to reconstruct projector
    """
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"[LOAD] Adapter directory not found: {adapter_dir}")
    if not os.path.exists(projector_path):
        raise FileNotFoundError(f"[LOAD] Projector weights not found: {projector_path}")

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=_select_compute_dtype(),
    )

    base = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, quantization_config=quant, device_map="auto"
    )

    print("[LOAD] Loading adapter via PeftModel.from_pretrained:", adapter_dir)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    # --- Inspect saved projector to validate D_cond and infer k ---
    state = torch.load(projector_path, map_location="cpu")
    # Expect keys like 'net.0.weight' / 'net.0.bias' for (Linear → GELU)
    if "net.0.weight" not in state:
        raise KeyError(
            "[LOAD] Projector state missing 'net.0.weight'. "
            "Did the projector architecture change?"
        )

    W = state["net.0.weight"]  # shape: [hidden*k, D_cond_saved]
    if W.ndim != 2:
        raise ValueError(
            f"[LOAD] Unexpected projector weight ndim={W.ndim}, expected 2."
        )

    hidden = int(model.config.hidden_size)
    out_features, D_cond_saved = int(W.shape[0]), int(W.shape[1])

    if out_features % hidden != 0:
        raise ValueError(
            "[LOAD] Projector out_features is not divisible by model hidden size:\n"
            f"        out_features={out_features}, hidden_size={hidden}\n"
            "        Cannot infer k (n_cond_tokens)."
        )
    k_saved = out_features // hidden

    if D_cond != D_cond_saved:
        raise ValueError(
            "[LOAD] D_cond mismatch:\n"
            f"        Provided D_cond={D_cond}\n"
            f"        Saved projector expects D_cond={D_cond_saved}\n"
            "        Pass the correct D_cond (e.g., D_eff + D_go + D_prot) to evaluation."
        )

    print(
        f"[LOAD] Projector check OK: hidden={hidden} k={k_saved} D_cond={D_cond_saved}"
    )

    # --- Rebuild projector exactly as trained and load weights strictly ---
    emb = model.get_input_embeddings().weight
    projector = CondProjector(D_cond_saved, hidden, k=k_saved).to(
        device=emb.device, dtype=emb.dtype
    )
    # Replace/attach
    model.add_module("cond_projector", projector)

    missing, unexpected = model.cond_projector.load_state_dict(state, strict=True)
    if missing or unexpected:
        # strict=True should raise before here, but keep a guard
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
