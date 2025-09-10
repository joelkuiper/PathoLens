# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from .modeling import CondProjector


def load_finetuned_model(
    model_id: str, D_cond: int, adapter_dir: str, projector_path: str
):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, quantization_config=quant, device_map="auto"
    )

    print("[LOAD] Loading adapter via PeftModel.from_pretrained:", adapter_dir)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    emb = model.get_input_embeddings().weight
    projector = CondProjector(D_cond, model.config.hidden_size, k=4).to(
        device=emb.device, dtype=emb.dtype
    )
    model.add_module("cond_projector", projector)

    psd = torch.load(projector_path, map_location="cpu")
    miss, unexp = model.cond_projector.load_state_dict(psd, strict=False)
    print(
        f"[LOAD] Projector missing={len(miss)} unexpected={len(unexp)} dtype={next(projector.parameters()).dtype}"
    )

    # align LoRA param dtype/device
    for n, p in model.named_parameters():
        if "lora_" in n and (p.dtype != emb.dtype or p.device != emb.device):
            p.data = p.data.to(dtype=emb.dtype, device=emb.device)

    model.config.use_cache = True
    model.eval()
    return tok, model
