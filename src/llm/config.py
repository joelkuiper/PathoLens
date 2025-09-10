# -*- coding: utf-8 -*-
from __future__ import annotations
import random
import numpy as np
import torch
from dataclasses import dataclass

# ===== System / prompt =====
CHAT_SYSTEM = """You are a concise genetics assistant specializing in variant-level reasoning from minimal context.

Your job: estimate clinical significance as a binary label from the given features.

Output format: write ONLY the label word, exactly one of:
Benign
Pathogenic
(no punctuation, no quotes, no extra text)
"""

BIN_LABELS = ["Benign", "Pathogenic"]

POS = {"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"}
NEG = {"benign", "likely benign", "benign/likely benign"}

PROMPT_TMPL = "Variant context (no phenotype):\n{ctx}\n\nReturn label now:"


def clinsig_to_binary(cs: str | None) -> int | None:
    if not isinstance(cs, str):
        return None
    s = cs.strip().lower()
    if s in POS:
        return 1
    if s in NEG:
        return 0
    return None


# ===== Run config =====
@dataclass
class LLMConfig:
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    out_dir: str = "out/qwen3_path_binary_lora"
    max_len: int = 384
    lr: float = 3e-4
    epochs: int = 1
    grad_accum: int = 16
    per_device_bs: int = 2
    eval_steps: int = 500
    save_steps: int = 1000
    bf16: bool = True
    n_cond_tokens: int = 4
    # dataloader/system knobs
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = False
    pin_memory: bool = True
    group_by_length: bool = False
    drop_last: bool = True
    seed: int = 42


def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
