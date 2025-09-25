# -*- coding: utf-8 -*-
from __future__ import annotations

import random

import numpy as np
import torch

from src.pipeline.config import LLMRunConfig

# ===== System / prompt =====
CHAT_SYSTEM = """You are a concise genetics assistant specializing in variant-level reasoning from minimal context.

Your job: estimate clinical significance as a binary label from the given features.

Output format: write ONLY the label word, exactly one of:
Benign
Pathogenic
(no punctuation, no quotes, no extra text)
"""

BIN_LABELS = ["Benign", "Pathogenic"]

COND_START_TOKEN = "<|cond|>"
COND_END_TOKEN = "<|endcond|>"

PROMPT_TMPL = "Variant context (no phenotype):\n{ctx}\n\nReturn label now:"


def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "CHAT_SYSTEM",
    "BIN_LABELS",
    "COND_START_TOKEN",
    "COND_END_TOKEN",
    "PROMPT_TMPL",
    "LLMRunConfig",
    "set_all_seeds",
]
