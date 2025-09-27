# -*- coding: utf-8 -*-
from __future__ import annotations

import random

import numpy as np
import torch

from src.pipeline.config import LLMRunConfig

# ===== System / prompt =====
CHAT_SYSTEM = """You are an assistant trained for clinical genetics.
Given minimal variant context, your task is to predict the variant’s
clinical significance as a binary label.

Output **only** one word: either "Benign" or "Pathogenic".
"""

BIN_LABELS = ["Benign", "Pathogenic"]

COND_START_TOKEN = "<|cond_start|>"
COND_END_TOKEN = "<|cond_end|>"

PROMPT_TMPL = "Variant context:\n{ctx}\n\nPredict label:"


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
