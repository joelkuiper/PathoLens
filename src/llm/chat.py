# -*- coding: utf-8 -*-
from __future__ import annotations
import re
from typing import List, Optional, Tuple

from .config import CHAT_SYSTEM

# Robust removal of hidden reasoning
_THINK_BLOCK_RE = re.compile(r"(?is)<think>.*?</think>")
_THINK_OPEN_OR_CLOSE_RE = re.compile(r"(?is)</?think>")


def strip_hidden_reasoning(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _THINK_OPEN_OR_CLOSE_RE.sub("", cleaned)
    return cleaned.strip()


def sanitize_model_text(text: str) -> str:
    s = strip_hidden_reasoning(text or "")
    if s.startswith("```"):
        parts = s.split("```", 2)
        s = parts[-1].strip() if len(parts) == 3 else s
    return s


def build_chat_strings(tokenizer, user_texts, assistant_texts=None):
    prompt_strings, full_strings = [], []
    for i, u in enumerate(user_texts):
        msgs = [
            {"role": "system", "content": CHAT_SYSTEM},
            {"role": "user", "content": u},
        ]
        # one templating pass that ends at the assistant header
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        prompt_strings.append(prompt)

        if assistant_texts is not None:
            # append EXACTLY our target + EOS, no second templating pass
            target = assistant_texts[i]
            eos = tokenizer.eos_token or ""
            full_strings.append(prompt + target + eos)

    return prompt_strings, full_strings
