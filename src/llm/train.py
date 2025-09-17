# src/llm/train.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)

from .config import LLMRunConfig, set_all_seeds
from .modeling import build_qwen_with_lora
from .data import CondTextDataset, make_collator


class CondTrainer(Trainer):
    def __init__(self, *args, train_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_train_sampler = train_sampler

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if self._custom_train_sampler is None:
            return super().get_train_dataloader()
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=self._custom_train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # -------------------
        # Unpack inputs
        # -------------------
        inputs = dict(inputs)
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask", None)
        labels = inputs.pop("labels", None)
        label_weights = inputs.pop("label_weights", None)
        cond_vec = inputs.pop("cond_vec")
        gold_y = inputs.pop("_y", None)  # sequence-level class label

        # ===== fast dimension guard (catches dataset/model mismatch) =====
        expected_d = getattr(getattr(model, "cond_projector", None), "net", [None])[0]
        if hasattr(expected_d, "in_features"):
            exp = int(expected_d.in_features)
            got = int(cond_vec.shape[-1])
            if got != exp:
                raise RuntimeError(
                    f"[CondTrainer] cond_vec dim mismatch: got {got}, expected {exp}. "
                    "Check D_eff/D_go/D_prot in dataset vs projector build."
                )

        # default attention mask if missing
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        # -------------------
        # Embeddings
        # -------------------
        tok_emb = model.get_input_embeddings()
        txt_emb = tok_emb(input_ids)
        cond_vec = cond_vec.to(device=txt_emb.device, dtype=txt_emb.dtype)

        # Projector now returns: cond tokens, aux logits
        cond_emb, aux_logits = model.cond_projector(cond_vec)  # (B, K, H), (B, 2)

        # ===== PROMPT DROPOUT =====
        p_drop = float(getattr(model, "prompt_dropout_prob", 0.0))
        if p_drop > 0.0 and model.training:
            B = txt_emb.size(0)
            drop_mask = torch.rand(B, device=txt_emb.device) < p_drop
            if drop_mask.any():
                txt_emb = txt_emb.clone()
                txt_emb[drop_mask] = 0.0

        # concat cond tokens before text tokens
        inputs_embeds = torch.cat([cond_emb, txt_emb], dim=1)

        # extend masks/labels to account for cond tokens
        if attention_mask is not None:
            B, K = attention_mask.size(0), cond_emb.size(1)
            ones = torch.ones(
                (B, K), dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask = torch.cat([ones, attention_mask], dim=1)
        if labels is not None:
            B, K = labels.size(0), cond_emb.size(1)
            ignore = torch.full((B, K), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([ignore, labels], dim=1)
        if label_weights is not None:
            B, K = label_weights.size(0), cond_emb.size(1)
            zeros = torch.zeros(
                (B, K), dtype=label_weights.dtype, device=label_weights.device
            )
            label_weights = torch.cat([zeros, label_weights], dim=1)

        # optional position ids
        pos_ids = torch.arange(
            inputs_embeds.size(1), device=inputs_embeds.device
        ).unsqueeze(0)

        # -------------------
        # Forward through model
        # -------------------
        out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=pos_ids,
            use_cache=False,
        )
        logits = out.logits

        # -------------------
        # LM loss
        # -------------------
        if labels is None:
            lm_loss = (
                out.loss
                if hasattr(out, "loss")
                else torch.tensor(0.0, device=logits.device)
            )
        else:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_w = (
                label_weights[:, 1:].contiguous()
                if label_weights is not None
                else torch.ones_like(shift_labels, dtype=torch.float32)
            )
            loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
            loss_tok = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view_as(shift_labels)
            valid = (shift_labels != -100).float()
            denom = valid.sum().clamp_min(1.0)
            lm_loss = (loss_tok * shift_w * valid).sum() / denom
            if torch.isnan(lm_loss) or torch.isinf(lm_loss):
                lm_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)

        # -------------------
        # Aux classification loss
        # -------------------
        aux_loss = None
        if gold_y is not None and aux_logits is not None:
            gold_y = gold_y.to(aux_logits.device, dtype=torch.long)
            aux_loss = F.cross_entropy(aux_logits, gold_y)

        # -------------------
        # Total loss
        # -------------------
        lambda_aux = float(getattr(model, "aux_loss_weight", 1.0))
        if aux_loss is not None:
            total_loss = lm_loss + lambda_aux * aux_loss
        else:
            total_loss = lm_loss

        return (total_loss, out) if return_outputs else total_loss
