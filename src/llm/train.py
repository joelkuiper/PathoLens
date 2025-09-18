# src/llm/train.py
# -*- coding: utf-8 -*-
from __future__ import annotations

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


class _FirstStepTimerCallback(TrainerCallback):
    def __init__(self):
        self._t0 = None
        self._stepped = False

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        print("[DEBUG] Trainer init complete.")
        return control

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._t0 = time.time()
        self._stepped = False
        print("[DEBUG] Training started ...")
        return control

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not self._stepped:
            self._stepped = True
            dt = time.time() - (self._t0 or time.time())
            print(f"[DEBUG] Time to first optimizer step: {dt:.2f}s")
        return control


def _true_label_from_meta(row) -> int | None:
    y = row.get("_y", None)
    try:
        return int(y) if y is not None else None
    except Exception:
        return None


def _make_balanced_sampler(train_ds: CondTextDataset) -> WeightedRandomSampler:
    ys = []
    for i in range(len(train_ds)):
        row = train_ds.meta.iloc[i]
        y = _true_label_from_meta(row)
        ys.append(int(0 if y is None else y))
    ys = np.asarray(ys)
    ys = (ys > 0).astype(np.int64)
    n_pos = max(1, int(ys.sum()))
    n_neg = max(1, int((ys == 0).sum()))
    w_pos, w_neg = 1.0 / n_pos, 1.0 / n_neg
    weights = np.where(ys == 1, w_pos, w_neg).astype("float32")
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights), num_samples=len(train_ds), replacement=True
    )


def smoke_checks(seq_ds_dict: Dict[str, object], tokenizer, cfg: LLMRunConfig):
    print("\n========== SMOKE CHECKS ==========")
    total_train, total_val = len(seq_ds_dict["train"]), len(seq_ds_dict["val"])
    print(f"[SMOKE] Train size: {total_train}  |  Val size: {total_val}")

    from collections import Counter

    def label_stats(ds, N=5000):
        ys = []
        for i in range(min(len(ds), N)):
            row = ds.meta.iloc[i]
            y = _true_label_from_meta(row)
            if y is not None:
                ys.append(int(y))
        c = Counter(ys)
        return {"n": len(ys), "pos": c.get(1, 0), "neg": c.get(0, 0)}

    tr = label_stats(seq_ds_dict["train"])
    va = label_stats(seq_ds_dict["val"])
    print(
        f"[SMOKE] Label coverage (≤5k): train n={tr['n']} (+:{tr['pos']}, -:{tr['neg']}), val n={va['n']} (+:{va['pos']}, -:{va['neg']})"
    )

    row0 = seq_ds_dict["train"].meta.iloc[0]
    ctx0 = row0.get("context", "")
    from .chat import build_chat_strings

    p0 = f"Variant context (no phenotype):\n{ctx0}\n\nReturn label now:"
    pr, fu = build_chat_strings(tokenizer, [p0], ["Benign"])
    print("[SMOKE] Example context:\n", ctx0)
    print("[SMOKE] Chat prompt (trunc 200):", pr[0][:200].replace("\n", "\\n"), "...")
    print("[SMOKE] Chat full   (trunc 200):", fu[0][:200].replace("\n", "\\n"), "...")


def run_llm_pipeline(cfg: LLMRunConfig, seq_ds: Dict[str, object]) -> Dict[str, Any]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(cfg.seed)

    # Compute *true* conditioning dim from dataset (includes protein if present)
    D_eff = int(seq_ds["train"].D_eff)
    D_go = int(seq_ds["train"].D_go)
    D_prot = int(getattr(seq_ds["train"], "D_prot", 0) or 0)
    D_cond = D_eff + D_go + D_prot
    print(f"[INFO] LLM pipeline: D_cond={D_cond} (eff={D_eff} go={D_go} prot={D_prot})")

    # build model with matching projector input
    tok, base_model, projector = build_qwen_with_lora(cfg, D_cond)

    # attach prompt dropout knob
    base_model.prompt_dropout_prob = float(cfg.prompt_dropout_prob)
    base_model.config.use_cache = False

    print("[DEBUG] Building CondTextDataset(train/val) ...")
    train_ds = CondTextDataset(seq_ds["train"], tokenizer=tok)
    val_ds = CondTextDataset(seq_ds["val"], tokenizer=tok)
    # sanity: dataset D_cond must match projector's in_features
    assert train_ds.D_cond == D_cond == val_ds.D_cond, (
        f"D_cond mismatch: dataset(train)={train_ds.D_cond}, dataset(val)={val_ds.D_cond}, "
        f"projector_in={projector.net[0].in_features}"
    )

    collator = make_collator(tok, max_len=cfg.max_len)
    smoke_checks(seq_ds, tok, cfg)

    sampler = _make_balanced_sampler(train_ds) if cfg.balanced_sampling else None

    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg.per_device_bs,
        per_device_eval_batch_size=cfg.per_device_bs,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        weight_decay=0.01,
        num_train_epochs=cfg.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=0.5,
        logging_steps=50,
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        bf16=(
            True
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else False
        ),
        gradient_checkpointing=False,
        use_reentrant=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=cfg.num_workers,
        dataloader_pin_memory=cfg.pin_memory,
        dataloader_prefetch_factor=(
            cfg.prefetch_factor if cfg.num_workers > 0 else None
        ),
        dataloader_persistent_workers=cfg.persistent_workers,
        group_by_length=cfg.group_by_length,
        optim="paged_adamw_8bit",
        tf32=True,
        seed=cfg.seed,
        data_seed=cfg.seed,
    )

    trainer = CondTrainer(
        model=base_model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[_FirstStepTimerCallback()],
        train_sampler=sampler,
    )

    dev = next(base_model.parameters()).device
    print(f"[INFO] Device: {dev} | CUDA={torch.cuda.is_available()} | bf16={args.bf16}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"[INFO] GPU mem: free={free / 1e9:.2f}G total={total / 1e9:.2f}G")
        print("[INFO] TF32:", torch.backends.cuda.matmul.allow_tf32)

    print("[DEBUG] Probing first training batch...")
    t0 = time.time()
    batch = next(iter(trainer.get_train_dataloader()))
    print(f"[DEBUG] Got first batch in {time.time() - t0:.2f}s")
    for k, v in batch.items():
        print(f"  {k}: shape={tuple(v.shape)} dtype={getattr(v, 'dtype', None)}")

    trainer.train()

    base_model.save_pretrained(str(out_dir / "adapter"))
    torch.save(projector.state_dict(), out_dir / "projector.pt")
    print("[INFO] Saved adapter + projector.")

    return {"out_dir": str(out_dir)}
