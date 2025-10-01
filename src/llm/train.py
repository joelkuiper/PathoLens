# src/llm/train.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

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
from .data import CondDataset, make_collator
from .context import build_ctx_from_row


class CondTrainer(Trainer):
    def __init__(self, *args, train_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_train_sampler = train_sampler
        self._reset_aux_metric_buffer()

    # ------------------------------------------------------------------
    # Auxiliary head logging helpers
    # ------------------------------------------------------------------
    def _reset_aux_metric_buffer(self):
        self._aux_loss_sum = 0.0
        self._aux_loss_count = 0
        self._aux_correct = 0
        self._aux_total = 0

    def _update_aux_metrics(
        self, aux_loss: torch.Tensor, aux_logits: torch.Tensor, gold_y: torch.Tensor
    ) -> None:
        with torch.no_grad():
            batch = int(gold_y.numel())
            if batch == 0:
                return
            self._aux_loss_sum += float(aux_loss.detach()) * batch
            self._aux_loss_count += batch
            preds = aux_logits.detach().argmax(dim=-1)
            correct = (preds == gold_y.detach()).sum().item()
            self._aux_correct += int(correct)
            self._aux_total += batch

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Inject auxiliary head stats into the standard Trainer log output."""
        has_metrics = (self._aux_loss_count > 0) or (self._aux_total > 0)
        if has_metrics:
            logs = dict(logs)
            if self._aux_loss_count > 0:
                logs["aux_loss"] = self._aux_loss_sum / self._aux_loss_count
            if self._aux_total > 0:
                logs["aux_acc"] = self._aux_correct / self._aux_total
                logs["aux_support"] = float(self._aux_total)
        super().log(logs, start_time)
        if has_metrics:
            self._reset_aux_metric_buffer()

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

    @staticmethod
    def _first_label_index(label_weights: torch.Tensor) -> torch.Tensor:
        """
        Given label_weights [B, T], return a LongTensor [B] with the index
        of the first supervised (weight > 0) position. If none found, returns T.
        """
        B, T = label_weights.shape
        mask = (label_weights > 0).to(label_weights.dtype)  # [B, T]
        first_idx = torch.argmax(mask, dim=1)  # [B]
        has_pos = mask.max(dim=1).values > 0
        first_idx = torch.where(has_pos, first_idx, torch.full_like(first_idx, T))
        return first_idx.long()

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
        gold_y = inputs.pop("_y", None)
        if gold_y is None:
            raise KeyError(
                "CondTrainer.compute_loss requires '_y' labels in the batch."
            )
        if not torch.is_tensor(gold_y):
            gold_y = torch.as_tensor(gold_y)
        if gold_y.dtype != torch.long:
            raise TypeError(
                f"CondTrainer expects '_y' dtype torch.long but received {gold_y.dtype}."
            )

        # ===== fast dimension guard  =====
        cond_proj = getattr(model, "cond_projector", None)
        expected_d = getattr(cond_proj, "d_in", None)
        if expected_d is not None:
            exp = int(expected_d)
            got = int(cond_vec.shape[-1])
            if got != exp:
                raise RuntimeError(
                    f"[CondTrainer] cond_vec dim mismatch: got {got}, expected {exp}. "
                    "Check D_eff/D_prot in dataset vs projector build."
                )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        # -------------------
        # Embeddings
        # -------------------
        tok_emb = model.get_input_embeddings()
        txt_emb = tok_emb(input_ids)
        cond_vec = cond_vec.to(device=txt_emb.device, dtype=txt_emb.dtype)

        # Projector
        cond_emb, aux_logits = model.cond_projector(cond_vec)
        if aux_logits is None:
            raise RuntimeError(
                "cond_projector must return auxiliary logits when '_y' supervision is provided."
            )
        if gold_y.ndim == 0:
            gold_y = gold_y.unsqueeze(0)
        if gold_y.ndim != 1:
            raise ValueError(
                f"CondTrainer expects '_y' to be 1-D (batch,), got shape {tuple(gold_y.shape)}."
            )

        # ===== PROMPT DROPOUT  =====
        p_drop = float(getattr(model, "prompt_dropout_prob", 0.0))
        if p_drop > 0.0 and model.training and (label_weights is not None):
            B, T = txt_emb.shape[:2]
            drop_mask = torch.rand(B, device=txt_emb.device) < p_drop
            if drop_mask.any():
                first_label_idx = self._first_label_index(label_weights).to(
                    txt_emb.device
                )  # [B]
                txt_emb = txt_emb.clone()
                attention_mask = attention_mask.clone()
                for b in torch.nonzero(drop_mask, as_tuple=False).flatten():
                    s = int(first_label_idx[b].item())
                    if s > 0:
                        # zero the prompt embeddings and make them non-attendable
                        txt_emb[b, :s, :] = 0.0
                        attention_mask[b, :s] = 0

        # -------------------
        # Concat cond + text, extend masks/labels
        # -------------------
        inputs_embeds = torch.cat([cond_emb, txt_emb], dim=1)
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

        pos_ids = torch.arange(
            inputs_embeds.size(1), device=inputs_embeds.device
        ).unsqueeze(0)

        # -------------------
        # Forward + losses
        # -------------------
        out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=pos_ids,
            use_cache=False,
        )
        logits = out.logits

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

        aux_loss = None
        if gold_y is not None:
            gold_y = gold_y.to(aux_logits.device)
            if gold_y.shape[0] != aux_logits.size(0):
                raise ValueError(
                    "Batch size mismatch between auxiliary logits and '_y' labels."
                    f" Got {aux_logits.size(0)} logits vs {gold_y.shape[0]} labels."
                )
            aux_loss = F.cross_entropy(aux_logits, gold_y)

        lambda_aux = float(getattr(model, "aux_loss_weight", 1.0))
        total_loss = lm_loss + (lambda_aux * aux_loss if aux_loss is not None else 0.0)

        if model.training and aux_loss is not None:
            self._update_aux_metrics(aux_loss, aux_logits, gold_y)

        return (total_loss, out) if return_outputs else total_loss


class _DeferredSmokeChecks(TrainerCallback):
    """Run tokenizer-heavy smoke checks only after DataLoader workers fork."""

    def __init__(self, runner: Callable[[], None]):
        self._runner = runner
        self._has_run = False

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not self._has_run:
            self._runner()
            self._has_run = True
        return control


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


def _make_balanced_sampler(train_ds: CondDataset) -> WeightedRandomSampler:
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


def smoke_checks(seq_ds_dict: Dict[str, object], tokenizer, cfg: "LLMRunConfig"):
    """
    Strict smoke test:
      - Relies on `_true_label_from_meta` and `build_ctx_from_row` (no fallbacks).
      - Prints split sizes and full label prevalence for train/val/test.
      - Shows one concrete chat example (context + prompt) and tokenization lengths.
      - Verifies special conditioning tokens map to exactly one tokenizer id.
      - Prints label-token diagnostics for BIN_LABELS.
      - Reports conditioning block dimensions from the training dataset.
    """
    import numpy as np

    # --- Imports from your codebase (required) ---
    from .chat import build_chat_strings
    from .config import BIN_LABELS, COND_START_TOKEN, COND_END_TOKEN

    print("\n========== SMOKE CHECKS ==========")

    # --- Split sizes ---
    for split in ("train", "val", "test"):
        if split in seq_ds_dict:
            print(f"[SMOKE] {split:5s} size: {len(seq_ds_dict[split])}")
    if "train" not in seq_ds_dict or "val" not in seq_ds_dict:
        raise KeyError("[SMOKE] Expected 'train' and 'val' splits in seq_ds_dict.")

    def label_stats(ds):
        # Apply in vectorized form instead of row-by-row
        ys = ds.meta.apply(_true_label_from_meta, axis=1).astype(int).to_numpy()

        n = ys.size
        if n == 0:
            return {"n": 0, "pos": 0, "neg": 0, "frac_pos": float("nan")}

        pos = int((ys == 1).sum())
        neg = n - pos
        frac_pos = pos / n
        return {"n": n, "pos": pos, "neg": neg, "frac_pos": frac_pos}

    tr = label_stats(seq_ds_dict["train"])
    va = label_stats(seq_ds_dict["val"])
    print(
        f"[SMOKE] Label coverage (full): train n={tr['n']} (+:{tr['pos']}, -:{tr['neg']}, pos%={tr['frac_pos']:.3f})"
    )
    print(
        f"[SMOKE] Label coverage (full): val n={va['n']} (+:{va['pos']}, -:{va['neg']}, pos%={va['frac_pos']:.3f})"
    )
    if "test" in seq_ds_dict:
        te = label_stats(seq_ds_dict["test"])
        print(
            f"[SMOKE] Label coverage (full): test n={te['n']} (+:{te['pos']}, -:{te['neg']}, pos%={te['frac_pos']:.3f})"
        )

    # --- Conditioning dims from training dataset ---
    ds_tr = seq_ds_dict["train"]
    D_eff = int(getattr(ds_tr, "D_eff"))
    D_prot = int(getattr(ds_tr, "D_prot", 0) or 0)
    D_cond = D_eff + D_prot
    print(f"[SMOKE] D_eff={D_eff}  D_prot={D_prot}  → D_cond={D_cond}")

    # --- Special tokens must be single ids ---
    def _single_id(token: str) -> int:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"[SMOKE] Token '{token}' must map to exactly one id; got {ids}"
            )
        return ids[0]

    cond_start_id = _single_id(COND_START_TOKEN)
    cond_end_id = _single_id(COND_END_TOKEN)
    print(f"[SMOKE] COND_START='{COND_START_TOKEN}' → id={cond_start_id}")
    print(f"[SMOKE] COND_END  ='{COND_END_TOKEN}' → id={cond_end_id}")

    # --- Label tokenization diagnostics ---
    for lw in BIN_LABELS:
        ids_no_space = tokenizer.encode(lw, add_special_tokens=False)
        ids_space = tokenizer.encode(" " + lw, add_special_tokens=False)
        print(f"[SMOKE] Label '{lw}': ids={ids_no_space}  |  ids(space)={ids_space}")

    # --- Example context + chat strings ---
    row0 = ds_tr.meta.iloc[0]
    ctx0 = build_ctx_from_row(row0)
    base_prompt = f"Variant context:\n{ctx0}\n\nPredict label:"
    prompts, prompts_full = build_chat_strings(tokenizer, [base_prompt], ["Benign"])

    print("[SMOKE] Example context:")
    print(ctx0)
    print(
        "[SMOKE] Chat prompt (trunc 300):", prompts[0][:300].replace("\n", "\\n"), "..."
    )
    print(
        "[SMOKE] Chat full   (trunc 300):",
        prompts_full[0][:300].replace("\n", "\\n"),
        "...",
    )

    # --- Token counts ---
    enc_prompt = tokenizer(
        prompts, return_tensors="pt", padding=False, truncation=False
    )
    enc_full = tokenizer(
        prompts_full, return_tensors="pt", padding=False, truncation=False
    )
    print(
        f"[SMOKE] Token count: prompt={enc_prompt['input_ids'].shape[1]}  |  prompt+label={enc_full['input_ids'].shape[1]}"
    )

    # --- First item feature sanity ---
    x0, *_ = ds_tr[0]
    x0 = x0 if isinstance(x0, np.ndarray) else x0.numpy()
    finite = np.isfinite(x0)
    print(
        f"[SMOKE] First feature vector: shape={tuple(x0.shape)}  finite%={(finite.mean() * 100):.2f}%  "
        f"min={np.nanmin(x0):.4f}  max={np.nanmax(x0):.4f}"
    )

    print("========== END SMOKE CHECKS ==========\n")


def run_llm_pipeline(cfg: LLMRunConfig, seq_ds: Dict[str, object]) -> Dict[str, Any]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(cfg.seed)

    # Compute conditioning dim from dataset (DNA + protein)
    train_split = seq_ds["train"]
    D_eff = int(train_split.D_eff)
    D_prot = int(train_split.D_prot)
    cond_spec = getattr(train_split, "cond_spec", None)
    if cond_spec is None:
        raise RuntimeError(
            "SequenceTowerDataset must expose 'cond_spec' for projector build"
        )
    D_cond = int(cond_spec.get("total_dim", D_eff + D_prot))
    print(f"[INFO] LLM pipeline: D_cond={D_cond} (eff={D_eff} prot={D_prot})")

    # build model with matching projector input
    tok, base_model, projector = build_qwen_with_lora(cfg, cond_spec)

    # attach prompt dropout knob
    base_model.prompt_dropout_prob = float(cfg.prompt_dropout_prob)
    base_model.config.use_cache = False

    print("[DEBUG] Building CondDataset(train/val) ...")
    train_ds = CondDataset(seq_ds["train"], tokenizer=tok)
    val_ds = CondDataset(seq_ds["val"], tokenizer=tok)
    # sanity: dataset D_cond must match projector's in_features
    assert train_ds.D_cond == D_cond == val_ds.D_cond == projector.d_in, (
        f"D_cond mismatch: dataset(train)={train_ds.D_cond}, dataset(val)={val_ds.D_cond}, "
        f"projector_in={projector.d_in}"
    )

    collator = make_collator(tok, max_len=cfg.max_len)
    smoke_cb = _DeferredSmokeChecks(lambda: smoke_checks(seq_ds, tok, cfg))

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
        callbacks=[_FirstStepTimerCallback(), smoke_cb],
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
    projector.save_pretrained(out_dir / "projector")
    print("[INFO] Saved adapter + projector.")

    return {"out_dir": str(out_dir)}
