# src/llm.py
# -*- coding: utf-8 -*-
from __future__ import annotations

# ============================================================
# Process / parallelism hygiene (set BEFORE imports that may touch tokenizers)
# ============================================================
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")  # keep HF tokenizer threads on

# ============================================================
# Standard imports
# ============================================================
import re, json, time, random, math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.clinvar import (
    pick_gene,
    extract_hgvsc,
    extract_hgvsp,
    classify_protein_change,
    parse_hgvsc_features,
)


# ============================================================
# Global speed/robustness toggles
# ============================================================
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass  # older torch


def _set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Binary task: Pathogenic vs Benign (VUS dropped upstream)
# ============================================================

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


def clinsig_to_binary(cs: str | None) -> int | None:
    if not isinstance(cs, str):
        return None
    s = cs.strip().lower()
    if s in POS:
        return 1
    if s in NEG:
        return 0
    return None


PROMPT_TMPL = "Variant context (no phenotype):\n{ctx}\n\nReturn label now:"


# ============================================================
# Utilities: chat template + think stripping + fast gen
# ============================================================


def build_chat_strings(
    tokenizer, user_texts: List[str], assistant_texts: Optional[List[str]]
):
    prompt_strings, full_strings = [], []
    for i, u in enumerate(user_texts):
        msgs = [
            {"role": "system", "content": CHAT_SYSTEM},
            {"role": "user", "content": u},
        ]
        prompt_strings.append(
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        )
        if assistant_texts is not None:
            msgs_full = msgs + [{"role": "assistant", "content": assistant_texts[i]}]
            full_strings.append(
                tokenizer.apply_chat_template(
                    msgs_full, tokenize=False, add_generation_prompt=False
                )
            )
    return prompt_strings, full_strings


_THINK_CLOSE_PAT = re.compile(r"</think>\s*", re.IGNORECASE)


def strip_think(decoded: str) -> str:
    m = _THINK_CLOSE_PAT.search(decoded)
    return decoded if not m else decoded[m.end() :]


def enable_fast_generate(model, tokenizer):
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    model.config.use_cache = (
        True  # generation: True (we turn it off inside training call)
    )
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


def build_ctx_from_name(row) -> str:
    name = row.get("Name", "") or ""
    gene = pick_gene(row)
    hgvsc = extract_hgvsc(name)
    hgvsp = extract_hgvsp(name)
    pcls = classify_protein_change(hgvsp)
    cfeat = parse_hgvsc_features(hgvsc)

    lines: List[str] = []
    if gene:
        lines.append(f"Gene: {gene}")
    if hgvsc:
        lines.append(f"HGVS.c: {hgvsc}")
    if hgvsp:
        lines.append(f"HGVS.p: {hgvsp}")
    cons = pcls or (
        "splice"
        if cfeat["splice"]
        else (cfeat["kind"] if cfeat["kind"] != "other" else "")
    )
    if cons:
        lines.append(f"Consequence: {cons}")
    if cfeat["region"] != "unknown":
        lines.append(f"Region: {cfeat['region']}")
    if cfeat["splice"]:
        lines.append("Splice-site: yes")
    if cfeat["indel_len"] > 0:
        lines.append(f"Indel length: {cfeat['indel_len']}")
    if cfeat["frameshift_guess"] is not None:
        lines.append(
            f"Frameshift (heuristic): {'yes' if cfeat['frameshift_guess'] else 'no'}"
        )
    if cfeat["snv_change"]:
        lines.append(f"SNV change: {cfeat['snv_change']}")
    return "\n".join(lines) if lines else "Variant context provided."


def row_to_example(meta_row) -> Tuple[str, str]:
    """
    Training target is the bare label string: "Benign" or "Pathogenic".
    No JSON, no rationale.
    """
    ctx = build_ctx_from_name(meta_row)
    prompt = PROMPT_TMPL.format(ctx=ctx)
    y = meta_row.get("_y", None)
    if y is None:
        y = clinsig_to_binary(meta_row.get("ClinicalSignificance", ""))
    cls = "Pathogenic" if y == 1 else "Benign"
    target = cls  # <-- only the label string
    return prompt, target


class CondTextDataset(Dataset):
    """
    Each item:
      - cond_vec: normalized [DNA ⊕ GO] vector (float32)
      - target: assistant label string
      - prompt: user prompt string (chat-template applied in collator)
    """

    def __init__(self, seq_ds, tokenizer=None):
        t0 = time.time()
        self.seq_ds = seq_ds
        self.meta = seq_ds.meta
        self.D_cond = seq_ds.D_eff + seq_ds.D_go
        print(f"[DEBUG] Dataset ready. Build time: {time.time()-t0:.2f}s")

    def __len__(self):
        return len(self.seq_ds)

    def __getitem__(self, i: int):
        x, _, _ = self.seq_ds[i]
        cond = x.numpy().astype("float32")
        cond /= np.linalg.norm(cond) + 1e-6
        prompt, target = row_to_example(self.meta.iloc[i])
        return {"cond_vec": cond, "prompt": prompt, "target": target}


def make_collator(tokenizer, max_len: int = 384):
    # tiny helpers for label token upweight (optional)
    tok_B = tokenizer.encode("Benign", add_special_tokens=False)
    tok_P = tokenizer.encode("Pathogenic", add_special_tokens=False)

    def _find_subseq(where_ids, what_ids, start_at):
        n, m = len(where_ids), len(what_ids)
        if m == 0 or n == 0 or n < m:
            return -1
        for i in range(start_at, n - m + 1):
            if where_ids[i : i + m] == what_ids:
                return i
        return -1

    def collate(batch):
        t0 = time.time()
        prompts = [str(b.get("prompt", "")) for b in batch]
        targets = [
            str(b.get("target", "") or "") for b in batch
        ]  # "Benign"/"Pathogenic"
        prompt_strings, full_strings = build_chat_strings(tokenizer, prompts, targets)

        enc_prompt = tokenizer(
            prompt_strings,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        enc_full = tokenizer(
            full_strings,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        input_ids = enc_full["input_ids"]
        attn_mask = enc_full["attention_mask"]
        labels = input_ids.clone()
        prompt_lens = enc_prompt["attention_mask"].sum(dim=1).tolist()
        for i, p_len in enumerate(prompt_lens):
            labels[i, : int(p_len)] = -100  # mask system+user

        # label token upweighting (since assistant is only the label, this mostly covers all assistant tokens)
        label_weights = torch.ones_like(labels, dtype=torch.float32)
        for i, p_len in enumerate(prompt_lens):
            label_weights[i, : int(p_len)] = 0.0  # no weight for prompt
            ids = input_ids[i].tolist()
            start = int(p_len)
            jB = _find_subseq(ids, tok_B, start)
            jP = _find_subseq(ids, tok_P, start)
            j = jB if (jB != -1 and (jP == -1 or jB <= jP)) else jP
            if j != -1:
                L = len(tok_B) if j == jB else len(tok_P)
                label_weights[i, j : j + L] *= 5.0  # boost classification span

        cond_vec = torch.from_numpy(
            np.stack([b["cond_vec"] for b in batch]).astype("float32")
        )
        dt = time.time() - t0
        if dt > 0.2:
            print(
                f"[DEBUG] Collator latency: {dt:.3f}s batch_size={len(batch)} (tokenizer likely the bottleneck)"
            )
        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
            "label_weights": label_weights,
            "cond_vec": cond_vec,
        }

    return collate


# ============================================================
# Model: Qwen + projector (registered submodule)
# ============================================================


class CondProjector(nn.Module):
    """Project [DNA ⊕ GO] -> K x hidden_size so we can prepend K virtual tokens."""

    def __init__(self, d_in: int, d_out: int, k: int = 4):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out * k, bias=True),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        y = self.net(x)  # [B, k*d_out]
        return y.view(B, self.k, -1)  # [B, k, d_out]


@dataclass
class LLMConfig:
    model_id: str = "Qwen/Qwen3-4B-Thinking-2507"
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
    num_workers: int = 0  # avoid pickling, keep HF threads
    prefetch_factor: int = 2
    persistent_workers: bool = False
    pin_memory: bool = True
    group_by_length: bool = False
    drop_last: bool = True
    seed: int = 42


def build_qwen_with_lora(cfg: LLMConfig, D_cond: int):
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print("[DEBUG] Loading tokenizer:", cfg.model_id)
    tok = AutoTokenizer.from_pretrained(
        cfg.model_id, use_fast=True, trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    print("[DEBUG] Loading 4-bit base model:", cfg.model_id)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        trust_remote_code=True,
        quantization_config=quant_cfg,
        device_map="auto",
    )
    try:
        base.config.pretraining_tp = 1
    except Exception:
        pass

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    base = prepare_model_for_kbit_training(base)
    base = get_peft_model(base, lora)

    hidden = base.config.hidden_size
    projector = CondProjector(D_cond, hidden, k=cfg.n_cond_tokens).to(
        next(base.parameters()).device
    )
    base.add_module("cond_projector", projector)
    print("[DEBUG] Model ready. Hidden:", hidden, "Cond tokens:", cfg.n_cond_tokens)
    return tok, base, projector


# ============================================================
# Custom Trainer: prepend K conditioning tokens
# ============================================================


class CondTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = dict(inputs)
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask", None)
        labels = inputs.pop("labels", None)
        label_weights = inputs.pop("label_weights", None)
        cond_vec = inputs.pop("cond_vec")

        txt_emb = model.get_input_embeddings()(input_ids)  # [B,T,H]
        cond_vec = cond_vec.to(device=txt_emb.device, dtype=txt_emb.dtype)
        cond_emb = model.cond_projector(cond_vec)  # [B,K,H]
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
        else:
            label_weights = None

        out = model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False
        )
        logits = out.logits

        if labels is None:
            loss = (
                out.loss
                if hasattr(out, "loss")
                else torch.tensor(0.0, device=logits.device)
            )
        else:
            # standard LM shift
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[
                :,
                1:,
            ].contiguous()
            if label_weights is not None:
                shift_w = label_weights[
                    :,
                    1:,
                ].contiguous()
            else:
                shift_w = torch.ones_like(shift_labels, dtype=torch.float32)

            loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
            loss_tok = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view_as(shift_labels)

            valid = (shift_labels != -100).float()
            loss = (loss_tok * shift_w * valid).sum() / valid.sum().clamp_min(1.0)

        return (loss, out) if return_outputs else loss


# ============================================================
# Inference utilities: label scoring, rationale->label (two-pass)
# ============================================================


@torch.inference_mode()
def _build_prompt_inputs_embeds(
    model, tokenizer, prompt_text: str, cond_vec_np: np.ndarray
):
    """Return (inputs_embeds, attention_mask, prompt_len_tokens) for system+user+<assistant> start,
    with K conditioning embeddings already prepended to inputs_embeds.
    """
    dev = next(model.parameters()).device
    msgs = [
        {"role": "system", "content": CHAT_SYSTEM},
        {"role": "user", "content": prompt_text},
    ]
    chat_str = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer([chat_str], return_tensors="pt").to(dev)

    txt_emb = model.get_input_embeddings()(enc["input_ids"])  # [1, T, H]

    cond = cond_vec_np.astype("float32")
    cond /= np.linalg.norm(cond) + 1e-6
    cond = torch.from_numpy(cond).unsqueeze(0).to(dev, dtype=txt_emb.dtype)  # [1, D]
    cond_emb = model.cond_projector(cond)  # [1, K, H]

    inputs_embeds = torch.cat([cond_emb, txt_emb], dim=1)  # [1, K+T, H]
    attn = enc["attention_mask"]
    K = cond_emb.size(1)
    ones = torch.ones((attn.size(0), K), dtype=attn.dtype, device=attn.device)
    attn = torch.cat([ones, attn], dim=1)
    return inputs_embeds, attn, enc["input_ids"].size(1)  # prompt_len=T (without K)


def _encode_label_variants(tokenizer, label_text: str):
    ids_a = tokenizer.encode(label_text, add_special_tokens=False)
    ids_b = tokenizer.encode(" " + label_text, add_special_tokens=False)
    out, seen = [], set()
    for ids in (ids_a, ids_b):
        t = tuple(ids)
        if ids and t not in seen:
            out.append(ids)
            seen.add(t)
    return out


@torch.inference_mode()
def score_label_logprob(
    model, tokenizer, prompt_text: str, cond_vec_np: np.ndarray, label_text: str
) -> float:
    """log P(label_text | prompt, cond) using teacher-forced scoring of the label tokens."""
    inputs_embeds, attn, _ = _build_prompt_inputs_embeds(
        model, tokenizer, prompt_text, cond_vec_np
    )
    dev = inputs_embeds.device
    E = model.get_input_embeddings()

    best_lp = None
    for label_ids in _encode_label_variants(tokenizer, label_text):
        lab_ids = torch.tensor([label_ids], device=dev)
        lab_emb = E(lab_ids)  # [1, L, H]

        full_emb = torch.cat([inputs_embeds, lab_emb], dim=1)
        full_attn = torch.cat(
            [
                attn,
                torch.ones(
                    (attn.size(0), lab_emb.size(1)),
                    dtype=attn.dtype,
                    device=attn.device,
                ),
            ],
            dim=1,
        )

        out = model(inputs_embeds=full_emb, attention_mask=full_attn, use_cache=False)
        logits = out.logits  # [1, S, V]

        L = lab_emb.size(1)
        label_logits = logits[:, -L - 1 : -1, :]  # last L predictions
        logprobs = torch.log_softmax(label_logits, dim=-1)
        token_lp = logprobs.gather(-1, lab_ids.unsqueeze(-1)).squeeze(-1)  # [1, L]
        lp = float(token_lp.sum().item())
        if (best_lp is None) or (lp > best_lp):
            best_lp = lp

    return best_lp if best_lp is not None else -1e9


@torch.inference_mode()
def predict_label_with_probs(
    model, tokenizer, prompt_text: str, cond_vec_np: np.ndarray
) -> Dict[str, Any]:
    lp_b = score_label_logprob(model, tokenizer, prompt_text, cond_vec_np, "Benign")
    lp_p = score_label_logprob(model, tokenizer, prompt_text, cond_vec_np, "Pathogenic")
    logits = torch.tensor([lp_b, lp_p])
    probs = torch.softmax(logits, dim=-1).tolist()
    label = "Benign" if probs[0] >= probs[1] else "Pathogenic"
    return {
        "label": label,
        "probs": {"Benign": probs[0], "Pathogenic": probs[1]},
        "logprobs": {"Benign": float(lp_b), "Pathogenic": float(lp_p)},
    }


@torch.inference_mode()
def generate_rationale_then_label(
    model,
    tokenizer,
    prompt_text: str,
    cond_vec_np: np.ndarray,
    *,
    rat_tokens: int = 80,
    temperature: float = 0.4,
    top_p: float = 0.9,
) -> Dict[str, Any]:
    """Two-pass inference: (1) short rationale; (2) pick label via scoring."""
    # Build prompt+cond embeds
    inputs_embeds, attn, _ = _build_prompt_inputs_embeds(
        model, tokenizer, prompt_text, cond_vec_np
    )

    # ---- Pass 1: rationale (free-form short decode) ----
    gen_rat = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_new_tokens=rat_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    text_full = tokenizer.decode(gen_rat[0], skip_special_tokens=True)
    rationale = strip_think(text_full).strip()

    # ---- Pass 2: score labels given the same prompt+cond (no rationale fed back) ----
    pred = predict_label_with_probs(model, tokenizer, prompt_text, cond_vec_np)
    return {
        "classification": pred["label"],
        "probs": pred["probs"],
        "rationale": rationale,
    }


# ============================================================
# Quick eval (sklearn) using label scoring (no JSON)
# ============================================================


def _true_label_from_meta(row) -> int | None:
    y = row.get("_y", None)
    if y is not None:
        try:
            return int(y)
        except Exception:
            pass
    return clinsig_to_binary(row.get("ClinicalSignificance", ""))


def quick_eval_pathogenicity_binary(
    model, tokenizer, val_seq_ds, n: int = 512, seed: int = 0
) -> Dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
        classification_report,
    )

    rng = np.random.default_rng(seed)
    idxs = rng.choice(
        len(val_seq_ds), size=min(n, len(val_seq_ds)), replace=False
    ).tolist()
    enable_fast_generate(model, tokenizer)

    y_true, y_pred, examples = [], [], []
    for i in idxs:
        x, _, _ = val_seq_ds[i]
        cond = x.numpy()
        row = val_seq_ds.meta.iloc[i]
        y = _true_label_from_meta(row)
        if y is None:
            continue
        ctx = build_ctx_from_name(row)
        prompt = PROMPT_TMPL.format(ctx=ctx)
        out = predict_label_with_probs(model, tokenizer, prompt, cond)
        cls = out["label"]
        yhat = 1 if cls == "Pathogenic" else 0
        y_true.append(int(y))
        y_pred.append(int(yhat))
        if len(examples) < 8:
            examples.append(
                {"idx": i, "truth": int(y), "pred": int(yhat), "probs": out["probs"]}
            )

    if not y_true:
        return {
            "n": 0,
            "note": "No labeled samples found in val meta (_y or mappable ClinicalSignificance).",
        }

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=BIN_LABELS, digits=3
    )
    return {
        "n": len(y_true),
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": cm,
        "report": report,
        "samples": examples,
    }


# ============================================================
# SMOKE CHECKS
# ============================================================


def _pct(x, n):
    return f"{100.0*x/n:.1f}%" if n else "n/a"


def smoke_checks(
    seq_ds_dict: Dict[str, object], tokenizer, cfg: LLMConfig, precompute_used: bool
):
    print("\n========== SMOKE CHECKS ==========")
    total_train = len(seq_ds_dict["train"])
    total_val = len(seq_ds_dict["val"])
    print(f"[SMOKE] Train size: {total_train}  |  Val size: {total_val}")

    def label_stats(ds):
        ys = []
        for i in range(min(len(ds), 5000)):
            row = ds.meta.iloc[i]
            y = _true_label_from_meta(row)
            if y is not None:
                ys.append(int(y))
        if not ys:
            return {"n": 0, "pos": 0, "neg": 0}
        ys = np.array(ys)
        return {"n": int(ys.size), "pos": int(ys.sum()), "neg": int((1 - ys).sum())}

    tr_stats = label_stats(seq_ds_dict["train"])
    va_stats = label_stats(seq_ds_dict["val"])
    print(
        f"[SMOKE] Label coverage (first ≤5k): train n={tr_stats['n']} (+:{tr_stats['pos']}, -:{tr_stats['neg']}), "
        f"val n={va_stats['n']} (+:{va_stats['pos']}, -:{va_stats['neg']})"
    )

    for split in ["train", "val"]:
        ds = seq_ds_dict[split]
        xs = []
        bad = 0
        for i in range(min(2000, len(ds))):
            x, _, _ = ds[i]
            v = x.numpy()
            if not np.all(np.isfinite(v)):
                bad += 1
            xs.append(np.linalg.norm(v))
        xs = np.array(xs) if xs else np.array([0.0])
        print(
            f"[SMOKE] {split} cond-norm: mean={xs.mean():.3f} std={xs.std():.3f} min={xs.min():.3f} "
            f"max={xs.max():.3f}  non-finite={bad}"
        )

    row0 = seq_ds_dict["train"].meta.iloc[0]
    ctx0 = build_ctx_from_name(row0)
    p0 = PROMPT_TMPL.format(ctx=ctx0)
    pr, fu = build_chat_strings(tokenizer, [p0], ["Benign"])
    print("[SMOKE] Example context:\n", ctx0)
    print(
        "[SMOKE] Chat prompt (truncated 200):", pr[0][:200].replace("\n", "\\n"), "..."
    )
    print(
        "[SMOKE] Chat full   (truncated 200):", fu[0][:200].replace("\n", "\\n"), "..."
    )

    enc = tokenizer(
        fu * 16,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.max_len,
    )
    lens = enc["attention_mask"].sum(dim=1).cpu().numpy()
    print(
        f"[SMOKE] Token lengths (full strings) min/mean/p95/max: "
        f"{lens.min():.0f}/{lens.mean():.1f}/{np.percentile(lens,95):.0f}/{lens.max():.0f} (max_len={cfg.max_len})"
    )

    tmp_ds = CondTextDataset(seq_ds_dict["train"], tokenizer=tokenizer)
    coll = make_collator(tokenizer, max_len=cfg.max_len)
    batch = [tmp_ds[i] for i in range(min(2, len(tmp_ds)))]
    out = coll(batch)
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"[SMOKE] Batch field {k}: shape={tuple(v.shape)} dtype={v.dtype}")

    print("[SMOKE] Tiny forward+loss on a micro-batch ...")
    return out


# ============================================================
# Orchestrator
# ============================================================


@dataclass
class LLMRunArgs:
    model_id: str = "Qwen/Qwen3-4B-Thinking-2507"
    epochs: int = 1
    max_len: int = 384
    per_device_bs: int = 2
    grad_accum: int = 16
    lr: float = 3e-4


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


def run_llm_pipeline(
    out_dir: str,
    seq_ds: Dict[str, object],
    *,
    model_id: str = "Qwen/Qwen3-4B-Thinking-2507",
    epochs: int = 1,
    max_len: int = 256,
    per_device_bs: int = 16,
    grad_accum: int = 8,
    lr: float = 3e-4,
    seed: int = 42,
    num_workers: int = 0,  # This does not work
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    pin_memory: bool = True,
    group_by_length: bool = False,
    drop_last: bool = True,
) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    _set_all_seeds(seed)

    cfg = LLMConfig(
        model_id=model_id,
        out_dir=out_dir,
        max_len=max_len,
        lr=lr,
        epochs=epochs,
        grad_accum=grad_accum,
        per_device_bs=per_device_bs,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        group_by_length=group_by_length,
        drop_last=drop_last,
        seed=seed,
    )

    tok, base_model, projector = build_qwen_with_lora(
        cfg, seq_ds["train"].D_eff + seq_ds["train"].D_go
    )
    base_model.config.use_cache = False  # training: turn off cache

    print("[DEBUG] Building CondTextDataset(train/val) ...")
    train_ds = CondTextDataset(seq_ds["train"], tokenizer=tok)
    val_ds = CondTextDataset(seq_ds["val"], tokenizer=tok)

    collator = make_collator(tok, max_len=cfg.max_len)

    batch_out = smoke_checks(seq_ds, tok, cfg, precompute_used=False)
    base_model.train(False)
    with torch.no_grad():
        txt_emb = base_model.get_input_embeddings()(
            batch_out["input_ids"].to(next(base_model.parameters()).device)
        )
        cond_emb = base_model.cond_projector(
            batch_out["cond_vec"].to(
                next(base_model.parameters()).device, dtype=txt_emb.dtype
            )
        )
        inputs_embeds = torch.cat([cond_emb, txt_emb], dim=1)
        attn = batch_out["attention_mask"].to(next(base_model.parameters()).device)
        B, K = cond_emb.size(0), cond_emb.size(1)
        ones = torch.ones((B, K), dtype=attn.dtype, device=attn.device)
        attn = torch.cat([ones, attn], dim=1)
        labels = torch.cat(
            [
                torch.full(
                    (B, K), -100, dtype=batch_out["labels"].dtype, device=attn.device
                ),
                batch_out["labels"].to(attn.device),
            ],
            dim=1,
        )
        out = base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            labels=labels,
            use_cache=False,
        )
        print(
            f"[SMOKE] Forward OK. loss={float(out.loss):.4f}  logits.shape={tuple(out.logits.shape)}"
        )
    base_model.train(True)

    args = TrainingArguments(
        output_dir=cfg.out_dir,
        per_device_train_batch_size=cfg.per_device_bs,
        per_device_eval_batch_size=cfg.per_device_bs,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        weight_decay=0.01,
        num_train_epochs=cfg.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=50,
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        bf16=cfg.bf16,
        gradient_checkpointing=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=cfg.num_workers,  # 0
        dataloader_pin_memory=cfg.pin_memory,
        dataloader_prefetch_factor=None,  # ignored when workers=0
        dataloader_persistent_workers=False,
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
    )

    dev = next(base_model.parameters()).device
    print(f"[INFO] Device: {dev} | CUDA={torch.cuda.is_available()} | bf16={cfg.bf16}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"[INFO] GPU mem: free={free/1e9:.2f}G total={total/1e9:.2f}G")
        print(
            "[INFO] CUDA capability available. TF32:",
            torch.backends.cuda.matmul.allow_tf32,
        )

    print("[DEBUG] Probing first training batch...")
    t0 = time.time()
    train_dl = trainer.get_train_dataloader()
    batch = next(iter(train_dl))
    dt = time.time() - t0
    print(f"[DEBUG] Got first batch in {dt:.2f}s")
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)}")

    trainer.train()

    base_model.save_pretrained(os.path.join(out_dir, "adapter"))
    torch.save(projector.state_dict(), os.path.join(out_dir, "projector.pt"))
    print("[INFO] Saved adapter + projector.")

    print("[INFO] Running quick eval on val split ...")
    eval_res = quick_eval_pathogenicity_binary(base_model, tok, seq_ds["val"], n=512)
    print(
        "[INFO] Eval:",
        json.dumps(
            {
                k: v
                for k, v in eval_res.items()
                if k in ("n", "accuracy", "precision", "recall", "f1")
            },
            indent=2,
        ),
    )
    return {"eval": eval_res, "out_dir": out_dir}


# ============================================================
# Load a finetuned model + projector for inference-only use
# ============================================================

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


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

    # Load LoRA adapter on top of plain base
    print("[LOAD] Loading adapter via PeftModel.from_pretrained:", adapter_dir)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    # Attach projector (match dtype/device)
    emb = model.get_input_embeddings().weight
    target_dtype, target_device = emb.dtype, emb.device
    projector = CondProjector(D_cond, model.config.hidden_size, k=4).to(
        device=target_device, dtype=target_dtype
    )
    model.add_module("cond_projector", projector)

    psd = torch.load(projector_path, map_location="cpu")
    miss, unexp = model.cond_projector.load_state_dict(psd, strict=False)
    print(
        f"[LOAD] Projector missing={len(miss)} unexpected={len(unexp)} dtype={next(projector.parameters()).dtype}"
    )

    # Ensure LoRA params are same dtype/device
    for n, p in model.named_parameters():
        if "lora_" in n and (p.dtype != target_dtype or p.device != target_device):
            p.data = p.data.to(dtype=target_dtype, device=target_device)

    model.config.use_cache = True  # for inference
    model.eval()
    return tok, model


# ============================================================
# Tiny inference smoke test (labels only; optional rationale)
# ============================================================


def quick_smoke_generate_labels(
    seq_ds: Dict[str, object],
    out_dir: str,
    n: int = 8,
    seed: int = 0,
    with_rationale: bool = False,
):
    from collections import Counter

    rng = np.random.default_rng(seed)
    adapter_dir = os.path.join(out_dir, "adapter")
    projector_path = os.path.join(out_dir, "projector.pt")

    D_cond = seq_ds["train"].D_eff + seq_ds["train"].D_go
    tok, model = load_finetuned_model(
        model_id="Qwen/Qwen3-4B-Thinking-2507",
        D_cond=D_cond,
        adapter_dir=adapter_dir,
        projector_path=projector_path,
    )
    enable_fast_generate(model, tok)

    val = seq_ds["val"]
    idxs = rng.choice(len(val), size=min(n, len(val)), replace=False).tolist()

    preds, raws = [], []
    for i in idxs:
        x, _, _ = val[i]
        cond = x.numpy()
        row = val.meta.iloc[i]
        ctx = build_ctx_from_name(row)
        prompt = PROMPT_TMPL.format(ctx=ctx)

        if with_rationale:
            out = generate_rationale_then_label(model, tok, prompt, cond, rat_tokens=80)
            cls, rat = out["classification"], out["rationale"]
        else:
            out = predict_label_with_probs(model, tok, prompt, cond)
            cls, rat = out["label"], ""

        preds.append((i, _true_label_from_meta(row), cls, rat))
        raws.append(out)

    print("[SMOKE] Pred label counts:", Counter([p[2] for p in preds]))
    for (i, y, cls, rat), o in zip(preds, raws):
        if with_rationale:
            print(f"- idx={i} truth={y} pred={cls}  rationale[:120]={rat[:120]!r}")
        else:
            print(f"- idx={i} truth={y} pred={cls}  probs={o['probs']}")
    return {"n": len(idxs), "preds": preds}
