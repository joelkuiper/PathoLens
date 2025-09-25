# src/llm/eval.py
# -*- coding: utf-8 -*-
# Batched evaluation + AUC + feather + optional label→rationale.
# Notes:
# - Ground truth is NEVER inserted into prompts or scoring. It is read only to compute metrics.
# - Rationales are generated AFTER predicting the label, and are conditioned on the PREDICTED label only.
# - LoRA + conditioning projector remain active for both scoring and rationale generation.

import os
import re
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from src.llm.config import PROMPT_TMPL, BIN_LABELS
from src.llm.load import load_finetuned_model
from src.llm.modeling import enable_fast_generate
from src.llm.infer import (
    encode_label_variants,
    generate_rationale_with_seed,
    predict_label_with_probs,
    prepare_prompt_embeddings,
    score_label_word_logprobs,
)
from .context import build_ctx_from_row

# Regexes used to sanity-check that the prompt actually looks blank
_RX_HGVS_P = re.compile(r"(?i)\bp\.(?!\[)[A-Za-z0-9_=*]+")
_RX_HGVS_C = re.compile(r"(?i)\bc\.(?!\[)[A-Za-z0-9_+\-*>=<]+")
_RX_LABELS = re.compile(r"(?i)\b(benign|pathogenic)\b")


def _probe_prompt_strings(prompts: List[str], tag: str = "probe"):
    """Print stats that prove emptiness (or not) of prompts."""
    n = len(prompts)
    if n == 0:
        print(f"[{tag}] no prompts")
        return
    # Basic stats
    lengths = [len(p) for p in prompts]
    n_blank = sum(1 for p in prompts if p.strip() == "")
    n_hgvsc = sum(1 for p in prompts if _RX_HGVS_C.search(p or ""))
    n_hgvsp = sum(1 for p in prompts if _RX_HGVS_P.search(p or ""))
    n_labels = sum(1 for p in prompts if _RX_LABELS.search(p or ""))

    print(
        f"[{tag}] prompts={n} blank={n_blank} "
        f"len[min/med/max]={min(lengths)}/{int(np.median(lengths))}/{max(lengths)} "
        f"HGVS.c hits={n_hgvsc} HGVS.p hits={n_hgvsp} label-word hits={n_labels}"
    )
    # Show up to 3 examples (hash + visible)
    for i in range(min(3, n)):
        s = prompts[i] or ""
        sha = _sha10(s)
        head = s.replace("\n", "\\n")[:180]
        print(f"[{tag}] i={i} sha={sha} text='{head}{'…' if len(s) > 180 else ''}'")


def _sha10(s: str) -> str:
    import hashlib

    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:10]


# -----------------------
# Main evaluator (compat with eval_old)
# -----------------------
@torch.inference_mode()
def evaluate_split_batched(
    seq_ds: Dict[str, Any],
    out_dir: str,
    split: str = "val",
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507",
    max_n: Optional[int] = None,
    batch_size: int = 64,
    sample_rationales: int = 0,  # generate AFTER predicting label
    rationale_tokens: int = 48,
    rationale_temp: float = 0.9,
    rationale_top_p: float = 0.92,
    guard_prompts: bool = True,
) -> Dict[str, Any]:
    """
    IMPORTANT: We do NOT insert ground-truth anywhere into the prompt or scoring.
    """
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        roc_auc_score,
        average_precision_score,
    )

    adapter_dir = os.path.join(out_dir, "adapter")
    projector_path = os.path.join(out_dir, "projector")

    train_split = seq_ds["train"]
    D_eff = int(train_split.D_eff)
    D_go = int(getattr(train_split, "D_go", 0) or 0)
    D_prot = int(train_split.D_prot)
    D_cond = D_eff + D_go + D_prot
    print(
        "[INFO] Eval: "
        f"D_cond={D_cond} (eff={D_eff} go={D_go} prot={D_prot})"
    )

    tok, model = load_finetuned_model(model_id, D_cond, adapter_dir, projector_path)
    enable_fast_generate(model, tok)

    ds = seq_ds[split]
    N_all = len(ds)

    # Evaluate all rows
    idxs = list(range(N_all))
    if max_n is not None:
        idxs = idxs[:max_n]
    N = len(idxs)
    if N == 0:
        return {"n": 0, "note": f"No samples in split '{split}'."}

    # Prepare tokenizations for labels once
    cand = {
        "Benign": encode_label_variants(tok, "Benign"),
        "Pathogenic": encode_label_variants(tok, "Pathogenic"),
    }

    y_true: List[int] = []
    y_pred: List[int] = []
    prob_pathogenic: List[float] = []

    # Batched scoring — matches eval_old path
    for bi in tqdm(range(0, N, batch_size), desc=f"Scoring [{split}]", unit="batch"):
        batch_idxs = idxs[bi : bi + batch_size]

        # Build prompts/conds WITHOUT ground-truth
        prompts, conds, y_batch = [], [], []
        for i in batch_idxs:
            x, *_ = ds[i]
            conds.append(x.numpy())

            # normal context build
            ctx = build_ctx_from_row(ds.meta.iloc[i])

            if getattr(ds, "_force_blank_prompts", False):
                prompt_text = getattr(
                    ds,
                    "_force_blank_prompt_template",
                    PROMPT_TMPL.format(ctx=""),
                )
            else:
                prompt_text = PROMPT_TMPL.format(ctx=ctx)

            prompts.append(prompt_text)
            y_batch.append(int(ds.meta.iloc[i]["_y"]))

        conds = np.stack(conds, axis=0)
        inp_emb, attn = prepare_prompt_embeddings(model, tok, prompts, conds)

        lp_b = score_label_word_logprobs(
            model,
            tok,
            inp_emb,
            attn,
            "Benign",
            label_variants=cand["Benign"],
        )
        lp_p = score_label_word_logprobs(
            model,
            tok,
            inp_emb,
            attn,
            "Pathogenic",
            label_variants=cand["Pathogenic"],
        )

        # Two-class normalization → calibrated probability for Pathogenic
        denom = np.logaddexp(lp_b, lp_p)
        prob_b = np.exp(lp_b - denom)
        prob_p = np.exp(lp_p - denom)

        yhat_batch = (lp_p > lp_b).astype(np.int32)

        y_true.extend(y_batch)
        y_pred.extend(yhat_batch.tolist())
        prob_pathogenic.extend(np.clip(prob_p, 0.0, 1.0).tolist())

    # Metrics
    acc = float(np.mean(np.array(y_pred) == np.array(y_true))) if y_true else 0.0
    p, r, f1 = torchmetrics_safe_prf(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=BIN_LABELS, digits=3
    )
    try:
        auc_roc = float(roc_auc_score(y_true, prob_pathogenic))
    except Exception:
        auc_roc = float("nan")
    try:
        auc_pr = float(average_precision_score(y_true, prob_pathogenic))
    except Exception:
        auc_pr = float("nan")

    # Save predictions alongside the original meta
    try:
        import pyarrow.feather as feather

        df = ds.meta.iloc[idxs].copy().reset_index(drop=True)
        df["_pred_pathogenic_prob"] = prob_pathogenic
        df["_pred_label"] = [int(x) for x in y_pred]
        out_path = os.path.join(out_dir, f"{split}_scored.feather")
        feather.write_feather(df, out_path)
        preds_path = out_path
    except Exception as e:
        preds_path = f"(save failed: {e})"

    out = {
        "split": split,
        "n": len(y_true),
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "confusion_matrix": cm,
        "report": report,
        "predictions_path": preds_path,
    }

    # Optional: rationales AFTER prediction (LoRA ON, cond used)
    if sample_rationales > 0:
        rng = np.random.default_rng(0)
        pool = rng.choice(idxs, size=min(sample_rationales, len(idxs)), replace=False)
        rats = []
        for i in pool:
            x, *_ = ds[i]
            cond_vec = x.numpy()
            row = ds.meta.iloc[i]
            ctx = build_ctx_from_row(row)
            if getattr(ds, "_force_blank_prompts", False):
                prompt = getattr(
                    ds,
                    "_force_blank_prompt_template",
                    PROMPT_TMPL.format(ctx=""),
                )
            else:
                prompt = PROMPT_TMPL.format(ctx=ctx)

            pred = predict_label_with_probs(model, tok, prompt, cond_vec)
            label_word = pred["label"]

            rat, _ = generate_rationale_with_seed(
                model,
                tok,
                prompt,
                cond_vec,
                label_word,
                seed_text=" Because",
                temperature=rationale_temp,
                top_p=rationale_top_p,
                max_new_tokens=rationale_tokens,
                min_new_tokens=12,
                repetition_penalty=1.05,
            )
            rats.append({"idx": int(i), "pred_label": label_word, "rationale": rat})
        out["rationales"] = rats

    return out


def torchmetrics_safe_prf(y_true, y_pred):
    """Helper: precision/recall/F1 with zero_division=0 and floats."""
    from sklearn.metrics import precision_recall_fscore_support

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return float(p), float(r), float(f1)


# -----------------------
# Convenience driver
# -----------------------
def run_all(
    seq_ds,
    out_dir: str = "artifacts/qwen3_hpo_lora",
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507",
    max_n_val: Optional[int] = None,
    max_n_test: Optional[int] = None,
    batch_size: int = 64,
    sample_rationales: int = 4,
):
    res_val = evaluate_split_batched(
        seq_ds,
        out_dir,
        split="val",
        model_id=model_id,
        max_n=max_n_val,
        batch_size=batch_size,
        sample_rationales=sample_rationales,
    )
    print("\nVAL metrics:")
    print(
        {
            "n": res_val["n"],
            "accuracy": res_val["accuracy"],
            "precision": res_val["precision"],
            "recall": res_val["recall"],
            "f1": res_val["f1"],
            "auc_roc": res_val.get("auc_roc"),
            "auc_pr": res_val.get("auc_pr"),
        }
    )
    print("Confusion:", res_val["confusion_matrix"])
    if res_val.get("predictions_path"):
        print("Saved VAL predictions to:", res_val["predictions_path"])

    res_test = evaluate_split_batched(
        seq_ds,
        out_dir,
        split="test",
        model_id=model_id,
        max_n=max_n_test,
        batch_size=batch_size,
        sample_rationales=sample_rationales,
    )
    print("\nTEST metrics:")
    print(
        {
            "n": res_test["n"],
            "accuracy": res_test["accuracy"],
            "precision": res_test["precision"],
            "recall": res_test["recall"],
            "f1": res_test["f1"],
            "auc_roc": res_test.get("auc_roc"),
            "auc_pr": res_test.get("auc_pr"),
        }
    )
    print("Confusion:", res_test["confusion_matrix"])
    if res_test.get("predictions_path"):
        print("Saved TEST predictions to:", res_test["predictions_path"])

    return {"val": res_val, "test": res_test}
