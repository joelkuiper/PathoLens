# src/ablation_probe.py
# -*- coding: utf-8 -*-
"""
Ablation MLP probe for PathoLens
---------------------------------------------------

Trains a small MLP on features from SequenceTowerDataset and evaluates:
  - dna
  - prot
  - dna+prot

Assumes SequenceTowerDataset packs features in this order per sample:
  [ dna_eff | prot_eff ]

Relies on dataset attributes:
  ds.D_eff, ds.D_prot

Usage:
    from src.mlp_test import run_ablation_probes, print_probe_table
    from src.pipeline.datasets import load_manifest_datasets

    cfg, manifest, datasets = load_manifest_datasets("configs/pipeline.local.toml")
    results = run_ablation_probes(datasets)
    print_probe_table(results)

See ``load_probe_datasets`` / ``run_probes_from_config`` below for convenience
wrappers that accept the unified pipeline configuration + manifest.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    confusion_matrix,
    balanced_accuracy_score,
)

from src.pipeline import PipelineConfig, PipelineManifest
from src.pipeline.datasets import load_manifest_datasets
from src.seq_dataset import SequenceTowerDataset


SequenceDatasetMap = Dict[str, SequenceTowerDataset]


# --------------------------
# Metrics helpers
# --------------------------
def _safe_auc(y, p, fn, default=np.nan):
    try:
        return float(fn(y, p))
    except Exception:
        return default


def _threshold_metrics(y_true, p_prob, thr: float) -> Dict[str, float]:
    yhat = (p_prob >= thr).astype(int)
    tn, fp, fn_, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    prec = precision_score(y_true, yhat, zero_division=0)
    rec = recall_score(y_true, yhat, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return dict(
        acc=accuracy_score(y_true, yhat),
        bal_acc=balanced_accuracy_score(y_true, yhat),
        f1=f1_score(y_true, yhat, zero_division=0),
        precision=prec,
        recall=rec,
        specificity=spec,
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn_),
    )


def _find_best_threshold(y_val, p_val, *, grid=201, criterion="f1"):
    """Pick threshold maximizing a metric on val (default: F1)."""
    ts = np.linspace(0.01, 0.99, grid)
    best_t, best_v = 0.5, -1.0
    for t in ts:
        if criterion == "f1":
            v = f1_score(y_val, (p_val >= t).astype(int), zero_division=0)
        elif criterion == "youden":
            yhat = (p_val >= t).astype(int)
            v = balanced_accuracy_score(y_val, yhat)
        else:
            raise ValueError("criterion must be 'f1' or 'youden'")
        if v > best_v:
            best_v, best_t = v, t
    return float(best_t), float(best_v)


# --------------------------
# Feature extraction
# --------------------------
def _feat_matrix(split, desc) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in tqdm(range(len(split)), desc=f"{desc}: features", unit="row", leave=False):
        v, *_ = split[i]
        v = v if isinstance(v, np.ndarray) else v.numpy()
        X.append(v)
        # label column is standardized to '_y' in meta
        y.append(int(split.meta.iloc[i]["_y"]))
    return np.vstack(X).astype(np.float32), np.array(y, dtype=np.int64)


def _slice_blocks(
    X: np.ndarray, D_eff: int, D_prot: int
) -> Tuple[np.ndarray, np.ndarray]:
    dna = X[:, :D_eff]
    prot = X[:, D_eff : D_eff + D_prot]
    return dna, prot


# --------------------------
# Model
# --------------------------
class _MLP(nn.Module):
    def __init__(self, d_in, hidden=512, dropout=0.2):
        super().__init__()
        h2 = max(32, hidden // 2)
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# --------------------------
# Core probe for one feature mode
# --------------------------
def _run_single_probe(
    seq_ds: SequenceDatasetMap,
    feat_mode: str,
    *,
    hidden=512,
    dropout=0.2,
    epochs=20,
    bs=8192,
    lr=3e-3,
    weight_decay=1e-4,
    patience=4,
    device: Optional[str] = None,
    seed: int = 0,
    tune_threshold_on="f1",  # 'f1' or 'youden'
    opt_metric: str = "pr",  # 'pr' or 'roc'
) -> Dict[str, dict]:
    """
    Train/evaluate an MLP on a chosen feature combination.
    Returns a dict with 'val', 'test', 'config'.
    """
    # device & seed
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    D_eff = int(seq_ds["train"].D_eff)
    D_prot = int(seq_ds["train"].D_prot)

    # pull features
    Xtr_all, ytr = _feat_matrix(seq_ds["train"], "[train]")
    Xv_all, yv = _feat_matrix(seq_ds["val"], "[val]")
    Xt_all, yt = _feat_matrix(seq_ds["test"], "[test]")

    tr_dna, tr_prot = _slice_blocks(Xtr_all, D_eff, D_prot)
    va_dna, va_prot = _slice_blocks(Xv_all, D_eff, D_prot)
    te_dna, te_prot = _slice_blocks(Xt_all, D_eff, D_prot)

    use_dna = feat_mode in ("dna", "dna+prot")
    use_prot = feat_mode in ("prot", "dna+prot")

    # z-score each active block separately (fit on train)
    def zfit(X):
        sc = StandardScaler()
        Xz = sc.fit_transform(X).astype(np.float32)
        return Xz, sc

    def zapply(X, sc):
        return sc.transform(X).astype(np.float32)

    if use_dna:
        tr_dna, sc_dna = zfit(tr_dna)
        va_dna = zapply(va_dna, sc_dna)
        te_dna = zapply(te_dna, sc_dna)
    if use_prot:
        tr_prot, sc_pr = zfit(tr_prot)
        va_prot = zapply(va_prot, sc_pr)
        te_prot = zapply(te_prot, sc_pr)

    def assemble(A, C):
        parts: List[np.ndarray] = []
        if use_dna:
            parts.append(A)
        if use_prot:
            parts.append(C)
        if not parts:
            raise ValueError(f"No features selected for mode '{feat_mode}'")
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=1)

    Xtr = assemble(tr_dna, tr_prot)
    Xv = assemble(va_dna, va_prot)
    Xt = assemble(te_dna, te_prot)

    # tensors
    trX = torch.tensor(Xtr)
    vaX = torch.tensor(Xv)
    teX = torch.tensor(Xt)
    ytr_t = torch.tensor(ytr)
    yv_t = torch.tensor(yv)
    yt_t = torch.tensor(yt)

    # class weights
    pos = max(1, int(ytr.sum()))
    w_pos = len(ytr) / (2.0 * pos)
    w_neg = len(ytr) / (2.0 * max(1, (len(ytr) - pos)))

    # training loop
    model = _MLP(trX.shape[1], hidden=int(hidden), dropout=float(dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=2
    )

    best = (-np.inf, None)
    bad = 0

    def eval_probs(X, y_np):
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(X.to(device))).cpu().numpy()
        roc = _safe_auc(y_np, p, roc_auc_score)
        pr = _safe_auc(y_np, p, average_precision_score)
        return p, roc, pr

    for ep in range(1, int(epochs) + 1):
        model.train()
        perm = torch.randperm(trX.size(0))
        losses = []
        for s in tqdm(
            range(0, trX.size(0), int(bs)),
            desc=f"MLP {feat_mode} (epoch {ep}/{epochs})",
            unit="batch",
            leave=False,
        ):
            idx = perm[s : s + int(bs)]
            xb, yb = trX[idx].to(device), ytr_t[idx].float().to(device)
            logits = model(xb)
            w = torch.where(
                yb == 1.0,
                torch.tensor(w_pos, device=device),
                torch.tensor(w_neg, device=device),
            )
            loss = F.binary_cross_entropy_with_logits(logits, yb, weight=w)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        p_val, roc_val, pr_val = eval_probs(vaX, yv)
        metric_val = pr_val if opt_metric == "pr" else roc_val
        sched.step(metric_val)

        tqdm.write(
            f"[MLP {feat_mode}] ep{ep:02d} "
            f"loss={np.mean(losses):.4f} valROC={roc_val:.4f} valPR={pr_val:.4f}"
        )

        if metric_val > best[0] + 1e-5:
            best = (
                metric_val,
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                tqdm.write("[MLP] early stopping")
                break

    if best[1] is not None:
        model.load_state_dict(best[1])

    # final eval
    p_val, roc_val, pr_val = eval_probs(vaX, yv)
    p_test, roc_test, pr_test = eval_probs(teX, yt)

    thr_best, _ = _find_best_threshold(yv, p_val, criterion=tune_threshold_on)

    m_val_05 = _threshold_metrics(yv, p_val, 0.5)
    m_tst_05 = _threshold_metrics(yt, p_test, 0.5)
    m_val_bt = _threshold_metrics(yv, p_val, thr_best)
    m_tst_bt = _threshold_metrics(yt, p_test, thr_best)

    # attach aucs
    for d in (m_val_05, m_val_bt):
        d["roc"] = roc_val
        d["pr"] = pr_val
    for d in (m_tst_05, m_tst_bt):
        d["roc"] = roc_test
        d["pr"] = pr_test

    return {
        "val": {
            "thr_best": float(thr_best),
            "at_0p5": m_val_05,
            "at_best_thr": m_val_bt,
        },
        "test": {"at_0p5": m_tst_05, "at_best_thr": m_tst_bt},
        "config": dict(
            feat=feat_mode,
            hidden=hidden,
            dropout=dropout,
            epochs=epochs,
            bs=bs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            device=device,
            seed=seed,
            tune_threshold_on=tune_threshold_on,
            opt_metric=opt_metric,
        ),
    }


# --------------------------
# Public runner
# --------------------------
def run_ablation_probes(
    seq_ds: SequenceDatasetMap,
    *,
    modes: Optional[List[str]] = None,
    hidden=512,
    dropout=0.2,
    epochs=20,
    bs=8192,
    lr=3e-3,
    weight_decay=1e-4,
    patience=4,
    device: Optional[str] = None,
    seed: int = 0,
    tune_threshold_on="f1",
    opt_metric="pr",
) -> Dict[str, dict]:
    """
    Run a grid of ablation probes. Returns {mode: result_dict}.
    """
    D_prot = int(seq_ds["train"].D_prot)
    default_modes = ["dna", "prot", "dna+prot"]
    if modes is None:
        modes = default_modes

    results: Dict[str, dict] = {}
    for m in modes:
        results[m] = _run_single_probe(
            seq_ds,
            m,
            hidden=hidden,
            dropout=dropout,
            epochs=epochs,
            bs=bs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            device=device,
            seed=seed,
            tune_threshold_on=tune_threshold_on,
            opt_metric=opt_metric,
        )
    return results


# --------------------------
# Pretty-print
# --------------------------
def print_probe_table(results: Dict[str, dict]) -> None:
    """
    Pretty-print a table of results from run_ablation_probes().
    Shows metrics at 0.5 and at best threshold (from val).
    """

    def _fmt(x, nd=4):
        if x is None:
            return "-"
        try:
            return f"{x:.{nd}f}"
        except Exception:
            return str(x)

    modes = list(results.keys())

    print("\nAblation Probe Results")
    print("=" * 80)
    print(
        f"{'Mode':12} | {'Split':6} | {'Thr':>6} | {'Acc':>6} {'BalAcc':>7} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Spec':>6} {'ROC':>6} {'PR':>6}"
    )
    print("-" * 80)

    for m in modes:
        res = results[m]
        for split in ("val", "test"):
            thr_best = res["val"]["thr_best"]
            row05 = res[split]["at_0p5"]
            rowbt = res[split]["at_best_thr"]
            # 0.5
            print(
                f"{m:12} | {split:6} | {('0.5'):>6} | "
                f"{_fmt(row05['acc']):>6} {_fmt(row05['bal_acc']):>7} {_fmt(row05['f1']):>6} "
                f"{_fmt(row05['precision']):>6} {_fmt(row05['recall']):>6} {_fmt(row05['specificity']):>6} "
                f"{_fmt(row05['roc']):>6} {_fmt(row05['pr']):>6}"
            )
            # best
            print(
                f"{'':12} | {split:6} | {('best'):>6} | "
                f"{_fmt(rowbt['acc']):>6} {_fmt(rowbt['bal_acc']):>7} {_fmt(rowbt['f1']):>6} "
                f"{_fmt(rowbt['precision']):>6} {_fmt(rowbt['recall']):>6} {_fmt(rowbt['specificity']):>6} "
                f"{_fmt(rowbt['roc']):>6} {_fmt(rowbt['pr']):>6}"
            )
        print("-" * 80)


def load_probe_datasets(
    config_path: str | Path,
    *,
    manifest_path: Optional[str | Path] = None,
    make_label: bool = True,
    label_col: str = "_y",
) -> Tuple[PipelineConfig, PipelineManifest, SequenceDatasetMap]:
    """Load config + manifest and build datasets for the ablation probes."""

    cfg, manifest, datasets = load_manifest_datasets(
        config_path,
        manifest_path=manifest_path,
        make_label=make_label,
        label_col=label_col,
    )
    return cfg, manifest, datasets


def run_probes_from_config(
    config_path: str | Path,
    *,
    manifest_path: Optional[str | Path] = None,
    make_label: bool = True,
    label_col: str = "_y",
    **probe_kwargs,
) -> Tuple[
    PipelineConfig,
    PipelineManifest,
    SequenceDatasetMap,
    Dict[str, dict],
]:
    """Convenience wrapper: load datasets and run the ablation probes."""

    cfg, manifest, datasets = load_probe_datasets(
        config_path,
        manifest_path=manifest_path,
        make_label=make_label,
        label_col=label_col,
    )
    results = run_ablation_probes(datasets, **probe_kwargs)
    return cfg, manifest, datasets, results
