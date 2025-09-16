# probe_sgd.py
# Fast out-of-sample linear probes using SGD (mini-batch logistic regression).
# - Progress bars for feature building, epochs, and batches
# - Class balancing via sample_weight
# - Early stopping on val ROC-AUC (patience)
# - Evaluates dna / go / both

import numpy as np
from typing import Dict, Any, Tuple
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
)
from sklearn.utils import shuffle as sk_shuffle
from sklearn.linear_model import SGDClassifier


# -----------------------------
# Data utils
# -----------------------------
def _to_numpy(v) -> np.ndarray:
    if isinstance(v, np.ndarray):
        arr = v
    else:
        arr = v.detach().cpu().numpy() if hasattr(v, "detach") else v.numpy()
    return arr.astype(np.float32, copy=False)


def _feat_matrix(split, desc: str) -> Tuple[np.ndarray, np.ndarray]:
    n = len(split)
    v0, *_ = split[0]
    d = _to_numpy(v0).shape[0]
    X = np.empty((n, d), dtype=np.float32)
    y = np.empty((n,), dtype=np.int64)
    for i in tqdm(range(n), desc=f"{desc}: features", unit="row"):
        v, *_ = split[i]
        X[i] = _to_numpy(v)
        y[i] = int(split.meta.iloc[i]["_y"])
    return X, y


def _split_feats(X: np.ndarray, D_eff: int, D_go: int) -> Dict[str, np.ndarray]:
    dna = X[:, :D_eff]
    go = X[:, D_eff : D_eff + D_go]
    both = X
    return {"dna": dna, "go": go, "both": both}


def _balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    # inverse-frequency weights → mean ~ 1
    n = y.size
    pos = max(1, int(y.sum()))
    neg = max(1, n - pos)
    w_pos = n / (2.0 * pos)
    w_neg = n / (2.0 * neg)
    w = np.where(y == 1, w_pos, w_neg).astype(np.float32)
    return w


# -----------------------------
# SGD logistic trainer
# -----------------------------
def _train_sgd_logit(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xval: np.ndarray,
    yval: np.ndarray,
    *,
    batch_size: int = 8192,
    epochs: int = 10,
    lr: float = 0.01,
    alpha: float = 1e-4,  # L2 reg
    early_stop_patience: int = 2,
    shuffle_each_epoch: bool = True,
    random_state: int = 0,
) -> SGDClassifier:
    """
    Trains an SGDClassifier (logistic loss) with mini-batches and early stopping on val ROC-AUC.
    """
    classes = np.array([0, 1], dtype=np.int64)
    clf = SGDClassifier(
        loss="log_loss",
        alpha=alpha,
        learning_rate="optimal",  # with log_loss often better than 'constant'
        eta0=lr,
        penalty="l2",
        fit_intercept=True,
        random_state=random_state,
        # n_jobs is not used by SGDClassifier; it's inherently fast & single-threaded
    )

    # Precompute per-class weights → per-sample weights per batch
    w_full = _balanced_sample_weight(ytr)

    best_val = -np.inf
    best_state = None
    bad_epochs = 0

    n = Xtr.shape[0]
    idx_all = np.arange(n)

    for ep in range(1, epochs + 1):
        if shuffle_each_epoch:
            idx_all = sk_shuffle(idx_all, random_state=random_state + ep)

        pbar = tqdm(
            range(0, n, batch_size),
            desc=f"SGD (epoch {ep}/{epochs})",
            unit="batch",
            leave=False,
        )
        for start in pbar:
            end = min(start + batch_size, n)
            idx = idx_all[start:end]
            Xb = Xtr[idx]
            yb = ytr[idx]
            wb = w_full[idx]

            if start == 0 and ep == 1:
                clf.partial_fit(Xb, yb, classes=classes, sample_weight=wb)
            else:
                clf.partial_fit(Xb, yb, sample_weight=wb)

        # ---- validation ROC for early stopping ----
        try:
            p_val = clf.predict_proba(Xval)[:, 1]
        except AttributeError:
            # some scikit builds expose decision_function only until calibrated;
            # fallback: sigmoid
            df = clf.decision_function(Xval)
            p_val = 1.0 / (1.0 + np.exp(-df))
        val_roc = roc_auc_score(yval, p_val)

        tqdm.write(f"[SGD] epoch {ep}: val ROC-AUC={val_roc:.4f}")

        if val_roc > best_val + 1e-4:
            best_val = val_roc
            # Save classifier state (copy coef/intercept)
            best_state = (
                clf.coef_.copy(),
                clf.intercept_.copy(),
                getattr(clf, "classes_", classes).copy(),
            )
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= early_stop_patience:
                tqdm.write("[SGD] early stopping.")
                break

    if best_state is not None:
        coef, intercept, classes = best_state
        clf.coef_ = coef
        clf.intercept_ = intercept
        clf.classes_ = classes
    return clf


# -----------------------------
# Top-level API
# -----------------------------
def linear_probe_oos_sgd(
    seq_ds: Dict[str, Any],
    *,
    batch_size: int = 8192,
    epochs: int = 10,
    lr: float = 0.01,
    alpha: float = 1e-4,
    early_stop_patience: int = 2,
    random_state: int = 0,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Train 3 SGD-logistic probes on TRAIN (dna / go / both),
    with standardization (fit on train, apply to val/test),
    early stop on val ROC-AUC. Evaluate on VAL and TEST.
    """
    D_eff = int(seq_ds["train"].D_eff)
    D_go = int(seq_ds["train"].D_go)

    # Build matrices
    Xtr, ytr = _feat_matrix(seq_ds["train"], desc="[train]")
    Xv, yv = _feat_matrix(seq_ds["val"], desc="[val]")
    Xt, yt = _feat_matrix(seq_ds["test"], desc="[test]")

    out: Dict[str, Dict[str, Dict[str, float]]] = {"val": {}, "test": {}}

    # Train a model per feature set
    for feat_name, split_tr in _split_feats(Xtr, D_eff, D_go).items():
        split_v = _split_feats(Xv, D_eff, D_go)[feat_name]
        split_t = _split_feats(Xt, D_eff, D_go)[feat_name]

        # Scale features (important for SGD)
        scaler = StandardScaler(with_mean=True, with_std=True)
        split_tr = scaler.fit_transform(split_tr)
        split_v = scaler.transform(split_v)
        split_t = scaler.transform(split_t)

        clf = _train_sgd_logit(
            split_tr,
            ytr,
            split_v,
            yv,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            alpha=alpha,
            early_stop_patience=early_stop_patience,
            random_state=random_state,
        )

        # Evaluate val & test
        def _eval(X, y):
            try:
                p = clf.predict_proba(X)[:, 1]
            except AttributeError:
                df = clf.decision_function(X)
                p = 1.0 / (1.0 + np.exp(-df))
            yh = (p >= 0.5).astype(int)
            return dict(
                acc=accuracy_score(y, yh),
                f1=f1_score(y, yh),
                roc=roc_auc_score(y, p),
                pr=average_precision_score(y, p),
            )

        out["val"][feat_name] = _eval(split_v, yv)
        out["test"][feat_name] = _eval(split_t, yt)

    return out


def print_probe_oos(
    out: Dict[str, Dict[str, Dict[str, float]]], title: str = "[oos probe / SGD]"
) -> None:
    print(title)
    for split in ("val", "test"):
        print(
            f"\n[{split}]  {'Feat':8} {'Acc':>8} {'F1':>8} {'ROC-AUC':>10} {'PR-AUC':>10}"
        )
        print("-" * 50)
        for k in ("dna", "go", "both"):
            r = out[split][k]
            print(
                f"{k:8} {r['acc']:8.4f} {r['f1']:8.4f} {r['roc']:10.4f} {r['pr']:10.4f}"
            )


# -----------------------------
# Example
# -----------------------------
# out = linear_probe_oos_sgd(
#     seq_ds,
#     batch_size=8192,   # bump to 16384/32768 if memory allows
#     epochs=10,
#     lr=0.01,
#     alpha=1e-4,
#     early_stop_patience=2,
#     random_state=0,
# )
# print_probe_oos(out)
