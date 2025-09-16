import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
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
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None


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
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn_,
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
            bacc = balanced_accuracy_score(y_val, yhat)
            v = bacc  # monotone with Youden's J under class balance scaling
        else:
            raise ValueError("criterion must be 'f1' or 'youden'")
        if v > best_v:
            best_v, best_t = v, t
    return float(best_t), float(best_v)


# --------------------------
# Feature extraction
# --------------------------
def _feat_matrix(split, desc):
    X, y = [], []
    for i in tqdm(range(len(split)), desc=f"{desc}: features", unit="row"):
        v, *_ = split[i]
        v = v if isinstance(v, np.ndarray) else v.numpy()
        X.append(v)
        y.append(int(split.meta.iloc[i]["_y"]))
    return np.vstack(X).astype(np.float32), np.array(y, dtype=np.int64)


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
# Main probe
# --------------------------
def mlp_probe(
    seq_ds,
    feat="both",  # 'dna' | 'go' | 'both'
    hidden=512,
    dropout=0.2,
    epochs=20,
    bs=8192,
    lr=3e-3,
    weight_decay=1e-4,
    patience=4,
    device: Optional[str] = None,
    seed: int = 0,
    pca_components: Optional[int] = None,  # e.g., 256 (applied after scaling)
    tune_threshold_on="f1",  # 'f1' or 'youden'
    # --- tiny HPO ---
    hpo_trials: int = 0,  # set >0 to try Optuna
    opt_metric: str = "pr",  # 'pr' or 'roc' to optimize on val
    grad_clip: float = 1.0,
):
    """
    Returns:
      {
        'val': {metrics_at_0.5..., 'thr_best', 'metrics_at_thr', 'roc', 'pr'},
        'test': {metrics_at_0.5..., 'metrics_at_thr', 'roc', 'pr'}
      }
    """
    # ---- device & seeds ----
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # ---- features ----
    D_eff, D_go = int(seq_ds["train"].D_eff), int(seq_ds["train"].D_go)
    Xtr, ytr = _feat_matrix(seq_ds["train"], "[train]")
    Xv, yv = _feat_matrix(seq_ds["val"], "[val]")
    Xt, yt = _feat_matrix(seq_ds["test"], "[test]")

    sl = (
        slice(0, D_eff)
        if feat == "dna"
        else slice(D_eff, D_eff + D_go) if feat == "go" else slice(None)
    )
    Xtr, Xv, Xt = Xtr[:, sl], Xv[:, sl], Xt[:, sl]

    # ---- scale (+ optional PCA) ----
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr).astype(np.float32)
    Xv = scaler.transform(Xv).astype(np.float32)
    Xt = scaler.transform(Xt).astype(np.float32)

    if pca_components is not None and PCA is not None and pca_components < Xtr.shape[1]:
        pca = PCA(n_components=int(pca_components), random_state=seed)
        Xtr = pca.fit_transform(Xtr).astype(np.float32)
        Xv = pca.transform(Xv).astype(np.float32)
        Xt = pca.transform(Xt).astype(np.float32)

    # ---- tensors ----
    tr = torch.tensor(Xtr)
    tv = torch.tensor(Xv)
    tt = torch.tensor(Xt)
    ytr_t = torch.tensor(ytr)
    yv_t = torch.tensor(yv)
    yt_t = torch.tensor(yt)

    # ---- class weights ----
    pos = max(1, int(ytr.sum()))
    w_pos = len(ytr) / (2.0 * pos)
    w_neg = len(ytr) / (2.0 * max(1, (len(ytr) - pos)))

    def train_eval_once(
        h=hidden, dr=dropout, lr_=lr, wd=weight_decay, bs_=bs, max_epochs=epochs
    ):
        model = _MLP(tr.shape[1], hidden=int(h), dropout=float(dr)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr_, weight_decay=wd)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=2
        )

        best = (-np.inf, None)
        bad = 0

        def eval_probs(X, y):
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(X.to(device))).cpu().numpy()
            roc = _safe_auc(y, p, roc_auc_score)
            pr = _safe_auc(y, p, average_precision_score)
            return p, roc, pr

        for ep in range(1, max_epochs + 1):
            model.train()
            perm = torch.randperm(tr.size(0))
            losses = []
            for s in tqdm(
                range(0, tr.size(0), bs_),
                desc=f"MLP {feat} (epoch {ep}/{max_epochs})",
                unit="batch",
                leave=False,
            ):
                idx = perm[s : s + bs_]
                xb, yb = tr[idx].to(device), ytr_t[idx].float().to(device)
                logits = model(xb)
                w = torch.where(
                    yb == 1.0,
                    torch.tensor(w_pos, device=device),
                    torch.tensor(w_neg, device=device),
                )
                loss = F.binary_cross_entropy_with_logits(logits, yb, weight=w)
                opt.zero_grad()
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                losses.append(loss.item())

            # validation
            p_val, roc_val, pr_val = eval_probs(tv, yv)
            metric_val = pr_val if opt_metric == "pr" else roc_val
            tqdm.write(
                f"[MLP {feat}] ep{ep:02d} "
                f"loss={np.mean(losses):.4f} valROC={roc_val:.4f} valPR={pr_val:.4f}"
            )
            sched.step(metric_val)

            if metric_val > best[0] + 1e-5:
                best = (
                    metric_val,
                    {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    },
                )
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    tqdm.write("[MLP] early stopping")
                    break

        if best[1] is not None:
            model.load_state_dict(best[1])

        # final probs/metrics
        p_val, roc_val, pr_val = eval_probs(tv, yv)
        p_test, roc_test, pr_test = eval_probs(tt, yt)
        return model, (p_val, roc_val, pr_val), (p_test, roc_test, pr_test)

    # ---- optional HPO via Optuna ----
    if hpo_trials and hpo_trials > 0:
        try:
            import optuna

            def objective(trial):
                h = trial.suggest_categorical("hidden", [256, 512, 768, 1024])
                dr = trial.suggest_float("dropout", 0.0, 0.5)
                lr_ = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
                wd = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)
                bs_ = trial.suggest_categorical("bs", [2048, 4096, 8192, 16384])
                ep_ = trial.suggest_categorical("epochs", [12, 16, 20, 24])

                _, (p_val, roc_val, pr_val), _ = train_eval_once(
                    h=h, dr=dr, lr_=lr_, wd=wd, bs_=bs_, max_epochs=ep_
                )
                return pr_val if opt_metric == "pr" else roc_val

            study = optuna.create_study(
                direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
            )
            study.optimize(objective, n_trials=int(hpo_trials), show_progress_bar=False)
            best = study.best_params
            hidden = best["hidden"]
            dropout = best["dropout"]
            lr = best["lr"]
            weight_decay = best["weight_decay"]
            bs = best["bs"]
            epochs = best["epochs"]
            tqdm.write(f"[HPO] best: {best}")
        except Exception as e:
            tqdm.write(
                f"[HPO] Optuna unavailable or failed ({e}); running default config."
            )

    # ---- train final ----
    model, (p_val, roc_val, pr_val), (p_test, roc_test, pr_test) = train_eval_once(
        h=hidden, dr=dropout, lr_=lr, wd=weight_decay, bs_=bs, max_epochs=epochs
    )

    # ---- metrics @ 0.5 and @ tuned threshold ----
    thr_best, _ = _find_best_threshold(yv, p_val, criterion=tune_threshold_on)
    m_val_05 = _threshold_metrics(yv, p_val, 0.5)
    m_tst_05 = _threshold_metrics(yt, p_test, 0.5)
    m_val_bt = _threshold_metrics(yv, p_val, thr_best)
    m_tst_bt = _threshold_metrics(yt, p_test, thr_best)

    # attach AUCs
    for d in (m_val_05, m_tst_05, m_val_bt, m_tst_bt):
        d["roc"] = None  # will fill below
        d["pr"] = None
    m_val_05["roc"] = roc_val
    m_val_05["pr"] = pr_val
    m_val_bt["roc"] = roc_val
    m_val_bt["pr"] = pr_val
    m_tst_05["roc"] = roc_test
    m_tst_05["pr"] = pr_test
    m_tst_bt["roc"] = roc_test
    m_tst_bt["pr"] = pr_test

    return {
        "val": {
            "thr_best": thr_best,
            "at_0p5": m_val_05,
            "at_best_thr": m_val_bt,
        },
        "test": {
            "at_0p5": m_tst_05,
            "at_best_thr": m_tst_bt,
        },
        "config": dict(
            feat=feat,
            hidden=hidden,
            dropout=dropout,
            epochs=epochs,
            bs=bs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            device=device,
            seed=seed,
            pca_components=pca_components,
            tune_threshold_on=tune_threshold_on,
            hpo_trials=hpo_trials,
            opt_metric=opt_metric,
        ),
    }


# Examples:
# res_dna  = mlp_probe(seq_ds, feat="dna",  hpo_trials=20)         # tiny HPO for fairness
# res_go   = mlp_probe(seq_ds, feat="go",   hpo_trials=20, pca_components=128)
# res_both = mlp_probe(seq_ds, feat="both", hpo_trials=20)
# import pprint
# pprint.pprint(res_both)
