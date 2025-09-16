# -*- coding: utf-8 -*-
# MLP probe with optional protein *gene-level* embeddings.
# No HGVS parsing, no ΔlogP, no ESM. Fast and simple.
#
# Modes:
#   'dna' | 'go' | 'both' (dna+go) | 'prot' | 'dna+prot' | 'dna+prot+go'
#
# Provide protein vectors via:
#   - gene2prot: dict[str, np.ndarray]  (same dim for all genes), OR
#   - prot_blocks: {'train': np.ndarray, 'val': np.ndarray, 'test': np.ndarray}
#
# If you don't request modes with 'prot', nothing protein-related is computed.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple
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


# -----------------------
# Helpers
# -----------------------
def _feat_matrix(split, desc):
    X, y = [], []
    for i in tqdm(range(len(split)), desc=f"{desc}: features", unit="row", leave=False):
        v, *_ = split[i]
        v = v if isinstance(v, np.ndarray) else v.numpy()
        X.append(v)
        y.append(int(split.meta.iloc[i]["_y"]))
    return np.vstack(X).astype(np.float32), np.array(y, dtype=np.int64)


def _safe_auc(y, p, fn, default=np.nan):
    try:
        return float(fn(y, p))
    except Exception:
        return default


def _threshold_metrics(y_true, p_prob, thr: float) -> Dict[str, float]:
    yhat = (p_prob >= thr).astype(int)
    tn, fp, fn_, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    return dict(
        acc=accuracy_score(y_true, yhat),
        bal_acc=balanced_accuracy_score(y_true, yhat),
        f1=f1_score(y_true, yhat, zero_division=0),
        precision=precision_score(y_true, yhat, zero_division=0),
        recall=recall_score(y_true, yhat, zero_division=0),
        specificity=tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn_,
    )


def _find_best_threshold(y_val, p_val, *, grid=201, criterion="f1"):
    ts = np.linspace(0.01, 0.99, grid)
    best_t, best_v = 0.5, -1.0
    for t in ts:
        if criterion == "f1":
            v = f1_score(y_val, (p_val >= t).astype(int), zero_division=0)
        elif criterion == "youden":
            v = balanced_accuracy_score(y_val, (p_val >= t).astype(int))
        else:
            raise ValueError("criterion must be 'f1' or 'youden'")
        if v > best_v:
            best_v, best_t = v, t
    return float(best_t), float(best_v)


# --- helper: pick a gene symbol from a row (robust to column names) ---
def _pick_gene_from_row(row) -> Optional[str]:
    for k in ("GeneSymbol", "GeneSymbolSimple", "Gene", "gene", "GENE"):
        v = row.get(k, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


# --- NEW: build protein block from a gene->vector dict ---
def build_gene_prot_block(
    split,
    gene2prot: Dict[str, np.ndarray],
    dim: Optional[int] = None,
    strict: bool = False,
) -> np.ndarray:
    """
    Build an (N, D) protein feature block aligned to `split.meta` by gene symbol.

    - Looks up gene symbol in gene2prot with several casings (exact, UPPER, Title).
    - If `dim` is None, infers D from the first vector in `gene2prot`.
    - Pads/truncates every vector to D and casts to float32.
    - Unknown/missing genes → zeros (unless strict=True, then raises).

    Returns: np.ndarray of shape (N, D), dtype float32.
    """
    # --- infer output dim if not provided ---
    if dim is None:
        try:
            first_vec = next(iter(gene2prot.values()))
            dim = int(np.asarray(first_vec).reshape(-1).shape[0])
        except StopIteration:
            raise ValueError("gene2prot is empty; cannot infer embedding dim.")
    D = int(dim)

    # --- helper: pick gene symbol from row ---
    def _pick_gene(row) -> Optional[str]:
        for k in ("GeneSymbol", "GeneSymbolSimple", "Gene", "Symbol", "gene"):
            val = row.get(k) if hasattr(row, "get") else None
            if val is None:
                try:
                    val = row[k]
                except Exception:
                    val = None
            if val is not None:
                s = str(val).strip()
                if s:
                    return s
        return None

    # --- helper: fetch vector by several casings ---
    def _lookup_vec(g: str) -> Optional[np.ndarray]:
        if g in gene2prot:
            return gene2prot[g]
        gU = g.upper()
        if gU in gene2prot:
            return gene2prot[gU]
        gT = g.title()
        if gT in gene2prot:
            return gene2prot[gT]
        return None

    # --- helper: normalize vector to length D ---
    def _normalize_vec(v: np.ndarray) -> np.ndarray:
        a = np.asarray(v).reshape(-1).astype(np.float32, copy=False)
        if a.shape[0] == D:
            return a
        if a.shape[0] < D:
            pad = np.zeros(D - a.shape[0], dtype=np.float32)
            return np.concatenate([a, pad], axis=0)
        return a[:D]

    # --- build block ---
    meta = getattr(split, "meta", None)
    if meta is None:
        # no metadata → return zeros
        return np.zeros((len(split), D), dtype=np.float32)

    out = np.zeros((len(split), D), dtype=np.float32)
    missing = 0
    for i in range(len(split)):
        row = meta.iloc[i]
        g = _pick_gene(row)
        if not g:
            missing += 1
            if strict:
                raise ValueError(f"No gene symbol at row {i}")
            continue
        vec = _lookup_vec(g)
        if vec is None:
            missing += 1
            if strict:
                raise KeyError(f"Gene '{g}' not found in gene2prot at row {i}")
            # leave zeros
            continue
        out[i] = _normalize_vec(vec)

    if missing:
        print(
            f"[gene2prot] filled {missing} / {len(split)} rows with zeros (no match)."
        )
    return out


# -----------------------
# Simple MLP
# -----------------------
class MLP(nn.Module):
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
            nn.Linear(h2, 1),  # (B,1)
        )

    def forward(self, x):
        return self.net(x)


def mlp_probe_plus(
    seq_ds,
    feat="dna+prot",  # 'dna' | 'go' | 'both' | 'prot' | 'dna+prot' | 'dna+prot+go'
    include_protein_len_col: Optional[str] = None,  # kept for API compat; unused here
    hidden=512,
    dropout=0.2,
    epochs=16,
    bs=8192,
    lr=3e-3,
    weight_decay=1e-4,
    patience=4,
    device: Optional[str] = None,
    seed=0,
    tune_threshold_on="f1",
    hpo_trials: int = 0,  # small Optuna sweep if >0
    opt_metric: str = "pr",  # 'pr' or 'roc' for model selection
    pca_components: Optional[int] = None,
    *,
    gene2prot: Optional[Dict[str, np.ndarray]] = None,  # NEW
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # --- base matrices from seq_ds ---
    D_eff, D_go = int(seq_ds["train"].D_eff), int(seq_ds["train"].D_go)
    Xtr_all, ytr = _feat_matrix(seq_ds["train"], "[train]")
    Xv_all, yv = _feat_matrix(seq_ds["val"], "[val]")
    Xt_all, yt = _feat_matrix(seq_ds["test"], "[test]")

    tr_dna, tr_go = Xtr_all[:, :D_eff], Xtr_all[:, D_eff : D_eff + D_go]
    va_dna, va_go = Xv_all[:, :D_eff], Xv_all[:, D_eff : D_eff + D_go]
    te_dna, te_go = Xt_all[:, :D_eff], Xt_all[:, D_eff : D_eff + D_go]

    # --- which blocks to use ---
    use_dna = feat in ("dna", "both", "dna+prot", "dna+prot+go")
    use_go = feat in ("go", "both", "dna+prot+go")
    use_prot = feat in ("prot", "dna+prot", "dna+prot+go")

    # --- protein block from gene2prot (aligned to split) ---
    if use_prot:
        if gene2prot is None:
            print(
                "[warn] feat includes 'prot' but no gene2prot provided → disabling prot block."
            )
            use_prot = False
            tr_prot = va_prot = te_prot = None
        else:
            tr_prot = build_gene_prot_block(seq_ds["train"], gene2prot)
            va_prot = build_gene_prot_block(
                seq_ds["val"], gene2prot, dim=tr_prot.shape[1]
            )
            te_prot = build_gene_prot_block(
                seq_ds["test"], gene2prot, dim=tr_prot.shape[1]
            )
    else:
        tr_prot = va_prot = te_prot = None

    # --- scale blocks separately (fit on train, apply to val/test) ---
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
    if use_go:
        tr_go, sc_go = zfit(tr_go)
        va_go = zapply(va_go, sc_go)
        te_go = zapply(te_go, sc_go)
    if use_prot and tr_prot is not None:
        tr_prot, sc_pr = zfit(tr_prot)
        va_prot = zapply(va_prot, sc_pr)
        te_prot = zapply(te_prot, sc_pr)

    # --- assemble final matrices ---
    def assemble(A, B, C):
        parts = []
        if use_dna:
            parts.append(A)
        if use_go:
            parts.append(B)
        if use_prot and C is not None:
            parts.append(C)
        return np.concatenate(parts, axis=1) if parts else A

    Xtr = assemble(tr_dna, tr_go, tr_prot)
    Xv = assemble(va_dna, va_go, va_prot)
    Xt = assemble(te_dna, te_go, te_prot)

    # --- optional PCA ---
    if pca_components:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=int(pca_components), random_state=seed)
        Xtr = pca.fit_transform(Xtr).astype(np.float32)
        Xv = pca.transform(Xv).astype(np.float32)
        Xt = pca.transform(Xt).astype(np.float32)

    # --- tensors ---
    trX = torch.tensor(Xtr)
    vaX = torch.tensor(Xv)
    teX = torch.tensor(Xt)
    ytr_t = torch.tensor(ytr)
    yv_t = torch.tensor(yv)
    yt_t = torch.tensor(yt)

    # --- trainer ---
    class _MLPTrainer:
        def __init__(self):
            self.best_state = None

        def train_eval_once(
            self,
            hid=hidden,
            dr=dropout,
            lr_=lr,
            wd=weight_decay,
            bs_=bs,
            max_epochs=epochs,
        ):
            model = MLP(trX.shape[1], hidden=int(hid), dropout=float(dr)).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr_, weight_decay=wd)
            best = (-np.inf, None)
            bad = 0

            def eval_probs(X, y_np):
                model.eval()
                with torch.no_grad():
                    logits = model(X.to(device))  # (N,1)
                    p = torch.sigmoid(logits).squeeze(-1)  # (N,)
                    p = p.cpu().numpy()
                roc = _safe_auc(y_np, p, roc_auc_score)
                pr = _safe_auc(y_np, p, average_precision_score)
                return p, roc, pr

            # class weights
            pos = max(1, int(ytr.sum()))
            w_pos = len(ytr) / (2.0 * pos)
            w_neg = len(ytr) / (2.0 * max(1, (len(ytr) - pos)))

            for ep in range(1, int(max_epochs) + 1):
                model.train()
                perm = torch.randperm(trX.size(0))
                for s in tqdm(
                    range(0, trX.size(0), int(bs_)),
                    desc=f"MLP {feat} ep{ep}/{max_epochs}",
                    unit="batch",
                    leave=False,
                ):
                    idx = perm[s : s + int(bs_)]
                    xb = trX[idx].to(device)
                    yb = ytr_t[idx].float().to(device)
                    logits = model(xb).squeeze(-1)
                    w = torch.where(
                        yb == 1.0,
                        torch.tensor(w_pos, device=device),
                        torch.tensor(w_neg, device=device),
                    )
                    loss = F.binary_cross_entropy_with_logits(logits, yb, weight=w)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                # val
                p_val, roc_val, pr_val = eval_probs(vaX, yv)
                metric_val = pr_val if opt_metric == "pr" else roc_val
                if metric_val > best[0] + 1e-6:
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
                        break

            if best[1] is not None:
                model.load_state_dict(best[1])

            p_val, roc_val, pr_val = eval_probs(vaX, yv)
            p_test, roc_test, pr_test = eval_probs(teX, yt)
            return model, (p_val, roc_val, pr_val), (p_test, roc_test, pr_test)

    trainer = _MLPTrainer()

    # --- tiny HPO (optional) ---
    if hpo_trials and hpo_trials > 0:
        try:
            import optuna

            def objective(trial):
                hid = trial.suggest_categorical("hidden", [256, 512, 768, 1024])
                dr = trial.suggest_float("dropout", 0.0, 0.6)
                lr_ = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
                wd = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)
                bs_ = trial.suggest_categorical("bs", [4096, 8192, 16384])
                ep_ = trial.suggest_categorical("epochs", [12, 16, 20])
                _, (p_val, roc_val, pr_val), _ = trainer.train_eval_once(
                    hid=hid, dr=dr, lr_=lr_, wd=wd, bs_=bs_, max_epochs=ep_
                )
                return pr_val if opt_metric == "pr" else roc_val

            study = optuna.create_study(
                direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
            )
            study.optimize(objective, n_trials=int(hpo_trials), show_progress_bar=False)
            bp = study.best_params
            hidden, dropout, lr, weight_decay, bs, epochs = (
                bp["hidden"],
                bp["dropout"],
                bp["lr"],
                bp["weight_decay"],
                bp["bs"],
                bp["epochs"],
            )
        except Exception as e:
            print(f"[HPO] skipped: {e}")

    # --- final train + report ---
    _, (p_val, roc_val, pr_val), (p_test, roc_test, pr_test) = trainer.train_eval_once()

    thr_best, _ = _find_best_threshold(yv, p_val, criterion=tune_threshold_on)

    val_05 = _threshold_metrics(yv, p_val, 0.5)
    val_05["roc"] = roc_val
    val_05["pr"] = pr_val
    val_bt = _threshold_metrics(yv, p_val, thr_best)
    val_bt["roc"] = roc_val
    val_bt["pr"] = pr_val

    tst_05 = _threshold_metrics(yt, p_test, 0.5)
    tst_05["roc"] = roc_test
    tst_05["pr"] = pr_test
    tst_bt = _threshold_metrics(yt, p_test, thr_best)
    tst_bt["roc"] = roc_test
    tst_bt["pr"] = pr_test

    return {
        "val": {"thr_best": float(thr_best), "at_0p5": val_05, "at_best_thr": val_bt},
        "test": {"at_0p5": tst_05, "at_best_thr": tst_bt},
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
            tune_threshold_on=tune_threshold_on,
            hpo_trials=hpo_trials,
            opt_metric=opt_metric,
            pca_components=pca_components,
            prot_source=(
                "gene2prot" if (use_prot and gene2prot is not None) else "none"
            ),
            prot_dim=(
                None if (not use_prot or tr_prot is None) else int(tr_prot.shape[1])
            ),
        ),
    }


def print_mlp_probe_summary(
    out: dict, title: str = "MLP Probe", show_counts: bool = False
):
    """
    Pretty-print the dict produced by mlp_probe_plus(...).

    Expected structure:
      out = {
        "val":  {"thr_best": float, "at_0p5": {...}, "at_best_thr": {...}},
        "test": {"at_0p5": {...},    "at_best_thr": {...}},
        "config": {...}
      }
    Each metrics dict may contain:
      acc, bal_acc, f1, precision, recall, specificity, roc, pr, tp, fp, tn, fn
    """

    def _fmt(x, nd=4):
        try:
            if x is None:
                return "-"
            if isinstance(x, float):
                return f"{x:.{nd}f}"
            return str(x)
        except Exception:
            return "-"

    def _row(split_name: str, key: str):
        d = out.get(split_name, {}).get(key, {}) or {}
        return dict(
            acc=d.get("acc"),
            bal_acc=d.get("bal_acc"),
            f1=d.get("f1"),
            precision=d.get("precision"),
            recall=d.get("recall"),
            specificity=d.get("specificity"),
            roc=d.get("roc"),
            pr=d.get("pr"),
            tp=d.get("tp"),
            fp=d.get("fp"),
            tn=d.get("tn"),
            fn=d.get("fn"),
        )

    # Header & config
    print(f"\n{title}")
    cfg = out.get("config", {}) or {}
    if cfg:
        keys = [
            "feat",
            "hidden",
            "dropout",
            "epochs",
            "bs",
            "lr",
            "weight_decay",
            "patience",
            "device",
            "tune_threshold_on",
            "hpo_trials",
            "opt_metric",
            "pca_components",
            "seed",
            "prot_source",
            "prot_dim",
        ]
        have = [k for k in keys if k in cfg]
        if have:
            print("Config:", ", ".join(f"{k}={cfg[k]}" for k in have))

    # Best threshold (from val)
    thr_best = out.get("val", {}).get("thr_best", None)
    if thr_best is not None:
        print(
            f"Best threshold (on val, by {cfg.get('tune_threshold_on','f1')}): {_fmt(thr_best)}"
        )

    # Columns
    cols = ["Acc", "BalAcc", "F1", "Prec", "Recall", "Spec", "ROC", "PR"]
    if show_counts:
        cols += ["TP", "FP", "TN", "FN"]

    def _print_split(split_name: str):
        r05 = _row(split_name, "at_0p5")
        rbt = _row(split_name, "at_best_thr")

        # Header
        header = (
            f"\n[{split_name.upper()}]\n"
            + f"{'Thr':>8} "
            + " ".join(f"{c:>8}" for c in cols)
        )
        print(header)
        print("-" * (len(header.splitlines()[-1])))

        def _vals(r: dict):
            base = [
                _fmt(r["acc"]),
                _fmt(r["bal_acc"]),
                _fmt(r["f1"]),
                _fmt(r["precision"]),
                _fmt(r["recall"]),
                _fmt(r["specificity"]),
                _fmt(r["roc"]),
                _fmt(r["pr"]),
            ]
            if show_counts:
                base += [
                    str(r.get("tp", 0) or 0),
                    str(r.get("fp", 0) or 0),
                    str(r.get("tn", 0) or 0),
                    str(r.get("fn", 0) or 0),
                ]
            return base

        print(f"{'0.5':>8} " + " ".join(f"{v:>8}" for v in _vals(r05)))
        print(f"{'best':>8} " + " ".join(f"{v:>8}" for v in _vals(rbt)))

        # Deltas
        try:
            d_acc = (rbt["acc"] or 0) - (r05["acc"] or 0)
            d_f1 = (rbt["f1"] or 0) - (r05["f1"] or 0)
            d_roc = (rbt["roc"] or 0) - (r05["roc"] or 0)
            d_pr = (rbt["pr"] or 0) - (r05["pr"] or 0)
            print(
                f"Δ(best-0.5): Acc={_fmt(d_acc)}  F1={_fmt(d_f1)}  ROC={_fmt(d_roc)}  PR={_fmt(d_pr)}"
            )
        except Exception:
            pass

    if "val" in out:
        _print_split("val")
    if "test" in out:
        _print_split("test")


from train import build_seq_datasets, RunCfg, OUT_DIR, FASTA_PATH, GO_NPZ

cfg = RunCfg(
    out_dir=OUT_DIR,
    fasta_path=FASTA_PATH,
    go_npz=GO_NPZ,
    epochs=1,
    force_dna=False,
)

seq_ds = build_seq_datasets(cfg)
