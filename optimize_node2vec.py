# fast_go_optimize.py
# -*- coding: utf-8 -*-
import os
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from go_node2vec import (
    load_go_term_edges,
    load_gene_term_edges,
    build_graph,
    graph_to_edge_index,
    train_node2vec,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# -------------------------
# Minimal memory logs (safe if no CUDA)
# -------------------------
def _cuda_mem_snapshot(tag: str):
    try:
        import torch

        if not torch.cuda.is_available():
            return
        free, total = torch.cuda.mem_get_info()
        alloc = torch.cuda.memory_allocated()
        resv = torch.cuda.memory_reserved()
        print(
            f"[mem] {tag}: alloc={alloc/2**20:.0f}MiB resv={resv/2**20:.0f}MiB "
            f"free={free/2**20:.0f}MiB/{total/2**20:.0f}MiB"
        )
    except Exception:
        pass


# -------------------------
# Feature helpers (for downstream probe)
# -------------------------
def _pick_gene(row) -> Optional[str]:
    g = row.get("GeneSymbol") or row.get("GeneSymbolSimple") or row.get("Gene") or ""
    g = str(g).strip()
    return g if g else None


def _split_dims(seq_ds) -> Tuple[int, int]:
    return int(seq_ds["train"].D_eff), int(seq_ds["train"].D_go)


def _preextract_split(split, D_eff: int, *, max_n: Optional[int] = None):
    n = len(split)
    idxs = np.arange(n)
    if max_n is not None and max_n < n:
        idxs = idxs[:max_n]

    dna, genes, y = [], [], []
    name = getattr(split, "name", None) or "split"
    pbar = tqdm(idxs, desc=f"[{name}] features", unit="row")
    for i in pbar:
        v, *_ = split[i]
        v = v if isinstance(v, np.ndarray) else v.numpy()
        dna.append(v[:D_eff].astype(np.float32))
        row = split.meta.iloc[i]
        genes.append(_pick_gene(row))
        y.append(int(row["_y"]))
    dna = np.vstack(dna).astype(np.float32)
    genes = np.array(genes, dtype=object)
    y = np.asarray(y, dtype=np.int64)
    return dna, genes, y


def _assemble_go_matrix(
    genes: np.ndarray, gene2vec: Dict[str, np.ndarray], dim: int
) -> np.ndarray:
    out = np.zeros((len(genes), dim), dtype=np.float32)
    for i, g in enumerate(genes):
        vec = gene2vec.get(g)
        if vec is not None:
            out[i] = vec
    return out


# -------------------------
# Downstream probe (fast, SGD)
# -------------------------
def _fit_eval_sgd(Xtr, ytr, Xte, yte) -> Dict[str, float]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=1000,
        tol=1e-3,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=0,
        n_jobs=-1,
    )
    clf.fit(Xtr_s, ytr)
    p = clf.predict_proba(Xte_s)[:, 1]
    return dict(
        roc=float(roc_auc_score(yte, p)),
        pr=float(average_precision_score(yte, p)),
    )


# -------------------------
# Hold-out splitter (G–T edges per gene)
# -------------------------
def _split_gene_term_edges_for_eval(
    edge_index, nodes, eval_frac: float = 0.10, seed: int = 0
):
    import torch

    rng = np.random.default_rng(seed)
    ei = edge_index.t().cpu().numpy()
    id2name = dict(enumerate(nodes))

    gt_idxs = []
    for i, (u, v) in enumerate(ei):
        nu, nv = id2name[u], id2name[v]
        if (nu.startswith("G::") and nv.startswith("T::")) or (
            nv.startswith("G::") and nu.startswith("T::")
        ):
            gt_idxs.append(i)
    gt_idxs = np.array(gt_idxs, dtype=np.int64)

    by_gene: Dict[int, List[int]] = {}
    for i in gt_idxs:
        u, v = ei[i]
        g = u if id2name[u].startswith("G::") else v
        by_gene.setdefault(g, []).append(i)

    keep_mask = np.ones(len(ei), dtype=bool)
    eval_take = []
    for g, idxs in by_gene.items():
        idxs = np.array(idxs, dtype=np.int64)
        k = max(1, int(round(len(idxs) * eval_frac)))
        sel = rng.choice(idxs, size=k, replace=False)
        eval_take.append(sel)
    eval_take = np.concatenate(eval_take) if eval_take else np.array([], dtype=np.int64)
    keep_mask[eval_take] = False

    train_ei = torch.from_numpy(ei[keep_mask]).t().contiguous()
    eval_ei = torch.from_numpy(ei[eval_take]).t().contiguous()
    return train_ei, eval_ei


# -------------------------
# Graph cache (CPU-only)
# -------------------------
class GraphCache:
    def __init__(self, go_json: str, gaf: str):
        self.go_json = go_json
        self.gaf = gaf
        self.term_edges = None
        self.gene_term_edges = None
        self._curated_only = None
        self.cache: Dict[
            Tuple[bool, bool, Optional[int]], Tuple["np.ndarray", list]
        ] = {}

    def _load_edges(self, curated_only: bool):
        if self.term_edges is None:
            self.term_edges = load_go_term_edges(self.go_json)
        if self.gene_term_edges is None or curated_only != self._curated_only:
            ev_wl = (
                {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC"}
                if curated_only
                else None
            )
            self.gene_term_edges = load_gene_term_edges(
                self.gaf,
                evidence_whitelist=ev_wl,
                exclude_not=True,
                aspect_whitelist={"P", "F"},
            )
            self._curated_only = curated_only

    def get_edge_index(
        self,
        *,
        curated_only: bool,
        no_term_edges: bool,
        drop_roots: bool,
        prune_term_degree: Optional[int],
    ):
        self._load_edges(curated_only)
        key = (no_term_edges, drop_roots, prune_term_degree)
        if key in self.cache:
            return self.cache[key]
        G = build_graph(
            [] if no_term_edges else self.term_edges,
            self.gene_term_edges,
            include_term_edges=not no_term_edges,
            drop_roots=drop_roots,
            prune_term_degree=prune_term_degree,
        )
        edge_index, nodes = graph_to_edge_index(G)
        self.cache[key] = (edge_index, nodes)
        return edge_index, nodes


# -------------------------
# Main optimization
# -------------------------
def optimize_go_for_seqds(
    seq_ds,
    go_json: str,
    gaf: str,
    out_prefix: str,
    *,
    n_trials: int = 12,
    refit_epochs: int = 12,
    seed: int = 0,
    use_optuna: Optional[bool] = None,
    objective_metric: str = "delta_roc",  # "delta_roc" or "delta_pr"
    max_train_probe: Optional[int] = 80000,
    max_val_probe: Optional[int] = 20000,
) -> Dict[str, Any]:
    import torch, random

    rng = np.random.default_rng(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)

    D_eff, _ = _split_dims(seq_ds)
    tr_dna, tr_genes, tr_y = _preextract_split(
        seq_ds["train"], D_eff, max_n=max_train_probe
    )
    va_dna, va_genes, va_y = _preextract_split(
        seq_ds["val"], D_eff, max_n=max_val_probe
    )

    graph_cache = GraphCache(go_json, gaf)

    search_space = dict(
        dim=[128, 256, 384],
        epochs=[8],
        batch_size=[256, 384],
        walk_len=[20, 30, 40],
        ctx_size=[4, 6, 8],
        walks_per_node=[8, 12, 16],
        neg_samples=[2, 5],
        curated_only=[True],
        no_term_edges=[False],
        drop_roots=[True],
        prune_term_degree=[800, 1000],
        device=[None],
    )

    if use_optuna is None:
        try:
            import optuna

            use_optuna = True
        except Exception:
            use_optuna = False

    def sample_cfg(trial=None):
        cfg = {}
        for k, vals in search_space.items():
            if trial is None:
                cfg[k] = rng.choice(vals)
            else:
                cfg[k] = trial.suggest_categorical(k, list(vals))
        return cfg

    def run_one_trial(cfg: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        edge_index_full, nodes = graph_cache.get_edge_index(
            curated_only=cfg["curated_only"],
            no_term_edges=cfg["no_term_edges"],
            drop_roots=cfg["drop_roots"],
            prune_term_degree=cfg["prune_term_degree"],
        )

        train_ei, _ = _split_gene_term_edges_for_eval(
            edge_index_full, nodes, eval_frac=0.10, seed=seed
        )

        emb = train_node2vec(
            train_ei,
            dim=cfg["dim"],
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            device=cfg["device"],
            walk_length=cfg["walk_len"],
            context_size=cfg["ctx_size"],
            walks_per_node=cfg["walks_per_node"],
            neg_samples=cfg["neg_samples"],
            num_nodes=len(nodes),
            seed=seed,
        )

        gene2vec = {
            name[3:]: emb[i].astype(np.float32)
            for i, name in enumerate(nodes)
            if name.startswith("G::")
        }
        dim_go = int(cfg["dim"])
        tr_go = _assemble_go_matrix(tr_genes, gene2vec, dim_go)
        va_go = _assemble_go_matrix(va_genes, gene2vec, dim_go)

        res_val = {
            "dna": _fit_eval_sgd(tr_dna, tr_y, va_dna, va_y),
            "go": _fit_eval_sgd(tr_go, tr_y, va_go, va_y),
            "both": _fit_eval_sgd(
                np.concatenate([tr_dna, tr_go], axis=1),
                tr_y,
                np.concatenate([va_dna, va_go], axis=1),
                va_y,
            ),
        }

        delta_pr = float(res_val["both"]["pr"] - res_val["dna"]["pr"])
        delta_roc = float(res_val["both"]["roc"] - res_val["dna"]["roc"])

        scores = {
            "probe_val": res_val,
            "delta_pr": delta_pr,
            "delta_roc": delta_roc,
        }

        if objective_metric == "delta_pr":
            obj = delta_pr
        else:
            obj = delta_roc
        return float(obj), scores

    results = []
    if use_optuna:
        import optuna

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        pbar = tqdm(total=n_trials, desc="[HPO] trials", unit="trial")

        def objective(trial):
            cfg = sample_cfg(trial)
            obj_val, res_val = run_one_trial(cfg)
            results.append({"cfg": cfg, "scores": res_val})
            trial.set_user_attr("val_delta_pr", res_val["delta_pr"])
            trial.set_user_attr("val_delta_roc", res_val["delta_roc"])
            pbar.update(1)
            return obj_val

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        pbar.close()
        best_cfg = {k: study.best_trial.params[k] for k in search_space.keys()}
        _, best_res_val = run_one_trial(best_cfg)
    else:
        raise ValueError("[HPO] Optuna not available")

    print(
        f"\n[REFIT] best {objective_metric}="
        f"{best_res_val.get(objective_metric, 0.0):.4f} → epochs={refit_epochs}"
    )
    refit_cfg = dict(best_cfg)
    refit_cfg["epochs"] = refit_epochs

    edge_index_full, nodes = graph_cache.get_edge_index(
        curated_only=refit_cfg["curated_only"],
        no_term_edges=refit_cfg["no_term_edges"],
        drop_roots=refit_cfg["drop_roots"],
        prune_term_degree=refit_cfg["prune_term_degree"],
    )
    train_ei, _ = _split_gene_term_edges_for_eval(
        edge_index_full, nodes, eval_frac=0.10, seed=seed
    )

    emb_final = train_node2vec(
        train_ei,
        dim=refit_cfg["dim"],
        epochs=refit_cfg["epochs"],
        batch_size=refit_cfg["batch_size"],
        device=refit_cfg["device"],
        walk_length=refit_cfg["walk_len"],
        context_size=refit_cfg["ctx_size"],
        walks_per_node=refit_cfg["walks_per_node"],
        neg_samples=refit_cfg["neg_samples"],
        num_nodes=len(nodes),
        seed=seed,
    )

    gene_idx = [i for i, n in enumerate(nodes) if n.startswith("G::")]
    gene_names = [nodes[i][3:] for i in gene_idx]
    mat = emb_final[gene_idx].astype(np.float32)
    genes = np.array(gene_names, dtype=object)
    np.savez_compressed(f"{out_prefix}_genes.npz", emb=mat, genes=genes)
    with open(f"{out_prefix}_genes.tsv", "w") as f:
        for g, vec in zip(genes, mat):
            f.write(g + "\t" + " ".join(f"{x:.6f}" for x in vec) + "\n")
    print(f"[SAVE] Final embeddings: {mat.shape} → {out_prefix}_genes.npz / .tsv")

    return {
        "best_cfg": refit_cfg,
        "val_scores": best_res_val,
        "final_path_npz": f"{out_prefix}_genes.npz",
        "final_path_tsv": f"{out_prefix}_genes.tsv",
        "all_trials": results,
    }


# ---------------------------
# Pretty printer
# ---------------------------
def print_go_opt_summary(summary: Dict[str, Any]):
    s = summary["val_scores"]
    print("\n[GO HPO] Validation scores (downstream probe)")
    if "probe_val" in s:
        pv = s["probe_val"]
        print(f"  dna : ROC={pv['dna']['roc']:.4f}  PR={pv['dna']['pr']:.4f}")
        print(f"  go  : ROC={pv['go']['roc']:.4f}  PR={pv['go']['pr']:.4f}")
        print(f"  both: ROC={pv['both']['roc']:.4f} PR={pv['both']['pr']:.4f}")
        if "delta_pr" in s:
            print(f"  ΔPR (both - dna): {s['delta_pr']:.4f}")
        if "delta_roc" in s:
            print(f"  ΔROC(both - dna): {s['delta_roc']:.4f}")

    print("\nBest (refit) config:")
    for k, v in summary["best_cfg"].items():
        print(f"  {k:16} = {v}")
    print(f"\nSaved: {summary['final_path_npz']}")
