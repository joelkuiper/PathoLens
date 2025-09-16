import os
import sys
import json
import gc
import time
from typing import Dict, Any, Optional, Tuple

import numpy as np
from tqdm import tqdm
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from go_node2vec import (
    load_go_term_edges,
    load_gene_term_edges,
    build_graph,
    graph_to_edge_index,
    train_node2vec,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# -------------------------
# Memory debug helpers
# -------------------------
def _cuda_mem_snapshot(tag: str):
    """Print a small CUDA memory snapshot (safe if no CUDA)."""
    try:
        import torch

        if not torch.cuda.is_available():
            print(f"[mem] {tag}: cuda not available")
            return
        device = torch.device("cuda:0")
        free, total = torch.cuda.mem_get_info(device)
        alloc = torch.cuda.memory_allocated(device)
        resv = torch.cuda.memory_reserved(device)
        peak = torch.cuda.max_memory_allocated(device)
        print(
            f"[mem] {tag}: "
            f"alloc={alloc/2**20:.1f}MiB  reserved={resv/2**20:.1f}MiB  "
            f"peak={peak/2**20:.1f}MiB  free={free/2**20:.1f}MiB / total={total/2**20:.1f}MiB"
        )
    except Exception as e:
        print(f"[mem] {tag}: <err {e}>")


def _host_rss_snapshot(tag: str):
    """Print process RSS (host) if psutil is present."""
    try:
        import psutil

        rss = psutil.Process(os.getpid()).memory_info().rss
        print(f"[rss] {tag}: rss={rss/2**20:.1f}MiB")
    except Exception:
        pass


# -------------------------
# Feature & scoring helpers
# -------------------------
def _pick_gene(row) -> Optional[str]:
    g = row.get("GeneSymbol") or row.get("GeneSymbolSimple") or row.get("Gene") or ""
    g = str(g).strip()
    return g if g else None


def _split_dims(seq_ds) -> Tuple[int, int]:
    return int(seq_ds["train"].D_eff), int(seq_ds["train"].D_go)


def _preextract_split(split, D_eff: int, *, max_n: Optional[int] = None):
    """One-time extraction: DNA array (N, D_eff), gene list (N,), labels (N,)."""
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
    """Vectorized gather: unknown genes → zeros."""
    out = np.zeros((len(genes), dim), dtype=np.float32)
    for i, g in enumerate(genes):
        vec = gene2vec.get(g)
        if vec is not None:
            out[i] = vec
    return out


def _fit_eval_hgb(Xtr, ytr, Xte, yte) -> Dict[str, float]:
    clf = HGB(
        loss="log_loss",
        max_depth=None,
        max_iter=300,
        learning_rate=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        l2_regularization=1e-4,
        max_bins=255,
        random_state=0,
    )
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    yh = (p >= 0.5).astype(int)
    return dict(
        acc=float(accuracy_score(yte, yh)),
        f1=float(f1_score(yte, yh)),
        roc=float(roc_auc_score(yte, p)),
        pr=float(average_precision_score(yte, p)),
    )


# -------------------------
# Intrinsic GO metric
# -------------------------
def _link_pred_score(
    edge_index,
    emb,
    nodes,
    num_neg=10000,
    seed=0,
    mode: str = "hard",  # "easy", "gt", or "hard"
) -> Dict[str, float]:
    """
    Intrinsic link prediction with selectable difficulty.

    mode:
      - "easy":   all edges (G–T, T–T, G–G) vs random pairs (original, saturates at ~0.99 ROC).
      - "gt":     only gene–term edges vs random gene–term non-edges (harder).
      - "hard":   only gene–term edges vs degree-matched gene–term non-edges (hardest).
    """
    rng = np.random.default_rng(seed)
    edges = edge_index.t().cpu().numpy()
    n_nodes = emb.shape[0]

    # Partition nodes
    gene_idx = [i for i, n in enumerate(nodes) if n.startswith("G::")]
    term_idx = [i for i, n in enumerate(nodes) if n.startswith("T::")]
    gene_idx = np.array(gene_idx, dtype=np.int64)
    term_idx = np.array(term_idx, dtype=np.int64)

    if mode == "easy":
        # all edges vs random pairs
        n_test = max(1, edges.shape[0] // 10)
        idxs = rng.choice(edges.shape[0], size=n_test, replace=False)
        pos_edges = edges[idxs]
        neg_edges = rng.integers(0, n_nodes, size=(num_neg, 2))

    else:
        # restrict to gene–term edges
        mask = [
            (nodes[u].startswith("G::") and nodes[v].startswith("T::"))
            or (nodes[v].startswith("G::") and nodes[u].startswith("T::"))
            for u, v in edges
        ]
        gt_edges = edges[np.array(mask)]

        # hold out 10% positives
        n_test = max(1, len(gt_edges) // 10)
        idxs = rng.choice(len(gt_edges), size=n_test, replace=False)
        pos_edges = gt_edges[idxs]

        # negatives
        edge_set = {tuple(sorted(e)) for e in map(tuple, gt_edges)}

        if mode == "gt":
            # uniform gene–term non-edges
            neg_edges = []
            while len(neg_edges) < num_neg:
                g = rng.choice(gene_idx)
                t = rng.choice(term_idx)
                e = tuple(sorted((g, t)))
                if e not in edge_set:
                    neg_edges.append((g, t))
            neg_edges = np.array(neg_edges, dtype=np.int64)

        elif mode == "hard":
            # degree-matched negatives
            deg = {}
            for g, t in gt_edges:
                g, t = (g, t) if nodes[g].startswith("G::") else (t, g)
                deg[t] = deg.get(t, 0) + 1
            neg_edges = []
            for g, t in pos_edges:
                g, t = (g, t) if nodes[g].startswith("G::") else (t, g)
                t_deg = deg.get(t, 1)
                cand_terms = [tt for tt in term_idx if abs(deg.get(tt, 1) - t_deg) <= 2]
                if not cand_terms:
                    cand_terms = term_idx
                tries = 0
                while True:
                    t2 = rng.choice(cand_terms)
                    e = tuple(sorted((g, t2)))
                    if e not in edge_set:
                        neg_edges.append((g, t2))
                        break
                    tries += 1
                    if tries > 10:
                        break
            neg_edges = np.array(neg_edges, dtype=np.int64)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # score edges
    pos_scores = np.sum(emb[pos_edges[:, 0]] * emb[pos_edges[:, 1]], axis=1)
    neg_scores = np.sum(emb[neg_edges[:, 0]] * emb[neg_edges[:, 1]], axis=1)

    y = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    s = np.concatenate([pos_scores, neg_scores])
    return dict(
        roc=float(roc_auc_score(y, s)),
        pr=float(average_precision_score(y, s)),
    )


# -------------------------
# Graph cache (CPU-only; does NOT touch CUDA)
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
                # Exclude Cellular Component, keep BP + MF
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
        edge_index, nodes = graph_to_edge_index(G)  # CPU tensor + python list
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
    max_train_probe: Optional[int] = 80000,
    max_val_probe: Optional[int] = 20000,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass

    # Pre-extract once (CPU)
    D_eff, _ = _split_dims(seq_ds)
    tr_dna, tr_genes, tr_y = _preextract_split(
        seq_ds["train"], D_eff, max_n=max_train_probe
    )
    va_dna, va_genes, va_y = _preextract_split(
        seq_ds["val"], D_eff, max_n=max_val_probe
    )

    graph_cache = GraphCache(go_json, gaf)

    search_space = dict(
        dim=[128],
        epochs=[8],
        batch_size=[256, 384],
        walk_len=[30, 40],
        ctx_size=[6, 8],
        walks_per_node=[12, 16],
        neg_samples=[2, 5],
        curated_only=[True],
        no_term_edges=[False],
        drop_roots=[True],
        prune_term_degree=[800, 1000],
        device=[None],
    )

    if use_optuna is None:
        try:
            import optuna  # noqa

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
        """Train Node2Vec once, score GO embeddings (intrinsic + downstream)."""
        edge_index, nodes = graph_cache.get_edge_index(
            curated_only=cfg["curated_only"],
            no_term_edges=cfg["no_term_edges"],
            drop_roots=cfg["drop_roots"],
            prune_term_degree=cfg["prune_term_degree"],
        )

        _cuda_mem_snapshot("trial/train begin")
        _host_rss_snapshot("trial/train begin")

        emb = train_node2vec(
            edge_index,
            dim=cfg["dim"],
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            device=cfg["device"],
            walk_length=cfg["walk_len"],
            context_size=cfg["ctx_size"],
            walks_per_node=cfg["walks_per_node"],
            neg_samples=cfg["neg_samples"],
            num_nodes=len(nodes),
        )

        _cuda_mem_snapshot("trial/after train (pre-clean)")
        _host_rss_snapshot("trial/after train (pre-clean)")

        # Intrinsic link prediction
        link_res = _link_pred_score(edge_index, emb, nodes, seed=seed, mode="hard")

        # Map to gene dict (CPU)
        gene2vec = {
            name[3:]: emb[i].astype(np.float32)
            for i, name in enumerate(nodes)
            if name.startswith("G::")
        }

        # Assemble GO mats (CPU)
        dim_go = int(cfg["dim"])
        tr_go = _assemble_go_matrix(tr_genes, gene2vec, dim_go)
        va_go = _assemble_go_matrix(va_genes, gene2vec, dim_go)

        # Downstream ClinVar probe
        res_val = {
            "dna": _fit_eval_hgb(tr_dna, tr_y, va_dna, va_y),
            "go": _fit_eval_hgb(tr_go, tr_y, va_go, va_y),
            "both": _fit_eval_hgb(
                np.concatenate([tr_dna, tr_go], axis=1),
                tr_y,
                np.concatenate([va_dna, va_go], axis=1),
                va_y,
            ),
        }

        return float(link_res["roc"]), {"link": link_res, **res_val}

    results = []
    if use_optuna:
        import optuna

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        pbar = tqdm(total=n_trials, desc="[HPO] trials", unit="trial")

        def objective(trial):
            cfg = sample_cfg(trial)
            link_roc, res_val = run_one_trial(cfg)
            results.append({"cfg": cfg, "scores": res_val})
            pbar.update(1)
            trial.set_user_attr("link_roc", res_val["link"]["roc"])
            trial.set_user_attr("link_pr", res_val["link"]["pr"])
            trial.set_user_attr("roc_both", res_val["both"]["roc"])
            return link_roc

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        pbar.close()
        best_cfg = {k: study.best_trial.params[k] for k in search_space.keys()}
        _, best_res_val = run_one_trial(best_cfg)
    else:
        raise ValueError("[HPO] Optuna not available; random search.")

    # ----- Refit best with more epochs -----
    print(
        f"\n[REFIT] best link ROC={best_res_val['link']['roc']:.4f} "
        f"→ epochs={refit_epochs}"
    )
    refit_cfg = dict(best_cfg)
    refit_cfg["epochs"] = refit_epochs

    edge_index, nodes = graph_cache.get_edge_index(
        curated_only=refit_cfg["curated_only"],
        no_term_edges=refit_cfg["no_term_edges"],
        drop_roots=refit_cfg["drop_roots"],
        prune_term_degree=refit_cfg["prune_term_degree"],
    )

    emb_final = train_node2vec(
        edge_index,
        dim=refit_cfg["dim"],
        epochs=refit_cfg["epochs"],
        batch_size=refit_cfg["batch_size"],
        device=refit_cfg["device"],
        walk_length=refit_cfg["walk_len"],
        context_size=refit_cfg["ctx_size"],
        walks_per_node=refit_cfg["walks_per_node"],
        neg_samples=refit_cfg["neg_samples"],
        num_nodes=len(nodes),
    )

    # Save genes only
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
    print("\n[GO HPO] Validation scores")
    print("Intrinsic link-pred (GO graph):")
    print(f"  ROC={s['link']['roc']:.4f}  PR={s['link']['pr']:.4f}")
    print("\nClinVar probe (HGB, downstream):")
    print(f"{'Feat':6} {'Acc':>8} {'F1':>8} {'ROC':>8} {'PR':>8}")
    print("-" * 42)
    for k in ("dna", "go", "both"):
        r = s[k]
        print(f"{k:6} {r['acc']:8.4f} {r['f1']:8.4f} {r['roc']:8.4f} {r['pr']:8.4f}")
    print("\nBest (refit) config:")
    for k, v in summary["best_cfg"].items():
        print(f"  {k:16} = {v}")
    print(f"\nSaved: {summary['final_path_npz']}")
