#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GO+GAF → bipartite (genes–terms) + optional term–term (is_a only) graph → Node2Vec.
- Only includes GO 'is_a' hierarchy edges (no 'part_of').
- Aspect whitelist for GAF (default: P,F). Pass --aspect PFC to include all.
- Optional pruning of high-degree term hubs and optional explicit term blacklist.
- Saves **gene** embeddings only (NPZ + TSV).

Example:
  python go_node2vec.py \
    --go-json data/raw/go.json \
    --gaf data/raw/goa_human.gaf.gz \
    --out-prefix data/processed/go_n2v_pf_isA \
    --dim 256 --epochs 8 --batch-size 384 \
    --walk-len 40 --walks-per-node 12 --ctx-size 6 --neg-samples 5 \
    --aspect PF --drop-roots --prune-term-degree 800
"""
import argparse
import gzip
import json
import re
import multiprocessing
from typing import List, Tuple, Optional, Set

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.nn.models import Node2Vec

N_PROC = multiprocessing.cpu_count()

GO_ID_RX = re.compile(r"(GO[:_]\d{7})")
GO_ROOTS = {"GO:0008150", "GO:0003674", "GO:0005575"}  # BP, MF, CC


# -------------------------
# Parsing helpers
# -------------------------
def norm_go_id(raw: str) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    m = GO_ID_RX.search(raw)
    return m.group(1).replace("_", ":") if m else None


def load_go_term_edges(go_json_path: str) -> List[Tuple[str, str]]:
    """
    Parse GO JSON → immediate is_a edges only (child -> parent).
    We ignore 'part_of' and other predicates by default.
    """
    with open(go_json_path, "r") as f:
        doc = json.load(f)
    edges: List[Tuple[str, str]] = []
    for g in doc.get("graphs", []) or []:
        for e in g.get("edges", []) or []:
            sub, obj = norm_go_id(e.get("sub", "")), norm_go_id(e.get("obj", ""))
            pred = (e.get("pred", "") or "").lower()
            if not (sub and obj):
                continue
            # robust is_a check (GO JSON usually has 'is_a', sometimes CURIE-ish)
            if pred.endswith("is_a") or pred == "is_a" or "is_a" in pred:
                edges.append((sub, obj))
    return edges


def load_gene_term_edges(
    gaf_path: str,
    *,
    taxon: str = "9606",
    evidence_whitelist: Optional[Set[str]] = None,
    aspect_whitelist: Optional[Set[str]] = None,  # {"P","F","C"} subset
    exclude_not: bool = True,
) -> List[Tuple[str, str]]:
    """
    Return (gene_symbol, GO_ID) edges from GAF.
    Filters by: taxon, evidence code whitelist, NOT-qualifier, and Aspect (P/F/C).
    GAF v2.2 columns used:
      3: Qualifier
      4: GO ID
      6: Evidence code
      8: Aspect (P/F/C)
      12: Taxon
      2: DB Object Symbol (gene symbol)
    """
    edges: List[Tuple[str, str]] = []
    opener = gzip.open if gaf_path.endswith(".gz") else open
    with opener(gaf_path, "rt") as f:
        for ln in f:
            if not ln or ln.startswith("!"):
                continue
            p = ln.rstrip("\n").split("\t")
            if len(p) < 13:
                continue

            qualifier = (p[3] or "").strip()
            if exclude_not and ("NOT" in qualifier.split("|")):
                continue

            gene, goid = (p[2] or "").strip(), (p[4] or "").strip()
            if not (gene and goid.startswith("GO:")):
                continue

            aspect = (p[8] or "").strip()  # P/F/C
            if aspect_whitelist and aspect not in aspect_whitelist:
                continue

            tax = (p[12] or "").split("|")[0].replace("taxon:", "")
            if tax != taxon:
                continue

            ec = (p[6] or "").strip()
            if evidence_whitelist and ec not in evidence_whitelist:
                continue

            edges.append((gene, goid))
    return edges


# -------------------------
# Graph construction
# -------------------------
def build_graph(
    term_edges_is_a: List[Tuple[str, str]],
    gene_term_edges: List[Tuple[str, str]],
    *,
    include_term_edges: bool,
    drop_roots: bool = True,
    prune_term_degree: Optional[int] = None,
    term_blacklist: Optional[Set[str]] = None,
) -> nx.Graph:
    """
    Build an undirected graph:
      - gene ↔ term (bipartite)
      - optional term ↔ term (is_a only) hierarchy edges
    Post-filters:
      - drop GO roots if requested
      - prune high-degree term hubs
      - drop explicitly blacklisted terms
    """
    G = nx.Graph()

    # Term-term edges (is_a only)
    if include_term_edges:
        for c, p in term_edges_is_a:
            if drop_roots and (c in GO_ROOTS or p in GO_ROOTS):
                continue
            if term_blacklist and (c in term_blacklist or p in term_blacklist):
                continue
            G.add_edge(f"T::{c}", f"T::{p}")

    # Gene-term edges
    for g, t in gene_term_edges:
        if drop_roots and (t in GO_ROOTS):
            continue
        if term_blacklist and (t in term_blacklist):
            continue
        G.add_edge(f"G::{g}", f"T::{t}")

    # Optional pruning of very high-degree terms (generic hubs)
    if prune_term_degree is not None:
        to_drop = [
            n
            for n, deg in G.degree()
            if n.startswith("T::") and deg >= prune_term_degree
        ]
        if to_drop:
            G.remove_nodes_from(to_drop)

    # Remove isolates created by pruning (genes that lost all term edges, etc.)
    iso = list(nx.isolates(G))
    if iso:
        G.remove_nodes_from(iso)
        print(f"Pruned isolates: {len(iso):,}")

    if G.number_of_nodes() == 0:
        raise RuntimeError("Graph is empty after filtering. Loosen filters and retry.")

    return G


def graph_to_edge_index(G: nx.Graph):
    """Convert NetworkX graph → PyG edge_index + node list (bidirectional)."""
    nodes = list(G.nodes())
    n2i = {n: i for i, n in enumerate(nodes)}
    edges = []
    for u, v in G.edges():
        iu, iv = n2i[u], n2i[v]
        edges.append((iu, iv))
        edges.append((iv, iu))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index, nodes


# -------------------------
# Node2Vec training
# -------------------------
def train_node2vec(
    edge_index,
    *,
    dim: int = 128,
    epochs: int = 2,
    batch_size: int = 128,
    device: Optional[str] = None,
    walk_length: int = 40,
    context_size: int = 5,
    walks_per_node: int = 5,
    neg_samples: int = 2,
    lr: float = 0.01,
    num_nodes: Optional[int] = None,
    seed: int = 0,
    oom_backoff: bool = True,
    min_batch_size: int = 64,
):
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

    # determinism-ish
    try:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    model = Node2Vec(
        edge_index,
        num_nodes=num_nodes,
        embedding_dim=dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=neg_samples,
        sparse=True,
    ).to(device)

    # fewer workers avoids VRAM spikes; keep 0 unless you know you need more
    num_workers = 0

    def make_loader(bs: int):
        return model.loader(
            batch_size=bs,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=False,
            pin_memory=(device == "cuda"),
        )

    loader = make_loader(batch_size)
    opt = torch.optim.SparseAdam(list(model.parameters()), lr=lr)

    model.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        it = iter(loader)
        pbar = tqdm(range(len(loader)), desc=f"epoch {ep}/{epochs}")
        for _ in pbar:
            try:
                pos_rw, neg_rw = next(it)
                opt.zero_grad(set_to_none=True)
                loss = model.loss(
                    pos_rw.to(device, non_blocking=True),
                    neg_rw.to(device, non_blocking=True),
                )
                loss.backward()
                opt.step()
                total += float(loss)
            except RuntimeError as e:
                if (
                    oom_backoff
                    and "out of memory" in str(e).lower()
                    and device == "cuda"
                ):
                    torch.cuda.empty_cache()
                    new_bs = max(min_batch_size, batch_size // 2)
                    if new_bs < batch_size:
                        batch_size = new_bs
                        loader = make_loader(batch_size)
                        # stop this epoch early; next epoch resumes with smaller bs
                        break
                    raise
                raise
        print(f"epoch {ep} avg loss {total / max(1, len(loader)):.4f}")

    with torch.no_grad():
        emb = model.embedding.weight.detach().cpu().numpy().astype("float32")
    return emb


def save_gene_embeddings(out_prefix: str, nodes: List[str], emb: np.ndarray):
    """Persist only gene embeddings with explicit mapping."""
    gene_names, gene_vecs = [], []
    for i, n in enumerate(nodes):
        if n.startswith("G::"):
            gene_names.append(n[3:])  # strip "G::"
            gene_vecs.append(emb[i])

    if not gene_vecs:
        raise RuntimeError("No gene embeddings found in the graph.")

    gene_vecs = np.vstack(gene_vecs).astype("float32")
    gene_names_arr = np.array(gene_names, dtype=object)
    np.savez_compressed(f"{out_prefix}_genes.npz", emb=gene_vecs, genes=gene_names_arr)
    with open(f"{out_prefix}_genes.tsv", "w") as f:
        for name, vec in zip(gene_names, gene_vecs):
            f.write(name + "\t" + " ".join(f"{x:.6f}" for x in vec) + "\n")
    print(f"Saved embeddings: {gene_vecs.shape} → {out_prefix}_genes.npz and .tsv")


# -------------------------
# CLI
# -------------------------
def _parse_aspect(aspect_str: str) -> Optional[Set[str]]:
    # Accept strings like "PF", "PFC", "P", "PFc" (case-insensitive)
    s = (aspect_str or "").upper().strip()
    if not s:
        return None
    allowed = {"P", "F", "C"}
    picked = {ch for ch in s if ch in allowed}
    return picked or None


def _parse_term_blacklist(bl: Optional[str]) -> Optional[Set[str]]:
    if not bl:
        return None
    return {tok.strip().replace("_", ":") for tok in bl.split(",") if tok.strip()}


def main():
    ap = argparse.ArgumentParser(description="GO+GAF Node2Vec (genes only output)")
    ap.add_argument("--go-json", required=True, help="GO JSON with graphs[].edges[].")
    ap.add_argument("--gaf", required=True, help="GAF file (can be .gz).")
    ap.add_argument("--out-prefix", required=True, help="Output prefix.")
    # Embedding / training
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--walk-len", type=int, default=40)
    ap.add_argument("--walks-per-node", type=int, default=5)
    ap.add_argument("--ctx-size", type=int, default=5)
    ap.add_argument("--neg-samples", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    # Filtering / structure
    ap.add_argument(
        "--aspect",
        default="PF",
        help="Aspect whitelist over {P,F,C}. Default PF (drop C). Use PFC to include all.",
    )
    ap.add_argument(
        "--curated-only",
        action="store_true",
        help="Keep only curated evidence (EXP/IDA/IPI/IMP/IGI/IEP/TAS/IC).",
    )
    ap.add_argument(
        "--no-term-edges",
        action="store_true",
        help="Do not include GO term-term edges (is_a).",
    )
    ap.add_argument(
        "--drop-roots",
        action="store_true",
        help="Drop GO root terms (recommended).",
    )
    ap.add_argument(
        "--prune-term-degree",
        type=int,
        default=None,
        help="Remove term nodes with degree >= this (prune hubs).",
    )
    ap.add_argument(
        "--term-blacklist",
        type=str,
        default=None,
        help="Comma-separated GO IDs to drop, e.g. 'GO:0005515,GO:0005829'.",
    )
    args = ap.parse_args()

    ev_wl = (
        {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC"}
        if args.curated_only
        else None
    )
    aspects = _parse_aspect(args.aspect)
    term_blacklist = _parse_term_blacklist(args.term_blacklist)

    print("Parsing GO (is_a only) + GAF…")
    term_edges = [] if args.no_term_edges else load_go_term_edges(args.go_json)
    gene_term_edges = load_gene_term_edges(
        args.gaf,
        evidence_whitelist=ev_wl,
        aspect_whitelist=aspects,
        exclude_not=True,
    )
    print(
        f"  term edges (is_a): {len(term_edges):,} | "
        f"gene-term edges (taxon=9606, aspects={''.join(sorted(aspects)) if aspects else 'ALL'}"
        f"{', curated' if ev_wl else ''}): {len(gene_term_edges):,}"
    )

    G = build_graph(
        term_edges,
        gene_term_edges,
        include_term_edges=not args.no_term_edges,
        drop_roots=args.drop_roots,
        prune_term_degree=args.prune_term_degree,
        term_blacklist=term_blacklist,
    )
    edge_index, nodes = graph_to_edge_index(G)
    print(f"Graph: {len(nodes):,} nodes, {G.number_of_edges():,} edges")

    try:
        emb = train_node2vec(
            edge_index,
            dim=args.dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            walk_length=args.walk_len,
            context_size=args.ctx_size,
            walks_per_node=args.walks_per_node,
            neg_samples=args.neg_samples,
            lr=args.lr,
            num_nodes=len(nodes),
            seed=args.seed,
        )
    except RuntimeError as e:
        if "MPS backend out of memory" in str(e):
            print("⚠️ MPS OOM. Retrying on CPU with smaller params…")
            emb = train_node2vec(
                edge_index,
                dim=max(64, args.dim // 2),
                epochs=max(1, args.epochs // 2),
                batch_size=min(64, args.batch_size),
                device="cpu",
                walk_length=max(20, args.walk_len // 2),
                context_size=max(4, args.ctx_size // 2),
                walks_per_node=max(3, args.walks_per_node // 2),
                neg_samples=max(1, args.neg_samples // 2),
                lr=args.lr,
                num_nodes=len(nodes),
                seed=args.seed,
            )
        else:
            raise

    save_gene_embeddings(args.out_prefix, nodes, emb)


if __name__ == "__main__":
    main()
