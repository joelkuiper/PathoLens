#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import gzip
import json
import re
from typing import List, Tuple
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.nn.models import Node2Vec
import multiprocessing
import sys


N_PROC = multiprocessing.cpu_count()

GO_ID_RX = re.compile(r"(GO[:_]\d{7})")
GO_ROOTS = {"GO:0008150", "GO:0003674", "GO:0005575"}  # BP, MF, CC


def norm_go_id(raw: str) -> str | None:
    if not isinstance(raw, str):
        return None
    m = GO_ID_RX.search(raw)
    return m.group(1).replace("_", ":") if m else None


def load_go_term_edges(go_json_path: str) -> List[Tuple[str, str]]:
    """Parse GO JSON graphs → immediate is_a/part_of edges (child -> parent)."""
    with open(go_json_path) as f:
        doc = json.load(f)
    edges = []
    for g in doc.get("graphs", []):
        for e in g.get("edges", []) or []:
            sub, obj = norm_go_id(e.get("sub", "")), norm_go_id(e.get("obj", ""))
            pred = e.get("pred", "") or ""
            if not (sub and obj):
                continue
            if pred.endswith("is_a") or pred.endswith("part_of"):
                edges.append((sub, obj))
    return edges


def load_gene_term_edges(
    gaf_path: str,
    taxon: str = "9606",
    evidence_whitelist: set[str] | None = None,
    exclude_not: bool = True,
) -> List[Tuple[str, str]]:
    """Return (gene_symbol, GO_ID) edges from GAF (filter taxon/evidence/NOT)."""
    edges = []
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

            gene, goid = p[2].strip(), p[4].strip()
            if not (gene and goid.startswith("GO:")):
                continue

            tax = p[12].split("|")[0].replace("taxon:", "")
            if tax != taxon:
                continue

            ec = (p[6] or "").strip()
            if evidence_whitelist and ec not in evidence_whitelist:
                continue

            edges.append((gene, goid))
    return edges


def build_graph(
    term_edges: List[Tuple[str, str]],
    gene_term_edges: List[Tuple[str, str]],
    include_term_edges: bool,
    drop_roots: bool = True,
    prune_term_degree: int | None = None,
) -> nx.Graph:
    """Build undirected bipartite graph + optional term-term edges; prune if requested."""
    G = nx.Graph()

    # Term-term (child-parent) edges
    if include_term_edges:
        for c, p in term_edges:
            if drop_roots and (c in GO_ROOTS or p in GO_ROOTS):
                continue
            G.add_edge(f"T::{c}", f"T::{p}")

    # Gene-term edges
    for g, t in gene_term_edges:
        if drop_roots and (t in GO_ROOTS):
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


def train_node2vec(
    edge_index,
    dim=128,
    epochs=2,
    batch_size=128,
    device=None,
    walk_length=40,
    context_size=5,
    walks_per_node=5,
    neg_samples=2,
    lr=0.01,
    num_nodes: int | None = None,
):
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

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

    num_workers = 0 if sys.platform == "darwin" else min(4, N_PROC)
    loader = model.loader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=(device == "cuda"),
    )

    opt = torch.optim.SparseAdam(list(model.parameters()), lr=lr)

    model.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        for pos_rw, neg_rw in tqdm(loader, desc=f"epoch {ep}/{epochs}"):
            opt.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            opt.step()
            total += float(loss)
        print(f"epoch {ep} avg loss {total / max(1, len(loader)):.4f}")

    with torch.no_grad():
        emb = model.embedding.weight.detach().cpu().numpy().astype("float32")
    return emb


def save_gene_embeddings(out_prefix: str, nodes: list[str], emb: np.ndarray):
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

    # Compact machine-readable bundle
    np.savez_compressed(f"{out_prefix}_genes.npz", emb=gene_vecs, genes=gene_names_arr)

    # Human-readable TSV
    with open(f"{out_prefix}_genes.tsv", "w") as f:
        for name, vec in zip(gene_names, gene_vecs):
            f.write(name + "\t" + " ".join(f"{x:.6f}" for x in vec) + "\n")

    print(f"Saved embeddings: {gene_vecs.shape} → {out_prefix}_genes.npz and .tsv")


def main():
    ap = argparse.ArgumentParser(description="GO+GAF Node2Vec (genes only output)")
    ap.add_argument("--go-json", required=True, help="GO JSON with graphs[].edges[].")
    ap.add_argument("--gaf", required=True, help="GAF file (can be .gz).")
    ap.add_argument("--out-prefix", required=True, help="Output prefix.")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--walk-len", type=int, default=40)
    ap.add_argument("--walks-per-node", type=int, default=5)
    ap.add_argument("--ctx-size", type=int, default=5)
    ap.add_argument("--neg-samples", type=int, default=2)
    ap.add_argument(
        "--curated-only",
        action="store_true",
        help="Keep only curated evidence (EXP/IDA/IPI/IMP/IGI/IEP/TAS/IC).",
    )
    ap.add_argument(
        "--no-term-edges",
        action="store_true",
        help="Do not include GO term-term edges.",
    )
    ap.add_argument(
        "--drop-roots", action="store_true", help="Drop GO root terms (recommended)."
    )
    ap.add_argument(
        "--prune-term-degree",
        type=int,
        default=None,
        help="Remove term nodes with degree >= this (prune hubs).",
    )
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    ev_wl = (
        {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC"}
        if args.curated_only
        else None
    )

    print("Parsing GO + GAF…")
    term_edges = [] if args.no_term_edges else load_go_term_edges(args.go_json)
    gene_term_edges = load_gene_term_edges(
        args.gaf, evidence_whitelist=ev_wl, exclude_not=True
    )
    print(
        f"  term edges: {len(term_edges):,}, gene-term edges: {len(gene_term_edges):,}"
    )

    G = build_graph(
        term_edges,
        gene_term_edges,
        include_term_edges=not args.no_term_edges,
        drop_roots=args.drop_roots,
        prune_term_degree=args.prune_term_degree,
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
            num_nodes=len(nodes),
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
            )
        else:
            raise

    save_gene_embeddings(args.out_prefix, nodes, emb)


if __name__ == "__main__":
    main()
