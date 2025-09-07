import math, time
from typing import Dict, Iterable, Tuple
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from src.clip_trainer import build_banks


# ---- uses your model API & helpers from clip_trainer.py ----
@torch.inference_mode()
def evaluate_clip_fast(
    model: torch.nn.Module,
    loader,  # val_loader_pair / test_loader_pair
    device: str = "cuda",
    k_list: Iterable[int] = (1, 5, 10, 50),
    block_cols: int = 8192,  # cols per block for streaming top-k
    use_fp16_on_device: bool = True,  # cast banks to fp16 on device for speed
    neg_samples_per_row: int = 1024,  # negatives per row for AUC/off_mean
    max_rows: int = None,  # optional cap for quick runs
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Fast CLIP eval without forming the full N×N sim matrix.

    Assumes loader yields (Xs, Xt, y[, meta]) and model(Xs,Xt)->(logits, zs, zt) with zs/zt L2-normalized.
    Metrics:
      - Recall@K (seq→text and text→seq) via streamed global top-k
      - Mean/median rank (approx) can be omitted for speed (not computed here)
      - diag_mean exact; off_mean & AUC via negative sampling
    """
    model.eval().to(device)

    # 1) Build banks (CPU float32), using your helper
    Zs_cpu, Zt_cpu = build_banks(loader, model)  # already L2-normalized by model
    if max_rows is not None:
        Zs_cpu = Zs_cpu[:max_rows].contiguous()
        Zt_cpu = Zt_cpu[:max_rows].contiguous()
    N, d = Zs_cpu.shape
    if verbose:
        print(f"[EvalFast] banks: N={N} d={d}")

    # diag mean exact (no matmul): ⟨zsi, zti⟩ = (Zs*Zt).sum(-1)
    diag_sim = (Zs_cpu * Zt_cpu).sum(dim=1).numpy()
    diag_mean = float(diag_sim.mean())

    # 2) Move Zs once; then stream Zt in column blocks
    Zs_dev = Zs_cpu.to(device, non_blocking=True)
    Zt_dev_full = None  # keep on CPU; we block-move slices
    if use_fp16_on_device and device == "cuda":
        Zs_dev = Zs_dev.half()

    Kmax = max(k_list) if len(tuple(k_list)) else 10
    # running global top-K per row for s2t
    topv_s2t = torch.full((N, Kmax), -float("inf"), device=device)
    topi_s2t = torch.full((N, Kmax), -1, dtype=torch.int32, device=device)

    # stream columns
    for c0 in range(0, N, block_cols):
        c1 = min(N, c0 + block_cols)
        Zt_blk = Zt_cpu[c0:c1].to(device, non_blocking=True)
        if use_fp16_on_device and device == "cuda":
            Zt_blk = Zt_blk.half()
        # [N, d] @ [d, B] -> [N, B]
        sims = Zs_dev @ Zt_blk.t()  # cosine since zs/zt are unit vectors

        # merge into running top-K
        # concat previous top-K values with new block then take top-K again
        cat_vals = torch.cat([topv_s2t, sims], dim=1)
        # build matching indices: old top-K indices already global, new need offset
        old_idx = topi_s2t
        new_idx = (
            torch.arange(c0, c1, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(N, c1 - c0)
        )
        cat_idx = torch.cat([old_idx, new_idx], dim=1)

        topv_s2t, top_pos = torch.topk(cat_vals, Kmax, dim=1)
        topi_s2t = torch.gather(cat_idx, 1, top_pos)

    # recall@K for s2t
    arangeN = torch.arange(N, device=device, dtype=torch.int32).unsqueeze(1)
    metrics: Dict[str, float] = {}
    for K in k_list:
        hits = (topi_s2t[:, :K] == arangeN).any(dim=1).float().mean().item()
        metrics[f"s2t_r@{K}"] = float(hits)

    # 3) text→seq (same trick, swap roles)
    topv_t2s = torch.full((N, Kmax), -float("inf"), device=device)
    topi_t2s = torch.full((N, Kmax), -1, dtype=torch.int32, device=device)

    Zt_dev = Zt_cpu.to(device, non_blocking=True)
    if use_fp16_on_device and device == "cuda":
        Zt_dev = Zt_dev.half()

    for c0 in range(0, N, block_cols):
        c1 = min(N, c0 + block_cols)
        Zs_blk = Zs_cpu[c0:c1].to(device, non_blocking=True)
        if use_fp16_on_device and device == "cuda":
            Zs_blk = Zs_blk.half()
        sims = Zt_dev @ Zs_blk.t()  # [N, d] @ [d, B] -> [N, B]

        cat_vals = torch.cat([topv_t2s, sims], dim=1)
        old_idx = topi_t2s
        new_idx = (
            torch.arange(c0, c1, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(N, c1 - c0)
        )
        cat_idx = torch.cat([old_idx, new_idx], dim=1)

        topv_t2s, top_pos = torch.topk(cat_vals, Kmax, dim=1)
        topi_t2s = torch.gather(cat_idx, 1, top_pos)

    for K in k_list:
        hits = (topi_t2s[:, :K] == arangeN).any(dim=1).float().mean().item()
        metrics[f"t2s_r@{K}"] = float(hits)

    # 4) Neg-sampled off_mean and AUC (fast)
    # 4) Neg-sampled off_mean and AUC (fast, all on-device, no diagonal)
    neg_k = min(neg_samples_per_row, N - 1)

    # Make a CUDA generator if we're on CUDA (to avoid device mismatch)
    gen = None
    if device == "cuda":
        gen = torch.Generator(device="cuda").manual_seed(0)
    else:
        gen = torch.Generator().manual_seed(0)

    # Sample integers in [0, N-2] and then "skip over" the diagonal per row:
    # idx = r + (r >= i)   where i is the row index
    r = torch.randint(0, max(1, N - 1), (N, neg_k), device=device, generator=gen)
    row_ids = torch.arange(N, device=device).unsqueeze(1)  # [N,1]
    neg_idx = r + (r >= row_ids)  # [N,neg_k], in [0..N-1]\{i}

    # positives (exact diag) — reuse diag_sim computed above
    pos_scores = torch.from_numpy(diag_sim).to(device)

    # negatives: batched dot(Zs_i, Zt_j) for j in neg_idx[i,*]
    # Move Zt once (we already have Zs_dev above)
    Zt_dev_all = Zt_cpu.to(device, non_blocking=True)
    if use_fp16_on_device and device == "cuda":
        Zt_dev_all = Zt_dev_all.half()

    neg_scores = []
    bs = 4096
    for r0 in range(0, N, bs):
        r1 = min(N, r0 + bs)
        Zs_blk = Zs_cpu[r0:r1].to(device, non_blocking=True)
        if use_fp16_on_device and device == "cuda":
            Zs_blk = Zs_blk.half()
        idx_blk = neg_idx[r0:r1]  # [B, neg_k]
        Zt_blk = Zt_dev_all.index_select(0, idx_blk.reshape(-1))  # [B*neg_k, d]
        Zt_blk = Zt_blk.view(idx_blk.size(0), idx_blk.size(1), -1)  # [B, neg_k, d]
        s = (Zs_blk.unsqueeze(1) * Zt_blk).sum(dim=-1)  # [B, neg_k]
        neg_scores.append(s)
    neg_scores = torch.cat(neg_scores, dim=0).reshape(-1)  # [N*neg_k]

    off_mean = float(neg_scores.mean().item())

    # AUC with sampled negatives
    y_true = torch.cat(
        [torch.ones(N, device=device), torch.zeros(N * neg_k, device=device)]
    )
    scores = torch.cat([pos_scores, neg_scores])
    auc = roc_auc_score(y_true.cpu().numpy(), scores.detach().float().cpu().numpy())
    metrics["pair_vs_impostor_auc"] = float(auc)

    metrics.update(
        {
            "diag_mean": diag_mean,
            "off_mean": off_mean,
        }
    )

    if verbose:
        print(
            "[EvalFast] "
            + " ".join([f"s2t r@{K}={metrics[f's2t_r@{K}']:.3f}" for K in k_list])
            + " | "
            + " ".join([f"t2s r@{K}={metrics[f't2s_r@{K}']:.3f}" for K in k_list])
            + f" | AUC={metrics['pair_vs_impostor_auc']:.3f} "
            f"(diag_mean={diag_mean:.4f}, off_mean~{off_mean:.4f}) "
        )
    return metrics
