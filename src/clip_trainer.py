# src/clip_trainer.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Model
# -------------------------
class SeqTextCLIP(nn.Module):
    """
    Simple CLIP-style heads:
      - proj_seq: (B, D_seq) -> (B, d_out)
      - proj_text: (B, D_txt) -> (B, d_out)
    Returns cosine-similarity logits scaled by exp(logit_scale).

    Notes:
      - We L2-normalize outputs before computing logits.
      - Temperature (logit_scale) is frozen by default (1/0.07).
    """

    def __init__(
        self,
        D_seq: int,
        D_txt: int,
        d_out: int = 512,
        hidden: int = 1024,
        p: float = 0.1,
        freeze_tau: bool = True,
        init_tau: float = 0.07,
    ):
        super().__init__()
        self.proj_seq = nn.Sequential(
            nn.LayerNorm(D_seq),
            nn.Linear(D_seq, hidden),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden, d_out),
        )
        self.proj_text = nn.Sequential(
            nn.LayerNorm(D_txt),
            nn.Linear(D_txt, hidden),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden, d_out),
        )
        # temperature
        self.logit_scale = nn.Parameter(
            torch.tensor(np.log(1.0 / float(init_tau)), dtype=torch.float32),
            requires_grad=not freeze_tau,
        )

    def forward(
        self, Xs: torch.Tensor, Xt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zs = self.proj_seq(Xs)  # (B, d)
        zt = self.proj_text(Xt)  # (B, d)
        # Normalize before logits
        zs = F.normalize(zs, dim=-1)
        zt = F.normalize(zt, dim=-1)
        logits = torch.exp(self.logit_scale) * (zs @ zt.t())
        return logits, zs, zt


# -------------------------
# Losses
# -------------------------
def clip_ce_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Classic CLIP symmetric cross-entropy with 1:1 positives on the diagonal.
    """
    B = logits.size(0)
    labels = torch.arange(B, device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i + loss_t)


def supcon_clip_loss(
    zs: torch.Tensor,
    zt: torch.Tensor,
    gene_ids: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised contrastive CLIP with multi-positives:
      - positives are pairs with the SAME gene_id (excluding self-pairs)
      - symmetric (seq->text and text->seq)

    Args:
      zs, zt: (B, d) unit-normalized
      gene_ids: (B,) int tensor; samples with same id are positives
    """
    sim = (zs @ zt.t()) / temperature  # (B,B)
    same = gene_ids.unsqueeze(1).eq(gene_ids.unsqueeze(0))  # (B,B) bool
    pos_mask = same & ~torch.eye(same.size(0), dtype=torch.bool, device=same.device)

    # seq -> text
    logp = sim.log_softmax(dim=1)
    denom = pos_mask.sum(dim=1).clamp(min=1)  # rows with at least 1 positive
    # if some rows have zero positives, they contribute 0 (masked_select then view with size 0)
    masked_vals = torch.zeros_like(denom, dtype=logp.dtype)
    if pos_mask.any():
        masked_vals = logp.masked_select(pos_mask).view(sim.size(0), -1).sum(dim=1)
    loss_st = -(masked_vals / denom).mean()

    # text -> seq
    logp_t = sim.t().log_softmax(dim=1)
    masked_vals_t = torch.zeros_like(denom, dtype=logp_t.dtype)
    if pos_mask.any():
        masked_vals_t = logp_t.masked_select(pos_mask).view(sim.size(0), -1).sum(dim=1)
    loss_ts = -(masked_vals_t / denom).mean()

    return 0.5 * (loss_st + loss_ts)


# -------------------------
# Utilities
# -------------------------
def _unpack_batch(
    b,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    """
    Accepts either (Xs, Xt, y) or (Xs, Xt, y, meta).
    Returns (Xs, Xt, y, meta_dict).
    """
    if len(b) == 3:
        Xs, Xt, y = b
        meta = {}
    elif len(b) == 4:
        Xs, Xt, y, meta = b
        if meta is None:
            meta = {}
    else:
        raise ValueError(f"Unexpected batch len={len(b)}")
    return Xs, Xt, y, meta


def _gene_ids_from_meta(
    meta: Dict[str, Any], device: torch.device
) -> Optional[torch.Tensor]:
    """
    Build per-batch integer ids from meta["gene"] (list[str]) if present.
    """
    if "gene" not in meta:
        return None
    genes = meta["gene"]
    if not isinstance(genes, (list, tuple)):
        return None
    # stable mapping within the batch
    uniq = {}
    gids = []
    for g in genes:
        k = str(g).upper()
        if k not in uniq:
            uniq[k] = len(uniq)
        gids.append(uniq[k])
    return torch.tensor(gids, device=device, dtype=torch.long)


# -------------------------
# Eval helpers
# -------------------------
@torch.inference_mode()
def clip_eval(
    loader, model: nn.Module, loss_mode: str = "ce", temperature: float = 0.07
) -> float:
    """
    Average validation loss over a loader.
      loss_mode: "supcon" (multi-positive) or "ce" (1:1)
    """
    model.eval()
    tot, n = 0.0, 0
    for b in loader:
        Xs, Xt, y, meta = _unpack_batch(b)
        Xs, Xt = Xs.cuda(non_blocking=True), Xt.cuda(non_blocking=True)
        logits, zs, zt = model(Xs, Xt)
        if loss_mode == "supcon":
            gid = _gene_ids_from_meta(meta, Xs.device)
            if gid is None:
                # fallback: assume 1:1
                loss = clip_ce_loss(logits)
            else:
                loss = supcon_clip_loss(zs, zt, gid, temperature=temperature)
        else:
            loss = clip_ce_loss(logits)
        bs = Xs.size(0)
        tot += float(loss) * bs
        n += bs
    return tot / max(1, n)


@torch.inference_mode()
def build_banks(loader, model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create L2-normalized banks Zs, Zt over a loader.
    """
    model.eval()
    Zs, Zt = [], []
    for b in loader:
        Xs, Xt, y, meta = _unpack_batch(b)
        Xs, Xt = Xs.cuda(non_blocking=True), Xt.cuda(non_blocking=True)
        _, zs, zt = model(Xs, Xt)
        Zs.append(zs.detach().cpu())
        Zt.append(zt.detach().cpu())
    return torch.cat(Zs, 0), torch.cat(Zt, 0)


def recall_at_k(sim: torch.Tensor, k: int) -> float:
    """
    Given similarity matrix sim (N x N), compute diagonal Recall@k.
    """
    topk = sim.topk(k, dim=1).indices
    N = sim.size(0)
    diag = torch.arange(N, device=sim.device).unsqueeze(1)
    hits = (topk == diag).any(dim=1).float().mean().item()
    return hits


# -------------------------
# Training
# -------------------------
@dataclass
class TrainConfig:
    epochs: int = 2
    lr: float = 3e-4
    wd: float = 5e-2
    grad_clip: float = 1.0
    amp_dtype: torch.dtype = torch.bfloat16
    loss_mode: str = "supcon"  # "supcon" or "ce"
    temperature: float = 0.07


def train_epoch(
    train_loader,
    val_loader,
    model: nn.Module,
    epochs: int = 1,
    lr: float = 3e-4,
    wd: float = 5e-2,
    grad_clip: float = 1.0,
    amp_dtype: torch.dtype = torch.bfloat16,
    loss_mode: str = "supcon",
    temperature: float = 0.07,
):
    """
    Trains for `epochs` and returns {"val_loss": float, "state": state_dict_on_cpu}.
    - loss_mode "supcon": multi-positive loss (same-gene positives).
    - loss_mode "ce": classic 1:1 CLIP loss.
    """
    cfg = TrainConfig(epochs, lr, wd, grad_clip, amp_dtype, loss_mode, temperature)

    model.cuda()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    # Cosine schedule across all steps
    total_steps = cfg.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps))

    scaler = torch.amp.GradScaler("cuda", enabled=True)  # new API

    best = {"val_loss": 1e9, "state": None}
    step = 0

    for ep in range(1, cfg.epochs + 1):
        model.train()
        for b in train_loader:
            Xs, Xt, y, meta = _unpack_batch(b)
            Xs = Xs.cuda(non_blocking=True)
            Xt = Xt.cuda(non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=cfg.amp_dtype):
                logits, zs, zt = model(Xs, Xt)

                if cfg.loss_mode == "supcon":
                    gid = _gene_ids_from_meta(meta, Xs.device)
                    if gid is None:
                        loss = clip_ce_loss(logits)
                    else:
                        loss = supcon_clip_loss(
                            zs, zt, gid, temperature=cfg.temperature
                        )
                else:
                    loss = clip_ce_loss(logits)

            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()

            step += 1
            if step % 100 == 0:
                print(
                    f"ep{ep} step{step} loss {loss.item():.4f} scale={model.logit_scale.exp().item():.3f}"
                )

        vloss = clip_eval(
            val_loader, model, loss_mode=cfg.loss_mode, temperature=cfg.temperature
        )
        print(f"[val] ep{ep} loss {vloss:.4f}")

        if vloss < best["val_loss"]:
            best["val_loss"] = vloss
            best["state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    return best
