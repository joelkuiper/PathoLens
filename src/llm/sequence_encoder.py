"""Lightweight 1D CNN encoders for DNA / protein token sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn


def _ensure_float(
    t: torch.Tensor,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Move tensor to the requested device/dtype while keeping floats lightweight."""

    target_dtype: torch.dtype
    if dtype is not None:
        target_dtype = dtype
    elif torch.is_floating_point(t):
        target_dtype = t.dtype
    else:
        target_dtype = torch.float32

    if device is None:
        device = t.device

    return t.to(device=device, dtype=target_dtype)


class ConvBlock(nn.Module):
    """Depthwise-style residual block with GELU activations."""

    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=1,
        )
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply the block, respecting optional padding mask."""

        if mask is not None:
            x = x * mask
        y = self.conv(x)
        y = y.transpose(1, 2)
        y = self.norm(y)
        y = self.act(y)
        y = y.transpose(1, 2)
        if mask is not None:
            y = y * mask
        return x + y


class QueryPooling(nn.Module):
    """Attention-style pooling using learnable query vectors."""

    def __init__(self, channels: int, k: int) -> None:
        super().__init__()
        self.k = int(k)
        self.scale = channels**-0.5
        self.queries = nn.Parameter(torch.randn(self.k, channels))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pool sequence features into ``k`` tokens."""

        B, L, C = x.shape
        queries = self.queries.to(x.dtype)
        scores = torch.einsum("bld,kd->bkl", x, queries) * self.scale
        if mask is not None:
            mask_bool = mask.bool()
            valid = mask_bool.any(dim=-1)
            finfo_min = torch.finfo(x.dtype).min
            if valid.all():
                pad = (~mask_bool).unsqueeze(1)
                scores = scores.masked_fill(pad, finfo_min)
                weights = torch.softmax(scores, dim=-1)
            else:
                weights = torch.zeros_like(scores)
                if valid.any():
                    pad = (~mask_bool[valid]).unsqueeze(1)
                    valid_scores = scores[valid].masked_fill(pad, finfo_min)
                    weights[valid] = torch.softmax(valid_scores, dim=-1)
                # rows with no valid positions remain zeros
        else:
            weights = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bkl,bld->bkd", weights, x)
        return pooled


@dataclass
class SequenceInputs:
    wt: torch.Tensor
    mt: torch.Tensor
    mask: torch.Tensor
    edit_mask: Optional[torch.Tensor] = None
    gap_mask: Optional[torch.Tensor] = None
    extra_masks: Optional[Sequence[torch.Tensor]] = None
    pos: Optional[torch.Tensor] = None


class SequenceCNNEncoder(nn.Module):
    """Encode WT/MT token sequences with a lightweight CNN."""

    def __init__(
        self,
        *,
        embed_dim: int,
        conv_dim: int,
        conv_layers: int,
        k_tokens: int,
        out_dim: int,
        scalar_features: int = 0,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.conv_dim = int(conv_dim)
        self.conv_layers = int(conv_layers)
        self.k_tokens = int(k_tokens)
        self.out_dim = int(out_dim)

        feature_dim = self.embed_dim * 3 + int(scalar_features)
        self.has_pos = False
        self.scalar_features = 0

        self.input_proj: Optional[nn.Linear] = None
        self.blocks = nn.ModuleList()
        self.pool = QueryPooling(self.conv_dim, self.k_tokens)
        self.out_proj = nn.Linear(self.conv_dim, self.out_dim)

        self._build(conv_layers, feature_dim)

    def _build(self, conv_layers: int, feature_dim: int) -> None:
        self.input_proj = nn.Linear(feature_dim, self.conv_dim)
        for _ in range(conv_layers):
            self.blocks.append(ConvBlock(self.conv_dim))

    def forward(self, inputs: SequenceInputs) -> torch.Tensor:
        target_device = inputs.wt.device
        target_dtype = (
            inputs.wt.dtype if torch.is_floating_point(inputs.wt) else torch.float32
        )

        wt = _ensure_float(inputs.wt, device=target_device, dtype=target_dtype)
        mt = _ensure_float(inputs.mt, device=target_device, dtype=target_dtype)
        mask = _ensure_float(inputs.mask, device=target_device, dtype=target_dtype)
        mask = mask.clamp(0.0, 1.0)
        mask = mask.unsqueeze(1)  # [B, 1, L]

        delta = mt - wt
        feats = [wt, mt, delta]

        if inputs.pos is not None:
            pos = _ensure_float(inputs.pos, device=target_device, dtype=target_dtype)
            feats.append(pos)
        if inputs.edit_mask is not None:
            feats.append(
                _ensure_float(
                    inputs.edit_mask, device=target_device, dtype=target_dtype
                )
            )
        if inputs.gap_mask is not None:
            feats.append(
                _ensure_float(inputs.gap_mask, device=target_device, dtype=target_dtype)
            )
        if inputs.extra_masks is not None:
            for extra in inputs.extra_masks:
                feats.append(
                    _ensure_float(extra, device=target_device, dtype=target_dtype)
                )

        x = torch.cat(feats, dim=-1)
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x, mask)
        x = x.transpose(1, 2)
        x = x * mask.transpose(1, 2)

        pooled = self.pool(x, mask.squeeze(1))
        tokens = self.out_proj(pooled)
        return tokens


__all__ = ["SequenceCNNEncoder", "SequenceInputs"]
