from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Set, Union

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TopKConfig:
    enabled: bool = False
    keep_rate: float = 1.0  # fraction of patch tokens to keep (CLS always kept)
    layers: Union[str, Iterable[int]] = "all"  # "all" or iterable of 0-based block indices
    score: str = "cls_attn"  # currently supported: "cls_attn"
    preserve_token_order: bool = True  # keep tokens in original order after selection


class AttentionWithCache(nn.Module):
    """
    Drop-in replacement for timm Attention modules that caches the attention
    matrix (after softmax, before dropout) as `last_attn` with shape [B, H, N, N].

    This reimplements the common timm attention forward:
      qkv -> reshape -> scaled dot-product -> softmax -> (attn @ v) -> proj
    using the *same* underlying parameters/modules from the original attention.
    """
    def __init__(self, attn: nn.Module):
        super().__init__()
        # Keep references to the original submodules/params to avoid divergence.
        self.qkv = attn.qkv
        self.num_heads = attn.num_heads
        self.scale = attn.scale
        self.attn_drop = attn.attn_drop
        self.proj = attn.proj
        self.proj_drop = attn.proj_drop

        self.last_attn: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)  # [B, N, 3*C]
        # [3, B, H, N, head_dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B, H, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        attn = attn.softmax(dim=-1)

        # Cache *pre-dropout* softmax attention for scoring
        self.last_attn = attn

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def _normalize_layers(layers: Union[str, Iterable[int]], num_blocks: int) -> Set[int]:
    if layers == "all":
        return set(range(num_blocks))
    out = set(int(i) for i in layers)
    # silently clamp to valid range
    return {i for i in out if 0 <= i < num_blocks}


def _topk_gather_tokens(
    x: torch.Tensor,
    scores: torch.Tensor,
    keep_rate: float,
    preserve_token_order: bool,
) -> torch.Tensor:
    """
    x: [B, N, C] tokens with CLS at index 0
    scores: [B, N-1] patch scores aligned to x[:, 1:, :]
    Returns pruned x with CLS + topK patches.
    """
    B, N, C = x.shape
    num_patches = N - 1
    if num_patches <= 1:
        return x

    keep_rate = float(keep_rate)
    if keep_rate >= 1.0:
        return x
    if keep_rate <= 0.0:
        # keep only CLS
        return x[:, :1, :]

    K = int(torch.ceil(torch.tensor(num_patches * keep_rate)).item())
    K = max(1, min(K, num_patches))

    topk_idx = scores.topk(K, dim=-1, largest=True, sorted=False).indices  # [B, K], in [0..num_patches-1]
    if preserve_token_order:
        topk_idx, _ = torch.sort(topk_idx, dim=-1)  # restore original spatial/token order

    gather_idx = torch.cat(
        [
            torch.zeros((B, 1), dtype=topk_idx.dtype, device=topk_idx.device),
            topk_idx + 1,  # shift by 1 because patches start at token index 1
        ],
        dim=1,
    )  # [B, K+1]

    # gather along token dimension
    gather_idx_exp = gather_idx.unsqueeze(-1).expand(-1, -1, C)  # [B, K+1, C]
    x_pruned = torch.gather(x, dim=1, index=gather_idx_exp)
    return x_pruned


class TopKWrapper(nn.Module):
    """
    Wraps a timm ViT/DeiT-like model and performs topK pruning between transformer blocks.

    Requirements for the wrapped model:
      - has .patch_embed, .pos_drop, .blocks (iterable), .norm
      - has .cls_token, .pos_embed
      - exposes .head or classifier in forward
    """
    def __init__(self, model: nn.Module, cfg: TopKConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg

        if not hasattr(model, "blocks"):
            raise TypeError("TopKWrapper expects a ViT/DeiT-style model with `.blocks`")

        self.layers = _normalize_layers(cfg.layers, num_blocks=len(model.blocks))

    def __getattr__(self, name: str):
        # Delegate attribute lookup to underlying model (keeps timm utilities working).
        if name in {"model", "cfg", "layers"}:
            return super().__getattr__(name)
        return getattr(self.model, name)

    @torch.no_grad()
    def _score_tokens(self, block_idx: int, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Returns scores [B, N-1] for patch tokens using cached attention from that block.
        """
        if self.cfg.score != "cls_attn":
            raise ValueError(f"Unsupported score method: {self.cfg.score}")

        attn_mod = getattr(self.model.blocks[block_idx], "attn", None)
        if attn_mod is None or not hasattr(attn_mod, "last_attn") or attn_mod.last_attn is None:
            # If we can't access cached attention, skip pruning (safe fallback)
            return None

        attn = attn_mod.last_attn  # [B, H, N, N]
        # CLS -> patches attention, mean over heads
        scores = attn[:, :, 0, 1:].mean(dim=1)  # [B, N-1]
        return scores

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        m = self.model

        # This mirrors typical timm VisionTransformer.forward_features()
        x = m.patch_embed(x)
        # Some timm models return (x, (H, W)) from patch_embed; handle that.
        if isinstance(x, (tuple, list)):
            x = x[0]

        B = x.shape[0]
        cls_tokens = m.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + m.pos_embed
        x = m.pos_drop(x)

        for i, blk in enumerate(m.blocks):
            x = blk(x)

            if (i in self.layers) and self.cfg.enabled and (self.cfg.keep_rate < 1.0):
                scores = self._score_tokens(i, x)
                if scores is not None:
                    x = _topk_gather_tokens(
                        x,
                        scores=scores,
                        keep_rate=self.cfg.keep_rate,
                        preserve_token_order=self.cfg.preserve_token_order,
                    )

        x = m.norm(x)
        # Most timm ViTs use CLS token as feature
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        # timm VisionTransformer usually has `head`
        if hasattr(self.model, "head"):
            return self.model.head(feats)
        # fallback to timm classifier getter
        cls = self.model.get_classifier()
        return cls(feats)


def apply_topk_pruning(model: nn.Module, cfg: TopKConfig) -> nn.Module:
    """
    In-place patch: replace each block.attn with AttentionWithCache so we can score tokens.
    Then return a wrapper that performs pruning between blocks.

    If cfg.enabled is False, returns the model unchanged.
    """
    if not cfg.enabled or cfg.keep_rate >= 1.0:
        return model

    if not hasattr(model, "blocks"):
        raise TypeError("apply_topk_pruning expects a ViT/DeiT-style model with `.blocks`")

    for blk in model.blocks:
        if hasattr(blk, "attn") and not isinstance(blk.attn, AttentionWithCache):
            blk.attn = AttentionWithCache(blk.attn)

    return TopKWrapper(model, cfg)
