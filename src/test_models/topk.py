from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import math
import torch
import torch.nn as nn


@dataclass(frozen=True)
class TopKConfig:
    enabled: bool = False

    # Reference-style controls:
    keep_rate: Sequence[float] = (1.0,)          # e.g. [0.7] or [0.9,0.8,...]
    reduction_loc: Sequence[int] = tuple()       # e.g. [3,6,9]
    # If keep_rate has length 1 and reduction_loc has >1, exponentiate:
    # token_ratio[i] = keep_rate[0] ** (idx+1), exactly like your reference.
    exponentiate_single_keep_rate: bool = True

    # Minor options:
    sorted_topk: bool = True                     # reference uses sorted=True
    init_n_override: Optional[int] = None        # if you ever want to force init_n


class AttentionTopKFromExisting(nn.Module):
    """
    Reference-aligned Attention_TopK that:
      - reuses the original timm attention weights/modules (qkv/proj/drop)
      - caches attention and returns (x_attn, index, idx) like the reference
      - uses absolute budget: left_tokens = int(keep_rate * init_n)
    """

    def __init__(self, attn: nn.Module, dim: int, keep_rate: float, init_n: int, num_special_tokens: int, sorted_topk: bool = True):
        super().__init__()
        self.num_heads = attn.num_heads
        self.scale = getattr(attn, "scale", (dim // attn.num_heads) ** -0.5)
        self.sorted_topk = bool(sorted_topk)

        # Reuse trained modules:
        self.qkv = attn.qkv
        self.attn_drop = attn.attn_drop
        self.proj = attn.proj
        self.proj_drop = attn.proj_drop

        self.keep_rate = float(keep_rate)
        assert 0.0 < self.keep_rate <= 1.0, f"keep_rate must be in (0,1], got {self.keep_rate}"

        # init_n = original number of patch tokens (e.g., 14*14 = 196 for 224/16)
        self.init_n = int(init_n)
        self.num_special_tokens = int(num_special_tokens)  # 1 (cls) or 2 (cls+dist)
        self.last_index: Optional[torch.Tensor] = None  # [B, K, C]
        self.last_idx: Optional[torch.Tensor] = None    # [B, K]

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        """
        Returns:
          x_attn: [B, N, C]
          index:  [B, left_tokens, C] gather index for patch tokens only (None if no pruning)
          idx:    [B, left_tokens] indices into patch-token sequence (None if no pruning)
        """
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
          attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)


        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)

        if self.keep_rate >= 1.0:
            return x_attn, None, None

        # Absolute budget based on original patch grid (reference behavior)
        left_tokens = int(self.keep_rate * self.init_n)

        # Current patch count may already be smaller due to earlier pruning
        cur_patches = N - self.num_special_tokens
        if cur_patches <= 1:
            return x_attn, None, None

        # If budget matches what we currently have, skip pruning
        if left_tokens >= cur_patches:
            return x_attn, None, None

        left_tokens = max(1, left_tokens)

        # Score patches by CLS attention to patches (exclude special tokens)
        # attn: [B, H, N, N]
        cls_attn = attn[:, :, 0, self.num_special_tokens:]          # [B, H, cur_patches]
        cls_attn = cls_attn.mean(dim=1)                             # [B, cur_patches]

        _, idx = torch.topk(
            cls_attn,
            k=left_tokens,
            dim=1,
            largest=True,
            sorted=True,  # reference uses sorted=True
        )                                                           # [B, left_tokens]

        self.last_idx = idx
        self.last_index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, K, C]
        return x_attn

class BlockTopKAdapter(nn.Module):
    """
    timm Block-compatible adapter (supports drop_path1/drop_path2 and ls1/ls2).
    Prunes immediately after attention residual, matching the reference.
    """

    def __init__(
        self,
        orig_block: nn.Module,
        embed_dim: int,
        keep_rate: float,
        init_n: int,
        num_special_tokens: int,
    ):
        super().__init__()
        self.norm1 = orig_block.norm1
        self.norm2 = orig_block.norm2
        self.mlp = orig_block.mlp

        # timm v0.9+ uses these names
        self.drop_path1 = getattr(orig_block, "drop_path1", nn.Identity())
        self.drop_path2 = getattr(orig_block, "drop_path2", nn.Identity())

        # LayerScale (can be Identity in some variants)
        self.ls1 = getattr(orig_block, "ls1", nn.Identity())
        self.ls2 = getattr(orig_block, "ls2", nn.Identity())

        self.num_special_tokens = int(num_special_tokens)

        self.attn = AttentionTopKFromExisting(
    attn=orig_block.attn,
    dim=embed_dim,
    keep_rate=keep_rate,
    init_n=init_n,
    num_special_tokens=num_special_tokens,
    sorted_topk=True,  # or cfg.sorted_topk via apply_topk_pruning
)


    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.drop_path1(self.ls1(attn_out))

        index = self.attn.last_index
        if index is not None:
            special = x[:, : self.num_special_tokens]
            patches = x[:, self.num_special_tokens :]
            kept = torch.gather(patches, dim=1, index=index)
            x = torch.cat([special, kept], dim=1)

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

def _compute_token_ratio_full(depth: int, cfg: TopKConfig) -> List[float]:
    keep_rate = list(float(x) for x in cfg.keep_rate)
    pruning_loc = list(int(x) for x in cfg.reduction_loc)

    if not pruning_loc:
        return [1.0 for _ in range(depth)]

    # Handle single keep_rate with multiple reduction locations
    if len(keep_rate) == 1 and len(pruning_loc) > 1:
        if cfg.exponentiate_single_keep_rate:
            # Reference behavior: exponentiate
            base = keep_rate[0]
            keep_rate = [base ** (i + 1) for i in range(len(pruning_loc))]
        else:
            # Replicate the same value across all locations
            keep_rate = [keep_rate[0] for _ in range(len(pruning_loc))]

    if len(keep_rate) != len(pruning_loc):
        raise ValueError(f"Mismatch: reduction_loc={pruning_loc} vs keep_rate={keep_rate}")

    token_ratio_full = [1.0 for _ in range(depth)]
    for r, loc in zip(keep_rate, pruning_loc):
        if 0 <= loc < depth:
            token_ratio_full[loc] = float(r)
    return token_ratio_full


def apply_topk_pruning(model: nn.Module, cfg: TopKConfig) -> nn.Module:
    """
    In-place transform: replaces model.blocks[i] with BlockTopKAdapter at all depths,
    with keep_rate schedule matching the reference (token_ratio_full).
    Off => returns model unchanged.

    IMPORTANT: Call this AFTER you shrink the head for ImageNet100 (as you already do),
    to keep concerns separated.
    """
    if not cfg.enabled:
        return model

    if not hasattr(model, "blocks"):
        raise TypeError("TopK pruning expects a timm ViT/DeiT-like model with `.blocks`")

    depth = len(model.blocks)
    token_ratio_full = _compute_token_ratio_full(depth, cfg)

    # Determine init_n (original patch tokens)
    if cfg.init_n_override is not None:
        init_n = int(cfg.init_n_override)
    else:
        if not hasattr(model, "patch_embed") or not hasattr(model.patch_embed, "num_patches"):
            raise TypeError("Model missing patch_embed.num_patches; cannot infer init_n")
        init_n = int(model.patch_embed.num_patches)

    # Special tokens: CLS always, dist token optionally
    num_special_tokens = 2 if hasattr(model, "dist_token") and model.dist_token is not None else 1

    # If all keep rates are 1.0, do nothing (keeps baseline identical)
    if all(abs(r - 1.0) < 1e-12 for r in token_ratio_full):
        return model

    # Replace blocks with adapters (reuse original weights/modules)
    for i in range(depth):
      keep_r = float(token_ratio_full[i])
      if keep_r >= 1.0:
          continue  # leave original block untouched

      orig_blk = model.blocks[i]
      embed_dim = getattr(model, "embed_dim", None)
      if embed_dim is None:
          embed_dim = orig_blk.norm1.normalized_shape[0]

      model.blocks[i] = BlockTopKAdapter(
          orig_block=orig_blk,
          embed_dim=int(embed_dim),
          keep_rate=keep_r,
          init_n=init_n,
          num_special_tokens=num_special_tokens,
      )


    return model
