"""
evit_tome.py  —  EViT-then-ToMe hybrid token reduction
=======================================================

Strategy
--------
Within each active block, two reduction stages run in sequence:

  1. **EViT stage** (inside attention):
       Score patch tokens by mean CLS-attention across heads.
       Keep the top-K patches; fuse the discarded ones into a single
       extra "fused" token via attention-weighted sum.
       → sequence length: S + K + 1  (S = num_special_tokens)

  2. **ToMe stage** (after residual add, before MLP):
       Run bipartite soft-matching on the surviving tokens (including the
       fused token) and merge r similar pairs.
       → sequence length: S + K + 1 - r

The two stages are controlled by *independent* schedules:
  - EVITToMeConfig.evit_keep_rate / evit_reduction_loc
  - EVITToMeConfig.tome_keep_rate / tome_reduction_loc

A block can have only EViT, only ToMe, both, or neither (original block).

Reuse policy
------------
- ``AttentionEViTFromExisting``  is imported directly from evit.py
- ``bipartite_soft_matching``, ``merge_wavg``, ``merge_source``
  are imported directly from tome.py
- No weight copies; all original timm modules are reused by reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Re-use building blocks from the two existing modules
# ---------------------------------------------------------------------------
from .evit import (
    AttentionEViTFromExisting,
    complement_idx,
    EVITConfig,
    _compute_token_ratio_full as _evit_token_ratio_full,
)
from .tome import (
    bipartite_soft_matching,
    merge_wavg,
    merge_source,
    ToMeConfig,
    _compute_keep_rate_full as _tome_keep_rate_full,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EVITToMeConfig:
    """
    Unified config for EViT-then-ToMe hybrid pruning.

    EViT controls
    -------------
    evit_enabled          : master switch for the EViT stage
    evit_keep_rate        : keep-rate schedule (same semantics as EVITConfig)
    evit_reduction_loc    : block indices where EViT fires
    evit_exponentiate     : exponentiate a single keep_rate across multiple locs
    evit_sorted_topk      : whether topk indices are returned sorted
    evit_init_n_override  : override patch count used for absolute budget

    ToMe controls
    -------------
    tome_enabled          : master switch for the ToMe stage
    tome_keep_rate        : keep-rate schedule (same semantics as ToMeConfig)
    tome_reduction_loc    : block indices where ToMe fires
    tome_exponentiate     : exponentiate a single keep_rate across multiple locs
    tome_prop_attn        : propagate proportional attention weights
    tome_init_n_override  : override patch count used for r computation

    Shared
    ------
    viz_mode              : collect per-block pruning decisions after forward
    """

    # --- EViT ---
    evit_enabled: bool = True
    evit_keep_rate: Sequence[float] = (0.7,)
    evit_reduction_loc: Sequence[int] = ()
    evit_exponentiate: bool = True
    evit_sorted_topk: bool = True
    evit_init_n_override: Optional[int] = None

    # --- ToMe ---
    tome_enabled: bool = True
    tome_keep_rate: Sequence[float] = (0.9,)
    tome_reduction_loc: Sequence[int] = ()
    tome_exponentiate: bool = True
    tome_prop_attn: bool = True
    tome_init_n_override: Optional[int] = None

    # --- Shared ---
    viz_mode: bool = False


# ---------------------------------------------------------------------------
# Hybrid attention module
# ---------------------------------------------------------------------------

class AttentionHybridFromExisting(nn.Module):
    """
    Hybrid attention: reuses timm weights and simultaneously computes both
    the CLS-attention signal required by the EViT stage and the mean-key
    metric required by the ToMe stage.

    When evit_keep_rate >= 1.0, the EViT caches are left as None (no pruning).
    The ToMe metric is *always* returned so BlockHybridAdapter can decide
    whether to merge.
    """

    def __init__(
        self,
        attn: nn.Module,
        dim: int,
        evit_keep_rate: float,
        init_n: int,
        num_special_tokens: int,
        sorted_topk: bool = True,
    ):
        super().__init__()
        self.num_heads = attn.num_heads
        self.scale = getattr(attn, "scale", (dim // attn.num_heads) ** -0.5)
        self.sorted_topk = bool(sorted_topk)

        # Reuse trained modules
        self.qkv       = attn.qkv
        self.attn_drop = attn.attn_drop
        self.proj      = attn.proj
        self.proj_drop = attn.proj_drop

        self.evit_keep_rate    = float(evit_keep_rate)
        self.init_n            = int(init_n)
        self.num_special_tokens = int(num_special_tokens)

        # EViT caches (consumed by BlockHybridAdapter)
        self.last_index:    Optional[torch.Tensor] = None  # [B, left_tokens, C]
        self.last_idx:      Optional[torch.Tensor] = None  # [B, left_tokens]
        self.last_cls_attn: Optional[torch.Tensor] = None  # [B, cur_patches]

    def forward(
        self,
        x: torch.Tensor,
        size: Optional[torch.Tensor] = None,
        attn_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    [B, N, C]
            size: [B, N, 1] proportional-attention weights (from ToMe stage)
        Returns:
            x_attn: [B, N, C]
            metric: [B, N, head_dim]  mean key — ToMe matching metric
        """
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Proportional attention bias (ToMe prop_attn)
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)

        # Mean key → ToMe metric
        metric = k.mean(dim=1)  # [B, N, head_dim]

        # Reset EViT caches
        self.last_index    = None
        self.last_idx      = None
        self.last_cls_attn = None

        # EViT: populate caches if pruning is requested
        if self.evit_keep_rate < 1.0:
            cur_patches = N - self.num_special_tokens
            left_tokens = math.ceil(self.evit_keep_rate * cur_patches)

            if cur_patches > 1 and left_tokens < cur_patches:
                left_tokens = max(1, left_tokens)

                cls_attn = attn[:, :, 0, self.num_special_tokens:]  # [B, H, cur_patches]
                cls_attn = cls_attn.mean(dim=1)                     # [B, cur_patches]

                _, idx = torch.topk(
                    cls_attn, k=left_tokens, dim=1,
                    largest=True, sorted=self.sorted_topk,
                )  # [B, left_tokens]

                self.last_idx      = idx
                self.last_index    = idx.unsqueeze(-1).expand(-1, -1, C)
                self.last_cls_attn = cls_attn

        return x_attn, metric


# ---------------------------------------------------------------------------
# Hybrid block adapter
# ---------------------------------------------------------------------------

class BlockHybridAdapter(nn.Module):
    """
    Drop-in replacement for a timm ViT block that runs:

        norm1 → HybridAttention (EViT prune + cache metric)
        residual add
        [EViT fusion: drop low-attn tokens, append fused token]
        [ToMe merge:  merge r similar token pairs]
        norm2 → MLP
        residual add

    Either or both stages can be disabled per block:
      - Set evit_keep_rate = 1.0  to skip EViT fusion in this block.
            - Set tome_keep_rate = 1.0 to skip ToMe merging in this block.

    attn_size propagation (ToMe prop_attn) is handled via _prev_hybrid_block,
    wired by apply_evit_tome_pruning — no changes to the host model's forward.
    """

    def __init__(
        self,
        orig_block: nn.Module,
        embed_dim: int,
        evit_keep_rate: float,          # 1.0 → EViT stage inactive
        tome_keep_rate: float,          # 1.0 → ToMe stage inactive
        init_n: int,
        num_special_tokens: int,
        sorted_topk: bool = True,
        prop_attn: bool = True,
    ):
        super().__init__()
        self.norm1 = orig_block.norm1
        self.norm2 = orig_block.norm2
        self.mlp   = orig_block.mlp

        self.drop_path1 = getattr(orig_block, "drop_path1", nn.Identity())
        self.drop_path2 = getattr(orig_block, "drop_path2", nn.Identity())
        self.ls1 = getattr(orig_block, "ls1", nn.Identity())
        self.ls2 = getattr(orig_block, "ls2", nn.Identity())

        self.num_special_tokens = int(num_special_tokens)
        self.tome_keep_rate     = float(tome_keep_rate)
        self.prop_attn          = bool(prop_attn)

        self.attn = AttentionHybridFromExisting(
            attn=orig_block.attn,
            dim=embed_dim,
            evit_keep_rate=evit_keep_rate,
            init_n=init_n,
            num_special_tokens=num_special_tokens,
            sorted_topk=sorted_topk,
        )

        # ToMe state propagation
        self._prev_hybrid_block: Optional["BlockHybridAdapter"] = None
        self._attn_size:         Optional[torch.Tensor]          = None

        # Viz caches
        self.last_evit_idx:     Optional[torch.Tensor] = None  # [B, left_tokens+1]
        self.last_evit_compl:   Optional[torch.Tensor] = None  # [B, n_pruned]
        self.last_tome_cluster: Optional[torch.Tensor] = None  # [B, N_merged]

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        B, N, C = x.shape

        # --- Retrieve propagated attn_size from previous hybrid block ---
        attn_size: Optional[torch.Tensor] = (
            self._prev_hybrid_block._attn_size
            if self._prev_hybrid_block is not None
            else None
        )

        # --- Attention (produces both EViT signal and ToMe metric) ---
        size_input = attn_size if self.prop_attn else None
        x_attn, metric = self.attn(self.norm1(x), size=size_input, attn_mask=attn_mask)
        x = x + self.drop_path1(self.ls1(x_attn))

        # Reset viz caches and attn_size
        self.last_evit_idx     = None
        self.last_evit_compl   = None
        self.last_tome_cluster = None
        self._attn_size        = attn_size  # updated below if ToMe fires

        # ----------------------------------------------------------------
        # Stage 1: EViT token fusion
        # ----------------------------------------------------------------
        index     = self.attn.last_index      # [B, left_tokens, C] or None
        idx       = self.attn.last_idx        # [B, left_tokens]    or None
        cls_attn  = self.attn.last_cls_attn   # [B, cur_patches]    or None

        if index is not None:
            special  = x[:, : self.num_special_tokens]      # [B, S, C]
            patches  = x[:, self.num_special_tokens:]       # [B, cur_patches, C]
            cur_patches = patches.shape[1]

            kept    = torch.gather(patches, 1, index)       # [B, left_tokens, C]
            compl   = complement_idx(idx, cur_patches)      # [B, n_pruned]
            non_topk = torch.gather(
                patches, 1,
                compl.unsqueeze(-1).expand(-1, -1, C)
            )                                               # [B, n_pruned, C]

            # Fuse pruned tokens into one extra token (attention-weighted sum)
            non_topk_attn = torch.gather(cls_attn, 1, compl)  # [B, n_pruned]
            extra_token   = torch.sum(
                non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True
            )                                               # [B, 1, C]

            x = torch.cat([special, kept, extra_token], dim=1)

            # Also trim metric and attn_size to match new token layout
            special_m  = metric[:, : self.num_special_tokens]
            patches_m  = metric[:, self.num_special_tokens:]
            kept_m     = torch.gather(patches_m, 1, idx.unsqueeze(-1).expand(-1, -1, metric.shape[-1]))
            extra_m    = torch.sum(
                torch.gather(patches_m, 1, compl.unsqueeze(-1).expand(-1, -1, metric.shape[-1]))
                * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True
            )
            metric = torch.cat([special_m, kept_m, extra_m], dim=1)

            # Trim attn_size in the same way
            if attn_size is not None:
                special_s = attn_size[:, : self.num_special_tokens]
                patches_s = attn_size[:, self.num_special_tokens:]
                kept_s    = torch.gather(patches_s, 1, idx.unsqueeze(-1))
                extra_s   = torch.sum(
                    torch.gather(patches_s, 1, compl.unsqueeze(-1))
                    * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True
                )
                attn_size = torch.cat([special_s, kept_s, extra_s], dim=1)
                self._attn_size = attn_size

            # Viz: sentinel (-1) marks the fused token
            sentinel = torch.full((B, 1), -1, device=idx.device, dtype=idx.dtype)
            self.last_evit_idx   = torch.cat([idx, sentinel], dim=1)
            self.last_evit_compl = compl

        # ----------------------------------------------------------------
        # Stage 2: ToMe token merging
        # ----------------------------------------------------------------
        cur_patches_for_tome = x.shape[1] - self.num_special_tokens
        if cur_patches_for_tome > 1 and self.tome_keep_rate < 1.0:
            left_tokens = math.ceil(self.tome_keep_rate * cur_patches_for_tome)
            left_tokens = max(1, left_tokens)
            r = cur_patches_for_tome - left_tokens

            if r <= 0:
                x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
                return x

            class_token   = self.num_special_tokens >= 1
            distill_token = self.num_special_tokens >= 2

            merge, _ = bipartite_soft_matching(
                metric, r, class_token, distill_token
            )

            # Viz: cluster assignment before merge
            t_orig        = x.shape[1]
            source_merged = merge_source(merge, x, None)     # [B, T_new, T_orig]
            cluster_idx   = (
                source_merged
                * torch.arange(1, t_orig + 1, device=x.device).float()[None, None, :]
            ).amax(dim=-1)

            if class_token:
                cluster_idx = (cluster_idx - 2)[:, 1:]
            else:
                cluster_idx = cluster_idx - 1

            x, new_size = merge_wavg(merge, x, attn_size)

            self._attn_size        = new_size
            self.last_tome_cluster = cluster_idx

        # --- MLP ---
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------

def _build_schedules(
    depth: int,
    cfg: EVITToMeConfig,
    init_n: int,
) -> Tuple[List[float], List[float]]:
    """
    Returns:
        evit_ratio_full: per-block EViT keep-rate  (1.0 → inactive)
        tome_keep_full:  per-block ToMe keep-rate (1.0 → inactive)
    """
    # --- EViT schedule ---
    if cfg.evit_enabled and cfg.evit_reduction_loc:
        evit_cfg = EVITConfig(
            enabled=True,
            keep_rate=cfg.evit_keep_rate,
            reduction_loc=cfg.evit_reduction_loc,
            exponentiate_single_keep_rate=cfg.evit_exponentiate,
        )
        evit_ratio_full = _evit_token_ratio_full(depth, evit_cfg)
    else:
        evit_ratio_full = [1.0] * depth

    # --- ToMe schedule ---
    if cfg.tome_enabled and cfg.tome_reduction_loc:
        tome_cfg = ToMeConfig(
            enabled=True,
            keep_rate=cfg.tome_keep_rate,
            reduction_loc=cfg.tome_reduction_loc,
            exponentiate_single_keep_rate=cfg.tome_exponentiate,
            prop_attn=cfg.tome_prop_attn,
        )
        tome_keep_full = _tome_keep_rate_full(depth, tome_cfg)
    else:
        tome_keep_full = [1.0] * depth

    return evit_ratio_full, tome_keep_full


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_evit_tome_pruning(model: nn.Module, cfg: EVITToMeConfig) -> nn.Module:
    """
    In-place transform: replaces model.blocks[i] with BlockHybridAdapter
    at any block where at least one of EViT or ToMe is active.

    Call AFTER shrinking the classification head.
    Returns model unchanged if both stages are disabled or have empty locs.

    Usage example
    -------------
    cfg = EVITToMeConfig(
        evit_enabled=True,
        evit_keep_rate=(0.7,),
        evit_reduction_loc=(3,),

        tome_enabled=True,
        tome_keep_rate=(0.9,),
        tome_reduction_loc=(6, 9),

        viz_mode=True,
    )
    model = apply_evit_tome_pruning(model, cfg)
    out   = model(images)
    viz   = collect_hybrid_viz(model)
    """
    if not cfg.evit_enabled and not cfg.tome_enabled:
        return model

    if not hasattr(model, "blocks"):
        raise TypeError(
            "Hybrid pruning expects a timm ViT/DeiT-like model with `.blocks`"
        )

    depth = len(model.blocks)

    # Resolve init_n
    evit_init_n = cfg.evit_init_n_override
    tome_init_n = cfg.tome_init_n_override
    if evit_init_n is None or tome_init_n is None:
        if not hasattr(model, "patch_embed") or not hasattr(
            model.patch_embed, "num_patches"
        ):
            raise TypeError(
                "Model missing patch_embed.num_patches; set "
                "evit_init_n_override and tome_init_n_override explicitly."
            )
        patch_n = int(model.patch_embed.num_patches)
        if evit_init_n is None:
            evit_init_n = patch_n
        if tome_init_n is None:
            tome_init_n = patch_n

    evit_ratio_full, tome_keep_full = _build_schedules(depth, cfg, tome_init_n)

    num_special_tokens = (
        2
        if hasattr(model, "dist_token") and model.dist_token is not None
        else 1
    )

    # Check if there is anything to do at all
    evit_active = any(abs(r - 1.0) > 1e-12 for r in evit_ratio_full)
    tome_active = any(abs(r - 1.0) > 1e-12 for r in tome_keep_full)
    if not evit_active and not tome_active:
        return model

    # Replace blocks that have at least one active stage
    for i in range(depth):
        evit_kr = float(evit_ratio_full[i])
        tome_keep = float(tome_keep_full[i])

        if evit_kr >= 1.0 and tome_keep >= 1.0:
            continue  # leave original block untouched

        orig_blk  = model.blocks[i]
        embed_dim = getattr(model, "embed_dim", None)
        if embed_dim is None:
            embed_dim = orig_blk.norm1.normalized_shape[0]

        model.blocks[i] = BlockHybridAdapter(
            orig_block=orig_blk,
            embed_dim=int(embed_dim),
            evit_keep_rate=evit_kr,
            tome_keep_rate=tome_keep,
            init_n=evit_init_n,
            num_special_tokens=num_special_tokens,
            sorted_topk=cfg.evit_sorted_topk,
            prop_attn=cfg.tome_prop_attn,
        )

    # Wire prev-block references so ToMe prop_attn propagates
    prev: Optional[BlockHybridAdapter] = None
    for blk in model.blocks:
        if isinstance(blk, BlockHybridAdapter):
            blk._prev_hybrid_block = prev
            prev = blk

    model._hybrid_viz_mode = cfg.viz_mode
    return model


def collect_hybrid_viz(model: nn.Module) -> dict:
    """
    After a forward pass on a model patched with apply_evit_tome_pruning,
    collect per-block pruning decisions.

    Returns
    -------
    {
        "EViT_Kept_Tokens":   {block_idx: np.ndarray [B, left_tokens+1]},
        "EViT_Fusion_Assign": {block_idx: np.ndarray [B, n_pruned]},
        "ToMe_Assignment":    {block_idx: np.ndarray [B, N_merged]},
    }
    """
    kept_tokens:   Dict[int, object] = {}
    fusion_assign: Dict[int, object] = {}
    tome_assign:   Dict[int, object] = {}

    if not hasattr(model, "blocks"):
        return {
            "EViT_Kept_Tokens":   kept_tokens,
            "EViT_Fusion_Assign": fusion_assign,
            "ToMe_Assignment":    tome_assign,
        }

    for i, blk in enumerate(model.blocks):
        if not isinstance(blk, BlockHybridAdapter):
            continue
        if blk.last_evit_idx is not None:
            kept_tokens[i]   = blk.last_evit_idx.clone().detach().cpu().numpy()
        if blk.last_evit_compl is not None:
            fusion_assign[i] = blk.last_evit_compl.clone().detach().cpu().numpy()
        if blk.last_tome_cluster is not None:
            tome_assign[i]   = blk.last_tome_cluster.clone().detach().cpu().numpy()

    return {
        "EViT_Kept_Tokens":   kept_tokens,
        "EViT_Fusion_Assign": fusion_assign,
        "ToMe_Assignment":    tome_assign,
    }
