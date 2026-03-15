#  my evit
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EVITConfig:
    enabled: bool = False

    # Reference-style controls:
    keep_rate: Sequence[float] = (1.0,)       # e.g. [0.7] or [0.9, 0.8, ...]
    reduction_loc: Sequence[int] = tuple()    # e.g. [3, 6, 9]
    # If keep_rate has length 1 and reduction_loc has >1, exponentiate:
    # token_ratio[i] = keep_rate[0] ** (idx+1), exactly like the reference.
    exponentiate_single_keep_rate: bool = True

    # Minor options:
    sorted_topk: bool = True
    init_n_override: Optional[int] = None     # force init_n if needed
    viz_mode: bool = False                    # enable visualization data ctollection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def complement_idx(idx: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute the complement: set(range(dim)) - set(idx).
    idx is a multi-dimensional tensor; complement is found along the trailing
    dimension, all other dimensions are treated as batch.

    Args:
        idx: input index, shape [B, *, K]
        dim: the total size to complement against
    Returns:
        compl: shape [B, *, dim-K], sorted ascending
    """
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1,)
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl


# ---------------------------------------------------------------------------
# Attention module
# ---------------------------------------------------------------------------

class AttentionEViTFromExisting(nn.Module):
    """
    EViT-style attention that reuses trained timm attention weights and:
      - returns the full attended output x_attn
      - caches (last_index, last_idx, last_cls_attn) for the block to use
    - computes progressive budget: left_tokens = ceil(keep_rate * cur_patches)
      - when keep_rate >= 1.0, passes through unchanged (index/idx/cls_attn = None)
    """

    def __init__(
        self,
        attn: nn.Module,
        dim: int,
        keep_rate: float,
        init_n: int,
        num_special_tokens: int,
        sorted_topk: bool = True,
    ):
        super().__init__()
        self.num_heads = attn.num_heads
        self.scale = getattr(attn, "scale", (dim // attn.num_heads) ** -0.5)
        self.sorted_topk = bool(sorted_topk)

        # Reuse trained modules (no weight copies)
        self.qkv = attn.qkv
        self.attn_drop = attn.attn_drop
        self.proj = attn.proj
        self.proj_drop = attn.proj_drop

        self.keep_rate = float(keep_rate)
        assert 0.0 < self.keep_rate <= 1.0, f"keep_rate must be in (0, 1], got {self.keep_rate}"

        self.init_n = int(init_n)                          # original patch grid size
        self.num_special_tokens = int(num_special_tokens)  # 1 (cls) or 2 (cls+dist)

        # Cached outputs consumed by BlockEViTAdapter.forward
        self.last_index: Optional[torch.Tensor] = None    # [B, left_tokens, C]
        self.last_idx: Optional[torch.Tensor] = None      # [B, left_tokens]
        self.last_cls_attn: Optional[torch.Tensor] = None # [B, cur_patches]

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        """
        Returns:
            x_attn: [B, N, C]  — full attended output (before pruning)
        Side-effects:
            Sets self.last_index, self.last_idx, self.last_cls_attn
            (all None when no pruning happens this step)
        """
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)

        # Reset cache
        self.last_index = None
        self.last_idx = None
        self.last_cls_attn = None

        if self.keep_rate >= 1.0:
            return x_attn

        cur_patches = N - self.num_special_tokens
        if cur_patches <= 1:
            return x_attn

        # Match reference behavior with progressive pruning.
        left_tokens = math.ceil(self.keep_rate * cur_patches)

        # Reference early-exit: budget keeps all current patches.
        if left_tokens == cur_patches:
            return x_attn

        # Nothing to prune if budget already met
        if left_tokens >= cur_patches:
            return x_attn

        left_tokens = max(1, left_tokens)

        # Score patches by mean CLS attention across heads (exclude special tokens)
        cls_attn = attn[:, :, 0, self.num_special_tokens:]  # [B, H, cur_patches]
        cls_attn = cls_attn.mean(dim=1)                     # [B, cur_patches]

        _, idx = torch.topk(
            cls_attn,
            k=left_tokens,
            dim=1,
            largest=True,
            sorted=self.sorted_topk,
        )  # [B, left_tokens]

        self.last_idx = idx
        self.last_index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]
        self.last_cls_attn = cls_attn

        return x_attn


# ---------------------------------------------------------------------------
# Block adapter
# ---------------------------------------------------------------------------

class BlockEViTAdapter(nn.Module):
    """
    timm Block-compatible adapter that implements EViT token fusion:
      - keeps the top-K tokens by CLS attention
      - fuses the remaining tokens into a single extra token (attention-weighted sum)
      - appends the extra token after the kept patches
      - stores (last_idx, last_compl) for viz_mode inspection

    Drop-path / LayerScale handled identically to BlockTopKAdapter.
    """

    def __init__(
        self,
        orig_block: nn.Module,
        embed_dim: int,
        keep_rate: float,
        init_n: int,
        num_special_tokens: int,
        sorted_topk: bool = True,
    ):
        super().__init__()
        self.norm1 = orig_block.norm1
        self.norm2 = orig_block.norm2
        self.mlp = orig_block.mlp

        # timm v0.9+ naming
        self.drop_path1 = getattr(orig_block, "drop_path1", nn.Identity())
        self.drop_path2 = getattr(orig_block, "drop_path2", nn.Identity())

        # LayerScale (Identity in most variants)
        self.ls1 = getattr(orig_block, "ls1", nn.Identity())
        self.ls2 = getattr(orig_block, "ls2", nn.Identity())

        self.num_special_tokens = int(num_special_tokens)

        self.attn = AttentionEViTFromExisting(
            attn=orig_block.attn,
            dim=embed_dim,
            keep_rate=keep_rate,
            init_n=init_n,
            num_special_tokens=num_special_tokens,
            sorted_topk=sorted_topk,
        )

        # Cached for viz_mode / external inspection
        self.last_idx: Optional[torch.Tensor] = None    # [B, left_tokens+1]  (+1 = fused marker)
        self.last_compl: Optional[torch.Tensor] = None  # [B, n_pruned]

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        """
        Returns:
            x:       [B, N', C]   where N' = num_special_tokens + left_tokens + 1 (fused)
                                  or original N if no pruning occurred
        """
        B, N, C = x.shape

        attn_out = self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.drop_path1(self.ls1(attn_out))

        # Reset viz cache
        self.last_idx = None
        self.last_compl = None

        index     = self.attn.last_index     # [B, left_tokens, C] or None
        idx       = self.attn.last_idx       # [B, left_tokens]    or None
        cls_attn  = self.attn.last_cls_attn  # [B, cur_patches]    or None

        if index is not None:
            special  = x[:, : self.num_special_tokens]                # [B, S, C]
            patches  = x[:, self.num_special_tokens:]                 # [B, cur_patches, C]
            cur_patches = patches.shape[1]

            # Keep top-K patches
            kept = torch.gather(patches, dim=1, index=index)          # [B, left_tokens, C]

            # Compute complement (pruned) indices
            compl = complement_idx(idx, cur_patches)                  # [B, n_pruned]
            non_topk = torch.gather(
                patches, dim=1,
                index=compl.unsqueeze(-1).expand(-1, -1, C)
            )                                                          # [B, n_pruned, C]

            # Fuse pruned tokens: attention-weighted sum → 1 extra token
            non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, n_pruned]
            extra_token = torch.sum(
                non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True
            )                                                            # [B, 1, C]

            x = torch.cat([special, kept, extra_token], dim=1)

            # Append sentinel (-1) to idx to mark the fused token
            sentinel = torch.full(
                (B, 1), -1, device=idx.device, dtype=idx.dtype
            )
            self.last_idx   = torch.cat([idx, sentinel], dim=1)  # [B, left_tokens+1]
            self.last_compl = compl

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# Schedule helpers  (mirrors topk.py exactly)
# ---------------------------------------------------------------------------

def _compute_token_ratio_full(depth: int, cfg: EVITConfig) -> List[float]:
    keep_rate  = [float(x) for x in cfg.keep_rate]
    pruning_loc = [int(x)  for x in cfg.reduction_loc]

    if not pruning_loc:
        return [1.0] * depth

    if len(keep_rate) == 1 and len(pruning_loc) > 1:
        if cfg.exponentiate_single_keep_rate:
            base = keep_rate[0]
            keep_rate = [base ** (i + 1) for i in range(len(pruning_loc))]
        else:
            keep_rate = [keep_rate[0]] * len(pruning_loc)

    if len(keep_rate) != len(pruning_loc):
        raise ValueError(
            f"Mismatch: reduction_loc={pruning_loc} vs keep_rate={keep_rate}"
        )

    token_ratio_full = [1.0] * depth
    for r, loc in zip(keep_rate, pruning_loc):
        if 0 <= loc < depth:
            token_ratio_full[loc] = float(r)
    return token_ratio_full


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_evit_pruning(model: nn.Module, cfg: EVITConfig) -> nn.Module:
    """
    In-place transform: replaces model.blocks[i] with BlockEViTAdapter at pruning
    locations, using the EViT token-fusion strategy.

    Call this AFTER shrinking the classification head (keeps concerns separated).
    If cfg.enabled is False, returns model unchanged.

    When cfg.viz_mode is True, a helper ``collect_evit_viz`` is available to
    extract per-block pruning decisions after a forward pass.
    """
    if not cfg.enabled:
        return model

    if not hasattr(model, "blocks"):
        raise TypeError(
            "EViT pruning expects a timm ViT/DeiT-like model with `.blocks`"
        )

    depth = len(model.blocks)
    token_ratio_full = _compute_token_ratio_full(depth, cfg)

    if cfg.init_n_override is not None:
        init_n = int(cfg.init_n_override)
    else:
        if not hasattr(model, "patch_embed") or not hasattr(
            model.patch_embed, "num_patches"
        ):
            raise TypeError(
                "Model missing patch_embed.num_patches; cannot infer init_n. "
                "Set EVITConfig.init_n_override explicitly."
            )
        init_n = int(model.patch_embed.num_patches)

    num_special_tokens = (
        2
        if hasattr(model, "dist_token") and model.dist_token is not None
        else 1
    )

    # If all keep rates are 1.0, nothing to do
    if all(abs(r - 1.0) < 1e-12 for r in token_ratio_full):
        return model

    for i in range(depth):
        keep_r = float(token_ratio_full[i])
        if keep_r >= 1.0:
            continue  # leave original block untouched

        orig_blk = model.blocks[i]
        embed_dim = getattr(model, "embed_dim", None)
        if embed_dim is None:
            embed_dim = orig_blk.norm1.normalized_shape[0]

        model.blocks[i] = BlockEViTAdapter(
            orig_block=orig_blk,
            embed_dim=int(embed_dim),
            keep_rate=keep_r,
            init_n=init_n,
            num_special_tokens=num_special_tokens,
            sorted_topk=cfg.sorted_topk,
        )

    # Attach viz_mode flag directly on the model for use in forward hooks
    model._evit_viz_mode = cfg.viz_mode

    return model


def collect_evit_viz(model: nn.Module) -> dict:
    """
    After a forward pass on a model patched with apply_evit_pruning, call this
    to collect per-block pruning decisions (matches the reference viz_data format).

    Returns:
        {
            "Kept_Tokens":   {block_idx: np.ndarray [B, left_tokens+1]},
            "Fusion_Assign": {block_idx: np.ndarray [B, n_pruned]},
        }
    """
    kept_tokens: dict    = {}
    fusion_assign: dict  = {}

    if not hasattr(model, "blocks"):
        return {"Kept_Tokens": kept_tokens, "Fusion_Assign": fusion_assign}

    for i, blk in enumerate(model.blocks):
        if not isinstance(blk, BlockEViTAdapter):
            continue
        if blk.last_idx is not None:
            kept_tokens[i]   = blk.last_idx.clone().detach().cpu().numpy()
        if blk.last_compl is not None:
            fusion_assign[i] = blk.last_compl.clone().detach().cpu().numpy()

    return {"Kept_Tokens": kept_tokens, "Fusion_Assign": fusion_assign}