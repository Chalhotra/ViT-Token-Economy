from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToMeConfig:
    enabled: bool = False

    # Reference-style controls:
    keep_rate: Sequence[float] = (1.0,)       # e.g. [0.7] or [0.9, 0.8, ...]
    reduction_loc: Sequence[int] = tuple()    # e.g. [3, 6, 9]
    # If keep_rate has length 1 and reduction_loc has >1, exponentiate:
    # token_ratio[i] = keep_rate[0] ** (idx+1), exactly like evit/topk references.
    exponentiate_single_keep_rate: bool = True

    # Minor options:
    prop_attn: bool = True                    # propagate attention sizes (weighted merge)
    init_n_override: Optional[int] = None     # force init_n if needed
    viz_mode: bool = False                    # enable cluster assignment collection


# ---------------------------------------------------------------------------
# ToMe core functions  (reference-faithful)
# ---------------------------------------------------------------------------

def do_nothing(x: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced bipartite matching set (50 / 50).

    Args:
        metric:        [B, N, C]  — the matching metric (mean key across heads)
        r:             number of tokens to *remove* (merge pairs → net -r tokens)
        class_token:   whether a CLS token is present at position 0
        distill_token: whether a distillation token is present
    Returns:
        merge, unmerge — callables that operate on [B, N, C] tensors
    """
    protected = int(class_token) + int(distill_token)

    # Maximum reduction is 50 % of un-protected tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]   # unmerged tokens
        src_idx = edge_idx[..., :r, :]   # tokens to merge away
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]  # keep CLS at front

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape
        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))
        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)
        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable,
    x: torch.Tensor,
    size: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Weighted-average merge: each token is weighted by its accumulated size so
    that larger (previously merged) tokens contribute proportionally.
    Returns (merged_x, new_sizes).
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x    = merge(x * size, mode="sum")
    size = merge(size,     mode="sum")
    x    = x / size
    return x, size


def merge_source(
    merge: Callable,
    x: torch.Tensor,
    source: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Tracks which original tokens each merged token represents.
    source is an adjacency matrix [B, T_orig, T_orig]; starts as identity.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None].expand(n, t, t)

    source = merge(source, mode="amax")
    return source


# ---------------------------------------------------------------------------
# Attention module
# ---------------------------------------------------------------------------

class AttentionToMeFromExisting(nn.Module):
    """
    ToMe-style attention that reuses trained timm attention weights and:
      - returns the full attended output x_attn
      - returns the mean key vector (the matching metric for bipartite matching)
      - supports proportional-attention size bias (prop_attn)
    """

    def __init__(
        self,
        attn: nn.Module,
        dim: int,
        num_special_tokens: int,
    ):
        super().__init__()
        self.num_heads = attn.num_heads
        self.scale = getattr(attn, "scale", (dim // attn.num_heads) ** -0.5)

        # Reuse trained modules (no weight copies)
        self.qkv      = attn.qkv
        self.attn_drop = attn.attn_drop
        self.proj      = attn.proj
        self.proj_drop = attn.proj_drop

        self.num_special_tokens = int(num_special_tokens)

    def forward(
        self,
        x: torch.Tensor,
        size: Optional[torch.Tensor] = None,
        attn_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    [B, N, C]
            size: [B, N, 1]  token size weights for proportional attention (or None)
        Returns:
            x_attn:  [B, N, C]
            metric:  [B, N, C//num_heads]  — mean key, used as ToMe matching metric
        """
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Proportional attention: up-weight tokens that represent more originals
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)

        # Mean key across heads → matching metric
        metric = k.mean(dim=1)  # [B, N, head_dim]

        return x_attn, metric


# ---------------------------------------------------------------------------
# Block adapter
# ---------------------------------------------------------------------------

class BlockToMeAdapter(nn.Module):
    """
    timm Block-compatible adapter that implements ToMe token merging.

    Returns only `x` so it is a drop-in replacement inside timm's standard
    `for blk in self.blocks: x = blk(x)` loop.

    attn_size (proportional attention) is propagated across ToMe blocks via a
    `_prev_tome_block` back-reference set by `apply_tome_merging` — no changes
    to the host model's forward loop are needed.
    """

    def __init__(
        self,
        orig_block: nn.Module,
        embed_dim: int,
        r: int,
        num_special_tokens: int,
        prop_attn: bool = True,
    ):
        super().__init__()
        self.norm1 = orig_block.norm1
        self.norm2 = orig_block.norm2
        self.mlp   = orig_block.mlp

        # timm v0.9+ naming
        self.drop_path1 = getattr(orig_block, "drop_path1", nn.Identity())
        self.drop_path2 = getattr(orig_block, "drop_path2", nn.Identity())

        # LayerScale (Identity in most variants)
        self.ls1 = getattr(orig_block, "ls1", nn.Identity())
        self.ls2 = getattr(orig_block, "ls2", nn.Identity())

        self.r                  = int(r)
        self.num_special_tokens = int(num_special_tokens)
        self.prop_attn          = bool(prop_attn)

        self.attn = AttentionToMeFromExisting(
            attn=orig_block.attn,
            dim=embed_dim,
            num_special_tokens=num_special_tokens,
        )

        # Linked to the previous ToMe block (if any) by apply_tome_merging so
        # we can read the accumulated attn_size it produced.
        self._prev_tome_block: Optional["BlockToMeAdapter"] = None

        # State written after each forward; read by the next ToMe block.
        self._attn_size: Optional[torch.Tensor] = None

        # Viz cache
        self.last_cluster_idx: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accepts and returns only `x` — compatible with timm's block loop.
        attn_size is read from the previous ToMe block's cached state and
        written back to self._attn_size for the next block to consume.
        """
        # --- Retrieve attn_size from the previous ToMe block (if any) ---
        attn_size: Optional[torch.Tensor] = (
            self._prev_tome_block._attn_size
            if self._prev_tome_block is not None
            else None
        )

        # --- Attention ---
        size_input = attn_size if self.prop_attn else None
        x_attn, metric = self.attn(self.norm1(x), size=size_input)
        x = x + self.drop_path1(self.ls1(x_attn))

        # Reset caches
        self.last_cluster_idx = None
        self._attn_size       = attn_size  # will be updated below if merging

        # --- ToMe merging ---
        if self.r > 0:
            class_token   = self.num_special_tokens >= 1
            distill_token = self.num_special_tokens >= 2

            merge, _ = bipartite_soft_matching(
                metric, self.r, class_token, distill_token
            )

            # Viz: build cluster assignment BEFORE x is merged
            t_orig = x.shape[1]
            source_merged = merge_source(merge, x, None)  # [B, T_new, T_orig]
            cluster_idx = (
                source_merged
                * torch.arange(1, t_orig + 1, device=x.device).float()[None, None, :]
            ).amax(dim=-1)  # [B, T_new]

            if class_token:
                cluster_idx = (cluster_idx - 2)[:, 1:]   # drop CLS column
            else:
                cluster_idx = cluster_idx - 1

            x, new_size = merge_wavg(merge, x, attn_size)

            self._attn_size       = new_size
            self.last_cluster_idx = cluster_idx

        # --- MLP ---
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# Schedule helpers  (mirrors evit.py / topk.py exactly)
# ---------------------------------------------------------------------------

def _compute_r_full(depth: int, cfg: ToMeConfig, init_n: int) -> List[int]:
    """
    Convert keep_rate schedule → per-block r (number of token pairs to merge).

    The reference computes r[loc] = prev_n_tokens - target_n_tokens so that
    after merging, exactly target_n_tokens patch tokens remain.
    """
    keep_rate  = [float(x) for x in cfg.keep_rate]
    pruning_loc = [int(x)  for x in cfg.reduction_loc]

    if not pruning_loc:
        return [0] * depth

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

    # Target token counts per pruning location
    target_counts = [int(init_n * r) for r in keep_rate]

    r_full = [0] * depth
    prev_n = init_n
    for i, loc in enumerate(pruning_loc):
        if 0 <= loc < depth:
            r_full[loc] = prev_n - target_counts[i]
            prev_n = target_counts[i]

    return r_full


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_tome_merging(model: nn.Module, cfg: ToMeConfig) -> nn.Module:
    """
    In-place transform: replaces model.blocks[i] with BlockToMeAdapter at
    merging locations, using the ToMe bipartite soft matching strategy.

    Call this AFTER shrinking the classification head (keeps concerns separated).
    If cfg.enabled is False, returns model unchanged.

    Use ``collect_tome_viz`` after a forward pass to gather cluster assignments.
    """
    if not cfg.enabled:
        return model

    if not hasattr(model, "blocks"):
        raise TypeError(
            "ToMe merging expects a timm ViT/DeiT-like model with `.blocks`"
        )

    depth = len(model.blocks)

    if cfg.init_n_override is not None:
        init_n = int(cfg.init_n_override)
    else:
        if not hasattr(model, "patch_embed") or not hasattr(
            model.patch_embed, "num_patches"
        ):
            raise TypeError(
                "Model missing patch_embed.num_patches; cannot infer init_n. "
                "Set ToMeConfig.init_n_override explicitly."
            )
        init_n = int(model.patch_embed.num_patches)

    r_full = _compute_r_full(depth, cfg, init_n)

    num_special_tokens = (
        2
        if hasattr(model, "dist_token") and model.dist_token is not None
        else 1
    )

    # Nothing to do if all r values are 0
    if all(r == 0 for r in r_full):
        return model

    for i in range(depth):
        r = r_full[i]
        if r <= 0:
            continue  # leave original block untouched

        orig_blk  = model.blocks[i]
        embed_dim = getattr(model, "embed_dim", None)
        if embed_dim is None:
            embed_dim = orig_blk.norm1.normalized_shape[0]

        model.blocks[i] = BlockToMeAdapter(
            orig_block=orig_blk,
            embed_dim=int(embed_dim),
            r=r,
            num_special_tokens=num_special_tokens,
            prop_attn=cfg.prop_attn,
        )

    # Wire prev-block references so attn_size propagates between ToMe blocks
    # without touching the host model's forward loop.
    prev: Optional[BlockToMeAdapter] = None
    for blk in model.blocks:
        if isinstance(blk, BlockToMeAdapter):
            blk._prev_tome_block = prev
            prev = blk

    # Attach flags for forward-pass helpers
    model._tome_viz_mode  = cfg.viz_mode
    model._tome_prop_attn = cfg.prop_attn

    return model


def collect_tome_viz(model: nn.Module) -> dict:
    """
    After a forward pass on a model patched with apply_tome_merging, call this
    to collect per-block cluster assignments (matches the reference viz_data format).

    Returns:
        {
            "Assignment_Maps": {block_idx: np.ndarray [B, N_merged]},
        }
    """
    assignments: dict = {}

    if not hasattr(model, "blocks"):
        return {"Assignment_Maps": assignments}

    for i, blk in enumerate(model.blocks):
        if not isinstance(blk, BlockToMeAdapter):
            continue
        if blk.last_cluster_idx is not None:
            assignments[i] = blk.last_cluster_idx.clone().detach().cpu().numpy()

    return {"Assignment_Maps": assignments}