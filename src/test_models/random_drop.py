from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import math
import torch
import timm
import math
import torch.nn as nn

class PrunedViT(nn.Module):
    def __init__(self, model_name='vit_tiny_patch16_224',
                 prune_layers=(2, 5, 8),
                 keep_ratios=(0.70, 0.70, 0.70)):  # same ratio applied at each stage
        super().__init__()

        self.model = timm.create_model(model_name, pretrained=True)
        self.prune_layers = prune_layers
        self.keep_ratios = keep_ratios

        assert len(prune_layers) == len(keep_ratios)

        self.num_special_tokens = self.model.num_tokens if hasattr(self.model, 'num_tokens') else 1

    def random_prune(self, x, keep_ratio):
        """
        x: (B, N, C)
        keep_ratio: fraction of CURRENT patch tokens to keep (EViT style)
        """
        B, N, C = x.shape

        special_tokens = x[:, :self.num_special_tokens, :]   # (B, num_special_tokens, C)
        patches = x[:, self.num_special_tokens:, :]          # (B, N - num_special_tokens, C)

        num_current_patches = patches.shape[1]

        # EViT style: apply keep_ratio to CURRENT number of patches, not original
        k = math.ceil(keep_ratio * num_current_patches)
        k = max(1, k)  # keep at least 1 token

        if k >= num_current_patches:
            return x  # nothing to prune

        # random indices per batch
        idx = torch.rand(B, num_current_patches, device=x.device).argsort(dim=1)
        idx = idx[:, :k]
        idx, _ = torch.sort(idx, dim=1)

        # gather kept patches
        patches = torch.gather(
            patches,
            1,
            idx.unsqueeze(-1).expand(-1, -1, C)
        )

        x = torch.cat([special_tokens, patches], dim=1)
        return x

    def forward_features(self, x):
        x = self.model.patch_embed(x)

        if hasattr(self.model, 'cls_token'):
            cls_token = self.model.cls_token
        else:
            cls_token = None

        if hasattr(self.model, 'dist_token'):
            dist_token = self.model.dist_token
        else:
            dist_token = None

        if cls_token is not None and dist_token is not None:
            special_tokens = torch.cat((cls_token, dist_token), dim=1).expand(x.shape[0], -1, -1)
        elif cls_token is not None:
            special_tokens = cls_token.expand(x.shape[0], -1, -1)
        else:
            special_tokens = None

        if special_tokens is not None:
            x = torch.cat((special_tokens, x), dim=1)

        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)

        prune_stage = 0

        for i, blk in enumerate(self.model.blocks):
            x = blk(x)

            if i in self.prune_layers:
                keep_ratio = self.keep_ratios[prune_stage]
                # EViT style: pass keep_ratio directly, applied to current token count
                x = self.random_prune(x, keep_ratio)
                prune_stage += 1

        x = self.model.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.model.head(x)
        return x