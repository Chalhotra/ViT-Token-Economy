from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import math
import torch
import timm
import torch.nn as nn

class PrunedViT(nn.Module):
    def __init__(self, model_name='vit_tiny_patch16_224',
                 prune_layers=(2, 5, 8),
                 keep_ratios=(0.70, 0.5, 0.25)):
        super().__init__()

        self.model = timm.create_model(model_name, pretrained=True)
        self.prune_layers = prune_layers
        self.keep_ratios = keep_ratios

        assert len(prune_layers) == len(keep_ratios)

        # Determine the number of special tokens (CLS, and potentially distillation token for DeiT)
        self.num_special_tokens = self.model.num_tokens if hasattr(self.model, 'num_tokens') else 1

    def random_prune(self, x, target_k):
        """
        x: (B, N, C)
        target_k: The absolute number of patch tokens to keep.
                  This should be calculated based on the original number of patches.
        """
        B, N, C = x.shape

        # Separate special tokens (CLS, and distillation token if present) from patch tokens
        special_tokens = x[:, :self.num_special_tokens, :]      # (B, num_special_tokens, C)
        patches = x[:, self.num_special_tokens:, :]            # (B, N - num_special_tokens, C)

        num_current_patches = patches.shape[1]

        # Ensure target_k doesn't exceed current available patches or is not negative
        k = min(target_k, num_current_patches)
        k = max(0, k) # Cannot keep negative patches

        # If k is 0, just keep the special tokens (and remove all patches)
        if k == 0:
            return special_tokens

        # random indices per batch
        idx = torch.rand(B, num_current_patches, device=x.device).argsort(dim=1)
        idx = idx[:, :k]
        idx, _ = torch.sort(idx, dim=1)

        # gather patches
        patches = torch.gather(
            patches,
            1,
            idx.unsqueeze(-1).expand(-1, -1, C)
        )

        x = torch.cat([special_tokens, patches], dim=1)
        return x

    def forward_features(self, x):
        x = self.model.patch_embed(x)

        # Expand CLS token (and distillation token if present)
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
            special_tokens = None # Should not happen for ViT/DeiT

        if special_tokens is not None:
            x = torch.cat((special_tokens, x), dim=1)

        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)

        prune_stage = 0

        for i, blk in enumerate(self.model.blocks):
            x = blk(x)

            if i in self.prune_layers:
                # Progressive pruning: budget is based on current patch count.
                keep_ratio = self.keep_ratios[prune_stage]
                cur_patch_tokens = x.shape[1] - self.num_special_tokens
                target_k_for_this_stage = math.ceil(keep_ratio * cur_patch_tokens)
                x = self.random_prune(x, target_k_for_this_stage)
                prune_stage += 1

        x = self.model.norm(x)

        # If there's a distillation token, the head typically expects only the CLS token (index 0)
        # or sometimes both. We'll return just the CLS token (or the first special token) for the head.
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.model.head(x)
        return x