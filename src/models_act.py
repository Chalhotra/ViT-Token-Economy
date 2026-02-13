import torch
import torch.nn as nn
from functools import partial
import timm
from timm.models.registry import register_model

# Import the class from your models folder
from models.topk import TopKVisionTransformer

def _get_pruning_params(kwargs):
    """
    Helper to extract pruning params from either direct kwargs 
    or the 'args' namespace if it exists (for compatibility).
    """
    args = kwargs.get('args', None)
    
    # Defaults
    locs = [3, 6, 9]
    rates = [0.7, 0.7, 0.7]

    # Try extracting from 'args' object (legacy style)
    if args is not None:
        if hasattr(args, 'pruning_locs'): locs = args.pruning_locs
        if hasattr(args, 'keep_rates'): rates = args.keep_rates
        
    # Overwrite if passed directly (modern style)
    locs = kwargs.get('pruning_locs', locs)
    rates = kwargs.get('keep_rates', rates)
    
    return locs, rates

# --- Tiny Variants ---

@register_model
def topk_deit_tiny_patch16_224(pretrained=True, **kwargs):
    pruning_locs, keep_rates = _get_pruning_params(kwargs)
    
    model = TopKVisionTransformer(
        pruning_locs=pruning_locs, keep_rates=keep_rates,
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    
    if pretrained:
        # Load standard DEIT weights
        print(f"Loading weights from deit_tiny_patch16_224...")
        base_model = timm.create_model('deit_tiny_patch16_224', pretrained=True, num_classes=kwargs.get('num_classes', 1000))
        model.load_state_dict(base_model.state_dict(), strict=False)
        
    return model

@register_model
def topk_vit_tiny_patch16_224(pretrained=True, **kwargs):
    # ViT Tiny is architecturally identical to DEIT Tiny usually, just different weights
    return topk_deit_tiny_patch16_224(pretrained, **kwargs)

# --- Small Variants ---

@register_model
def topk_deit_small_patch16_224(pretrained=True, **kwargs):
    pruning_locs, keep_rates = _get_pruning_params(kwargs)

    model = TopKVisionTransformer(
        pruning_locs=pruning_locs, keep_rates=keep_rates,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )

    if pretrained:
        print(f"Loading weights from deit_small_patch16_224...")
        base_model = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=kwargs.get('num_classes', 1000))
        model.load_state_dict(base_model.state_dict(), strict=False)

    return model

# --- Base Variants ---

@register_model
def topk_base_patch16_224(pretrained=True, **kwargs):
    pruning_locs, keep_rates = _get_pruning_params(kwargs)

    model = TopKVisionTransformer(
        pruning_locs=pruning_locs, keep_rates=keep_rates,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )

    if pretrained:
        # Check for distillation request like the reference repo
        args = kwargs.get('args', None)
        distilled = args.distillation_type != 'none' if (args and hasattr(args, 'distillation_type')) else False
        
        base_name = 'deit_base_distilled_patch16_224' if distilled else 'deit_base_patch16_224'
        
        print(f"Loading weights from {base_name}...")
        base_model = timm.create_model(base_name, pretrained=True, num_classes=kwargs.get('num_classes', 1000))
        model.load_state_dict(base_model.state_dict(), strict=False)

    return model