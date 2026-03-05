import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import _cfg, default_cfgs
from timm.models.registry import register_model

from src.models.topk import TopKVisionTransformer

deit_url_paths = {"deit_tiny_patch16_224": "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                  "deit_tiny_distilled_patch16_224": "https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
                  "deit_small_patch16_224": "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                  "deit_small_distilled_patch16_224": "https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
                  "deit_base_patch16_224": "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                  "deit_base_distilled_patch16_224": "https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
                  }


__all__ = [
    'deit_tiny_patch16_224_local', \
    'deit_small_patch16_224_local', \
    'topk_deit_tiny_patch16_224', \
    'topk_deit_small_patch16_224', \
        ]


@register_model
def deit_tiny_patch16_224_local(pretrained=True, **kwargs):

    from timm.models.vision_transformer import VisionTransformer

    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    kwargs.pop("args", None)

    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), distilled = deit_distillation, **kwargs)
        
    if deit_distillation:
        key = "deit_tiny_distilled_patch16_224"
    else:
        key = "deit_tiny_patch16_224"

    model.default_cfg = default_cfgs[key]
    if pretrained:

        # note that this part loads DEIT weights, not A-ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )

        print(checkpoint["model"].keys())
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def deit_small_patch16_224_local(pretrained=True, **kwargs):

    from timm.models.vision_transformer import VisionTransformer

    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    kwargs.pop("args", None)

    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), distilled = deit_distillation, **kwargs)
    
    if deit_distillation:
        key = "deit_small_distilled_patch16_224"
    else:
        key = "deit_small_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not A-ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


# Helper class to pass arguments to TopKVisionTransformer
class Args:
    def __init__(self, keep_rate, reduction_loc, distillation_type='none', viz_mode=False):
        self.keep_rate = keep_rate
        self.reduction_loc = reduction_loc
        self.distillation_type = distillation_type
        self.viz_mode = viz_mode


@register_model
def topk_deit_tiny_patch16_224(pretrained=True, **kwargs):
    """DeiT Tiny with TopK token pruning"""
    
    # Extract pruning parameters
    pruning_locs = kwargs.pop('pruning_locs', [3, 6, 9])
    keep_rates = kwargs.pop('keep_rates', [0.7, 0.7, 0.7])
    
    # Create args object expected by TopKVisionTransformer
    args = Args(keep_rate=keep_rates, reduction_loc=pruning_locs)
    
    model = TopKVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), args=args, **kwargs)
    
    model.default_cfg = default_cfgs["deit_tiny_patch16_224"]
    
    if pretrained:
        print(f"Loading pretrained weights for topk_deit_tiny_patch16_224...")
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths["deit_tiny_patch16_224"],
            model_dir="./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
        print("Loaded base DeiT weights (some parameters may not match due to pruning)")
    
    return model


@register_model
def topk_deit_small_patch16_224(pretrained=True, **kwargs):
    """DeiT Small with TopK token pruning"""
    
    # Extract pruning parameters
    pruning_locs = kwargs.pop('pruning_locs', [3, 6, 9])
    keep_rates = kwargs.pop('keep_rates', [0.7, 0.7, 0.7])
    
    # Create args object expected by TopKVisionTransformer
    args = Args(keep_rate=keep_rates, reduction_loc=pruning_locs)
    
    model = TopKVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), args=args, **kwargs)
    
    model.default_cfg = default_cfgs["deit_small_patch16_224"]
    
    if pretrained:
        print(f"Loading pretrained weights for topk_deit_small_patch16_224...")
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths["deit_small_patch16_224"],
            model_dir="./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
        print("Loaded base DeiT weights (some parameters may not match due to pruning)")
    
    return model
