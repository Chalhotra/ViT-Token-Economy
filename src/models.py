from __future__ import annotations
from typing import Dict
import torch
import timm

# Import your configs
from src.configs import ModelConfig

# CRITICAL: Triggers registration of topk_* models in timm
import src.models_act 

def create_model(cfg: ModelConfig) -> torch.nn.Module:
    """
    Factory that handles standard models AND custom pruning models transparently.
    """
    
    # 1. Base arguments for every model
    model_kwargs = {
        "pretrained": cfg.pretrained,
        "num_classes": cfg.num_classes,
    }

    # 2. Inject Pruning Arguments if they exist
    if cfg.pruning is not None:
        # Converts TopKConfig(locs=[3], rates=[0.7]) -> {'pruning_locs': [3], 'keep_rates': [0.7]}
        # This gets passed to the constructor in models_act.py
        model_kwargs.update(cfg.pruning.to_kwargs())

    # 3. Create Model
    # If model_id is 'topk_deit_tiny...', timm uses src.models_act + these kwargs
    # If model_id is 'resnet50', timm ignores the extra kwargs (or warns)
    return timm.create_model(cfg.model_id, **model_kwargs)

def shrink_imagenet1k_head_to_imagenet100(model: torch.nn.Module, new_to_old_map: Dict[int, int], num_classes: int = 100) -> torch.nn.Module:
    # (Same implementation as before)
    head = getattr(model, "head", None)
    if head is None:
        head = model.get_classifier()

    if not isinstance(head, torch.nn.Linear):
        raise TypeError(f"Expected Linear head, got: {type(head)}")

    in_features = head.in_features
    old_weight = head.weight.detach().clone()
    old_bias = head.bias.detach().clone() if head.bias is not None else None

    new_head = torch.nn.Linear(in_features, num_classes)

    with torch.no_grad():
        new_head.weight.zero_()
        if new_head.bias is not None:
            new_head.bias.zero_()
        for new_idx, old_idx in new_to_old_map.items():
            new_head.weight[new_idx].copy_(old_weight[old_idx])
            if old_bias is not None:
                new_head.bias[new_idx].copy_(old_bias[old_idx])

    if hasattr(model, "head"):
        model.head = new_head
    else:
        model.reset_classifier(num_classes=num_classes)
        model.get_classifier().load_state_dict(new_head.state_dict())

    return model