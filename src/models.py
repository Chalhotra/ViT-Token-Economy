from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import torch
import timm
import torch.nn as nn

@dataclass
class ModelConfig:
    model_id: str  # e.g. 'vit_tiny_patch16_224' or 'deit_tiny_patch16_224'
    num_classes: int = 100
    pretrained: bool = True

def create_model(cfg: ModelConfig) -> torch.nn.Module:
    """Create a model using timm.
    
    Args:
        cfg: ModelConfig with model_id and pretrained flag
        
    Returns:
        The created model
        
    Raises:
        RuntimeError: If model creation fails (e.g., invalid model_id)
    """
    try:
        model = timm.create_model(cfg.model_id, pretrained=cfg.pretrained)
        return model
    except Exception as e:
        # Show a few example models (avoid listing all thousands)
        example_models = "vit_tiny_patch16_224, deit_tiny_patch16_224, resnet50, etc."
        raise RuntimeError(
            f"Failed to create model '{cfg.model_id}'. "
            f"Please check the model name. "
            f"Example available models: {example_models} "
            f"Original error: {e}"
        ) from e

def shrink_imagenet1k_head_to_imagenet100(model: torch.nn.Module, new_to_old_map: Dict[int, int], num_classes: int = 100) -> torch.nn.Module:
    """Replace 1000-way head with 100-way head, copying weights per new_to_old_map.

    Functionality matches the notebook exactly:
    - model starts with a 1000-class head
    - we create new Linear(in_features, 100)
    - fill weights/biases by copying rows old_idx -> new_idx
    """
    # timm models expose classifier via get_classifier in many cases,
    # but the notebook uses .head. We'll support both without changing behavior.
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

    # assign back
    if hasattr(model, "head"):
        model.head = new_head
    else:
        model.reset_classifier(num_classes=num_classes)
        # ensure it is set
        model.get_classifier().load_state_dict(new_head.state_dict())

    return model
