"""Tests for model creation and head adaptation."""
from __future__ import annotations
import pytest
import torch
from src.models import ModelConfig, create_model, shrink_imagenet1k_head_to_imagenet100


def test_create_model():
    """Test basic model creation."""
    cfg = ModelConfig(model_id='vit_tiny_patch16_224', pretrained=False)
    model = create_model(cfg)
    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_create_model_invalid_id():
    """Test that invalid model ID raises error."""
    cfg = ModelConfig(model_id='invalid_model_name_xyz', pretrained=False)
    with pytest.raises(RuntimeError) as exc_info:
        create_model(cfg)
    assert "Failed to create model" in str(exc_info.value)


def test_shrink_head():
    """Test head shrinking from 1000 to 100 classes."""
    cfg = ModelConfig(model_id='vit_tiny_patch16_224', pretrained=False)
    model = create_model(cfg)
    
    # Create a simple mapping (first 100 classes)
    new_to_old_map = {i: i for i in range(100)}
    
    model = shrink_imagenet1k_head_to_imagenet100(model, new_to_old_map, num_classes=100)
    
    # Check head has 100 classes
    head = getattr(model, 'head', None)
    if head is None:
        head = model.get_classifier()
    
    assert isinstance(head, torch.nn.Linear)
    assert head.out_features == 100


def test_shrink_head_weights_copied():
    """Test that head weights are actually copied correctly."""
    cfg = ModelConfig(model_id='vit_tiny_patch16_224', pretrained=False)
    model = create_model(cfg)
    
    # Get original head weights
    original_head = getattr(model, 'head', None)
    if original_head is None:
        original_head = model.get_classifier()
    
    original_weight = original_head.weight.detach().clone()
    
    # Create mapping
    new_to_old_map = {i: i * 10 for i in range(100)}  # Map to every 10th class
    
    model = shrink_imagenet1k_head_to_imagenet100(model, new_to_old_map, num_classes=100)
    
    # Check new head
    new_head = getattr(model, 'head', None)
    if new_head is None:
        new_head = model.get_classifier()
    
    # Verify weights were copied correctly
    for new_idx, old_idx in new_to_old_map.items():
        assert torch.allclose(new_head.weight[new_idx], original_weight[old_idx])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
