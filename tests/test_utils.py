"""Tests for utility functions."""
from __future__ import annotations
import torch
from src.utils import get_device, num_params


def test_get_device():
    """Test device detection."""
    device = get_device()
    assert device in ["cuda", "cpu"]


def test_num_params():
    """Test parameter counting."""
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.Linear(20, 5)
    )
    
    expected = 10 * 20 + 20 + 20 * 5 + 5  # weights + biases
    assert num_params(model) == expected


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
