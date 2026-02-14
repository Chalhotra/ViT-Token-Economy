"""Tests for ImageNet mapping functionality."""
from __future__ import annotations
import pytest
from src.imagenet_mapping import build_imagenet100_to_1k_map


def test_build_imagenet100_to_1k_map():
    """Test that mapping builds correctly with expected properties."""
    maps = build_imagenet100_to_1k_map()
    
    # Check we have 100 WNIDs
    assert len(maps.imagenet100_wnids) == 100
    
    # Check we have 1000 ImageNet-1K classes
    assert len(maps.wnid_to_imagenet1k_idx) == 1000
    
    # Check mapping has 100 entries
    assert len(maps.new_to_old_map) == 100
    
    # Check all new indices are in range [0, 99]
    assert all(0 <= idx < 100 for idx in maps.new_to_old_map.keys())
    
    # Check all old indices are in range [0, 999]
    assert all(0 <= idx < 1000 for idx in maps.new_to_old_map.values())
    
    # Check mapping is complete (no missing indices)
    assert set(maps.new_to_old_map.keys()) == set(range(100))


def test_imagenet100_wnids_are_strings():
    """Test that WNIDs are strings."""
    maps = build_imagenet100_to_1k_map()
    assert all(isinstance(wnid, str) for wnid in maps.imagenet100_wnids)
    # WNIDs should be in format like 'n01440764'
    assert all(wnid.startswith('n') and len(wnid) == 9 for wnid in maps.imagenet100_wnids)


def test_mapping_consistency():
    """Test that mapping is consistent across multiple calls."""
    maps1 = build_imagenet100_to_1k_map()
    maps2 = build_imagenet100_to_1k_map()
    
    assert maps1.imagenet100_wnids == maps2.imagenet100_wnids
    assert maps1.new_to_old_map == maps2.new_to_old_map


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
