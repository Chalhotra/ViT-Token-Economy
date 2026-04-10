from __future__ import annotations
import torch
from torchvision.transforms import v2

# ──────────────────────────────────────────────
# Optimized GPU Pipelines
# ──────────────────────────────────────────────

def all_augmentations() -> v2.Compose:
    """Full GPU-optimized pipeline: flip → jitter → rotation → noise."""
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        v2.RandomRotation(degrees=15),
        v2.GaussianNoise(sigma=0.05),
        # Ensure output is float32 and scaled [0, 1] for the model
        v2.ToDtype(torch.float32, scale=True),
    ])

def horizontal_flip_only() -> v2.Compose:
    return v2.Compose([
        v2.RandomHorizontalFlip(p=1.0),
    ])

def rotation_only(max_degrees:int=15, random:bool=False) -> v2.Compose:
    # Set p=1.0 inside RandomApply or just use RandomRotation
    low = max_degrees
    if random:
        low=-max_degrees
    hi = max_degrees
    return v2.Compose([
        v2.RandomRotation(degrees=(low, hi)), # Fixed 15 deg if you want consistency
    ])

def jitter_only() -> v2.Compose:
    return v2.Compose([
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    ])

def gaussian_noise_only(sigma: int=0.05) -> v2.Compose:
    return v2.Compose([
        v2.GaussianNoise(sigma=sigma),
    ])

def colour_only() -> v2.Compose:
    return v2.Compose([
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        v2.GaussianNoise(sigma=0.05),
    ])

def geometric_only() -> v2.Compose:
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
    ])

def no_augmentation() -> v2.Identity:
    return v2.Identity()