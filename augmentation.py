# augmentations.py
from __future__ import annotations

import random
import numpy as np
from dataclasses import dataclass
from typing import Sequence

from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T


# ──────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────

class Augmentation:
    """Base class for all augmentations. Override __call__ and __repr__."""

    def __call__(self, img: Image.Image) -> Image.Image:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__


# ──────────────────────────────────────────────
# Individual augmentations
# ──────────────────────────────────────────────

@dataclass
class HorizontalFlip(Augmentation):
    """Randomly flip the image horizontally."""
    p: float = 1

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return TF.hflip(img)
        return img

    def __repr__(self) -> str:
        return f"HorizontalFlip(p={self.p})"


@dataclass
class ColourJitter(Augmentation):
    """Randomly jitter brightness, contrast, saturation, and hue."""
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.05
    p: float = 1

    def __post_init__(self):
        self._transform = T.ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue,
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return self._transform(img)
        return img

    def __repr__(self) -> str:
        return (
            f"ColourJitter(brightness={self.brightness}, contrast={self.contrast}, "
            f"saturation={self.saturation}, hue={self.hue}, p={self.p})"
        )


@dataclass
class SlightRotation(Augmentation):
    """Rotate by a random angle within [-max_degrees, +max_degrees]. For our consistency purposes, we'fe set it to be constant"""
    max_degrees: float = 15.0
    p: float = 1
    fill: int = 0  # pixel fill value for areas outside original image

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            angle = self.max_degrees # can change to (-m,m) for unif random
            return TF.rotate(img, angle, fill=self.fill)
        return img

    def __repr__(self) -> str:
        return f"SlightRotation(max_degrees={self.max_degrees}, p={self.p})"


@dataclass
class GaussianNoise(Augmentation):
    """Add Gaussian noise in pixel space (applied after converting to numpy)."""
    std: float = 0.05       # noise std relative to [0, 1] pixel range
    p: float = 1
    clip: bool = True       # clamp to valid [0, 255] range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            arr = np.array(img, dtype=np.float32) / 255.0
            noise = np.random.normal(0, self.std, arr.shape).astype(np.float32)
            arr = arr + noise
            if self.clip:
                arr = np.clip(arr, 0.0, 1.0)
            return Image.fromarray((arr * 255).astype(np.uint8))
        return img

    def __repr__(self) -> str:
        return f"GaussianNoise(std={self.std}, p={self.p})"


# ──────────────────────────────────────────────
# Pipeline composer
# ──────────────────────────────────────────────

class AugmentationPipeline:
    """Chains a sequence of Augmentation objects. Applied in order."""

    def __init__(self, augmentations: Sequence[Augmentation]):
        self.augmentations = list(augmentations)

    def __call__(self, img: Image.Image) -> Image.Image:
        for aug in self.augmentations:
            img = aug(img)
        return img

    def __repr__(self) -> str:
        aug_str = "\n  ".join(repr(a) for a in self.augmentations)
        return f"AugmentationPipeline(\n  {aug_str}\n)"


# ──────────────────────────────────────────────
# Preset pipelines (plug-and-play)
# ──────────────────────────────────────────────

def all_augmentations() -> AugmentationPipeline:
    """Full pipeline: flip → jitter → rotation → noise."""
    return AugmentationPipeline([
        HorizontalFlip(p=0.5),
        ColourJitter(p=0.8),
        SlightRotation(max_degrees=15.0, p=0.5),
        GaussianNoise(std=0.05, p=0.5),
    ])

def horizontal_flip_only() -> AugmentationPipeline:
    return AugmentationPipeline([
        HorizontalFlip(),
    ])


def rotation_only() -> AugmentationPipeline:
    return AugmentationPipeline([
        SlightRotation(max_degrees=15),
    ])

def jitter_only() -> AugmentationPipeline:
    return AugmentationPipeline([
        ColourJitter(),
    ])

def gaussian_noise_only() -> AugmentationPipeline:
    return AugmentationPipeline([
        GaussianNoise(std=1),
    ])

def colour_only() -> AugmentationPipeline:
    """Only photometric augmentations."""
    return AugmentationPipeline([
        ColourJitter(p=0.8),
        GaussianNoise(std=0.05, p=0.5),
    ])
def geometric_only() -> AugmentationPipeline:
    """Only spatial augmentations — useful for colour-sensitive experiments."""
    return AugmentationPipeline([
        HorizontalFlip(p=0.5),
        SlightRotation(max_degrees=15.0, p=0.5),
    ])


def no_augmentation() -> AugmentationPipeline:
    """Identity pipeline — baseline / eval mode."""
    return AugmentationPipeline([])