from __future__ import annotations
from dataclasses import dataclass
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from timm.data import resolve_data_config, create_transform
# from augmentation import (
#     AugmentationPipeline,
#     all_augmentations,
#     geometric_only,
#     colour_only,
#     no_augmentation,
# )
@dataclass
class DataConfig:
    dataset_id: str = "clane9/imagenet-100"
    split: str = "validation"
    batch_size: int = 64
    num_workers: int = 2
    pin_memory: bool = False
    shuffle: bool = False  # eval: keep False

def build_transform_for_model(model):
    cfg = resolve_data_config({}, model=model)
    # Match notebook: use model's recommended eval preprocessing
    return create_transform(**cfg, is_training=False)

def load_imagenet100_split(cfg: DataConfig):
    """Load ImageNet-100 dataset from Hugging Face.
    
    Args:
        cfg: DataConfig with dataset_id and split
        
    Returns:
        The loaded dataset
        
    Raises:
        Exception: If dataset loading fails
    """
    try:
        return load_dataset(cfg.dataset_id, split=cfg.split)
    except Exception as e:
        raise Exception(
            f"Failed to load dataset '{cfg.dataset_id}' (split: '{cfg.split}'). "
            f"Please check your internet connection and dataset availability. "
            f"Original error: {e}"
        ) from e
def apply_timm_preprocess(ds, transform): # Removed aug_pipeline
    label_names = None
    if hasattr(ds, "features") and "label" in ds.features:
        label_feature = ds.features["label"]
        label_names = getattr(label_feature, "names", None)

    def preprocess(example, idx):
        # We only do the bare minimum here to keep the CPU fast
        img = example["image"].convert("RGB")
        example["pixel_values"] = np.array(img, dtype=np.uint8)
        example["image_id"] = f"image_{idx:06d}"
        example["ground_truth_label"] = label_names[int(example["label"])] if label_names else str(example["label"])
        return example

    ds2 = ds.map(preprocess, with_indices=True, remove_columns=["image"])
    ds2.set_format("numpy")
    return ds2, transform

def make_collate_fn(transform):
    def collate_fn(batch):
        from PIL import Image
        # timm transform converts PIL -> Tensor
        imgs = [Image.fromarray(b["pixel_values"]) for b in batch]
        pixel_values = torch.stack([transform(img) for img in imgs])
        
        return {
            "pixel_values": pixel_values,
            "label": torch.tensor([b["label"] for b in batch]),
            "image_id": [b["image_id"] for b in batch],
            "ground_truth_label": [b["ground_truth_label"] for b in batch],
        }
    return collate_fn


def build_loader(ds, cfg: DataConfig, collate_fn=None) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
        persistent_workers=cfg.num_workers > 0,
    )
