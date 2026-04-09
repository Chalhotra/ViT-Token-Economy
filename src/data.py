from __future__ import annotations
from dataclasses import dataclass
from datasets import load_dataset
from torch.utils.data import DataLoader
from timm.data import resolve_data_config, create_transform
from augmentation import (
    AugmentationPipeline,
    all_augmentations,
    geometric_only,
    colour_only,
    no_augmentation,
)
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
def apply_timm_preprocess(ds, transform, aug_pipeline: AugmentationPipeline | None = None):
    """
    Matches notebook behavior: map transforms ahead of DataLoader.
    Optionally runs aug_pipeline on the raw PIL image before timm preprocessing.
    """
    label_names = None
    if hasattr(ds, "features") and "label" in ds.features:
        label_feature = ds.features["label"]
        label_names = getattr(label_feature, "names", None)

    def preprocess(example, idx):
        img = example["image"].convert("RGB")
        if aug_pipeline is not None:
            img = aug_pipeline(img)                   # augment on PIL image
        example["pixel_values"] = transform(img)      # then timm normalise/resize
        example["image_id"] = f"image_{idx:06d}"
        if label_names is not None:
            example["ground_truth_label"] = label_names[int(example["label"])]
        else:
            example["ground_truth_label"] = str(example["label"])
        return example

    ds2 = ds.map(preprocess, with_indices=True, remove_columns=["image"])
    return ds2


def build_loader(ds, cfg: DataConfig) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
