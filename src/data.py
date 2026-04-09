from __future__ import annotations
from dataclasses import dataclass
import torch
import numpy as np
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
def apply_timm_preprocess(ds, transform, aug_pipeline=None):
    label_names = None
    if hasattr(ds, "features") and "label" in ds.features:
        label_feature = ds.features["label"]
        label_names = getattr(label_feature, "names", None)

    def preprocess(example, idx):
        import numpy as np          # local import — survives pickling
        from PIL import Image

        img = example["image"].convert("RGB")
        if aug_pipeline is not None:
            img = aug_pipeline(img)
        example["pixel_values"] = np.array(img, dtype=np.uint8)
        example["image_id"] = f"image_{idx:06d}"
        if label_names is not None:
            example["ground_truth_label"] = label_names[int(example["label"])]
        else:
            example["ground_truth_label"] = str(example["label"])
        return example

    ds2 = ds.map(preprocess, with_indices=True, remove_columns=["image"])
    ds2.set_format("numpy")
    return ds2, transform

def make_collate_fn(transform):
    def collate_fn(batch):
        import numpy as np          # local import — survives pickling
        from PIL import Image
        imgs = [Image.fromarray(b["pixel_values"]) for b in batch]
        pixel_values = torch.stack([transform(img) for img in imgs])
        labels = torch.tensor([b["label"] for b in batch])
        image_ids = [b["image_id"] for b in batch]
        ground_truth_labels = [b["ground_truth_label"] for b in batch]
        return {
            "pixel_values": pixel_values,
            "label": labels,
            "image_id": image_ids,
            "ground_truth_label": ground_truth_labels,
        }
    return collate_fn


# def build_loader(ds, cfg: DataConfig, collate_fn=None) -> DataLoader:
#     return DataLoader(
#         ds,
#         batch_size=cfg.batch_size,
#         shuffle=cfg.shuffle,
#         num_workers=cfg.num_workers,
#         pin_memory=cfg.pin_memory,
#         collate_fn=collate_fn,
#         persistent_workers=cfg.num_workers > 0,
#     )

def build_loader(ds, cfg: DataConfig, collate_fn=None) -> DataLoader:
    """Build DataLoader with custom collate function.
    
    Args:
        ds: Preprocessed dataset (from apply_timm_preprocess)
        cfg: DataConfig with batch_size, num_workers, etc.
        collate_fn: Custom collate function. If provided, num_workers is forced to 0
                    (torch multiprocessing doesn't support complex closures).
    
    Returns:
        DataLoader instance
    """
    # When using custom collate_fn with closures, must use num_workers=0
    num_workers = 0 if collate_fn is not None else cfg.num_workers
    
    if collate_fn is not None and cfg.num_workers > 0:
        print(f"⚠️  Warning: Forcing num_workers=0 (custom collate_fn requires single-process)")
    
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
    )
