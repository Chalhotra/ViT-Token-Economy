from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset
from torch.utils.data import DataLoader
from timm.data import resolve_data_config, create_transform

@dataclass
class DataConfig:
    dataset_id: str = "clane9/imagenet-100"
    split: str = "validation"
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = False  # eval: keep False

def build_transform_for_model(model):
    cfg = resolve_data_config({}, model=model)
    # Match notebook: use model's recommended eval preprocessing
    return create_transform(**cfg, is_training=False)

def load_imagenet100_split(cfg: DataConfig):
    return load_dataset(cfg.dataset_id, split=cfg.split)

def apply_timm_preprocess(ds, transform):
    """Matches notebook behavior: map transforms ahead of DataLoader."""
    def preprocess(example):
        example["pixel_values"] = transform(example["image"].convert("RGB"))
        return example

    ds2 = ds.map(preprocess, remove_columns=["image"])
    ds2.set_format(type="torch", columns=["pixel_values", "label"])
    return ds2

def build_loader(ds, cfg: DataConfig) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
