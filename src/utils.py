from __future__ import annotations
import os
import random
import numpy as np
import torch

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism trades speed; keep defaults unless needed.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def cuda_sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

def num_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
