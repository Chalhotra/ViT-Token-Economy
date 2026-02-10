from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import requests

IMAGENET100_WNIDS_URL = "https://raw.githubusercontent.com/HobbitLong/CMC/master/imagenet100.txt"
# index -> [wnid, label]
# IMAGENET_CLASS_INDEX_URL = "https://raw.githubusercontent.com/pytorch/vision/main/torchvision/models/imagenet_class_index.json"
IMAGENET_CLASS_INDEX_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"


@dataclass(frozen=True)
class ImagenetMaps:
    imagenet100_wnids: List[str]
    wnid_to_imagenet1k_idx: Dict[str, int]
    new_to_old_map: Dict[int, int]

def build_imagenet100_to_1k_map(timeout_s: int = 30) -> ImagenetMaps:
    """Build mapping {new_idx (0..99) -> old_idx (0..999)}.

    Functionality matches the notebook intent:
    - new_idx is the position in HobbitLong imagenet100 WNID list
    - old_idx is the ImageNet-1K class index used by torchvision/timm pretrained heads
    """
    # 1) Load ImageNet-100 WNIDs (ordered)
    resp = requests.get(IMAGENET100_WNIDS_URL, timeout=timeout_s)
    resp.raise_for_status()
    imagenet100_wnids = [line.strip() for line in resp.text.splitlines() if line.strip()]

    # 2) Load ImageNet-1K class index mapping
    resp = requests.get(IMAGENET_CLASS_INDEX_URL, timeout=timeout_s)
    resp.raise_for_status()
    class_index = resp.json()  # keys: "0".."999", values: [wnid, label]

    wnid_to_idx: Dict[str, int] = {}
    for k, v in class_index.items():
        idx = int(k)
        wnid = v[0]
        wnid_to_idx[wnid] = idx

    # 3) Build new->old map
    new_to_old: Dict[int, int] = {}
    for new_idx, wnid in enumerate(imagenet100_wnids):
        if wnid not in wnid_to_idx:
            raise KeyError(f"WNID {wnid} not found in ImageNet-1K index mapping.")
        new_to_old[new_idx] = wnid_to_idx[wnid]

    return ImagenetMaps(
        imagenet100_wnids=imagenet100_wnids,
        wnid_to_imagenet1k_idx=wnid_to_idx,
        new_to_old_map=new_to_old,
    )
