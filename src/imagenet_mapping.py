from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

IMAGENET100_WNIDS_URL = "https://raw.githubusercontent.com/HobbitLong/CMC/master/imagenet100.txt"
# index -> [wnid, label]
# IMAGENET_CLASS_INDEX_URL = "https://raw.githubusercontent.com/pytorch/vision/main/torchvision/models/imagenet_class_index.json"
IMAGENET_CLASS_INDEX_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"


@dataclass(frozen=True)
class ImagenetMaps:
    imagenet100_wnids: List[str]
    wnid_to_imagenet1k_idx: Dict[str, int]
    new_to_old_map: Dict[int, int]

def _get_session_with_retries(retries: int = 3, backoff_factor: float = 0.3) -> requests.Session:
    """Create a requests session with retry logic for robustness."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        # Also retry on connection errors
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def build_imagenet100_to_1k_map(timeout_s: int = 30) -> ImagenetMaps:
    """Build mapping {new_idx (0..99) -> old_idx (0..999)}.

    Functionality matches the notebook intent:
    - new_idx is the position in HobbitLong imagenet100 WNID list
    - old_idx is the ImageNet-1K class index used by torchvision/timm pretrained heads
    
    Raises:
        requests.RequestException: If network requests fail after retries
        ValueError: If response data is malformed
        KeyError: If ImageNet-100 WNID not found in ImageNet-1K mapping
    """
    session = _get_session_with_retries()
    
    try:
        # 1) Load ImageNet-100 WNIDs (ordered)
        resp = session.get(IMAGENET100_WNIDS_URL, timeout=timeout_s)
        resp.raise_for_status()
        imagenet100_wnids = [line.strip() for line in resp.text.splitlines() if line.strip()]
        
        if not imagenet100_wnids:
            raise ValueError("ImageNet-100 WNID list is empty")
        if len(imagenet100_wnids) != 100:
            raise ValueError(f"Expected 100 WNIDs, got {len(imagenet100_wnids)}")

        # 2) Load ImageNet-1K class index mapping
        resp = session.get(IMAGENET_CLASS_INDEX_URL, timeout=timeout_s)
        resp.raise_for_status()
        class_index = resp.json()  # keys: "0".."999", values: [wnid, label]
        
        if not class_index or len(class_index) != 1000:
            raise ValueError(f"Expected 1000 ImageNet classes, got {len(class_index)}")

        wnid_to_idx: Dict[str, int] = {}
        for k, v in class_index.items():
            try:
                idx = int(k)
                wnid = v[0]
                wnid_to_idx[wnid] = idx
            except (ValueError, IndexError, TypeError) as e:
                raise ValueError(f"Malformed class index entry: {k}={v}") from e

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
    except requests.RequestException as e:
        raise requests.RequestException(
            f"Failed to fetch ImageNet mapping data. Please check your internet connection. Error: {e}"
        ) from e
    finally:
        session.close()
