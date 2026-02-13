from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal

@dataclass
class PruningConfig:
    """Base class for all pruning configurations."""
    method: str = "none"

    def to_kwargs(self):
        """Convert config to dictionary for timm model constructor."""
        return {k: v for k, v in asdict(self).items() if k != 'method'}

@dataclass
class TopKConfig(PruningConfig):
    method: Literal["topk"] = "topk"
    pruning_locs: List[int] = field(default_factory=lambda: [3, 6, 9])
    keep_rates: List[float] = field(default_factory=lambda: [0.7, 0.7, 0.7])

@dataclass
class EViTConfig(PruningConfig):
    method: Literal["evit"] = "evit"
    pruning_locs: List[int] = field(default_factory=lambda: [3, 6, 9])
    keep_rates: List[float] = field(default_factory=lambda: [0.7, 0.7, 0.7])
    fuse_token: bool = True  # Specific to EViT

@dataclass
class ModelConfig:
    model_id: str  
    num_classes: int = 100
    pretrained: bool = True
    # The polymorphic slot: can hold TopKConfig, EViTConfig, or None
    pruning: Optional[PruningConfig] = None