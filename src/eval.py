from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Any
import time
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
from .utils import cuda_sync

@dataclass
class EvalConfig:
    warmup_batches: int = 0  # notebook did no warmup; keep 0 to match functionality
    max_batches: Optional[int] = None

def evaluate_accuracy_latency_throughput(model: torch.nn.Module, loader, device: str, cfg: EvalConfig = EvalConfig()) -> Dict[str, float]:
    model.eval()

    correct = 0
    total = 0
    total_time = 0.0

    i = 0
    with torch.inference_mode():
        for batch in loader:
            images = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            cuda_sync(device)
            start = time.perf_counter()
            outputs = model(images)
            cuda_sync(device)
            end = time.perf_counter()

            # Match notebook behavior: always count timing + accuracy for all batches
            total_time += (end - start)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            i += 1
            if cfg.max_batches is not None and i >= cfg.max_batches:
                break

    acc1 = 100.0 * correct / total if total else 0.0
    latency_ms = 1000.0 * total_time / total if total else 0.0
    throughput = total / total_time if total_time > 0 else 0.0
    return {"acc1": acc1, "latency_ms": latency_ms, "throughput": throughput}

def compute_gflops(model: torch.nn.Module, input_tensor: torch.Tensor) -> float:
    flops = FlopCountAnalysis(model, input_tensor)
    return float(flops.total()) / 1e9

def compute_flops_table(model: torch.nn.Module, input_tensor: torch.Tensor) -> str:
    flops = FlopCountAnalysis(model, input_tensor)
    return flop_count_table(flops)
