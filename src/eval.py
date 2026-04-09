from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
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


def evaluate_with_topk_predictions(
    model: torch.nn.Module,
    loader,
    device: str,
    class_names: Sequence[str],
    topk: int = 10,
    cfg: EvalConfig = EvalConfig(),
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    model.eval()

    correct = 0
    total = 0
    total_time = 0.0
    prediction_rows: List[Dict[str, object]] = []

    i = 0
    with torch.inference_mode():
        for batch in loader:
            images = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            image_ids = batch.get("image_id")
            gt_labels = batch.get("ground_truth_label")

            cuda_sync(device)
            start = time.perf_counter()
            outputs = model(images)
            cuda_sync(device)
            end = time.perf_counter()

            total_time += (end - start)

            probs = torch.softmax(outputs, dim=1)
            k = min(int(topk), probs.shape[1])
            top_probs, top_indices = torch.topk(probs, k=k, dim=1)
            preds = top_indices[:, 0]
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)

            for row_idx in range(images.size(0)):
                image_id = image_ids[row_idx] if image_ids is not None else f"image_{total - images.size(0) + row_idx:06d}"
                ground_truth_label = gt_labels[row_idx] if gt_labels is not None else class_names[int(labels[row_idx].item())]
                top1_conf = float(top_probs[row_idx, 0].item())
                entropy_value = float(entropy[row_idx].item())

                for rank_idx in range(k):
                    class_idx = int(top_indices[row_idx, rank_idx].item())
                    predicted_class = class_names[class_idx]
                    confidence = float(top_probs[row_idx, rank_idx].item())
                    prediction_rows.append({
                        "image_id": image_id,
                        "ground_truth_label": ground_truth_label,
                        "rank": rank_idx + 1,
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "is_correct": class_idx == int(labels[row_idx].item()),
                        "entropy": entropy_value,
                        "conf_gap_to_rank1": top1_conf - confidence,
                    })

            i += 1
            if cfg.max_batches is not None and i >= cfg.max_batches:
                break

    acc1 = 100.0 * correct / total if total else 0.0
    latency_ms = 1000.0 * total_time / total if total else 0.0
    throughput = total / total_time if total_time > 0 else 0.0
    return ({"acc1": acc1, "latency_ms": latency_ms, "throughput": throughput}, prediction_rows)

def compute_gflops(model: torch.nn.Module, input_tensor: torch.Tensor) -> float:
    flops = FlopCountAnalysis(model, input_tensor)
    return float(flops.total()) / 1e9

def compute_flops_table(model: torch.nn.Module, input_tensor: torch.Tensor) -> str:
    flops = FlopCountAnalysis(model, input_tensor)
    return flop_count_table(flops)
