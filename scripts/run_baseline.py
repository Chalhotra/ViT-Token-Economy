from __future__ import annotations
import argparse
import torch

from src.utils import get_device, seed_everything, num_params
from src.imagenet_mapping import build_imagenet100_to_1k_map
from src.data import DataConfig, load_imagenet100_split, build_transform_for_model, apply_timm_preprocess, build_loader
from src.eval import evaluate_accuracy_latency_throughput, compute_gflops

# Import new config structure
from src.configs import ModelConfig, TopKConfig
from src.models_smthing import create_model, shrink_imagenet1k_head_to_imagenet100

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model ID (e.g. topk_deit_tiny_patch16_224, deit_tiny_patch16_224)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--seed", type=int, default=42)
    
    # --- TopK Pruning Arguments (only used if model name starts with topk_) ---
    ap.add_argument("--pruning-locs", type=int, nargs="+", default=[3, 6, 9],
                    help="Layer indices where pruning occurs")
    ap.add_argument("--keep-rates", type=float, nargs="+", default=[0.7, 0.7, 0.7],
                    help="Token keep rates at each pruning location")

    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()

    # --- Auto-detect if pruning config is needed based on model name ---
    pruning_cfg = None
    
    if args.model.startswith("topk_"):
        pruning_cfg = TopKConfig(
            pruning_locs=args.pruning_locs,
            keep_rates=args.keep_rates
        )
        print(f"TopK Pruning Config: locs={args.pruning_locs}, rates={args.keep_rates}")

    # --- Create Model Config ---
    config = ModelConfig(
        model_id=args.model,
        pretrained=True,
        pruning=pruning_cfg
    )

    print(f"Creating model: {args.model}")
    model = create_model(config)
    
    maps = build_imagenet100_to_1k_map()
    model = shrink_imagenet1k_head_to_imagenet100(model, maps.new_to_old_map, num_classes=100)
    model = model.to(device).eval()

    # --- Load Data ---
    ds = load_imagenet100_split(DataConfig(split=args.split))
    transform = build_transform_for_model(model)
    ds_t = apply_timm_preprocess(ds, transform)
    loader = build_loader(ds_t, DataConfig(batch_size=args.batch_size, split=args.split, shuffle=False))

    # --- Evaluate ---
    print(f"\nEvaluating on ImageNet-100 {args.split} split...")
    metrics = evaluate_accuracy_latency_throughput(model, loader, device)
    
    # --- Compute FLOPs (handle custom models that might not support profiling) ---
    try:
        sample = ds_t[0]["pixel_values"].unsqueeze(0).to(device)
        gflops = compute_gflops(model, sample)
    except Exception as e:
        print(f"Warning: Could not compute FLOPs ({e})")
        gflops = 0.0

    # --- Print Results ---
    print(f"\n{'='*60}")
    print(f"Model: {args.model}")
    print(f"Params: {num_params(model)/1e6:.2f} M")
    print(f"FLOPs:  {gflops:.2f} GFLOPs")
    print(f"Acc@1:  {metrics['acc1']:.2f}%")
    print(f"Throughput: {metrics['throughput']:.2f} im/s")
    print(f"Latency:    {metrics['latency_ms']:.2f} ms/img")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()