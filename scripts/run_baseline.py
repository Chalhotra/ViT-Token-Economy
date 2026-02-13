from __future__ import annotations
import argparse
import torch

from src.utils import get_device, seed_everything, num_params
from src.imagenet_mapping import build_imagenet100_to_1k_map
from src.data import DataConfig, load_imagenet100_split, build_transform_for_model, apply_timm_preprocess, build_loader
from src.eval import evaluate_accuracy_latency_throughput, compute_gflops

# Import new config structure
from src.configs import ModelConfig, TopKConfig, EViTConfig
from src.models_smthing import create_model, shrink_imagenet1k_head_to_imagenet100

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model ID (e.g. topk_deit_tiny_patch16_224)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--seed", type=int, default=42)
    
    # --- Dynamic Pruning Arguments ---
    ap.add_argument("--pruning-method", type=str, default="none", choices=["none", "topk", "evit"],
                    help="Which pruning method config to use")
    
    # Shared pruning args
    ap.add_argument("--pruning-locs", type=int, nargs="+", default=[3, 6, 9])
    ap.add_argument("--keep-rates", type=float, nargs="+", default=[0.7, 0.7, 0.7])
    
    # Method-specific args (e.g. for EViT)
    ap.add_argument("--fuse-token", action="store_true", help="Enable token fusion (EViT only)")

    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()

    # --- 1. Construct the Specific Pruning Config ---
    pruning_cfg = None
    
    if args.pruning_method == "topk":
        pruning_cfg = TopKConfig(
            pruning_locs=args.pruning_locs,
            keep_rates=args.keep_rates
        )
    elif args.pruning_method == "evit":
        pruning_cfg = EViTConfig(
            pruning_locs=args.pruning_locs,
            keep_rates=args.keep_rates,
            fuse_token=args.fuse_token
        )

    # --- 2. Create Model Config ---
    config = ModelConfig(
        model_id=args.model,
        pretrained=True,
        pruning=pruning_cfg  # Pass the specific config object
    )

    print(f"Creating model: {args.model}")
    if pruning_cfg:
        print(f"Pruning Strategy: {pruning_cfg}")

    model = create_model(config)
    
    maps = build_imagenet100_to_1k_map()
    model = shrink_imagenet1k_head_to_imagenet100(model, maps.new_to_old_map, num_classes=100)
    model = model.to(device).eval()

    # (Rest of the pipeline remains identical)
    ds = load_imagenet100_split(DataConfig(split=args.split))
    transform = build_transform_for_model(model)
    ds_t = apply_timm_preprocess(ds, transform)
    loader = build_loader(ds_t, DataConfig(batch_size=args.batch_size, split=args.split, shuffle=False))

    metrics = evaluate_accuracy_latency_throughput(model, loader, device)
    
    # Robust GFLOPs check (handle custom models that might break standard profilers)
    try:
        sample = ds_t[0]["pixel_values"].unsqueeze(0).to(device)
        gflops = compute_gflops(model, sample)
    except Exception as e:
        print(f"Warning: Could not compute FLOPs for dynamic model ({e})")
        gflops = 0.0

    print(f"Model: {args.model}")
    print(f"Params: {num_params(model)/1e6:.2f} M")
    print(f"FLOPs:  {gflops:.2f} GFLOPs")
    print(f"Acc@1:  {metrics['acc1']:.2f}%")
    print(f"Throughput: {metrics['throughput']:.2f} im/s")
    print(f"Latency:    {metrics['latency_ms']:.2f} ms/img")

if __name__ == "__main__":
    main()