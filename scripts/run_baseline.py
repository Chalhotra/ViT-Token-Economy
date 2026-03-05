from __future__ import annotations
import argparse
import torch

from src.utils import get_device, seed_everything, num_params
from src.imagenet_mapping import build_imagenet100_to_1k_map
from src.data import DataConfig, load_imagenet100_split, build_transform_for_model, apply_timm_preprocess, build_loader
from src.eval import evaluate_accuracy_latency_throughput, compute_gflops

from src.test_models.topk import TopKConfig, apply_topk_pruning  # NEW

def _parse_int_list(s: str) -> list[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split(",") if x.strip() != ""]


def _parse_layers(s: str):
    if s.strip().lower() == "all":
        return "all"
    if not s.strip():
        return "all"
    return [int(x) for x in s.split(",") if x.strip() != ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model ID (e.g. topk_deit_tiny_patch16_224, deit_tiny_patch16_224)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--seed", type=int, default=42)

    # TopK pruning flags (NEW)
    ap.add_argument("--topk", action="store_true", help="enable Top-K token pruning")
    ap.add_argument(
        "--keep-rate",
        nargs="+",
        type=float,
        default=[1.0],
        help="keep rates; one value or list aligned with --reduction-loc (reference behavior supported)",
    )
    ap.add_argument(
        "--reduction-loc",
        type=str,
        default="",
        help='comma-separated block indices where pruning is applied, e.g. "3,6,9"',
    )
    ap.add_argument(
        "--no-exp-keep-rate",
        action="store_true",
        help="disable reference behavior where single keep-rate is exponentiated across reduction locations",
    )

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

    # Create + shrink head (baseline unchanged)
    model = create_model(ModelConfig(model_id=args.model, pretrained=True))
    model = shrink_imagenet1k_head_to_imagenet100(model, maps.new_to_old_map, num_classes=100)

    topk_cfg = TopKConfig(
        enabled=bool(args.topk),
        keep_rate=list(args.keep_rate),
        reduction_loc=_parse_int_list(args.reduction_loc),
        exponentiate_single_keep_rate=not bool(args.no_exp_keep_rate),
    )
    model = apply_topk_pruning(model, topk_cfg)

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