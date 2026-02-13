from __future__ import annotations
import argparse
import torch

from src.utils import get_device, seed_everything, num_params
from src.imagenet_mapping import build_imagenet100_to_1k_map
from src.models import ModelConfig, create_model, shrink_imagenet1k_head_to_imagenet100
from src.data import DataConfig, load_imagenet100_split, build_transform_for_model, apply_timm_preprocess, build_loader
from src.eval import evaluate_accuracy_latency_throughput, compute_gflops

from src.pruning import TopKConfig, apply_topk_pruning  # NEW


def _parse_layers(s: str):
    if s.strip().lower() == "all":
        return "all"
    if not s.strip():
        return "all"
    return [int(x) for x in s.split(",") if x.strip() != ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="timm model id, e.g. vit_tiny_patch16_224 or deit_tiny_patch16_224")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--seed", type=int, default=42)

    # TopK pruning flags (NEW)
    ap.add_argument("--topk", action="store_true", help="enable Top-K token pruning")
    ap.add_argument("--keep-rate", type=float, default=1.0, help="fraction of patch tokens kept (CLS always kept)")
    ap.add_argument("--topk-layers", type=str, default="all", help='block indices to prune after: "all" or comma list, e.g. "2,5,8"')
    ap.add_argument("--preserve-token-order", action="store_true", help="keep selected tokens in original order (recommended)")

    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()

    maps = build_imagenet100_to_1k_map()

    # Create + shrink head (baseline unchanged)
    model = create_model(ModelConfig(model_id=args.model, pretrained=True))
    model = shrink_imagenet1k_head_to_imagenet100(model, maps.new_to_old_map, num_classes=100)

    # Apply TopK (feature toggle, orthogonal)
    topk_cfg = TopKConfig(
        enabled=bool(args.topk),
        keep_rate=float(args.keep_rate),
        layers=_parse_layers(args.topk_layers),
        preserve_token_order=bool(args.preserve_token_order),
    )
    model = apply_topk_pruning(model, topk_cfg)

    model = model.to(device).eval()

    ds = load_imagenet100_split(DataConfig(split=args.split))
    transform = build_transform_for_model(model)
    ds_t = apply_timm_preprocess(ds, transform)

    loader = build_loader(ds_t, DataConfig(batch_size=args.batch_size, split=args.split, shuffle=False))

    metrics = evaluate_accuracy_latency_throughput(model, loader, device)

    # FLOPs on one sample (matches notebook behavior)
    sample = ds_t[0]["pixel_values"].unsqueeze(0).to(device)
    gflops = compute_gflops(model, sample)

    print(f"Model: {args.model}")
    print(f"Params: {num_params(model)/1e6:.2f} M")
    print(f"FLOPs:  {gflops:.2f} GFLOPs (1x3x224x224)")
    print(f"Acc@1:  {metrics['acc1']:.2f}%")
    print(f"Throughput: {metrics['throughput']:.2f} im/s")
    print(f"Latency:    {metrics['latency_ms']:.2f} ms/img")


if __name__ == "__main__":
    main()
