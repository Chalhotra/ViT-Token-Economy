# ViT/DeiT Tiny baselines (ImageNet-100)

This repo reimplements your notebook as a clean, config-driven pipeline with **no change in functionality**:
- Loads **ImageNet-100 validation** from Hugging Face (`clane9/imagenet-100`)
- Uses `timm`'s `resolve_data_config` + `create_transform` (eval transforms)
- Creates a **pretrained ImageNet-1K model** (ViT-Tiny or DeiT-Tiny)
- Replaces the 1000-class head with a 100-class head by **copying the corresponding rows** using a `new_to_old_map`
- Evaluates **top-1 accuracy**, **throughput**, **latency**
- Computes **FLOPs** with `fvcore`

## Quickstart (Colab/Kaggle)

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
pip install -e .
python scripts/run_baseline.py --model vit_tiny_patch16_224
python scripts/run_baseline.py --model deit_tiny_patch16_224
```

## Outputs
Prints:
- Accuracy (%)
- Throughput (images/sec)
- Latency (ms/image)
- FLOPs (GFLOPs) for a single image
