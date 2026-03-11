# ViT/DeiT Tiny baselines (ImageNet-100)

This repo reimplements your notebook as a clean, config-driven pipeline with **no change in functionality**:
- Loads **ImageNet-100 validation** from Hugging Face (`clane9/imagenet-100`)
- Uses `timm`'s `resolve_data_config` + `create_transform` (eval transforms)
- Creates a **pretrained ImageNet-1K model** (ViT-Tiny or DeiT-Tiny)
- Replaces the 1000-class head with a 100-class head by **copying the corresponding rows** using a `new_to_old_map`
- Evaluates **top-1 accuracy**, **throughput**, **latency**
- Computes **FLOPs** with `fvcore`

## Quickstart (Local/Kaggle)

```bash
git clone https://github.com/Chalhotra/ViT-Token-Economy.git
cd ViT-Token-Economy
pip install -r requirements.txt
pip install -e .
python scripts/run_baseline.py --model vit_tiny_patch16_224
python scripts/run_baseline.py --model deit_tiny_patch16_224
```

## Quickstart (Google Colab)

1. Open the notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chalhotra/ViT-Token-Economy/blob/main/notebooks/01_baseline.ipynb)
2. Run the setup cell to clone the repository
   - For **public repos**: No token needed
   - For **private repos**: You'll be prompted for a GitHub token (or set `GITHUB_TOKEN` in Colab secrets)
3. Run the remaining cells to evaluate models

## Outputs
Prints:
- Accuracy (%)
- Throughput (images/sec)
- Latency (ms/image)
- FLOPs (GFLOPs) for a single image

## Project Structure

```
ViT-Token-Economy/
├── src/                      # Core modules
│   ├── data.py              # Dataset loading and preprocessing
│   ├── models.py            # Model creation and head adaptation
│   ├── eval.py              # Evaluation metrics
│   ├── imagenet_mapping.py  # ImageNet-100 to ImageNet-1K mapping
│   └── utils.py             # Utility functions
├── scripts/
│   └── run_baseline.py      # CLI entry point
├── notebooks/
│   └── 01_baseline.ipynb    # Colab-ready notebook
├── requirements.txt         # Dependencies
└── pyproject.toml          # Package configuration
```
