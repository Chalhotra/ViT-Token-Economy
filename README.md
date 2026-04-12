# ViT Token Economy (ImageNet-100)

A config-driven pipeline for evaluating **token economy methods** on Vision Transformers using ImageNet-100. Starting from pretrained ViT/DeiT baselines, the project implements and compares several strategies for reducing the number of tokens processed at each layer, and measures how each method interacts with input augmentation:

| Method | Description |
|--------|-------------|
| **Top-K Pruning** | Keeps the highest CLS-attention tokens; drops the rest |
| **EViT (Token Fusion)** | Keeps top-K tokens and fuses the remainder into a single extra token |
| **ToMe (Token Merging)** | Bipartite soft-matching that merges the most similar token pairs |
| **EViT + ToMe (Hybrid)** | Two-stage: EViT fusion followed by ToMe merging |
| **Random Drop** | Random token dropping baseline for comparison |

Core features:
- Loads **ImageNet-100** from Hugging Face (`clane9/imagenet-100`)
- Uses `timm` pretrained models (ViT-Tiny, ViT-Base, DeiT-Tiny, and others)
- Replaces the 1000-class head with a 100-class head by copying the corresponding rows
- Evaluates **top-1 accuracy**, **throughput**, **latency**, and **FLOPs**
- Custom **augmentation pipeline** (horizontal flip, colour jitter, slight rotation) applied at evaluation time to study robustness

## Quickstart (Local/Kaggle)

```bash
git clone https://github.com/Chalhotra/ViT-Token-Economy.git
cd ViT-Token-Economy
pip install -r requirements.txt
pip install -e .

# Baseline evaluation (no pruning)
python scripts/run_baseline.py --model vit_tiny_patch16_224
python scripts/run_baseline.py --model deit_tiny_patch16_224

# Top-K pruning with 70% keep rate at blocks 3, 6, 9
python scripts/run_baseline.py --model deit_tiny_patch16_224 \
    --topk --keep-rate 0.7 --reduction-loc "3,6,9"
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | *(required)* | `timm` model ID (e.g. `vit_tiny_patch16_224`, `deit_tiny_patch16_224`) |
| `--batch-size` | `64` | Batch size for evaluation |
| `--split` | `validation` | Dataset split |
| `--seed` | `42` | Random seed |
| `--topk` | off | Enable Top-K token pruning |
| `--keep-rate` | `1.0` | Keep rate(s); single value or list aligned with `--reduction-loc` |
| `--reduction-loc` | `""` | Comma-separated block indices for pruning (e.g. `"3,6,9"`) |
| `--no-exp-keep-rate` | off | Disable exponentiating a single keep-rate across reduction locations |

## Quickstart (Google Colab)

1. Open a notebook:
   - Baseline: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chalhotra/ViT-Token-Economy/blob/main/notebooks/01_baseline.ipynb)
   - Top-K: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chalhotra/ViT-Token-Economy/blob/main/notebooks/02_topk_testing.ipynb)
   - Random Masking: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chalhotra/ViT-Token-Economy/blob/main/notebooks/03_random_masking_testing.ipynb)
   - EViT: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chalhotra/ViT-Token-Economy/blob/main/notebooks/04_evit_testing.ipynb)
   - ToMe: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chalhotra/ViT-Token-Economy/blob/main/notebooks/05_tome_testing.ipynb)
   - EViT + ToMe: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chalhotra/ViT-Token-Economy/blob/main/notebooks/06_evit_tome_testing.ipynb)
   - EViT + ToMe (Augmentation): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chalhotra/ViT-Token-Economy/blob/main/notebooks/07_evit_tome_testing_augmentation.ipynb)
   - Baseline Augmentation Sweep: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chalhotra/ViT-Token-Economy/blob/main/notebooks/08_baseline_augmentation_testing.ipynb)
   - EViT Augmentation Sweep: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chalhotra/ViT-Token-Economy/blob/main/notebooks/09_evit_testing_augmentation.ipynb)
   - ToMe Augmentation Sweep: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chalhotra/ViT-Token-Economy/blob/main/notebooks/10_tome_testing_augmentation.ipynb)
   - Hybrid (No Augmentation): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chalhotra/ViT-Token-Economy/blob/main/notebooks/11_hybrid_no_augmentation_testing.ipynb)
   - Baseline (No Augmentation): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chalhotra/ViT-Token-Economy/blob/main/notebooks/12_baseline_no_augmentation.ipynb)
2. Run the setup cell to clone the repository
   - For **public repos**: No token needed
   - For **private repos**: You'll be prompted for a GitHub token (or set `GITHUB_TOKEN` in Colab secrets)
3. Run the remaining cells to evaluate models

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_baseline.ipynb` | Baseline evaluation of ViT/DeiT models on ImageNet-100 |
| `02_topk_testing.ipynb` | Top-K token pruning across different keep rates with visualizations |
| `03_random_masking_testing.ipynb` | Random token dropping baseline with visualization |
| `04_evit_testing.ipynb` | EViT token fusion evaluation and visualization |
| `05_tome_testing.ipynb` | ToMe bipartite soft-matching merging evaluation |
| `06_evit_tome_testing.ipynb` | Hybrid EViT + ToMe strategy evaluation |
| `07_evit_tome_testing_augmentation.ipynb` | EViT + ToMe hybrid with augmentation sweep and two-stage visualizations |
| `08_baseline_augmentation_testing.ipynb` | Baseline augmentation sweep (ViT-Tiny & ViT-Base × 3 augmentations) |
| `09_evit_testing_augmentation.ipynb` | EViT token fusion augmentation sweep |
| `10_tome_testing_augmentation.ipynb` | ToMe merging augmentation sweep |
| `11_hybrid_no_augmentation_testing.ipynb` | EViT + ToMe hybrid sweep without augmentation |
| `12_baseline_no_augmentation.ipynb` | Baseline (no augmentation) with CSV-logging for per-prediction analysis |
| `RAD(Relative Accuracy Degradation) calculation.ipynb` | Computes Relative Accuracy Degradation across all methods and keep rates |

## Augmentation

The project includes a custom augmentation module (`augmentation.py`) for evaluating the robustness of each token economy method under distribution shift.

### Available Augmentations

| Class | Description |
|-------|-------------|
| `HorizontalFlip` | Random horizontal flip |
| `ColourJitter` | Random brightness, contrast, saturation, and hue jitter |
| `SlightRotation` | Rotation by a fixed angle within a configurable range |
| `AugmentationPipeline` | Chains any sequence of augmentations in order |

### Preset Pipelines

| Function | Pipeline |
|----------|----------|
| `all_augmentations()` | Flip → ColourJitter → SlightRotation |
| `horizontal_flip_only()` | HorizontalFlip only |
| `rotation_only()` | SlightRotation only |
| `jitter_only()` | ColourJitter only |
| `colour_only()` | ColourJitter (photometric only) |
| `geometric_only()` | HorizontalFlip + SlightRotation (spatial only) |
| `no_augmentation()` | Identity pipeline — baseline / eval mode |

## Outputs

- **Accuracy** — top-1 accuracy (%)
- **Throughput** — images per second
- **Latency** — milliseconds per image
- **FLOPs** — GFLOPs for a single 3×224×224 image
- **Params** — total model parameters

Notebooks additionally produce per-layer visualizations of token retention, fusion, and merging patterns.

## Project Structure

```
ViT-Token-Economy/
├── src/                          # Core modules
│   ├── data.py                  # Dataset loading and preprocessing
│   ├── models.py                # Model creation and head adaptation
│   ├── eval.py                  # Accuracy, throughput, latency, FLOPs
│   ├── imagenet_mapping.py      # ImageNet-100 ↔ ImageNet-1K mapping
│   ├── utils.py                 # Device, seeding, parameter counting
│   └── test_models/             # Token economy methods
│       ├── topk.py              # Top-K attention-based pruning
│       ├── evit.py              # EViT token fusion
│       ├── tome.py              # ToMe bipartite token merging
│       ├── evit_tome.py         # Hybrid EViT + ToMe
│       └── random_drop.py       # Random token dropping baseline
├── augmentation.py              # Custom augmentation classes and preset pipelines
├── scripts/
│   └── run_baseline.py          # CLI entry point
├── notebooks/                   # Colab-ready experiment notebooks
│   ├── 01_baseline.ipynb
│   ├── 02_topk_testing.ipynb
│   ├── 03_random_masking_testing.ipynb
│   ├── 04_evit_testing.ipynb
│   ├── 05_tome_testing.ipynb
│   ├── 06_evit_tome_testing.ipynb
│   ├── 07_evit_tome_testing_augmentation.ipynb
│   ├── 08_baseline_augmentation_testing.ipynb
│   ├── 09_evit_testing_augmentation.ipynb
│   ├── 10_tome_testing_augmentation.ipynb
│   ├── 11_hybrid_no_augmentation_testing.ipynb
│   ├── 12_baseline_no_augmentation.ipynb
│   └── RAD(Relative Accuracy Degradation) calculation.ipynb
├── tests/                       # Unit tests
│   ├── test_models.py           # Model creation and head adaptation
│   ├── test_imagenet_mapping.py # ImageNet-100 mapping validation
│   └── test_utils.py            # Utility function tests
├── requirements.txt             # Dependencies
└── pyproject.toml               # Package configuration
```
