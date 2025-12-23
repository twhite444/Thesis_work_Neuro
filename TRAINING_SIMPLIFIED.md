# Simplified Training Script Interface

## Overview
The training script has been simplified to have just **two model choices**: `mlp` and `cnn`.

PCA support is now handled transparently through the `--use-pca` flag - you no longer need to specify `pca_mlp` as a model type.

## Quick Start

### Basic Usage (Raw Maps)
```bash
# MLP on raw spatial maps
python scripts/train_baseline_nn.py --model mlp --epochs 100

# CNN on raw spatial maps  
python scripts/train_baseline_nn.py --model cnn --epochs 100
```

### PCA Mode (Faster, Smaller)
```bash
# MLP with PCA (auto-selects optimized pca_mlp internally)
python scripts/train_baseline_nn.py --model mlp --use-pca --epochs 100

# CNN with PCA - NOT ALLOWED
# CNN requires 2D spatial structure, incompatible with 1D PCA components
```

### K-Fold Cross-Validation
```bash
# 5-fold CV with PCA
python scripts/train_baseline_nn.py --model mlp --use-pca --k-folds 5 --epochs 100

# 3-fold CV with raw maps
python scripts/train_baseline_nn.py --model mlp --k-folds 3 --epochs 100
```

## Model Selection Logic

| User Input | Internal Model | Notes |
|------------|---------------|-------|
| `--model mlp` | `mlp` | Standard MLP on raw maps |
| `--model mlp --use-pca` | `pca_mlp` | Auto-selected for PCA |
| `--model cnn` | `cnn` | Standard CNN on raw maps |
| `--model cnn --use-pca` | ❌ ERROR | CNN incompatible with PCA |

## Training Modes

### Single Split Mode (Default)
```bash
python scripts/train_baseline_nn.py --model mlp --use-pca --epochs 100
```
- Splits data: 70% train, 15% val, 15% test
- Saves best model based on validation loss
- Output: `experiments/baseline_nn/best_model.pth`

### K-Fold Cross-Validation Mode
```bash
python scripts/train_baseline_nn.py --model mlp --use-pca --k-folds 5 --epochs 100
```
- Splits data into K folds
- Trains K models, each on different train/val split
- Reports average performance across folds
- Output: `experiments/baseline_nn/fold_N/` and `cv_results.json`

## All Options

```
usage: train_baseline_nn.py [-h] --model {mlp,cnn} 
                            [--use-pca] [--n-components N]
                            [--k-folds K] [--random-seed SEED]
                            [--epochs N] [--batch-size N] [--lr LR]
                            [--dropout P] [--weight-decay W]
                            [--patience N] [--quiet]

Required:
  --model {mlp,cnn}        Model architecture (mlp or cnn)

PCA Options:
  --use-pca               Use PCA-transformed maps (~170x smaller, faster)
  --n-components N        Number of PCA components (default: 50)

Training Mode:
  --k-folds K             K-fold CV (K ≥ 2). Omit for single train/val/test split
  --random-seed SEED      Random seed for reproducibility (default: 42)

Hyperparameters:
  --epochs N              Number of epochs (default: 100)
  --batch-size N          Batch size (default: 32)
  --lr LR                 Learning rate (default: 0.005)
  --dropout P             Dropout probability (default: 0.35)
  --weight-decay W        L2 regularization (default: 1e-5)
  --patience N            Early stopping patience (default: 20)

Output:
  --quiet                 Suppress per-epoch progress bars
```

## Examples

### Quick Test (2 epochs)
```bash
python scripts/train_baseline_nn.py --model mlp --use-pca --epochs 2 --quiet
```

### Full Training with PCA
```bash
python scripts/train_baseline_nn.py --model mlp --use-pca --epochs 100 --lr 0.001
```

### 5-Fold Cross-Validation
```bash
python scripts/train_baseline_nn.py --model mlp --use-pca --k-folds 5 --epochs 100 --quiet
```

### CNN on Raw Maps (High Quality, Slower)
```bash
python scripts/train_baseline_nn.py --model cnn --epochs 100 --batch-size 16
```

## What Changed?

### Before (Old Interface)
```bash
# Had to explicitly choose pca_mlp
python scripts/train_baseline_nn.py --model pca_mlp --n-components 50

# K-fold was separate script
python scripts/train_baseline_nn_kfold.py --model pca_mlp --k-folds 5
```

### After (New Interface)
```bash
# Just use --use-pca flag with mlp
python scripts/train_baseline_nn.py --model mlp --use-pca --n-components 50

# K-fold integrated into main script
python scripts/train_baseline_nn.py --model mlp --use-pca --k-folds 5
```

**Benefits:**
- ✅ Simpler: Only 2 model choices (mlp, cnn)
- ✅ Intuitive: `--use-pca` flag clearly indicates PCA mode
- ✅ Unified: Single script for both single-split and K-fold
- ✅ Safe: Automatic validation prevents invalid combinations
- ✅ Clean: Deprecated old K-fold script, reduced duplication

## Known Issues

### Poor PCA Performance (R² ≈ 0.3)
**Problem:** PCA components have different scales:
- Component 0: std = 17.5, range [-42, +37]
- Component 9: std = 5.7, range [-18, +14]

Neural network focuses on high-variance components (easier MSE reduction) and ignores low-variance components.

**Solution:** Needs per-component normalization or weighted loss function (not yet implemented).

**Workaround:** Use raw maps (`--model cnn` or `--model mlp` without `--use-pca`) for better accuracy until scaling is fixed.

## File Locations

- **Main Script:** `scripts/train_baseline_nn.py`
- **Deprecated:** `scripts/train_baseline_nn_kfold.py.deprecated`
- **Documentation:** 
  - `TRAINING_QUICK_REF.md` - Quick reference
  - `UNIFIED_TRAINING_SCRIPT.md` - Full documentation
  - `TRAINING_SCRIPTS_UPDATE.md` - Initial PCA support docs

## Next Steps

To improve PCA performance:
1. Add per-component normalization in `pca_transform.py`
2. Or add `nn.BatchNorm1d(n_components)` in `pca_mlp` model
3. Or implement weighted MSE loss (weight by 1/variance)

Expected improvement: R² from 0.3 → 0.5+ (matching raw map performance).
