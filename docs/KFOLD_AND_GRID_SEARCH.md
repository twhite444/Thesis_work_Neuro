# K-Fold Cross-Validation and Grid Search Guide

## Overview

New advanced training features for robust model evaluation and hyperparameter optimization:

1. **Early Stopping** - Automatically stop training when validation performance plateaus
2. **K-Fold Cross-Validation** - Match reference paper methodology with 5-fold CV  
3. **Grid Search** - Explore any hyperparameter combinations systematically

---

## 1. Early Stopping

Automatically stops training when validation loss doesn't improve for N epochs.

### Usage

```bash
# Stop if no improvement for 15 epochs
python scripts/train_baseline_nn.py --model mlp --epochs 200 --early-stopping 15

# Stop if no improvement for 20 epochs  
python scripts/train_baseline_nn.py --model cnn --epochs 300 --early-stopping 20
```

### Benefits

- **Saves time**: No need to manually monitor training
- **Prevents overfitting**: Stops before model starts memorizing training data
- **Optimal checkpoints**: Always saves best model before stopping

### Recommended Values

- **Short runs (< 50 epochs)**: `--early-stopping 5-10`
- **Medium runs (50-150 epochs)**: `--early-stopping 15-20`
- **Long runs (> 150 epochs)**: `--early-stopping 25-30`

---

## 2. K-Fold Cross-Validation

Implements the same methodology as the reference paper for robust evaluation.

### Why Use K-Fold CV?

✅ **Matches reference paper** - Direct comparison with published results  
✅ **More robust estimates** - Mean ± std across folds  
✅ **Small dataset friendly** - Uses all 287 samples efficiently  
✅ **Reduces overfitting** - Multiple independent validation sets  
✅ **Thesis appropriate** - Expected by reviewers  

### Basic Usage

```bash
# Standard 5-fold CV (matches reference paper)
python scripts/train_baseline_nn_kfold.py --model mlp --n-folds 5 --epochs 100

# 3-fold CV for faster iteration
python scripts/train_baseline_nn_kfold.py --model mlp --n-folds 3 --epochs 50

# 10-fold CV for maximum robustness (but slower)
python scripts/train_baseline_nn_kfold.py --model mlp --n-folds 10 --epochs 100
```

### With All Options

```bash
python scripts/train_baseline_nn_kfold.py \
    --model mlp \
    --n-folds 5 \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.005 \
    --dropout 0.35 \
    --weight-decay 1e-5 \
    --early-stopping 15 \
    --output-dir experiments/thesis_kfold_cv \
    --random-seed 42
```

### Output Files

```
experiments/baseline_nn_kfold/
├── cv_results.json           # Aggregated results
├── fold_1/
│   ├── best_model.pth       # Best model for fold 1
│   ├── metrics.json         # Metrics for fold 1
│   └── logs/                # Tensorboard logs
├── fold_2/
│   └── ...
├── fold_3/
│   └── ...
├── fold_4/
│   └── ...
└── fold_5/
    └── ...
```

### Results Format

The `cv_results.json` contains:

```json
{
  "mean_metrics": {
    "best_val_loss": 0.2555,
    "best_val_correlation": 0.548,
    "best_val_r2": 0.338,
    "best_val_mae": 0.312
  },
  "std_metrics": {
    "best_val_loss": 0.0124,
    "best_val_correlation": 0.023,
    "best_val_r2": 0.015,
    "best_val_mae": 0.008
  },
  "best_fold": 3,
  "fold_results": [...]
}
```

### Reporting Results

For your thesis, report as:

> "We evaluated the baseline MLP using 5-fold cross-validation. The model achieved a mean correlation of **0.548 ± 0.023** (mean ± std) and R² of **0.338 ± 0.015** across folds."

---

## 3. Grid Search

Flexible hyperparameter optimization that searches any parameter combinations.

### What Can You Search?

**Model Parameters:**
- `dropout` - Dropout rate
- `hidden_dims` - Architecture (MLP only)
- `latent_dim` - Latent dimension (CNN only)

**Training Parameters:**
- `learning_rate` - Optimizer learning rate
- `weight_decay` - L2 regularization strength
- `batch_size` - Batch size
- `num_epochs` - Number of epochs

### Basic Examples

**Search dropout rates:**
```bash
python scripts/grid_search_baseline.py \
    --model mlp \
    --param dropout 0.2 0.35 0.5 \
    --epochs 50
```

**Search learning rates:**
```bash
python scripts/grid_search_baseline.py \
    --model mlp \
    --param learning_rate 0.001 0.003 0.005 0.007 0.01 \
    --epochs 50
```

**Search regularization:**
```bash
python scripts/grid_search_baseline.py \
    --model mlp \
    --param dropout 0.3 0.35 0.4 \
    --param weight_decay 0.0 1e-5 1e-4 1e-3 \
    --epochs 50
```

**Search architectures:**
```bash
python scripts/grid_search_baseline.py \
    --model mlp \
    --param hidden_dims "[512,256,128]" "[1024,512,256]" "[256,128,64]" \
    --epochs 50
```

### Advanced: Multi-Parameter Search

Search multiple parameters simultaneously:

```bash
python scripts/grid_search_baseline.py \
    --model mlp \
    --param dropout 0.2 0.35 0.5 \
    --param learning_rate 0.001 0.005 0.01 \
    --param weight_decay 0.0 1e-5 1e-4 \
    --use-kfold --n-folds 5 \
    --epochs 100 \
    --output-dir experiments/comprehensive_grid_search
```

This searches: 3 × 3 × 3 = **27 combinations** with 5-fold CV each = **135 total training runs**

### With K-Fold CV (Recommended for Thesis)

```bash
# Enable K-fold CV for robust hyperparameter selection
python scripts/grid_search_baseline.py \
    --model mlp \
    --param dropout 0.2 0.35 0.5 \
    --param learning_rate 0.003 0.005 0.007 \
    --use-kfold --n-folds 5 \
    --epochs 100
```

### Fast Iteration (Single Split)

```bash
# Disable K-fold for faster iteration during development
python scripts/grid_search_baseline.py \
    --model mlp \
    --param dropout 0.2 0.35 0.5 \
    --no-kfold \
    --epochs 30
```

### With Early Stopping

```bash
# Use early stopping to speed up grid search
python scripts/grid_search_baseline.py \
    --model mlp \
    --param dropout 0.2 0.3 0.35 0.4 0.5 \
    --param learning_rate 0.001 0.003 0.005 0.007 0.01 \
    --epochs 200 \
    --early-stopping 20
```

### Output Files

```
experiments/grid_search/
├── grid_search_results.json   # Detailed results for all combinations
├── grid_search_summary.csv    # Table format (easy to view in Excel)
├── exp_001/                   # First parameter combination
│   ├── fold_1/ (if using k-fold)
│   └── ...
├── exp_002/                   # Second parameter combination
│   └── ...
└── ...
```

### Analyzing Results

**1. View summary table:**
```bash
# View in terminal
column -t -s, experiments/grid_search/grid_search_summary.csv | less -S

# Or open in Excel/Numbers/LibreOffice
```

**2. Load in Python:**
```python
import pandas as pd
import json

# Load summary table
summary = pd.read_csv('experiments/grid_search/grid_search_summary.csv')

# Sort by performance
summary.sort_values('mean_correlation', ascending=False)

# Load detailed results
with open('experiments/grid_search/grid_search_results.json') as f:
    results = json.load(f)

print(f"Best params: {results['best_params']}")
print(f"Best score: {results['best_score']:.3f}")
```

### Thesis Recommendations

1. **Initial Exploration (Fast)**:
   ```bash
   python scripts/grid_search_baseline.py --model mlp \
       --param dropout 0.2 0.3 0.35 0.4 0.5 \
       --param learning_rate 0.001 0.003 0.005 0.007 0.01 \
       --no-kfold --epochs 50
   ```
   → Identifies promising ranges quickly

2. **Refined Search (Robust)**:
   ```bash
   python scripts/grid_search_baseline.py --model mlp \
       --param dropout 0.30 0.35 0.40 \
       --param learning_rate 0.003 0.005 0.007 \
       --param weight_decay 0.0 1e-5 1e-4 \
       --use-kfold --n-folds 5 --epochs 100
   ```
   → Final hyperparameters with CV validation

3. **Report Best Model**:
   - Train final model with best hyperparameters using K-fold CV
   - Report mean ± std across folds
   - Use best fold's model for final predictions

---

## Recommended Workflow for Thesis

### Step 1: Baseline with K-Fold CV

Establish baseline performance with reference paper settings:

```bash
python scripts/train_baseline_nn_kfold.py \
    --model mlp \
    --n-folds 5 \
    --epochs 100 \
    --lr 0.005 \
    --dropout 0.35 \
    --weight-decay 0.0 \
    --output-dir experiments/baseline_reference_cv \
    --random-seed 42
```

**Thesis section**: "We first established a baseline using the reference architecture..."

### Step 2: Fast Hyperparameter Exploration

Quick grid search to identify promising ranges:

```bash
python scripts/grid_search_baseline.py \
    --model mlp \
    --param dropout 0.2 0.3 0.35 0.4 0.5 \
    --param learning_rate 0.001 0.003 0.005 0.007 0.01 \
    --no-kfold \
    --epochs 50 \
    --early-stopping 10 \
    --output-dir experiments/grid_search_initial
```

**Thesis section**: "We conducted a preliminary grid search to identify optimal hyperparameter ranges..."

### Step 3: Refined Search with CV

Focused search in promising range with robust evaluation:

```bash
python scripts/grid_search_baseline.py \
    --model mlp \
    --param dropout 0.30 0.35 0.40 \
    --param learning_rate 0.003 0.005 0.007 \
    --param weight_decay 0.0 1e-5 1e-4 \
    --use-kfold --n-folds 5 \
    --epochs 100 \
    --early-stopping 15 \
    --output-dir experiments/grid_search_refined_cv
```

**Thesis section**: "Final hyperparameter optimization was performed using 5-fold cross-validation..."

### Step 4: Final Model Evaluation

Train final model with best hyperparameters:

```bash
python scripts/train_baseline_nn_kfold.py \
    --model mlp \
    --n-folds 5 \
    --epochs 150 \
    --lr 0.005 \  # (from grid search)
    --dropout 0.35 \  # (from grid search)
    --weight-decay 1e-5 \  # (from grid search)
    --early-stopping 20 \
    --output-dir experiments/final_model_cv \
    --random-seed 42
```

**Thesis section**: "The final optimized model achieved..."

---

## Performance Tips

### Faster Development Iteration

1. **Use smaller n-folds**: `--n-folds 3` instead of 5
2. **Disable K-fold**: `--no-kfold` for single split
3. **Fewer epochs**: `--epochs 30` during exploration
4. **Enable early stopping**: `--early-stopping 10`
5. **Smaller grids**: Test 2-3 values instead of 5

### Optimal for Thesis

1. **Use 5-fold CV**: Matches reference paper
2. **Sufficient epochs**: `--epochs 100-150`
3. **Early stopping**: `--early-stopping 15-20` prevents waste
4. **Fixed random seed**: `--random-seed 42` for reproducibility

### Time Estimates (on MPS)

With current optimizations (batch_size=32, num_workers=0):

- **1 epoch**: ~1 second
- **100 epochs**: ~2 minutes  
- **5-fold CV (100 epochs)**: ~10 minutes
- **Grid search (3×3×3, 5-fold CV, 100 epochs)**: ~4.5 hours

---

## Troubleshooting

### "Out of memory" errors

Reduce batch size:
```bash
--batch-size 16  # or even 8
```

### Grid search taking too long

1. Reduce grid size (fewer parameter values)
2. Use `--no-kfold` initially
3. Add `--early-stopping 10`
4. Reduce `--epochs 30`

### Want to resume interrupted grid search

Grid search saves each experiment independently. You can:
1. Check which experiments completed in the output directory
2. Manually exclude completed combinations and re-run

---

## Summary

| Feature | Command | When to Use |
|---------|---------|-------------|
| **Basic Training** | `train_baseline_nn.py` | Quick experiments |
| **Early Stopping** | `--early-stopping 15` | Long training runs |
| **K-Fold CV** | `train_baseline_nn_kfold.py` | Robust evaluation, thesis results |
| **Grid Search** | `grid_search_baseline.py` | Hyperparameter optimization |
| **Grid Search + CV** | `--use-kfold --n-folds 5` | Final hyperparameter selection |

For your thesis, the typical workflow is:
1. Baseline with K-fold CV
2. Grid search exploration (fast)
3. Grid search refinement (with CV)
4. Final model with best hyperparameters (K-fold CV)

This ensures rigorous methodology matching the reference paper! 🎓
