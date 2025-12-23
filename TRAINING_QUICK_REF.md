# Quick Reference - Unified Training Script

## One Script, All Modes! 🚀

```bash
python scripts/train_baseline_nn.py [OPTIONS]
```

## Common Commands

### 🏃 Quick Start (Fast PCA Training)
```bash
python scripts/train_baseline_nn.py --model pca_mlp --use-pca --epochs 50
```

### 📊 Robust Evaluation (K-Fold CV)
```bash
python scripts/train_baseline_nn.py --model pca_mlp --use-pca --k-folds 5 --epochs 100
```

### 🎯 High Accuracy (Raw Maps)
```bash
python scripts/train_baseline_nn.py --model mlp --epochs 200 --early-stopping 20
```

### 📈 Publication Quality (K-Fold + Raw Maps)
```bash
python scripts/train_baseline_nn.py --model mlp --k-folds 5 --epochs 100
```

## Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model {mlp,cnn,pca_mlp}` | Model architecture | **Required** |
| `--use-pca` | Use PCA components (faster) | False |
| `--k-folds N` | K-fold CV (None = single split) | None |
| `--epochs N` | Number of epochs | 100 |
| `--lr` | Learning rate | 0.005 |
| `--batch-size` | Batch size | 32 |
| `--dropout` | Dropout rate | 0.35 |
| `--early-stopping N` | Early stopping patience | 0 (off) |
| `--quiet` | Suppress progress bars | False |

## Model + PCA Compatibility

| Model | use_pca=False | use_pca=True |
|-------|---------------|--------------|
| `mlp` | ✅ Spatial maps (3397-dim) | ❌ Error |
| `cnn` | ✅ Spatial maps (79×43) | ❌ Error |
| `pca_mlp` | ⚠️ Auto-enables PCA | ✅ PCA components (50-dim) |

**Rule**: Always use `--model pca_mlp` with `--use-pca`

## Examples by Use Case

### Hyperparameter Search
```bash
# Fast iterations on PCA
for lr in 0.001 0.005 0.01; do
    python scripts/train_baseline_nn.py --model pca_mlp --use-pca --lr $lr --epochs 50
done
```

### Architecture Comparison
```bash
# Compare MLP vs CNN on raw maps
python scripts/train_baseline_nn.py --model mlp --epochs 100 --output-dir experiments/mlp
python scripts/train_baseline_nn.py --model cnn --epochs 100 --output-dir experiments/cnn
```

### Robust Cross-Validation
```bash
# 5-fold CV for reliable metrics
python scripts/train_baseline_nn.py --model pca_mlp --use-pca --k-folds 5 --epochs 100
```

### Production Training
```bash
# Best accuracy with early stopping
python scripts/train_baseline_nn.py --model mlp --epochs 300 --early-stopping 30 --lr 0.001
```

## Output Locations

### Single Split Mode
```
experiments/baseline_nn/
├── training_curves.png
├── best_model.pth
├── training_metrics.csv
└── final_metrics.json
```

### K-Fold Mode
```
experiments/baseline_nn/
├── cv_results.json          # Overall CV results
├── fold_1/
│   ├── training_curves.png
│   ├── best_model.pth
│   └── metrics.csv
├── fold_2/
│   └── ...
└── fold_N/
    └── ...
```

## Tips

💡 **Use PCA for**: Rapid prototyping, hyperparameter search, quick experiments  
💡 **Use Raw Maps for**: Final models, publication results, best accuracy  
💡 **Use K-Fold for**: Robust evaluation, small datasets, cross-validation  
💡 **Use Single Split for**: Fast training, large datasets, quick tests  

## Troubleshooting

### "Cannot use 'mlp' model with --use-pca"
**Fix**: Use `--model pca_mlp` instead of `--model mlp`

### "k-folds must be at least 2"
**Fix**: Use `--k-folds 3` or higher (or remove flag for single split)

### Poor performance (R² < 0.2)
**Try**: Lower learning rate `--lr 0.001`, more epochs `--epochs 200`, or remove `--use-pca`

---

**For full documentation**, see `UNIFIED_TRAINING_SCRIPT.md`
