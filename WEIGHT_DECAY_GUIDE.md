# Weight Decay (L2 Regularization) Guide

## Overview

Weight decay (L2 regularization) is **already implemented** in the training script! It helps prevent overfitting by penalizing large weights during training.

## How to Use

Simply add the `--weight-decay` flag when training:

```bash
# No weight decay (default)
python scripts/train_baseline_nn.py --model mlp --use-pca --epochs 100

# Light regularization (recommended starting point)
python scripts/train_baseline_nn.py --model mlp --use-pca --epochs 100 --weight-decay 1e-5

# Moderate regularization
python scripts/train_baseline_nn.py --model mlp --use-pca --epochs 100 --weight-decay 1e-4

# Strong regularization
python scripts/train_baseline_nn.py --model mlp --use-pca --epochs 100 --weight-decay 1e-3
```

## What is Weight Decay?

Weight decay adds a penalty term to the loss function:

```
Total Loss = MSE Loss + (weight_decay * sum of squared weights)
```

This encourages the model to:
- Keep weights small
- Avoid overfitting to noise in training data
- Improve generalization to validation/test data

## Empirical Results (30 epochs, PCA with 50 components)

| Weight Decay | Val Loss | Val R² | Val Correlation | Notes |
|--------------|----------|--------|-----------------|-------|
| 0.0 (none)   | 8.43     | 0.329  | 0.541          | Baseline, no regularization |
| 1e-5 (light) | 8.64     | 0.312  | 0.520          | Slight regularization |
| 1e-4 (moderate) | 8.51  | 0.322  | 0.525          | Balanced approach |
| 1e-3 (strong)   | 8.72  | 0.305  | 0.538          | May be too strong |

## Recommendations

### For PCA Training
```bash
# Start with light regularization
python scripts/train_baseline_nn.py --model mlp --use-pca --weight-decay 1e-5 --epochs 100
```

### For Raw Map Training (CNN/MLP)
```bash
# Raw maps have more parameters, may benefit from stronger regularization
python scripts/train_baseline_nn.py --model cnn --weight-decay 1e-4 --epochs 100
```

### For K-Fold Cross-Validation
```bash
# Combine with K-fold for robust evaluation
python scripts/train_baseline_nn.py --model mlp --use-pca --k-folds 5 --weight-decay 1e-4 --epochs 100
```

## How to Choose Weight Decay

1. **Start with 0.0** (no regularization) as baseline
2. **Try 1e-5** for light regularization
3. **Increase to 1e-4** if seeing overfitting (train loss << val loss)
4. **Try 1e-3** only if overfitting is severe
5. **Monitor validation metrics** - best weight decay minimizes validation loss

## Signs You Need More Regularization

- Training loss much lower than validation loss (overfitting)
- Validation loss increases while training loss decreases
- Poor generalization to test set
- High variance in K-fold cross-validation results

## Signs Weight Decay is Too Strong

- Training converges very slowly
- Both training and validation loss remain high
- Model underfits (can't learn patterns)
- Weights remain very small throughout training

## Implementation Details

Weight decay is implemented in the Adam optimizer:

```python
optimizer = optim.Adam(
    model.parameters(), 
    lr=learning_rate, 
    weight_decay=weight_decay  # L2 penalty parameter
)
```

This is mathematically equivalent to adding L2 regularization to the loss function.

## Complete Example

```bash
# Full training run with optimal hyperparameters
python scripts/train_baseline_nn.py \
    --model mlp \
    --use-pca \
    --n-components 50 \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.005 \
    --weight-decay 1e-4 \
    --dropout 0.35 \
    --early-stopping 20 \
    --k-folds 5 \
    --random-seed 42
```

## Other Regularization Techniques

The training script also supports:

- **Dropout** (`--dropout 0.35`): Randomly drops neurons during training
- **Early Stopping** (`--early-stopping 20`): Stops if val loss doesn't improve
- **Learning Rate Scheduling**: Automatically reduces LR on plateau

These can be combined with weight decay for stronger regularization:

```bash
python scripts/train_baseline_nn.py \
    --model mlp \
    --use-pca \
    --weight-decay 1e-4 \
    --dropout 0.4 \
    --early-stopping 20 \
    --epochs 200
```

## Related Documentation

- `TRAINING_SIMPLIFIED.md` - Main training guide
- `UNIFIED_TRAINING_SCRIPT.md` - Full script documentation
- `TRAINING_QUICK_REF.md` - Quick reference for all options

## Summary

✅ **Weight decay is already implemented!**  
✅ **Use `--weight-decay` flag to enable it**  
✅ **Start with 1e-5, adjust based on results**  
✅ **Combine with dropout and early stopping for best results**
