# Training Script Unification - Complete! ✅

## Summary

Successfully merged K-fold cross-validation into the main `train_baseline_nn.py` script, creating a unified, clean, and simple-to-use training interface.

## What Changed

### Before (2 separate scripts):
- `scripts/train_baseline_nn.py` - Single train/val/test split
- `scripts/train_baseline_nn_kfold.py` - K-fold cross-validation

### After (1 unified script):
- `scripts/train_baseline_nn.py` - Does both! Toggle with `--k-folds`

## New Unified Interface

### Single Split Mode (Default)
```bash
# Train on raw activity maps
python scripts/train_baseline_nn.py --model mlp --epochs 100

# Train on PCA components (faster)
python scripts/train_baseline_nn.py --model pca_mlp --use-pca --epochs 100
```

### K-Fold Cross-Validation Mode
```bash
# 5-fold CV on raw maps
python scripts/train_baseline_nn.py --model mlp --k-folds 5 --epochs 100

# 5-fold CV on PCA components
python scripts/train_baseline_nn.py --model pca_mlp --use-pca --k-folds 5 --epochs 100

# 3-fold CV (faster evaluation)
python scripts/train_baseline_nn.py --model cnn --k-folds 3 --epochs 50
```

## Features

### ✅ Clean and Simple
- **One script to rule them all** - No need to remember which script to use
- **Clear mode indicators** - Visual banners show which mode is active
- **Consistent interface** - Same arguments for both modes
- **Better error messages** - Validates PCA/model compatibility

### ✅ Flexible
- Toggle between single-split and K-fold with one flag
- All hyperparameters work in both modes
- PCA support in both modes
- All model types supported (mlp, cnn, pca_mlp)

### ✅ Robust
- Enforces pca_mlp for PCA targets (prevents shape mismatches)
- Validates K-fold argument (must be ≥2)
- Automatic use_pca enabling for pca_mlp model
- Comprehensive dataset info display

## Usage Examples

### Quick Prototyping (PCA + Single Split)
```bash
python scripts/train_baseline_nn.py --model pca_mlp --use-pca --epochs 20 --quiet
```

### Robust Evaluation (PCA + K-Fold)
```bash
python scripts/train_baseline_nn.py --model pca_mlp --use-pca --k-folds 5 --epochs 50
```

### Production Training (Raw Maps + Single Split)
```bash
python scripts/train_baseline_nn.py --model mlp --epochs 200 --early-stopping 20
```

### Paper-Style Evaluation (Raw Maps + K-Fold)
```bash
python scripts/train_baseline_nn.py --model mlp --k-folds 5 --epochs 100
```

## Output Display

### Single Split Mode
```
============================================================
SINGLE SPLIT MODE (train/val/test)
============================================================

Dataset info:
  Train samples: 200
  Val samples: 43
  Test samples: 44
  Feature dimension: 268
  Output dimension: 50
    (Using 50 PCA components)

Creating PCA_MLP model (dropout=0.35)...
  Output: 50 PCA components

...

============================================================
TRAINING COMPLETE
============================================================
Best validation loss: 30.0279
Best validation correlation: 0.345
Best validation R²: 0.103
```

### K-Fold Mode
```
============================================================
K-FOLD CROSS-VALIDATION MODE (3 folds)
============================================================

Dataset info:
  Total samples: 287
  Feature dimension: 268
  Output dimension: 50
    (Using 50 PCA components)
  Will be split into 3 folds

...

============================================================
K-FOLD CROSS-VALIDATION COMPLETE
============================================================
Results saved to: experiments/baseline_nn
  - cv_results.json (detailed results)
  - fold_N/ (individual fold checkpoints)
```

## Migration Guide

### Old Command → New Command

**K-fold CV:**
```bash
# OLD
python scripts/train_baseline_nn_kfold.py --model pca_mlp --use-pca --n-folds 5

# NEW
python scripts/train_baseline_nn.py --model pca_mlp --use-pca --k-folds 5
```

**Single Split (unchanged):**
```bash
# SAME
python scripts/train_baseline_nn.py --model pca_mlp --use-pca --epochs 100
```

### Argument Changes
- `--n-folds` → `--k-folds` (more conventional naming)
- Added `--random-seed` for reproducibility
- Everything else is identical

## Files Changed

### Modified
- ✅ `scripts/train_baseline_nn.py` - Unified training script
  - Added K-fold support
  - Enhanced output formatting
  - Better validation and error messages
  - Comprehensive dataset info display

### Deprecated
- ⚠️ `scripts/train_baseline_nn_kfold.py` → `scripts/train_baseline_nn_kfold.py.deprecated`
  - Old K-fold script renamed
  - Kept for reference but not recommended

## Testing Results

### Single Split Mode ✅
```bash
$ python scripts/train_baseline_nn.py --model pca_mlp --use-pca --epochs 2 --quiet
# Works perfectly - R² = 0.103 (2 epochs, expected)
```

### K-Fold Mode ✅
```bash
$ python scripts/train_baseline_nn.py --model pca_mlp --use-pca --k-folds 3 --epochs 2 --quiet
# Works perfectly - Creates 3 folds, trains each
```

### Error Handling ✅
```bash
$ python scripts/train_baseline_nn.py --model mlp --use-pca
# ERROR: Cannot use 'mlp' model with --use-pca flag!
# (Clear, helpful error message)
```

## Benefits

### For Users
1. **Simpler mental model** - One script, one interface
2. **Easier experimentation** - Quick toggle between modes
3. **Less confusion** - No need to choose between scripts
4. **Better discoverability** - All options in one `--help`

### For Developers
1. **Less code duplication** - DRY principle
2. **Single point of maintenance** - Fix once, works everywhere
3. **Consistent behavior** - Same validation logic
4. **Easier testing** - Test one script thoroughly

## Next Steps

### Recommended Usage
1. **Rapid prototyping**: Use `--model pca_mlp --use-pca` (fast)
2. **Hyperparameter search**: Use `--model pca_mlp --use-pca` (fast iterations)
3. **Robust evaluation**: Add `--k-folds 5` (more reliable metrics)
4. **Final training**: Use `--model mlp` without PCA (best accuracy)
5. **Publication**: Use `--model mlp --k-folds 5` (robust + accurate)

### Future Enhancements
- Could add `--stratified` flag for stratified K-fold
- Could add `--nested-cv` for nested cross-validation
- Could add `--ensemble` to create ensemble from K-fold models
- Could add automatic hyperparameter optimization

---

**Date**: December 18, 2025  
**Status**: Complete and tested ✅  
**Backwards Compatible**: Yes (except `--n-folds` → `--k-folds`)  
**Clean and Simple**: Yes! ✨
