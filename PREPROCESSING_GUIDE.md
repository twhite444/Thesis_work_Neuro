# Modular Preprocessing Pipeline Guide

## Overview

The preprocessing pipeline has been **unified and made fully configurable**! All feature selection now happens in the correct order (BEFORE standardization) within the `preprocess.py` script.

## Why the Change?

**Problem**: Previously, `VarianceThreshold` was applied AFTER `StandardScaler`, which made it useless:
- StandardScaler forces all features to have variance ≈ 1.0
- Applying variance threshold after standardization does nothing
- We were filtering on meaningless, normalized variances

**Solution**: Unified pipeline that applies steps in the correct order:
1. Featurization (SMILES → Mordred descriptors)
2. Drop NaN columns
3. Drop zero-only columns
4. **Variance threshold (BEFORE standardization!)**
5. Standardization (optional)

## Quick Start

### Default (Recommended)
```bash
# Drop NaN/zeros, remove constants, standardize
python scripts/preprocess.py --use-cached
```

**Output**: 287 samples × 1187 features

### Remove Low-Variance Features
```bash
# Filter out features with variance < 0.01
python scripts/preprocess.py --use-cached --variance-threshold 0.01
```

**Output**: 287 samples × 1023 features (164 removed)

### Save Intermediate Results
```bash
# Save both unscaled and scaled features
python scripts/preprocess.py --use-cached --variance-threshold 0.01 --save-intermediate
```

**Outputs**:
- `data/02_processed/unscaled_features.csv` (before standardization)
- `data/02_processed/cleaned_data.csv` (after standardization)

### Experiment with Different Thresholds
```bash
# Very aggressive filtering (variance < 0.1)
python scripts/preprocess.py --use-cached --variance-threshold 0.1

# Only remove constants (default)
python scripts/preprocess.py --use-cached --variance-threshold 0.0
```

## All Options

```bash
python scripts/preprocess.py \
    --use-cached \                    # Use cached molecules (fast)
    --variance-threshold 0.01 \       # Remove features with var < 0.01
    --save-intermediate \             # Save unscaled features too
    --no-drop-nan \                   # Keep NaN columns (not recommended)
    --no-drop-zero \                  # Keep zero-only columns
    --no-standardize                  # Don't standardize (keep raw values)
```

### Parameter Details

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--variance-threshold` | 0.0 | Minimum variance for features. 0.0 = remove only constants |
| `--no-drop-nan` | False | Keep columns with NaN (default: drop them) |
| `--no-drop-zero` | False | Keep zero-only columns (default: drop them) |
| `--no-standardize` | False | Don't standardize (default: apply StandardScaler) |
| `--save-intermediate` | False | Save unscaled features to separate file |
| `--use-cached` | True | Use cached molecules (fast NPZ loading) |
| `--force-download` | False | Force fresh download from Pyrfume |

## Understanding Variance Threshold

Variance threshold removes features with low variance across samples. Applied on **raw (unscaled)** data:

```python
# Example feature variances BEFORE standardization:
Feature A: variance = 0.005   ← REMOVED (< 0.01)
Feature B: variance = 0.15    ← KEPT
Feature C: variance = 2500    ← KEPT

# After StandardScaler:
Feature B: variance = 1.0
Feature C: variance = 1.0  ← All normalized to 1.0!
```

**Why this matters**:
- Low-variance features provide little information (nearly constant across samples)
- Removing them reduces dimensionality without losing much signal
- Must be done BEFORE standardization (when variance is still meaningful)

## Recommended Workflows

### For Neural Network Training
```bash
# Moderate filtering + standardization (recommended)
python scripts/preprocess.py --use-cached --variance-threshold 0.01
```

### For Exploratory Analysis
```bash
# Save both versions to compare
python scripts/preprocess.py \
    --use-cached \
    --variance-threshold 0.01 \
    --save-intermediate
```

### For Linear Models
```bash
# More aggressive filtering (linear models sensitive to collinearity)
python scripts/preprocess.py --use-cached --variance-threshold 0.1
```

### For Debugging
```bash
# Minimal processing to see raw data
python scripts/preprocess.py \
    --use-cached \
    --variance-threshold 0.0 \
    --no-standardize \
    --save-intermediate
```

## Pipeline Output

The script saves several files to `data/02_processed/`:

1. **`cleaned_data.csv`** - Final processed features (always saved)
   - Used by training scripts
   - Standardized (variance = 1.0 per feature)

2. **`unscaled_features.csv`** - Intermediate unscaled features (if `--save-intermediate`)
   - After variance threshold, before standardization
   - Useful for analysis/debugging

3. **`scaler_stats.json`** - StandardScaler parameters (if standardized)
   - Mean and scale for each feature
   - Needed to transform new data the same way

4. **`preprocess_metadata.json`** - Pipeline configuration
   - All parameters used
   - Useful for reproducibility

## Example Results

### Default (variance_threshold=0.0)
```
Featurizing 287 molecules with Mordred descriptors...
  Generated 1826 descriptors
  Dropped 432 columns with NaN values
  Dropped 207 zero-only columns

Standardizing features...
✓ Saved standardized features to data/02_processed/cleaned_data.csv

Final feature set: 287 samples × 1187 features
```

### With Filtering (variance_threshold=0.01)
```
Featurizing 287 molecules with Mordred descriptors...
  Generated 1826 descriptors
  Dropped 432 columns with NaN values
  Dropped 207 zero-only columns

Applying variance threshold: 0.01
  Variance range: [0.0000, 25132874.6289]
  Removed 164 low-variance features
  Kept 1023 features

Standardizing features...
✓ Saved standardized features to data/02_processed/cleaned_data.csv

Final feature set: 287 samples × 1023 features
```

**Impact**: Reduced from 1187 to 1023 features (14% reduction) by removing low-variance features.

## Migration from Old Workflow

### Old (Broken) Workflow
```bash
# Step 1: Featurize and standardize
python scripts/preprocess.py

# Step 2: Try to filter (DOESN'T WORK - all variance = 1.0!)
python scripts/select_features.py --threshold 0.99
```

### New (Correct) Workflow
```bash
# Single step: filter THEN standardize (correct order!)
python scripts/preprocess.py --use-cached --variance-threshold 0.01
```

## Deprecated Scripts

- ❌ **`scripts/select_features.py`** - No longer needed, use `scripts/preprocess.py` instead
- ❌ **`src/neuro_foundation/pipeline/feature_select.py`** - Integrated into `preprocess.py`

## Advanced Usage

### Custom Feature Engineering Pipeline

You can modify `src/neuro_foundation/pipeline/preprocess.py` to add more preprocessing steps:

```python
# Add custom filtering logic
def featurize_and_standardize(
    molecules: pd.DataFrame,
    variance_threshold: float = 0.0,
    correlation_threshold: float = 0.95,  # NEW: Remove highly correlated
    # ...
):
    # ... existing code ...
    
    # Add correlation-based filtering before standardization
    if correlation_threshold < 1.0:
        correlation_matrix = filtered.corr().abs()
        upper_tri = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        to_drop = [col for col in upper_tri.columns 
                   if any(upper_tri[col] > correlation_threshold)]
        filtered = filtered.drop(columns=to_drop)
    
    # Then standardize...
```

## Summary

✅ **Unified pipeline** - All preprocessing in one place  
✅ **Correct order** - Variance threshold before standardization  
✅ **Fully configurable** - Experiment with all parameters  
✅ **Saves metadata** - Reproducible results  
✅ **Backward compatible** - Default behavior unchanged  

**Use `python scripts/preprocess.py --help` to see all options!**
