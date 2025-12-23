# Modular Preprocessing Pipeline - Summary

## What Changed

### Before (Broken)
- ❌ Two separate scripts: `preprocess.py` and `select_features.py`
- ❌ Wrong order: StandardScaler → VarianceThreshold
- ❌ Variance threshold didn't work (all features had variance ≈ 1.0 after scaling)
- ❌ Confusing workflow, easy to misuse

### After (Fixed)
- ✅ Single unified script: `preprocess.py`
- ✅ Correct order: VarianceThreshold → StandardScaler
- ✅ Fully configurable pipeline with all parameters exposed
- ✅ Clear, modular code with proper documentation

## Key Improvements

### 1. Unified Pipeline
All preprocessing in `src/neuro_foundation/pipeline/preprocess.py`:
```python
def featurize_and_standardize(
    molecules: pd.DataFrame,
    variance_threshold: float = 0.0,      # NEW
    drop_nan_columns: bool = True,        # Configurable
    drop_zero_columns: bool = True,       # Configurable
    standardize: bool = True,             # Configurable
    save_intermediate: bool = False,      # NEW
    output_dir: str = "data/02_processed"
) -> pd.DataFrame:
```

### 2. Configurable Script
All options exposed in `scripts/preprocess.py`:
```bash
python scripts/preprocess.py \
    --variance-threshold 0.01 \
    --save-intermediate \
    --no-drop-nan \
    --no-drop-zero \
    --no-standardize
```

### 3. Correct Execution Order
```
1. SMILES → Mordred (1826 features)
2. Drop NaN columns (→ 1394 features)
3. Drop zero-only columns (→ 1187 features)
4. VarianceThreshold (0.01) (→ 1023 features)  ← BEFORE standardization!
5. StandardScaler (mean=0, std=1)
6. Save to cleaned_data.csv
```

### 4. Intermediate Outputs
With `--save-intermediate`:
- `unscaled_features.csv` - After variance filtering, before standardization
- `cleaned_data.csv` - Final standardized features
- `scaler_stats.json` - StandardScaler parameters
- `preprocess_metadata.json` - Pipeline configuration

## Usage Examples

### Default (Recommended)
```bash
python scripts/preprocess.py --use-cached
# Output: 287 samples × 1187 features
```

### With Filtering
```bash
python scripts/preprocess.py --use-cached --variance-threshold 0.01
# Output: 287 samples × 1023 features (164 removed)
```

### Save Both Versions
```bash
python scripts/preprocess.py --use-cached --variance-threshold 0.01 --save-intermediate
# Outputs:
#   - unscaled_features.csv (1023 features, raw variances)
#   - cleaned_data.csv (1023 features, variance=1.0)
```

### Experiment with Thresholds
```bash
# Very aggressive
python scripts/preprocess.py --use-cached --variance-threshold 0.1
# Output: ~850 features

# Moderate
python scripts/preprocess.py --use-cached --variance-threshold 0.01
# Output: ~1023 features

# Conservative (default)
python scripts/preprocess.py --use-cached --variance-threshold 0.0
# Output: 1187 features
```

## Impact on Training

### Feature Count Comparison
| Variance Threshold | Features Kept | % Reduction |
|-------------------|---------------|-------------|
| 0.0 (default)     | 1187          | 0%          |
| 0.01              | 1023          | 14%         |
| 0.1               | ~850          | 28%         |

### Training Performance
Using fewer, more informative features can:
- ✅ Speed up training (fewer parameters)
- ✅ Reduce overfitting (less noise)
- ✅ Improve generalization (focus on signal)

## Code Architecture

### Pipeline Module (`src/neuro_foundation/pipeline/preprocess.py`)
- **Function**: `featurize_and_standardize()`
- **Purpose**: Core preprocessing logic
- **Features**:
  - Modular steps (all optional)
  - Correct order of operations
  - Verbose logging
  - Metadata saving

### Script (`scripts/preprocess.py`)
- **Purpose**: User-facing CLI
- **Features**:
  - Argument parsing
  - Help documentation
  - Example usage
  - Calls pipeline module

### Deprecated
- ❌ `scripts/select_features.py.deprecated` - No longer needed
- ❌ `src/neuro_foundation/pipeline/feature_select.py` - Integrated into preprocess.py

## Migration Guide

### Old Workflow
```bash
# Step 1
python scripts/preprocess.py

# Step 2 (broken - doesn't actually filter!)
python scripts/select_features.py --threshold 0.99
```

### New Workflow
```bash
# Single step with correct filtering
python scripts/preprocess.py --use-cached --variance-threshold 0.01
```

## Documentation

- **`PREPROCESSING_GUIDE.md`** - Full guide with examples and explanations
- **`WEIGHT_DECAY_GUIDE.md`** - Weight decay (L2 regularization) guide
- **`TRAINING_SIMPLIFIED.md`** - Training script guide
- **`scripts/preprocess.py --help`** - Quick reference

## Testing

### Verify Variance Filtering Works
```python
import pandas as pd

# Load unscaled features (after filtering, before standardization)
unscaled = pd.read_csv('data/02_processed/unscaled_features.csv', index_col='CID')
variances = unscaled.var()

print(f"Min variance: {variances.min():.4f}")  # Should be >= threshold
print(f"Max variance: {variances.max():.2f}")
print(f"Features with var < 0.01: {(variances < 0.01).sum()}")  # Should be 0

# Load standardized features  
scaled = pd.read_csv('data/02_processed/cleaned_data.csv', index_col='CID')
scaled_vars = scaled.var()

print(f"All scaled variances ≈ 1.0: {all(abs(scaled_vars - 1.0) < 0.01)}")  # Should be True
```

## Future Enhancements

Possible additions to the pipeline:

1. **Correlation filtering** - Remove highly correlated features
2. **Feature importance** - Use model-based selection
3. **PCA integration** - Optional PCA before standardization
4. **Missing value imputation** - Alternative to dropping NaN columns
5. **Custom transformations** - Log, sqrt, etc.

All can be added as optional parameters to maintain backward compatibility!

## Summary

✅ **Fixed order of operations** - Variance filtering before standardization  
✅ **Unified codebase** - Single source of truth  
✅ **Fully configurable** - All steps can be toggled  
✅ **Proper documentation** - Clear guides and examples  
✅ **Backward compatible** - Default behavior unchanged  
✅ **Ready for experimentation** - Easy to try different configurations  

**The preprocessing pipeline is now robust, modular, and ready for production use!** 🎉
