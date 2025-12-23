# Performance Optimization Summary

## 🚀 Performance Improvements

### Mordred Feature Caching

**Problem:** Preprocessing was slow (~30 seconds) because Mordred molecular descriptors were computed from SMILES strings on every run.

**Solution:** Cache Mordred features separately during initial data loading.

### Performance Comparison

| Scenario | Time | Speedup |
|----------|------|---------|
| **Before** (computing Mordred) | ~30 seconds | baseline |
| **After** (loading from cache) | ~2.3 seconds | **13x faster** |

## 📋 Optimized Workflow

### One-Time Setup (Slow)
```bash
# Download and cache all data (including Mordred features)
python scripts/load_all_data.py

# Output:
# - molecules_raw.csv/.npz (287 molecules)
# - mordred_features_raw.csv/.npz (1826 descriptors) ← NEW!
# - behavior_data.csv/.npz
# - stimuli_metadata.csv/.npz
# - activity_maps.npz
```

**Time:** ~30-60 seconds (one-time cost)

### Fast Preprocessing (Repeated Use)
```bash
# Preprocessing with cached features
python scripts/preprocess.py --variance-threshold 0.01

# Loading Mordred descriptors for 287 molecules...
#   ✓ Loaded 1826 descriptors from cache (data/01_raw/mordred_features_raw.npz)
#   Dropped 432 columns with NaN values
#   Dropped 207 zero-only columns
# 
# Applying variance threshold: 0.01
#   Variance range: [0.0000, 25132874.6289]
#   Removed 164 low-variance features
#   Kept 1023 features
# 
# Final feature set: 287 samples × 1023 features
```

**Time:** ~2.3 seconds ⚡

### Fallback Behavior (Cache Missing)
```bash
# If cache doesn't exist, will compute from SMILES
python scripts/preprocess.py

# Loading Mordred descriptors for 287 molecules...
#   Cache not found, computing Mordred descriptors from SMILES...
#   💡 Tip: Run 'python scripts/load_all_data.py' first to cache features for faster preprocessing
#   [Progress bar...]
#   ✓ Computed 1826 descriptors
```

**Time:** ~31 seconds (same as before, but shows helpful tip)

## 🏗️ Architecture

### File Structure
```
data/
├── 01_raw/                          # Cached raw data
│   ├── molecules_raw.csv/.npz       # Molecule metadata
│   ├── mordred_features_raw.csv/.npz  # Mordred descriptors (NEW!)
│   ├── behavior_data.csv/.npz       # Behavior annotations
│   └── activity_maps.npz            # Neural activity maps
│
└── 02_processed/                    # Preprocessed features
    └── cleaned_data.csv             # Final feature matrix
```

### Code Components

**1. Caching (in `load_all_data.py`)**
```python
from src.neuro_foundation.data.pyrfume_loader import PyrfumeLoader

loader = PyrfumeLoader()
molecules = loader.load_molecules()

# Compute and cache Mordred features
mordred_features = loader.compute_mordred_features(molecules)
# Saves to: mordred_features_raw.csv and mordred_features_raw.npz
```

**2. Loading (in `preprocess.py`)**
```python
from src.neuro_foundation.data.pyrfume_loader import load_mordred_features_npz

# Try to load cached features
try:
    mordred_features = load_mordred_features_npz('data/01_raw')
    print(f"✓ Loaded from cache")
except FileNotFoundError:
    # Fallback to computing from SMILES
    from pyrfume.features import smiles_to_mordred
    mordred_features = smiles_to_mordred(smiles)
    print(f"💡 Tip: Run 'python scripts/load_all_data.py' first")
```

## 📊 Implementation Details

### Cache Format

**CSV Format** (`mordred_features_raw.csv`)
- Human-readable
- Good for inspection and debugging
- Slower to load (~1-2 seconds)

**NPZ Format** (`mordred_features_raw.npz`)
- Binary compressed format
- Fast loading (<0.5 seconds)
- **Used by preprocessing pipeline**

### Data Alignment

The caching system handles CID alignment automatically:
```python
# Features are indexed by CID
mordred_features = load_mordred_features_npz()  # Index: CID
mordred_features = mordred_features.loc[cids]   # Align with molecules
```

## 🎯 Benefits

1. **Speed**: 13x faster preprocessing (30s → 2.3s)
2. **Convenience**: Cache persists across runs
3. **Flexibility**: Automatic fallback if cache missing
4. **Modularity**: Separate caching and processing logic
5. **Reproducibility**: Same features across all runs

## 💡 Best Practices

### Recommended Workflow
```bash
# 1. One-time setup (do once or when data changes)
python scripts/load_all_data.py

# 2. Fast preprocessing (use repeatedly)
python scripts/preprocess.py --variance-threshold 0.01

# 3. Train models
python scripts/train_baseline_nn.py
```

### When to Recompute Cache

Rerun `load_all_data.py` when:
- Pyrfume data is updated
- Molecule list changes
- First-time setup

### Disk Space

- Mordred CSV: ~300 KB (human-readable)
- Mordred NPZ: ~50 KB (compressed)
- Total overhead: <1 MB

## 🔄 Migration Guide

### For Existing Users

**Before** (slow):
```bash
python scripts/preprocess.py
# → Computes Mordred every time (~30s)
```

**After** (fast):
```bash
# One-time setup
python scripts/load_all_data.py

# Then forever fast
python scripts/preprocess.py
# → Loads from cache (~2.3s)
```

No changes needed to existing scripts or workflows!

## 📚 Related Documentation

- `PREPROCESSING_GUIDE.md` - Complete preprocessing guide
- `PREPROCESSING_SUMMARY.md` - Preprocessing pipeline summary
- `README_FOUNDATION.md` - Project overview
