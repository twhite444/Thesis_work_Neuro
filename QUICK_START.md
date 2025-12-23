# Quick Start Guide - Optimized Workflow

## 🚀 Fast Track (Recommended)

### First Time Setup (One-Time)
```bash
# Download and cache all data (~30-60 seconds)
python scripts/load_all_data.py
```

**Output:**
- ✅ molecules_raw.csv/.npz (287 molecules)
- ✅ **mordred_features_raw.csv/.npz** (1826 descriptors) ← Speeds up preprocessing!
- ✅ behavior_data.csv/.npz (405 entries)
- ✅ stimuli_metadata.csv/.npz (432 entries)
- ✅ activity_maps.npz (405 maps)

### Every Time After (Fast!)
```bash
# Preprocessing (~2.3 seconds)
python scripts/preprocess.py --variance-threshold 0.01

# Training
python scripts/train_baseline_nn.py
```

## ⚡ Performance

| Task | Time | Notes |
|------|------|-------|
| `load_all_data.py` | 30-60s | One-time setup |
| `preprocess.py` (cached) | **2.3s** | ✅ 13x faster! |
| `preprocess.py` (uncached) | 31s | Falls back to computation |

## 🎯 Common Tasks

### Default Preprocessing (Recommended)
```bash
python scripts/preprocess.py
```
- Loads Mordred from cache (fast!)
- Drops NaN columns
- Drops zero-only columns
- Removes constant features (variance_threshold=0.0)
- Standardizes features
- Output: 1187 features

### Aggressive Feature Reduction
```bash
python scripts/preprocess.py --variance-threshold 0.01
```
- Same as default, but removes low-variance features
- Output: 1023 features (14% reduction)

### Minimal Processing (Raw Features)
```bash
python scripts/preprocess.py --no-drop-zero --no-standardize --variance-threshold 0
```
- Only drops NaN columns
- No standardization
- Useful for exploratory analysis

### Save Intermediate Data
```bash
python scripts/preprocess.py --save-intermediate
```
- Saves both unscaled and scaled features
- Useful for comparing preprocessing effects

## 🔧 Training Examples

### Default Training (Includes Weight Decay)
```bash
python scripts/train_baseline_nn.py
```
- Uses weight_decay=1e-5 (light L2 regularization)
- Prevents overfitting
- Good default for most cases

### Custom Training
```bash
# Stronger regularization
python scripts/train_baseline_nn.py --weight-decay 1e-4

# No regularization
python scripts/train_baseline_nn.py --weight-decay 0

# Larger network
python scripts/train_baseline_nn.py --hidden-dim 256 --weight-decay 1e-5

# More epochs
python scripts/train_baseline_nn.py --epochs 200 --weight-decay 1e-5
```

## 📊 Verify Your Setup

```bash
# Check cache exists
ls -lh data/01_raw/*.npz

# Should see:
# - molecules_raw.npz
# - mordred_features_raw.npz  ← Important!
# - behavior_data.npz
# - stimuli_metadata.npz
# - activity_maps.npz
```

## 🆘 Troubleshooting

### Preprocessing is Slow (>10 seconds)
**Problem:** Mordred features not cached

**Solution:**
```bash
python scripts/load_all_data.py
```

Look for this message:
```
✓ Computed 1826 descriptors for 287 molecules
✓ Saved to data/01_raw/mordred_features_raw.npz
```

### "FileNotFoundError: mordred_features_raw.npz not found"
**Problem:** Cache doesn't exist

**Solution:**
```bash
python scripts/load_all_data.py
```

This will create the cache. Preprocessing will be fast afterward.

### Want to Force Fresh Download
```bash
# Redownload everything (will also recompute Mordred)
python scripts/load_all_data.py

# Or just for preprocessing
python scripts/preprocess.py --force-download
```

## 📚 Full Documentation

- **COMPLETE_IMPLEMENTATION_SUMMARY.md** - Comprehensive overview
- **PREPROCESSING_GUIDE.md** - Detailed preprocessing guide
- **PERFORMANCE_OPTIMIZATION_SUMMARY.md** - Performance details
- **WEIGHT_DECAY_GUIDE.md** - L2 regularization guide

## ✨ Key Features

### Preprocessing Pipeline
- ✅ **13x faster** with Mordred caching
- ✅ Correct feature selection (variance threshold before standardization)
- ✅ Fully configurable (all steps optional)
- ✅ Reproducible (saves metadata)
- ✅ Cache by default (opt-out with `--force-download`)

### Training
- ✅ Weight decay (L2 regularization) enabled by default
- ✅ Prevents overfitting
- ✅ Configurable hyperparameters
- ✅ Comprehensive logging

## 🎯 Typical Workflow

```bash
# 1. One-time setup (first time only)
python scripts/load_all_data.py

# 2. Preprocess data (fast!)
python scripts/preprocess.py --variance-threshold 0.01

# 3. Train model
python scripts/train_baseline_nn.py --epochs 100 --weight-decay 1e-5

# 4. Iterate on model
# - Try different architectures
# - Adjust hyperparameters
# - No need to re-run preprocessing (data is cached!)
python scripts/train_baseline_nn.py --hidden-dim 256 --weight-decay 1e-4
```

## 💡 Pro Tips

1. **Always run `load_all_data.py` first** - One-time setup makes everything else fast
2. **Use variance threshold wisely** - Start with 0.01, increase if you need fewer features
3. **Weight decay helps** - Default (1e-5) is a good starting point
4. **Save intermediate data** - Use `--save-intermediate` to inspect preprocessing effects
5. **Check metadata** - Look at `data/02_processed/preprocess_metadata.json` to verify settings

## 🔄 Updates & Changes

### What's New?
- ✅ Mordred feature caching (13x faster preprocessing)
- ✅ Fixed variance threshold bug (was applied after standardization)
- ✅ Unified preprocessing pipeline (all steps in one place)
- ✅ Cache by default (better performance)
- ✅ Weight decay enabled by default (better generalization)

### Migration from Old Workflow
No changes needed! Everything is backward compatible.

**Old workflow still works:**
```bash
python scripts/preprocess.py  # Now fast with cache!
python scripts/train_baseline_nn.py  # Now includes weight decay!
```

**But you'll get better performance with:**
```bash
python scripts/load_all_data.py  # One-time setup
python scripts/preprocess.py  # Then super fast!
```
