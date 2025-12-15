# Quick Start Guide - Activity Maps Pipeline

## TL;DR

```bash
# 1. Process activity maps (run once)
python scripts/run_activity_maps.py

# 2. Train models (uses processed maps)
python scripts/train_baseline_nn.py --model mlp --epochs 100
python scripts/train_baseline_nn.py --model cnn --epochs 100
```

## Complete Workflow

### Step 1: Load Raw Data (one-time setup)
```bash
python scripts/load_all_data.py
# Downloads and saves activity maps to data/01_raw/activity_maps_csv/
```

### Step 2: Preprocess Molecules
```bash
# Generate molecular features
python scripts/preprocess.py

# Select informative features
python scripts/select_features.py
# Outputs: data/02_processed/selected_features.csv (287 × 268)
```

### Step 3: Process Activity Maps ⭐ NEW
```bash
# Generate processed activity maps
python scripts/run_activity_maps.py
# Outputs: data/02_processed/processed_maps.npz (287 × 79 × 43)
```

### Step 4: Train Models
```bash
# MLP baseline
python scripts/train_baseline_nn.py --model mlp --epochs 100

# CNN decoder
python scripts/train_baseline_nn.py --model cnn --epochs 100
```

## Common Use Cases

### Default (Recommended)
```bash
python scripts/run_activity_maps.py
# Uses best_quality selection, 0.5 coverage threshold
```

### Try Different Selection Strategies
```bash
# Quality-based (default)
python scripts/run_activity_maps.py --strategy best_quality

# Average all maps
python scripts/run_activity_maps.py --strategy average

# Median (robust)
python scripts/run_activity_maps.py --strategy median
```

### Adjust Coverage
```bash
# More permissive (includes more brain regions)
python scripts/run_activity_maps.py --coverage-threshold 0.3

# More restrictive (only high-confidence regions)
python scripts/run_activity_maps.py --coverage-threshold 0.7
```

### Speed Tips
```bash
# Skip visualizations for faster processing
python scripts/run_activity_maps.py --no-visualizations
```

## What Gets Generated?

After running `run_activity_maps.py`:

```
data/02_processed/
├── processed_maps.npz           ← Main output: 287 processed maps
├── processed_maps_metadata.csv  ← Human-readable metadata
├── global_mask.npy              ← Reusable mask (79 × 43)
├── global_mask_metadata.json    ← Mask parameters
└── *.png                        ← Visualizations (if enabled)
```

## Troubleshooting

### Error: "Processed maps not found"
**Solution:** Run `python scripts/run_activity_maps.py` first

### Error: "Selected features not found"
**Solution:** Run preprocessing first:
```bash
python scripts/preprocess.py
python scripts/select_features.py
```

### Want to regenerate with different settings?
**Just run again:** The pipeline overwrites old outputs
```bash
python scripts/run_activity_maps.py --strategy average
```

## Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--strategy` | `best_quality` | How to select maps: `best_quality`, `average`, `median`, `first` |
| `--coverage-threshold` | `0.5` | Fraction of maps required (0.0-1.0). Lower = more permissive |
| `--min-region-size` | `100` | Minimum brain region size in pixels |
| `--no-visualizations` | (flag) | Skip generating PNG visualizations for speed |
| `--verbose` | (flag) | Print detailed processing information |

## Full Help
```bash
python scripts/run_activity_maps.py --help
```

## Next Steps

See full documentation: `docs/ACTIVITY_MAPS_PIPELINE.md`
