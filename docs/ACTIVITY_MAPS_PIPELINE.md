# Activity Maps Preprocessing Pipeline

## Overview

The activity maps preprocessing pipeline is a modular, flexible system for preparing brain activity maps for training. It's **completely separate from molecule preprocessing**, making it ideal for use with both descriptor-based models (MLP/CNN) and graph-based models (GNN).

## Architecture

```
data/01_raw/
├── activity_maps_csv/        # Input: All raw activity maps (405 maps)
└── behavior_data.csv          # Metadata: which maps belong to which CID

                    ↓ [run_activity_maps.py]
                    
src/olfactory_modeling/pipeline/activity_maps.py
├── Selection Strategies (pluggable)
│   ├── select_best_by_quality()      # Composite score method (default)
│   ├── select_by_averaging()         # Average all maps per CID
│   ├── select_by_median()            # Median (robust to outliers)
│   └── select_first_map()            # Simple baseline
│
├── Global Masking
│   ├── compute_and_save_global_mask()  # Create & save reusable mask
│   └── apply_mask()                     # Apply mask to maps
│
└── Pipeline Orchestration
    └── process_activity_maps()          # Main pipeline function

                    ↓
                    
data/02_processed/
├── processed_maps.npz         # Selected & masked maps (287 × 79 × 43)
├── processed_maps_metadata.csv # Metadata: CID, selection_method, etc.
├── global_mask.npy            # Reusable binary mask (79 × 43)
└── global_mask_metadata.json  # Mask parameters and statistics
```

## Key Features

### ✅ Separation of Concerns
- **Molecule preprocessing** (SMILES → features) is separate from **map preprocessing**
- Maps pipeline is independent and works with any model architecture
- Can regenerate maps without touching molecular features

### ✅ Flexibility
- **4 selection strategies** to choose from
- **Configurable coverage threshold** (0.0 to 1.0)
- **Easy experimentation** via CLI arguments

### ✅ Reusability
- Global mask computed once and saved
- Processed maps saved in efficient NPZ format
- No redundant computation during training

### ✅ Clear Data Flow
```
Raw Data (01_raw) → Process Maps (pipeline) → Processed Data (02_processed) → Train Models
```

## Selection Strategies

### 1. Best Quality (Default) - `best_quality`

**How it works:**
- Computes quality metrics for each map: coverage, range, mean activity
- Calculates composite score: `z(coverage) + z(range) - 0.5 * z(mean_active)`
- Selects map with highest score per CID

**When to use:**
- Default choice for most use cases
- Picks the most informative map based on multiple quality criteria
- Good for maximizing signal quality

**Results:**
- 287 molecules (230 single-map, 57 multi-map CIDs)
- Automatically selects best map when multiple maps exist

### 2. Average - `average`

**How it works:**
- Averages all maps for each CID element-wise
- Most conservative approach

**When to use:**
- Want to reduce noise by averaging multiple measurements
- Prefer stability over picking single "best" map
- Have concerns about individual map quality

**Trade-offs:**
- May blur spatial patterns if maps are misaligned
- More conservative but potentially less sharp

### 3. Median - `median`

**How it works:**
- Takes element-wise median across all maps per CID
- More robust to outliers than averaging

**When to use:**
- Suspect outlier maps with anomalous values
- Want robustness to extreme values
- Similar benefits to averaging but less sensitive to outliers

### 4. First - `first`

**How it works:**
- Simply uses the first available map for each CID
- No quality assessment

**When to use:**
- Quick baseline for testing
- Debugging pipeline without complex selection
- All maps are known to be similar quality

## Global Masking

### Coverage Threshold

The coverage threshold determines which brain regions to include based on how many molecules show activity there.

**Formula:**
```
pixel_is_active = (num_maps_with_activity / total_maps) >= threshold
```

**Tested configurations:**

| Threshold | Mask Coverage | Description | Use Case |
|-----------|--------------|-------------|----------|
| **0.3** | 66.76% (2268 pixels) | Lenient - includes more regions | Exploratory analysis, maximize coverage |
| **0.5** | 65.12% (2212 pixels) | **Balanced (default)** | General use, good trade-off |
| **0.7** | 63.56% (2159 pixels) | Strict - only high-coverage regions | Conservative, high-confidence regions only |

**Recommendation:** Start with 0.5 (default) for balanced coverage.

## Usage Examples

### Basic Usage

```bash
# Default: best quality selection, 50% coverage
python scripts/run_activity_maps.py
```

### Experiment with Strategies

```bash
# Use averaging instead of quality-based selection
python scripts/run_activity_maps.py --strategy average

# Use median (robust to outliers)
python scripts/run_activity_maps.py --strategy median

# Simple baseline (first map only)
python scripts/run_activity_maps.py --strategy first
```

### Adjust Coverage Threshold

```bash
# Stricter masking (70% coverage required)
python scripts/run_activity_maps.py --coverage-threshold 0.7

# More lenient masking (30% coverage required)
python scripts/run_activity_maps.py --coverage-threshold 0.3

# Very strict (80% coverage required)
python scripts/run_activity_maps.py --coverage-threshold 0.8
```

### Speed Up Testing

```bash
# Skip visualizations for faster processing
python scripts/run_activity_maps.py --no-visualizations

# Combine with strategy and threshold
python scripts/run_activity_maps.py --strategy median --coverage-threshold 0.6 --no-visualizations
```

### Full Control

```bash
# Specify all parameters
python scripts/run_activity_maps.py \
  --strategy best_quality \
  --coverage-threshold 0.5 \
  --min-region-size 100 \
  --output-dir data/02_processed \
  --verbose
```

## Pipeline Outputs

After running the pipeline, you'll find these files in `data/02_processed/`:

### 1. `processed_maps.npz` (Main output)
- **Format:** NumPy compressed archive
- **Contents:**
  - `maps`: Array of shape (287, 79, 43) - processed activity maps
  - `cids`: Array of shape (287,) - corresponding CIDs
  - Metadata fields (selection_strategy, coverage_threshold, etc.)
- **Size:** ~1.5 MB (efficient storage)
- **Loading:**
  ```python
  data = np.load('data/02_processed/processed_maps.npz')
  maps = data['maps']  # (287, 79, 43)
  cids = data['cids']  # (287,)
  ```

### 2. `processed_maps_metadata.csv`
- Human-readable metadata for each processed map
- Columns: CID, selection_strategy, coverage_threshold, map_shape_h, map_shape_w

### 3. `global_mask.npy`
- **Format:** NumPy binary array
- **Shape:** (79, 43)
- **Type:** boolean
- **Purpose:** Reusable mask showing which brain regions are included
- **Loading:**
  ```python
  mask = np.load('data/02_processed/global_mask.npy')
  ```

### 4. `global_mask_metadata.json`
- Mask creation parameters and statistics
- Example:
  ```json
  {
    "coverage_threshold": 0.5,
    "min_region_size": 100,
    "n_maps": 405,
    "active_pixels": 2212,
    "total_pixels": 3397,
    "coverage_fraction": 0.6512
  }
  ```

### 5. Visualizations (if enabled)
- `global_mask.png` - Visual representation of mask
- `processed_map_example.png` - Example processed map
- `processed_maps_gallery.png` - Gallery of 6 example maps

## Integration with Training

The dataset automatically loads pre-processed maps:

```python
from src.olfactory_modeling.data.activity_map_dataset import get_dataloaders

# Dataloaders automatically use processed maps from data/02_processed/
train_loader, val_loader, test_loader = get_dataloaders(
    processed_dir="data/02_processed",
    batch_size=32
)

# Training works seamlessly
for features, activity_map, metadata in train_loader:
    # features: (batch, 268) - pre-computed molecular features
    # activity_map: (batch, 79, 43) - pre-processed activity maps
    # metadata: dict with 'cid' and 'index'
    pass
```

**Key benefits:**
- No on-the-fly map loading during training
- Faster data loading (pre-processed NPZ format)
- Consistent preprocessing across all runs
- Easy to regenerate with different strategies

## Testing Results

### Pipeline Validation ✅

**Test run (5 epochs, MLP model):**
```
Loaded 287 molecules with 268 features and aligned maps
Train split: 200 samples
Val split: 43 samples
Test split: 44 samples

Epoch 5/5:
  Train - Loss: 0.2603, Corr: 0.515, R²: 0.308
  Val   - Loss: 0.2789, Corr: 0.510, R²: 0.277
```

**Conclusion:** Pipeline works perfectly! Dataset loads processed maps correctly and training runs smoothly.

### Strategy Comparison

All strategies successfully processed 287 molecules:

| Strategy | Maps Processed | Single-Map CIDs | Multi-Map CIDs |
|----------|---------------|-----------------|----------------|
| best_quality | 287 | 230 | 57 |
| average | 287 | N/A | N/A |
| median | 287 | N/A | N/A |
| first | 287 | N/A | N/A |

### Coverage Threshold Comparison

| Threshold | Mask Coverage | Active Pixels |
|-----------|--------------|---------------|
| 0.3 | 66.76% | 2268 / 3397 |
| 0.5 | 65.12% | 2212 / 3397 |
| 0.7 | 63.56% | 2159 / 3397 |

## Workflow Recommendations

### 1. Initial Setup
```bash
# Generate processed maps with default settings
python scripts/run_activity_maps.py
```

### 2. Training
```bash
# Train models using processed maps
python scripts/train_baseline_nn.py --model mlp --epochs 100
python scripts/train_baseline_nn.py --model cnn --epochs 100
```

### 3. Experimentation
```bash
# Try different selection strategies
python scripts/run_activity_maps.py --strategy average
python scripts/train_baseline_nn.py --model mlp --epochs 100 --output-dir experiments/average_maps

python scripts/run_activity_maps.py --strategy median
python scripts/train_baseline_nn.py --model mlp --epochs 100 --output-dir experiments/median_maps

# Compare results
```

### 4. Coverage Analysis
```bash
# Test different coverage thresholds
for threshold in 0.3 0.5 0.7; do
    python scripts/run_activity_maps.py --coverage-threshold $threshold --no-visualizations
    python scripts/train_baseline_nn.py --model mlp --epochs 50 --output-dir experiments/threshold_$threshold
done
```

## Future Extensions

### Potential New Strategies

1. **Weighted Average** - Weight maps by quality score
2. **Consensus** - Only include pixels that agree across maps
3. **Best Per Region** - Select best map separately for each brain region

### Advanced Masking

1. **Region-specific thresholds** - Different thresholds for different brain areas
2. **Anatomical constraints** - Use known brain structure information
3. **Adaptive masking** - Learn optimal mask during training

## Code Organization

```
src/olfactory_modeling/pipeline/activity_maps.py
├── SelectionStrategy (Enum)
├── ActivityMapRecord (dataclass)
│
├── Loading Functions
│   ├── load_directory_csv()
│   └── load_activity_maps()
│
├── Selection Strategies
│   ├── select_best_by_quality()
│   ├── select_by_averaging()
│   ├── select_by_median()
│   ├── select_first_map()
│   └── select_maps_by_strategy()
│
├── Masking Functions
│   ├── compute_global_mask()
│   ├── compute_and_save_global_mask()
│   ├── load_global_mask()
│   └── apply_mask()
│
├── Save/Load Functions
│   ├── save_processed_maps()
│   └── load_processed_maps()
│
├── Visualization Functions
│   ├── visualize_map()
│   ├── visualize_global_mask()
│   ├── visualize_coverage()
│   ├── visualize_coverage_histogram()
│   └── visualize_processing_results()
│
└── Main Pipeline
    └── process_activity_maps()
```

## Summary

The activity maps preprocessing pipeline provides:

✅ **Modularity** - Easy to swap selection strategies and parameters  
✅ **Reproducibility** - Saves all parameters and metadata  
✅ **Efficiency** - Pre-processes once, use many times  
✅ **Flexibility** - Works with any model architecture  
✅ **Clarity** - Separate from molecule preprocessing  
✅ **Testing** - Validated with actual training runs  

**Default configuration works well** for most use cases:
```bash
python scripts/run_activity_maps.py
# Uses: best_quality strategy, 0.5 coverage threshold
# Produces: 287 maps, 65.12% mask coverage
```

For questions or issues, see the pipeline code at:
- `src/olfactory_modeling/pipeline/activity_maps.py`
- `scripts/run_activity_maps.py`
