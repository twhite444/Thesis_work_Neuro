# Scripts Directory

This directory contains all executable scripts for the neuroimaging pipeline.

## Directory Structure

```
scripts/
├── examples/          # Example scripts showing how to use the data loaders
├── exploration/       # Interactive tools for data inspection and visualization
├── load_all_data.py   # Download and cache all data from Pyrfume
├── preprocess.py      # Feature extraction with Mordred descriptors
├── run_activity_maps.py # Activity maps masking pipeline
├── select_features.py # Variance-based feature selection
└── train_linear.py    # Linear model training
```

## Main Pipeline Scripts

### Data Loading

**`load_all_data.py`** - Download and cache all data from Pyrfume
```bash
# Download all data (molecules, behavior, stimuli, activity maps)
python scripts/load_all_data.py

# Skip activity maps download (405 CSVs)
python scripts/load_all_data.py --skip-activity-maps
```

**Output**: 
- `data/01_raw/molecules_raw.csv` and `.npz` (287 unique molecules)
- `data/01_raw/behavior_data.csv` and `.npz` (405 entries)
- `data/01_raw/stimuli_metadata.csv` and `.npz` (432 stimuli)
- `data/01_raw/activity_maps.npz` and `activity_maps_csv/*.csv` (405 maps)

### Processing Pipeline

**`preprocess.py`** - Feature extraction with Mordred molecular descriptors
```bash
python scripts/preprocess.py \
  --input-csv data/01_raw/molecules_raw.csv \
  --output-dir data/02_processed
```

**Output**: `data/02_processed/cleaned_data.csv` (287 molecules × 1187 features)

---

**`run_activity_maps.py`** - Activity maps masking and averaging pipeline
```bash
# Run with default coverage threshold (0.5)
python scripts/run_activity_maps.py

# Custom coverage threshold
python scripts/run_activity_maps.py --coverage-threshold 0.3 --verbose
```

**Output**:
- `data/02_processed/global_mask.png` - Refined brain mask
- `data/02_processed/coverage_counts.png` - Pixel coverage heatmap
- `data/02_processed/coverage_histogram.png` - Coverage distribution
- `data/02_processed/masked_averaged_example.png` - Example masked map
- `data/02_processed/masked_averaged_gallery.png` - Gallery of 6 maps

---

**`select_features.py`** - Variance-based feature selection
```bash
python scripts/select_features.py \
  --input-csv data/02_processed/cleaned_data.csv \
  --threshold 1.0 \
  --output-dir data/02_processed
```

**Output**: `data/02_processed/selected_features.csv` (filtered features)

---

**`train_linear.py`** - Train linear regression model
```bash
python scripts/train_linear.py \
  --input-csv data/02_processed/selected_features.csv \
  --output-dir data/02_processed
```

**Output**: Model coefficients and predictions

## Example Scripts

See **`examples/`** folder for:
- Data loading examples (NPZ vs CSV performance)
- CID-based activity map loading
- Stimuli metadata usage
- Batch processing examples

## Exploration Tools

See **`exploration/`** folder for:
- Interactive activity map inspection (`inspect_activity_map.py`)
- Visualization tools
- Data quality checks

## Complete Workflow

```bash
# 1. Download all data
python scripts/load_all_data.py

# 2. Extract molecular features
python scripts/preprocess.py

# 3. Process activity maps
python scripts/run_activity_maps.py

# 4. Select features
python scripts/select_features.py

# 5. Train model
python scripts/train_linear.py

# Optional: Explore the data
python scripts/exploration/inspect_activity_map.py --cid 180 --show-images
python scripts/examples/example_load_by_cid.py
```

## Requirements

All scripts require the Python environment specified in `requirements.txt`:
```bash
pip install -r requirements.txt
```

Key dependencies:
- `pyrfume` - Data loading
- `mordred` - Molecular descriptor calculation
- `pandas`, `numpy` - Data processing
- `scikit-learn` - ML and scaling
- `matplotlib` - Visualization
- `scipy` - Image processing

## See Also

- **Source Code**: `src/neuro_foundation/` - Reusable modules
- **Tests**: `tests/` - Unit and integration tests
- **Documentation**: `README_FOUNDATION.md` - Project overview
