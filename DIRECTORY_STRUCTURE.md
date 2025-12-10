# Project Directory Structure

This document describes the organization of the Thesis_work_Neuro project.

## Overview

```
Thesis_work_Neuro/
├── data/                   # All data files (raw, interim, processed)
│   ├── 01_raw/            # Original source data (never edited)
│   ├── 02_interim/        # Cached temporary outputs
│   └── 03_processed/      # Final feature matrices, train/test splits
│
├── viz/                    # All visualization outputs (auto-generated)
│   ├── molecules/         # Molecular structure visualizations
│   ├── maps/              # Activity map visualizations
│   └── reports/           # Analysis reports and figures
│
├── src/                    # Source code (importable package)
│   └── neuro_foundation/
│       ├── data/          # Data loading and graph generation
│       └── pipeline/      # Processing pipeline modules
│
├── scripts/               # Executable scripts (thin wrappers)
├── tests/                 # Unit tests
├── legacy/                # Deprecated/old code (for reference)
└── experiments/           # Experimental notebooks and prototypes
```

## Data Directory (`data/`)

### `01_raw/` - Original Data
**Never modify files in this directory!**

Contents:
- `molecules.csv` - Molecule metadata and SMILES
- `behavior_directory.csv` - Activity map file paths
- `activity_maps_csv/` - Individual activity map CSV files
- `molecular_graphs.npz` - Pre-computed molecular graphs

### `02_interim/` - Intermediate Results
Temporary cached outputs during processing:
- Embeddings
- Partial computations
- Intermediate transformations

### `03_processed/` - Final Processed Data
Ready-to-use feature matrices and datasets:
- Training/test splits
- Feature matrices
- Normalized datasets
- Model-ready inputs

## Visualization Directory (`viz/`)

**All visualization outputs are auto-generated.** Directories are created automatically when needed.

### `molecules/` - Molecular Visualizations

**Static Images (PNG)**
- `CID_{id}.png` - Static molecular structures
  - Simple 2D: Clean RDKit drawing
  - Simple 3D: Ultra-high-quality PyMOL rendering (2000×2000, ray traced)
  - Detailed 2D/3D: Molecule + comprehensive graph info
- `gallery.png` - Gallery of multiple molecules

**Interactive Viewers (HTML)**
- `CID_{id}.html` - Interactive 3D viewers (py3Dmol)
  - Rotate, zoom, pan in browser
  - Multiple rendering styles

**Generation:**
```python
from src.neuro_foundation.data.molecular_graphs import load_graph_by_cid, visualize_molecule_interactive

# Static visualization
load_graph_by_cid(1183, mode='simple', projection='3d', save_image=True)

# Interactive visualization
visualize_molecule_interactive(1183, style='sphere')
```

### `maps/` - Activity Map Visualizations

Activity pattern analysis:
- `coverage_counts.png` - Pixel coverage across all maps
- `coverage_histogram.png` - Coverage distribution
- `global_mask.png` - Analysis region mask
- `masked_averaged_example.png` - Example averaged map
- `masked_averaged_gallery.png` - Gallery of averaged maps

**Generation:**
```python
from src.neuro_foundation.pipeline.activity_maps import load_and_mask_maps

averaged_maps, cids, mask = load_and_mask_maps(
    'data/01_raw/behavior_directory.csv',
    output_dir='data/03_processed'
)
# Visualizations automatically saved to viz/maps/
```

### `reports/` - Analysis Reports

Training metrics, model comparisons, and analysis figures:
- Performance plots
- Feature importance visualizations
- Comparison charts
- Training curves

## Source Code (`src/neuro_foundation/`)

### `data/` - Data Loading and Graph Generation

**Core Modules:**
- `pyrfume_loader.py` - Load molecule metadata
- `molecular_graphs.py` - Generate molecular graphs, load/visualize
- `graph_viz.py` - Visualization utilities (PyMOL, py3Dmol, matplotlib)
- `interfaces.py` - Data structure definitions

**Key Functions:**
```python
# Load molecules
from src.neuro_foundation.data.pyrfume_loader import load_molecules_npz
molecules = load_molecules_npz('data/01_raw')

# Generate graphs
from src.neuro_foundation.data.molecular_graphs import generate_molecular_graphs
generate_molecular_graphs('data/01_raw')

# Visualize specific molecule
from src.neuro_foundation.data.molecular_graphs import load_graph_by_cid
load_graph_by_cid(1183, mode='detailed', projection='3d')
```

### `pipeline/` - Processing Pipeline

**Core Modules:**
- `activity_maps.py` - Load and process activity maps
- `preprocess.py` - Data cleaning and normalization
- `feature_select.py` - Feature selection methods
- `train_linear.py` - Linear model training

**Typical Pipeline:**
```python
# 1. Load and process activity maps
from src.neuro_foundation.pipeline.activity_maps import load_and_mask_maps
maps, cids, mask = load_and_mask_maps('data/01_raw/behavior_directory.csv',
                                       output_dir='data/03_processed')

# 2. Preprocess features
from src.neuro_foundation.pipeline.preprocess import preprocess_features
X_clean = preprocess_features(X_raw)

# 3. Select features
from src.neuro_foundation.pipeline.feature_select import select_features
X_selected = select_features(X_clean, y, method='variance')

# 4. Train model
from src.neuro_foundation.pipeline.train_linear import train_linear_model
model, metrics = train_linear_model(X_train, y_train)
```

## Scripts (`scripts/`)

Thin executable wrappers around source code:
- `load_data.py` - Download and load all data
- `preprocess.py` - Run preprocessing pipeline
- `select_features.py` - Run feature selection
- `train_linear.py` - Train linear models

**Usage:**
```bash
python scripts/load_data.py
python scripts/preprocess.py
python scripts/select_features.py
python scripts/train_linear.py
```

## Tests (`tests/`)

Unit tests for all modules:
- `test_preprocess.py`
- `test_feature_select.py`
- `test_train_linear.py`
- `conftest.py` - Shared test fixtures

**Run tests:**
```bash
pytest tests/
pytest tests/test_preprocess.py -v
```

## Configuration Files

- `requirements.txt` - Python dependencies
- `environment.yml` - Conda environment specification
- `.gitignore` - Git ignore rules
- `README_FOUNDATION.md` - Project overview
- `VISUALIZATION_GUIDE.md` - Detailed visualization documentation

## Best Practices

### Data Management
1. **Never modify `data/01_raw/`** - Keep original data intact
2. **Auto-generate intermediate** - Let pipeline create `02_interim/`
3. **Version control processed** - Track `03_processed/` if deterministic

### Visualization
1. **Auto-generated only** - All `viz/` contents are regenerated
2. **Don't commit viz/** - Excluded in `.gitignore`
3. **Use provided functions** - Automatic directory creation

### Code Organization
1. **Logic in `src/`** - All reusable code goes here
2. **Thin scripts** - Scripts are simple wrappers
3. **Test everything** - Add tests for new functionality

### Git Workflow
1. **Track source code** - Commit `src/`, `scripts/`, `tests/`
2. **Track configs** - Commit `requirements.txt`, `environment.yml`
3. **Exclude generated** - Don't commit `viz/`, large data files
4. **Document changes** - Update READMEs when structure changes

## Quick Start

```bash
# 1. Set up environment
conda env create -f environment.yml
conda activate thesis-neuro

# 2. Load data
python scripts/load_data.py

# 3. Generate molecular graphs
python -c "from src.neuro_foundation.data.molecular_graphs import generate_molecular_graphs; generate_molecular_graphs('data/01_raw')"

# 4. Visualize a molecule
python -c "from src.neuro_foundation.data.molecular_graphs import load_graph_by_cid; load_graph_by_cid(1183, mode='simple', projection='3d')"

# 5. Process activity maps
python scripts/preprocess.py

# 6. Run analysis
python scripts/train_linear.py
```

## Need Help?

- **Visualization**: See `VISUALIZATION_GUIDE.md`
- **Pipeline**: See `README_FOUNDATION.md`
- **Legacy code**: Check `legacy/README_LEGACY.md`
