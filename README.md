# Olfactory Modeling

A modular, well-tested foundation for neuroscience molecular data analysis, supporting both classical machine learning and graph neural network approaches.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/twhite444/Thesis_work_Neuro/workflows/Tests/badge.svg)](https://github.com/twhite444/Thesis_work_Neuro/actions)
[![Linting](https://github.com/twhite444/Thesis_work_Neuro/workflows/Lint/badge.svg)](https://github.com/twhite444/Thesis_work_Neuro/actions)

## Scope and Assumptions

This package is designed for modeling olfactory datasets in the
[Pyrfume](https://pyrfume.org) ecosystem.

Currently supported:
- Leon dataset (primary)
- Other Pyrfume datasets may be compatible but can require
  loader-specific adjustments.

This is not a general-purpose neuroscience or ML framework.

## ⚡ What's New (Dec 2024)

**Performance Optimizations:**
- 🚀 **13x faster preprocessing** (30s → 2.3s) with Mordred feature caching
- 🐛 Fixed critical variance threshold bug (now applied before standardization)
- 🔧 Unified preprocessing pipeline with full configurability
- 📊 Weight decay (L2 regularization) enabled by default for better generalization

**New Documentation:**
- 📚 **[QUICK_START.md](QUICK_START.md)** - Fast track for new users ⭐
- 📖 **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** - Complete guide index
- � See all guides in the root directory

**Quick Migration:** Just run `python scripts/load_all_data.py` once, then enjoy 13x faster preprocessing!

## �📋 Overview

This project provides tools for analyzing molecular odorant data and neural activity maps from the Pyrfume database. It supports multiple complementary approaches:

### 1. **Classical ML Pipeline** (Mordred Descriptors) ⚡ Optimized!
- Fixed-size molecular descriptors (1826 → 1187 features after cleaning)
- **Smart caching** - Mordred computed once, reused forever
- Configurable variance threshold feature selection
- Classical models (Linear, Ridge, Lasso) with regularization
- Fast, interpretable, works on CPU

### 2. **Molecular Graph Pipeline** ✨
- Graph-based molecular representations from SMILES
- 137-dimensional node features (atom properties)
- 10-dimensional edge features (bond properties)
- Advanced visualization tools
- Ready for Graph Neural Networks (GNNs)

### 3. **Activity Maps Pipeline** ✨
- Pre-processes brain activity maps (79×43 spatial patterns)
- **4 selection strategies**: best_quality, average, median, first
- **Configurable masking**: adjust coverage threshold (0.0-1.0)
- **Efficient storage**: NPZ format (1.5 MB vs 50 MB CSVs)
- **Separate from molecule preprocessing**: works with both descriptors and graphs
- See: [`docs/ACTIVITY_MAPS_PIPELINE.md`](docs/ACTIVITY_MAPS_PIPELINE.md) for details

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/twhite444/Thesis_work_Neuro.git
   cd Thesis_work_Neuro
   ```

2. **Create and activate environment:**
   ```bash
   # Using conda (PRIMARY - recommended for full compatibility)
   conda env create -f environment.yml
   conda activate neuro-foundation
   
   # Alternative: using pip (may have dependency conflicts)
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   
   > **Note:** This project uses conda as the primary package manager for reproducible environments and complex scientific dependencies. The `environment.yml` file contains all required packages with compatible versions.

3. **Verify installation:**
   ```bash
   pytest tests/  # Should pass 100% of tests
   ```

### Basic Usage

#### Classical ML Pipeline

```bash
# 1. Load raw data from Pyrfume
python scripts/load_data.py --output-dir data/01_raw

# 2. Extract molecular descriptors and process
python scripts/preprocess.py --output-dir data/02_processed --use-cached

# 3. Select features based on variance
python scripts/select_features.py \
    --input-csv data/02_processed/cleaned_data.csv \
    --threshold 1.0 \
    --output-dir data/02_processed

# 4. Train linear baseline model
python scripts/train_linear.py \
    --input-csv data/02_processed/cleaned_data.csv \
    --target-column PC1 \
    --output-dir experiments/baseline_linear
```

#### Molecular Graph Pipeline

```bash
# 1. Generate molecular graphs (one-time)
python scripts/generate_graph_data.py

# 2. Visualize molecules
python scripts/visualize_graphs.py --cids 180 240 7991

# 3. Create gallery view
python scripts/visualize_graphs.py --gallery

# 4. Compare structure vs graph representation
python scripts/visualize_graphs.py --compare 180

# 5. View dataset summary
python scripts/visualize_graphs.py --summary
```

#### Activity Map Processing

```bash
# 1. Process activity maps (select best, apply mask, save)
python scripts/run_activity_maps.py

# 2. Try different selection strategies
python scripts/run_activity_maps.py --strategy average
python scripts/run_activity_maps.py --strategy median

# 3. Adjust coverage threshold
python scripts/run_activity_maps.py --coverage-threshold 0.7  # Stricter
python scripts/run_activity_maps.py --coverage-threshold 0.3  # Lenient

# 4. Train neural networks using processed maps
python scripts/train_baseline_nn.py --model mlp --epochs 100
python scripts/train_baseline_nn.py --model cnn --epochs 100

# 5. K-fold cross-validation (recommended for thesis)
python scripts/train_baseline_nn_kfold.py --model mlp --n-folds 5 --epochs 100

# 6. Grid search for hyperparameter tuning
python scripts/grid_search_baseline.py --model mlp \
    --param lr 0.001 0.005 0.01 \
    --param dropout 0.3 0.35 0.4 \
    --use-kfold

# See docs/ACTIVITY_MAPS_PIPELINE.md for full guide
```

**All training automatically generates publication-quality visualizations (300 DPI):**
- Training curves (loss, correlation, R²)
- K-fold CV analysis with mean ± std across folds
- Grid search heatmaps and rankings
- **Feature importance** rankings based on first-layer weights 🆕
- See [`docs/VISUALIZATION_NN_GUIDE.md`](docs/VISUALIZATION_NN_GUIDE.md) for complete guide

#### Performance Profiling ⏱️

```bash
# Complete profiling (dataloader + device comparison + training breakdown)
python scripts/profile_performance.py --model mlp --profile-epochs 3 --compare-devices

# Quick dataloader check
python scripts/profile_performance.py --model mlp --profile-batches 20

# Compare CPU vs MPS performance
python scripts/profile_performance.py --model mlp --compare-devices
```

**Key findings:**
- ✅ 15-60x speedup over legacy implementation
- ✅ DataLoader is very fast (5-8ms/batch, 120-190 batches/s)
- ✅ For our small models, CPU is 4.5x faster than MPS for single inference
- ✅ No bottlenecks identified - training is well optimized

See [`docs/PROFILING_QUICKSTART.md`](docs/PROFILING_QUICKSTART.md) for details.
```

## 📚 Documentation

### Core Documentation

- **[Activity Maps Pipeline](docs/ACTIVITY_MAPS_PIPELINE.md)** - Complete guide to map preprocessing ✨
- **[Quick Start Guide](docs/QUICK_START.md)** - Fast reference for activity maps ✨
- **[Performance Improvements](docs/PERFORMANCE_IMPROVEMENTS.md)** - 15-60x speedup details ✨
- **[K-Fold CV & Grid Search](docs/KFOLD_AND_GRID_SEARCH.md)** - Hyperparameter optimization guide ✨
- **[NN Visualization Guide](docs/VISUALIZATION_NN_GUIDE.md)** - Complete visualization tutorial 📊 NEW
- **[Profiling Guide](docs/PROFILING_GUIDE.md)** - Complete profiling tutorial (700+ lines) ⏱️
- **[Profiling Quick Start](docs/PROFILING_QUICKSTART.md)** - Quick reference for profiling tools ⏱️
- **[API Documentation](docs/API_DOCUMENTATION.md)** - Complete API reference for all modules
- **[Testing Guide](docs/TESTING_GUIDE.md)** - How to run and write tests
- **[Visualization Guide](docs/VISUALIZATION_GUIDE.md)** - Using visualization tools
- **[Quick Reference](docs/QUICK_REFERENCE.md)** - Common commands and workflows
- **[Directory Structure](docs/DIRECTORY_STRUCTURE.md)** - Project organization
- **[Final Report](docs/FINAL_REPORT.md)** - Detailed project analysis

### Module-Specific READMEs

- **[scripts/README.md](scripts/README.md)** - Command-line tools
- **[scripts/examples/README.md](scripts/examples/README.md)** - Usage examples
- **[scripts/exploration/README.md](scripts/exploration/README.md)** - Data exploration tools
- **[viz/README.md](viz/README.md)** - Visualization outputs

## 🏗️ Project Structure

```
Thesis_work_Neuro/
├── src/olfactory_modeling/          # Core library code
│   ├── data/                       # Data loading and interfaces
│   │   ├── interfaces.py           # Abstract base classes
│   │   ├── pyrfume_loader.py       # Pyrfume data loader
│   │   ├── molecular_graphs.py     # Graph generation
│   │   └── graph_viz.py            # Graph visualization
│   └── pipeline/                   # Analysis pipeline
│       ├── preprocess.py           # Feature extraction
│       ├── feature_select.py       # Feature selection
│       ├── train_linear.py         # Model training
│       └── activity_maps.py        # Activity map processing
├── scripts/                        # Command-line tools
│   ├── load_data.py               # Data download
│   ├── preprocess.py              # Preprocessing
│   ├── generate_graph_data.py     # Graph generation
│   ├── visualize_graphs.py        # Visualization
│   ├── examples/                  # Usage examples
│   └── exploration/               # Data exploration
├── tests/                         # Test suite (100% passing)
├── docs/                          # Documentation
│   ├── API_DOCUMENTATION.md       # Complete API reference
│   ├── TESTING_GUIDE.md           # Testing guide
│   ├── VISUALIZATION_GUIDE.md     # Visualization guide
│   └── ...                        # Additional documentation
├── data/                          # Data storage
│   ├── 01_raw/                    # Raw downloaded data
│   └── 02_processed/              # Processed features
├── viz/                           # Visualization outputs
│   ├── graphs/                    # Molecular graphs
│   ├── molecules/                 # 3D structures
│   └── maps/                      # Activity maps
└── experiments/                   # Model outputs and metrics
```

## 🧪 Testing

The project has comprehensive test coverage with 100% passing tests:

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m slow          # Slow tests only

# Run with coverage
pytest --cov=src/olfactory_modeling --cov-report=html

# Run specific test file
pytest tests/test_preprocess.py -v
```

See [docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md) for more details.

## 📊 Data

This project uses the **Pyrfume** database, which provides:

- **287 unique molecules** with SMILES structures
- **405 brain activity maps** from olfactory experiments
- **432 stimuli** with experimental metadata
- Molecular properties (MW, IUPAC names, etc.)

Data is cached locally in both CSV (human-readable) and NPZ (fast loading) formats.

## 🎨 Visualization Tools

The project includes powerful visualization capabilities:

### Molecular Graphs
- **2D structure diagrams** with atom indices
- **3D ball-and-stick models** with PyMOL rendering
- **Interactive 3D viewers** (HTML-based)
- **Graph connectivity matrices**
- **Gallery views** for multiple molecules

### Neural Activity Maps
- **Heatmap visualizations** of brain activity
- **Statistical summaries** (intensity, coverage)
- **Batch processing** for multiple maps
- **Mask visualization** and quality metrics

See [docs/VISUALIZATION_GUIDE.md](docs/VISUALIZATION_GUIDE.md) for examples.

## 🔬 Use Cases

### Classical ML Workflow
```python
from olfactory_modeling.data.pyrfume_loader import PyrfumeLoader, load_molecules_npz
from olfactory_modeling.pipeline.preprocess import featurize_smiles_batch

# Load data
loader = PyrfumeLoader(output_dir='data/01_raw')
molecules = loader.load_molecules()

# Extract features
features_df = featurize_smiles_batch(
    molecules['IsomericSMILES'].tolist(),
    molecules['CID'].tolist()
)

# Continue with feature selection and modeling...
```

### Graph-Based Workflow
```python
from olfactory_modeling.data.molecular_graphs import load_graph_data, get_graph_by_cid

# Load pre-generated graphs
graph_data = load_graph_data('data/01_raw')

# Get specific molecule's graph
graph = get_graph_by_cid(180, graph_data)
print(f"Nodes: {graph['num_nodes']}, Edges: {graph['num_edges']}")
print(f"Node features shape: {graph['node_features'].shape}")
print(f"Edge features shape: {graph['edge_attr'].shape}")

# Use with PyTorch Geometric for GNNs
# (See examples in scripts/examples/)
```

### Activity Map Analysis
```python
from olfactory_modeling.pipeline.activity_maps import pipeline_load_and_mask

# Process activity maps for molecules
results = pipeline_load_and_mask(
    data_dir='data/01_raw',
    output_dir='viz/maps',
    cids=[180, 240, 7991],
    visualize=True
)

# Access processed data
for result in results:
    print(f"CID {result['cid']}: {result['num_active_voxels']} active voxels")
```

## 🛠️ Development

### Running Linters

```bash
# Ruff (format + lint)
ruff check src/ tests/ scripts/
ruff format src/ tests/ scripts/

# Mypy (type checking)
mypy --ignore-missing-imports src/
```

### Adding New Features

1. Implement in `src/olfactory_modeling/`
2. Add tests in `tests/`
3. Update documentation
4. Run linters and tests
5. Submit pull request

### Code Style

- **Type hints** for all functions
- **Docstrings** (Google style)
- **Pure functions** where possible
- **Single responsibility** principle
- **Comprehensive tests**

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## 📝 License

This project is part of a thesis work. Please contact the author for usage permissions.

## 👤 Author

**Tom White**
- GitHub: [@twhite444](https://github.com/twhite444)

## 🙏 Acknowledgments

- **Pyrfume Project** for the olfactory database
- **RDKit** for molecular informatics tools
- **PyMOL** for 3D molecular visualization

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@misc{white2025neuro,
  author = {White, Tom},
  title = {Neuro Foundation: Molecular Analysis for Olfactory Neuroscience},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/twhite444/Thesis_work_Neuro}
}
```

---

## 📌 Notes

- **Legacy code** in `legacy/` is preserved for reference but not maintained
- All new development happens in the modular `src/` structure
- Both classical ML and graph approaches coexist harmoniously
- Data is cached to avoid repeated downloads
- All visualizations are saved with timestamps to prevent overwriting
