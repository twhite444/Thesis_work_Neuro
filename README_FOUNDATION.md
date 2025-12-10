# Neuro Foundation

A minimal, modular foundation to rebuild the thesis pipeline without touching `legacy/`.

## 🚀 Two Parallel Pipelines

This project now supports **two complementary approaches** for molecular property prediction:

### 1. **Classical ML Pipeline** (Mordred Descriptors)
- Fixed-size molecular descriptors (1613 features)
- Feature selection and dimensionality reduction
- Classical models (Linear, Ridge, Lasso)
- Fast, interpretable, works on CPU

### 2. **GNN Pipeline** (Graph Neural Networks) ✨ NEW!
- Graph-based molecular representations
- 137-dim node features + 10-dim edge features
- Deep learning with PyTorch Geometric
- Captures spatial molecular structure
- See [GNN_PIPELINE.md](GNN_PIPELINE.md) for details

## Structure
- `src/neuro_foundation/data/` – dataset loader interface and implementations (supports both pipelines)
- `src/neuro_foundation/pipeline/` – pure functions for preprocessing, feature selection, training
- `scripts/` – small CLIs composing steps (both classical ML and GNN)
- `data/` – output folders (`01_raw`, `02_processed`)
- `experiments/` – metrics and coefficients

## Quick Start

1. Load raw data via Pyrfume:
   ```
   python scripts/load_data.py --output-dir data/01_raw
   ```
2. Featurize and standardize SMILES:
   ```
   python scripts/preprocess.py --output-dir data/02_processed --use-cached
   ```
3. Feature selection:
   ```
   python scripts/select_features.py --input-csv data/02_processed/cleaned_data.csv --threshold 1.0 --output-dir data/02_processed
   ```
4. Train linear baseline (requires target column in the CSV):
   ```
   python scripts/train_linear.py --input-csv data/02_processed/cleaned_data.csv --target-column PC1 --output-dir experiments/baseline_linear
   ```

## Swap Data Sources Later
Implement another loader (e.g., `CsvLoader`) that conforms to `DatasetLoader` and change the CLI to import it. No pipeline code changes needed.

## GNN Quick Start

For Graph Neural Network models:

1. **Generate molecular graphs** (one-time):
   ```bash
   python scripts/generate_graph_data.py
   ```
   
2. **Test the GNN pipeline**:
   ```bash
   python scripts/test_gnn_data.py
   ```
   
3. **Run example GNN training**:
   ```bash
   python scripts/example_gnn_training.py
   ```

4. **Build your own GNN model** - see [GNN_PIPELINE.md](GNN_PIPELINE.md) for complete guide

## Notes
- `legacy/` is read-only. This foundation is a fresh, CS-oriented rebuild.
- Keep each step single-purpose and deterministic; cache artifacts in `data/` and `experiments/`.
- Both pipelines (classical ML and GNN) can coexist and complement each other.
