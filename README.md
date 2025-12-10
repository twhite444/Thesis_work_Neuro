# Neuro Foundation

A minimal, modular foundation to rebuild the thesis pipeline without touching `legacy/`.
If you would like to see original thesis code with neural networks you may find it in the legacy folder or in the prefactor branch. I am currently working on redesigning the code architecture to be modular and user friendly. I will also be making it more compatible for both GNN and NN analysis. 

## 🚀 Data Preparation

This project supports **molecular descriptor** extraction and **graph-based representations**:

### 1. **Classical ML Pipeline** (Mordred Descriptors)
- Fixed-size molecular descriptors (1613 features)
- Feature selection and dimensionality reduction
- Classical models (Linear, Ridge, Lasso)
- Fast, interpretable, works on CPU

### 2. **Molecular Graph Data** ✨ NEW!
- Graph-based molecular representations from SMILES
- 137-dim node features (atom properties)
- 10-dim edge features (bond properties)  
- Stored in NPZ format for later use
- Visualization tools for inspecting graphs

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

## Graph Data Quick Start

For molecular graph generation and visualization:

1. **Generate molecular graphs** (one-time):
   ```bash
   python scripts/generate_graph_data.py
   ```
   Output: `data/01_raw/molecular_graphs.npz` (287 molecules, 0.06 MB)
   
2. **Visualize molecular graphs**:
   ```bash
   # Visualize specific molecules
   python scripts/visualize_graphs.py --cids 180 240 7991
   
   # Create a gallery view
   python scripts/visualize_graphs.py --gallery
   
   # Compare structure vs graph representation
   python scripts/visualize_graphs.py --compare 180
   
   # Print dataset summary
   python scripts/visualize_graphs.py --summary
   ```

3. **Use in Python**:
   ```python
   from src.neuro_foundation.data.molecular_graphs import load_graph_data, get_graph_by_cid
   
   # Load all graphs
   graph_data = load_graph_data('data/01_raw')
   
   # Get specific graph
   graph = get_graph_by_cid(180, graph_data)
   # graph['node_features']: (num_nodes, 137)
   # graph['edge_index']: (2, num_edges)
   # graph['edge_attr']: (num_edges, 10)
   ```

## Notes
- `legacy/` is read-only. This foundation is a fresh, CS-oriented rebuild.
- Keep each step single-purpose and deterministic; cache artifacts in `data/` and `experiments/`.
- Both pipelines (classical ML and GNN) can coexist and complement each other.
