# GNN Data Pipeline Documentation

## Overview

The GNN (Graph Neural Network) data pipeline provides a parallel workflow to the existing Mordred-based feature extraction pipeline. While the Mordred pipeline generates molecular descriptors as fixed-size feature vectors, the GNN pipeline converts molecular SMILES strings into graph representations suitable for graph neural networks.

**Key Features:**
- ✅ Converts 287 molecules from SMILES to graph format
- ✅ 137-dimensional node features (atom properties)
- ✅ 10-dimensional edge features (bond properties)
- ✅ PyTorch Geometric compatible format
- ✅ Efficient NPZ storage (0.06 MB for all graphs)
- ✅ Ready-to-use DataLoader utilities
- ✅ Train/val/test splitting functionality

## Architecture

### Data Flow

```
Raw Data (molecules_raw.npz)
    ↓
SMILES Strings (IsomericSMILES column)
    ↓
RDKit Molecular Graph Conversion
    ↓
Node Features (137-dim) + Edge Features (10-dim)
    ↓
PyTorch Geometric Data Objects
    ↓
GNN Model Training
```

### Comparison: Mordred vs GNN Pipeline

| Aspect | Mordred Pipeline | GNN Pipeline |
|--------|------------------|--------------|
| **Input** | SMILES → Molecular descriptors | SMILES → Graph structure |
| **Output Format** | Fixed-size vector (1613 dims) | Variable-size graph |
| **Model Type** | Classical ML (Linear, Ridge, etc.) | Graph Neural Networks |
| **Preprocessing** | Feature selection, PCA | None (learns from graph) |
| **Spatial Info** | No (descriptors are summary stats) | Yes (preserves molecular structure) |
| **Use Case** | Fast, interpretable models | Complex pattern recognition |

Both pipelines can coexist and complement each other!

## File Structure

```
src/neuro_foundation/data/
├── molecular_graphs.py      # Graph conversion (SMILES → graphs)
├── gnn_utils.py             # PyTorch Geometric utilities
└── pyrfume_loader.py        # Data loading (used by both)

scripts/
├── generate_graph_data.py   # Convert all molecules to graphs
├── test_gnn_data.py         # Test GNN data loading
└── example_gnn_training.py  # Example GNN model

data/01_raw/
├── molecules_raw.npz        # Original molecular data
└── molecular_graphs.npz     # Generated graph data (287 molecules)
```

## Node Features (137 dimensions)

Each atom in the molecule is represented by a 137-dimensional feature vector:

### 1. Atomic Number (100 dims)
One-hot encoding for atoms 1-100 (H, C, N, O, etc.)

### 2. Degree (11 dims)
Number of bonded neighbors (0-10), one-hot encoded

### 3. Formal Charge (5 dims)
Charge state: -2, -1, 0, +1, +2 (one-hot)

### 4. Hybridization (5 dims)
Orbital hybridization: SP, SP2, SP3, SP3D, SP3D2 (one-hot)

### 5. Aromaticity (1 dim)
Binary: is the atom aromatic?

### 6. Total Hydrogens (5 dims)
Number of bonded hydrogens: 0-4 (one-hot)

### 7. Radical Electrons (5 dims)
Number of radical electrons: 0-4 (one-hot)

### 8. Ring Membership (1 dim)
Binary: is the atom in a ring?

### 9. Chirality (4 dims)
Stereochemistry: R, S, unspecified, other (one-hot)

**Total: 100 + 11 + 5 + 5 + 1 + 5 + 5 + 1 + 4 = 137 dimensions**

## Edge Features (10 dimensions)

Each bond in the molecule is represented by a 10-dimensional feature vector:

### 1. Bond Type (4 dims)
Single, double, triple, aromatic (one-hot)

### 2. Conjugation (1 dim)
Binary: is the bond conjugated?

### 3. Ring Membership (1 dim)
Binary: is the bond in a ring?

### 4. Stereochemistry (4 dims)
None, any, Z, E (one-hot)

**Total: 4 + 1 + 1 + 4 = 10 dimensions**

## Usage Guide

### 1. Generate Graph Data

First, convert all molecules to graph format:

```bash
python scripts/generate_graph_data.py
```

This creates `data/01_raw/molecular_graphs.npz` containing:
- Node feature matrices for all 287 molecules
- Edge indices (connectivity)
- Edge feature matrices
- Compound IDs and metadata

**Output:**
```
Converting 287 molecules to graphs...
Successfully converted 287/287 molecules (100.0%)
Average nodes per molecule: 23.7
Average edges per molecule: 23.1
File size: 0.06 MB
```

### 2. Load and Explore Data

```python
from src.neuro_foundation.data.gnn_utils import (
    create_pyg_dataset,
    print_dataset_info
)

# Load dataset
dataset = create_pyg_dataset(data_dir='data/01_raw')
print(f"Loaded {len(dataset)} molecules")

# Print statistics
print_dataset_info(dataset)

# Access individual graphs
data = dataset[0]
print(f"CID: {data.cid}")
print(f"Nodes: {data.num_nodes}")
print(f"Edges: {data.edge_index.shape[1]}")
print(f"Node features: {data.x.shape}")
print(f"Edge features: {data.edge_attr.shape}")

# Access by compound ID
data_180 = dataset.get_by_cid(180)
```

### 3. Split Dataset for Training

```python
from src.neuro_foundation.data.gnn_utils import create_train_val_test_split

train_ds, val_ds, test_ds = create_train_val_test_split(
    dataset,
    train_ratio=0.7,   # 70% training
    val_ratio=0.15,    # 15% validation
    test_ratio=0.15,   # 15% test
    seed=42            # For reproducibility
)

print(f"Train: {len(train_ds)} molecules")
print(f"Val: {len(val_ds)} molecules")
print(f"Test: {len(test_ds)} molecules")
```

### 4. Create DataLoaders

```python
from torch_geometric.loader import DataLoader

# Create data loaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# Iterate through batches
for batch in train_loader:
    print(f"Batch size: {batch.num_graphs}")
    print(f"Total nodes: {batch.x.shape[0]}")
    print(f"Total edges: {batch.edge_index.shape[1]}")
    break
```

### 5. Build a GNN Model

```python
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, Dropout
from torch_geometric.nn import GCNConv, global_mean_pool

class MolecularGNN(torch.nn.Module):
    def __init__(self, num_node_features=137, hidden_dim=64, num_classes=1):
        super().__init__()
        
        # Graph convolutional layers
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Prediction head
        self.mlp = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Dropout(0.2),
            Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        
        # Global pooling (aggregate node features to graph level)
        x = global_mean_pool(x, batch)
        
        # Final prediction
        x = self.mlp(x)
        return x

# Initialize model
model = MolecularGNN(num_node_features=137, hidden_dim=64, num_classes=1)
```

### 6. Train the Model

```python
# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# Training loop
model.train()
for epoch in range(100):
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch)
        
        # Compute loss (assuming batch.y contains target values)
        loss = criterion(out, batch.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
```

## Testing the Pipeline

### Quick Test

```bash
python scripts/test_gnn_data.py
```

This will:
- Load the molecular graph dataset
- Display dataset statistics
- Test individual graph access
- Test train/val/test split
- Test DataLoader batching
- Show feature definitions

### Full Training Example

```bash
python scripts/example_gnn_training.py
```

This demonstrates:
- Complete training pipeline
- Simple GNN architecture
- Training loop implementation
- Model evaluation

## Advanced Usage

### Custom Graph Featurization

Modify `src/neuro_foundation/data/molecular_graphs.py` to add custom features:

```python
def get_custom_atom_features(atom):
    """Add custom atom features."""
    features = get_atom_features(atom)  # Base features
    
    # Add custom features
    features.extend([
        atom.GetMass(),
        atom.GetIsotope(),
        # ... more custom features
    ])
    
    return features
```

### Edge Features Control

Generate graphs without edge features to save space:

```bash
python scripts/generate_graph_data.py --no-edge-features
```

### Custom SMILES Column

Use a different SMILES column:

```bash
python scripts/generate_graph_data.py --smiles-col CanonicalSMILES
```

## Dataset Statistics

**Generated from 287 molecules:**
- **Total atoms**: 6,789
- **Total bonds**: 6,638
- **Atoms per molecule**: 23.7 ± 8.4 (range: 5-61)
- **Bonds per molecule**: 23.1 ± 8.4 (range: 4-60)
- **File size**: 0.06 MB (compressed NPZ)

## Common GNN Architectures

The pipeline is compatible with many GNN architectures from PyTorch Geometric:

### 1. Graph Convolutional Network (GCN)
```python
from torch_geometric.nn import GCNConv
self.conv = GCNConv(in_channels, out_channels)
```

### 2. Graph Attention Network (GAT)
```python
from torch_geometric.nn import GATConv
self.conv = GATConv(in_channels, out_channels, heads=8)
```

### 3. GraphSAGE
```python
from torch_geometric.nn import SAGEConv
self.conv = SAGEConv(in_channels, out_channels)
```

### 4. Graph Isomorphism Network (GIN)
```python
from torch_geometric.nn import GINConv
self.conv = GINConv(nn_module)
```

### 5. Message Passing Neural Network
```python
from torch_geometric.nn import MessagePassing
class CustomMPNN(MessagePassing):
    # Custom implementation
```

## Pooling Strategies

### Global Pooling
```python
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

# Mean pooling (most common)
x = global_mean_pool(x, batch)

# Max pooling
x = global_max_pool(x, batch)

# Sum pooling
x = global_add_pool(x, batch)
```

### Hierarchical Pooling
```python
from torch_geometric.nn import TopKPooling, SAGPooling

# TopK pooling
x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None, batch)

# Self-attention pooling
x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None, batch)
```

## Integration with Existing Pipeline

The GNN pipeline works alongside the existing Mordred pipeline:

```python
# Option 1: Use Mordred features for classical ML
from src.neuro_foundation.data.pyrfume_loader import load_molecules_npz
molecules = load_molecules_npz()
# ... use molecular descriptors for Ridge regression, etc.

# Option 2: Use GNN pipeline for deep learning
from src.neuro_foundation.data.gnn_utils import create_pyg_dataset
dataset = create_pyg_dataset()
# ... use graph representations for GNN models

# Option 3: Ensemble both approaches
# Train both types of models and combine predictions
```

## Troubleshooting

### Issue: Import Error for PyTorch Geometric

**Error:**
```
ImportError: No module named 'torch_geometric'
```

**Solution:**
```bash
conda activate neuro
pip install torch torch-geometric
```

### Issue: RDKit Not Found

**Error:**
```
ImportError: No module named 'rdkit'
```

**Solution:**
```bash
conda install -c conda-forge rdkit
```

### Issue: Graph Conversion Fails

**Error:**
```
Successfully converted 285/287 molecules (99.3%)
```

**Solution:**
This is expected. Some SMILES may be invalid or cause parsing errors. The pipeline handles this gracefully with a `valid_mask` in the output.

### Issue: Out of Memory During Training

**Solution:**
Reduce batch size:
```python
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)  # Smaller batch
```

## Performance Tips

1. **Use GPU**: GNNs benefit greatly from GPU acceleration
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ```

2. **Optimize Batch Size**: Find the largest batch that fits in memory
   ```python
   # Try: 16, 32, 64, 128
   batch_size = 32
   ```

3. **Use Mixed Precision**: For faster training on compatible GPUs
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

4. **Profile Your Model**: Identify bottlenecks
   ```python
   with torch.profiler.profile() as prof:
       model(batch)
   ```

## Further Reading

- **PyTorch Geometric Documentation**: https://pytorch-geometric.readthedocs.io/
- **RDKit Documentation**: https://www.rdkit.org/docs/
- **Graph Neural Networks**: A Review of Methods and Applications (Zhou et al., 2020)
- **Molecular Property Prediction**: A Review of Current Methods (Yang et al., 2019)

## Summary

The GNN data pipeline is now fully operational and provides:

✅ **Complete data flow**: SMILES → Graphs → PyTorch Geometric → GNN training  
✅ **Rich features**: 137-dim nodes, 10-dim edges  
✅ **Easy to use**: Simple API with comprehensive utilities  
✅ **Well tested**: All components verified and working  
✅ **Documented**: Complete examples and guides  
✅ **Scalable**: Ready for production use  

You can now build sophisticated GNN models for molecular property prediction!
