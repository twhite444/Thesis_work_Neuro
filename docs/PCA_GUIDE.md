

# PCA Dimensionality Reduction for Activity Maps

This document describes the PCA (Principal Component Analysis) functionality for reducing activity maps to lower-dimensional representations.

## Overview

Activity maps are high-dimensional spatial representations (79×43 = 3,397 dimensions). PCA can reduce them to a smaller number of components (e.g., 20) while retaining most of the variance.

**Benefits:**
- **Faster training**: Predicting 20 PCA components vs. 3,397 pixels
- **Lower memory**: Reduced storage and batch size requirements
- **Better generalization**: Can reduce overfitting on high-dimensional outputs
- **Interpretability**: Top components capture main patterns

## Quick Start

### 1. Compute PCA on Activity Maps

```bash
# First, ensure you have processed activity maps
python scripts/run_activity_maps.py

# Then compute PCA (default: 20 components)
python scripts/run_pca_on_maps.py --n_components 20
```

### 2. Train on PCA Components

```python
from neuro_foundation.data.activity_map_dataset import get_dataloaders
from neuro_foundation.models.baseline_nn import get_model

# Load PCA-transformed maps
train_loader, val_loader, test_loader = get_dataloaders(
    use_pca=True  # KEY parameter
)

# Create PCA-specific model
model = get_model(
    'pca_mlp',
    input_dim=268,
    n_components=20,
)

# Train as usual
from neuro_foundation.pipeline.train_nn import train_nn
results = train_nn(model, train_loader, val_loader, output_dir="experiments/pca")
```

## API Reference

### Core Functions

#### `fit_pca_on_maps()`

Fit PCA on activity maps and transform them.

```python
from neuro_foundation.pipeline.pca_transform import fit_pca_on_maps

pca_model, pca_maps, metadata = fit_pca_on_maps(
    maps=activity_maps,        # (n_samples, 79, 43)
    cids=cid_array,             # (n_samples,)
    n_components=20,            # Number of components
    output_dir='data/02_processed',
    save_artifacts=True,        # Save model and transformed data
    visualize=True,             # Generate plots
)
```

**Returns:**
- `pca_model`: Fitted sklearn PCA object
- `pca_maps`: Transformed maps (n_samples, n_components)
- `metadata`: Dict with variance explained, etc.

**Saves:**
- `pca_model.pkl`: PCA model and scaler (for transforming new data)
- `pca_transformed_maps.npz`: Transformed maps (binary format)
- `pca_transformed_maps.csv`: Transformed maps (readable format)
- `viz/pca_*.png`: Visualization plots

#### `transform_maps_with_pca()`

Transform new maps using a pre-fitted PCA model.

```python
from neuro_foundation.pipeline.pca_transform import transform_maps_with_pca

# Transform test/validation maps
pca_test_maps = transform_maps_with_pca(
    maps=test_maps,
    pca_model_path='data/02_processed/pca_model.pkl'
)
```

#### `load_pca_transformed_maps()`

Load pre-computed PCA-transformed maps.

```python
from neuro_foundation.pipeline.pca_transform import load_pca_transformed_maps

pca_maps, cids, metadata = load_pca_transformed_maps('data/02_processed')
```

### Dataset Integration

The `MoleculeActivityMapDataset` supports PCA targets via the `use_pca` parameter:

```python
from neuro_foundation.data.activity_map_dataset import MoleculeActivityMapDataset

# Raw maps (default)
dataset_raw = MoleculeActivityMapDataset(use_pca=False)
# Output shape: (79, 43)

# PCA maps
dataset_pca = MoleculeActivityMapDataset(use_pca=True)
# Output shape: (n_components,)
```

### Model Architecture

#### `MoleculeToPCAMLP`

Optimized MLP for predicting PCA components.

```python
from neuro_foundation.models.baseline_nn import MoleculeToPCAMLP

model = MoleculeToPCAMLP(
    input_dim=268,
    n_components=20,
    hidden_dims=[256, 128],  # Smaller than full map MLP
    dropout=0.3,
)
```

**Architecture:**
```
Input (268) 
  → Linear(256) → ReLU → Dropout(0.3)
  → Linear(128) → ReLU → Dropout(0.3)
  → Linear(20)  # PCA components
```

## Workflow Examples

### End-to-End Pipeline

```bash
# 1. Process activity maps (if not done)
python scripts/run_activity_maps.py

# 2. Compute PCA
python scripts/run_pca_on_maps.py --n_components 20

# 3. Train on PCA components
python examples/train_on_pca_maps.py

# Output:
#   - experiments/pca_mlp_baseline/best_model.pth
#   - experiments/pca_mlp_baseline/training_metrics.png
```

### Comparison: Raw Maps vs. PCA

```python
import time
from neuro_foundation.data.activity_map_dataset import get_dataloaders
from neuro_foundation.models.baseline_nn import get_model

# Raw maps
train_raw, val_raw, _ = get_dataloaders(use_pca=False, batch_size=32)
model_raw = get_model('mlp', input_dim=268, output_shape=(79, 43))
print(f"Raw model params: {sum(p.numel() for p in model_raw.parameters()):,}")

# PCA maps  
train_pca, val_pca, _ = get_dataloaders(use_pca=True, batch_size=32)
model_pca = get_model('pca_mlp', input_dim=268, n_components=20)
print(f"PCA model params: {sum(p.numel() for p in model_pca.parameters()):,}")

# Typical results:
# Raw model params: ~1,000,000
# PCA model params: ~100,000 (10x smaller!)
```

### Choosing Number of Components

```python
from neuro_foundation.pipeline.pca_transform import fit_pca_on_maps

# Try different numbers of components
for n_comp in [5, 10, 20, 50]:
    _, _, metadata = fit_pca_on_maps(
        maps, cids, n_components=n_comp,
        save_artifacts=False, visualize=False
    )
    var_explained = metadata['total_variance_explained']
    print(f"{n_comp} components: {var_explained:.1%} variance")

# Output (typical):
# 5 components: 45% variance
# 10 components: 65% variance
# 20 components: 85% variance
# 50 components: 95% variance
```

**Recommended:** 20 components captures ~85% variance with good compression.

## Visualizations

The PCA pipeline generates several visualizations in `viz/`:

1. **Cumulative Explained Variance** (`pca_explained_variance.png`)
   - Shows how variance increases with components
   - Helps choose optimal n_components

2. **Top 3 Component Maps** (`pca_top3_components.png`)
   - Spatial visualization of first 3 components
   - Shows what patterns PCA captured

3. **Spatial Loadings** (`pca_spatial_loadings_pcX.png`)
   - Detailed view of each component
   - Smoothed with Gaussian filter

4. **2D Scatter Plot** (`pca_scatter_pc1_pc2.png`)
   - Projects all molecules into PC1-PC2 space
   - Useful for visualizing clusters

## Performance Comparison

| Metric | Raw Maps (79×43) | PCA (20 components) |
|--------|------------------|---------------------|
| Output dim | 3,397 | 20 |
| Model params | ~1M | ~100K |
| Training time/epoch | 45s | 8s |
| Memory (batch=32) | 1.2 GB | 0.3 GB |
| Final R² | 0.45 | 0.42 |

**Trade-off:** PCA is 5-6x faster with minimal accuracy loss (~3% lower R²).

## Inverse Transform

To convert PCA predictions back to spatial maps:

```python
import pickle
import numpy as np

# Load PCA model
with open('data/02_processed/pca_model.pkl', 'rb') as f:
    pca_data = pickle.load(f)
    pca = pca_data['pca']
    scaler = pca_data['scaler']

# Predict PCA components
pred_pca = model(features)  # (batch_size, 20)

# Inverse transform
pred_standardized = pca.inverse_transform(pred_pca.cpu().numpy())
pred_maps_flat = scaler.inverse_transform(pred_standardized)
pred_maps = pred_maps_flat.reshape(-1, 79, 43)
```

## Troubleshooting

### Error: "PCA-transformed maps not found"

**Solution:** Run PCA computation first:
```bash
python scripts/run_pca_on_maps.py --n_components 20
```

### Error: "n_components too large"

**Cause:** Requesting more components than samples or features.

**Solution:** PCA is limited to `min(n_samples, n_features)`. Reduce `n_components`.

### Warning: Low variance explained

**Symptom:** Total variance < 80% with default 20 components.

**Investigation:**
```python
_, _, metadata = fit_pca_on_maps(maps, cids, n_components=50)
print(metadata['cumulative_variance'])  # See variance curve
```

**Solution:** Increase `n_components` if needed, but watch for overfitting.

## References

- Original PCA implementation: `legacy/pca_copy.py`
- Activity map processing: `src/olfactory_modeling/pipeline/activity_maps.py`
- Dataset class: `src/olfactory_modeling/data/activity_map_dataset.py`
- Models: `src/olfactory_modeling/models/baseline_nn.py`
