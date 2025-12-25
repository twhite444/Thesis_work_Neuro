# Neural Network Visualization Guide

Comprehensive guide to visualizing neural network training, cross-validation, grid search, and predictions.

## Table of Contents

- [Quick Start](#quick-start)
- [Automatic Visualizations](#automatic-visualizations)
- [Visualization Functions](#visualization-functions)
- [Examples](#examples)
- [Customization](#customization)
- [Best Practices](#best-practices)

---

## Quick Start

**Training automatically generates visualizations:**

```bash
# Train with auto-visualization
python scripts/train_baseline_nn.py --model mlp --epochs 100
# Generates: data/03_models/mlp_baseline/training_curves.png

# K-fold CV with auto-visualization
python scripts/train_baseline_nn_kfold.py --model mlp --n-folds 5 --epochs 100
# Generates: data/03_models/mlp_kfold/cv_analysis.png

# Grid search with auto-visualization
python scripts/grid_search_baseline.py --model mlp --param dropout 0.3 0.35 0.4
# Generates: data/03_models/grid_search/grid_search_analysis.png
```

**All visualizations are publication-quality (300 DPI) and ready for thesis/papers!**

---

## Automatic Visualizations

### 1. Training Curves (from `train_nn()`)

**Automatically generated when training:**
- File: `{output_dir}/training_curves.png`
- 3-panel figure showing:
  - Loss over epochs (train vs validation)
  - Correlation over epochs
  - R² score over epochs
- Best epoch marked with red line
- Includes legend and grid

**Example:**
```python
from src.olfactory_modeling.pipeline.train_nn import train_nn

metrics = train_nn(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    output_dir="experiments/my_model",
    num_epochs=100,
)
# Auto-generates: experiments/my_model/training_curves.png
```

### 2. Cross-Validation Analysis (from `train_nn_kfold()`)

**Automatically generated:**
- File: `{output_dir}/cv_analysis.png`
- Comprehensive 6-panel figure:
  - Individual fold loss curves
  - Individual fold correlation curves
  - Best correlation per fold (bar chart)
  - Summary statistics table
  - Mean learning curve across folds (with ± 1 std band)

**Example:**
```python
from src.olfactory_modeling.pipeline.train_nn import train_nn_kfold

cv_results = train_nn_kfold(
    model_factory=model_factory,
    dataset=dataset,
    output_dir="experiments/my_cv",
    n_splits=5,
    num_epochs=100,
)
# Auto-generates: experiments/my_cv/cv_analysis.png
```

### 3. Grid Search Analysis (from `grid_search()`)

**Automatically generated:**
- File: `{output_dir}/grid_search_analysis.png`
- Comprehensive multi-panel figure:
  - Top N configurations ranked (horizontal bar chart)
  - 2D parameter heatmap (if 2 parameters)
  - Best configuration summary
  - Score distribution histogram
  - Learning curves for top 3 configurations

**Example:**
```python
from src.olfactory_modeling.pipeline.train_nn import grid_search

grid_results = grid_search(
    model_factory_template=model_factory_template,
    dataset=dataset,
    param_grid={'dropout': [0.3, 0.35, 0.4], 'learning_rate': [0.003, 0.005, 0.007]},
    output_dir="experiments/my_grid_search",
)
# Auto-generates: experiments/my_grid_search/grid_search_analysis.png
```

---

## Visualization Functions

All visualization functions can also be used standalone for custom analysis.

### 1. `plot_training_curves()`

**Purpose:** Visualize training progress over epochs.

**Usage:**
```python
from src.olfactory_modeling.visualization import plot_training_curves

# After training
metrics = train_nn(...)

# Generate standalone plot
fig = plot_training_curves(
    metrics,
    output_path='my_training_curves.png',
    show_r2=True,  # Include R² subplot
    figsize=(14, 5)
)
```

**Parameters:**
- `metrics_dict`: Dict with training history (from `train_nn()`)
- `output_path`: Where to save figure (optional)
- `show_r2`: Whether to show R² subplot (default: True)
- `figsize`: Figure size (width, height)

**Returns:** matplotlib Figure object

### 2. `plot_cv_results()`

**Purpose:** Analyze K-fold cross-validation results.

**Usage:**
```python
from src.olfactory_modeling.visualization import plot_cv_results

fig = plot_cv_results(
    'experiments/my_cv/cv_results.json',
    output_path='my_cv_analysis.png',
    figsize=(14, 10)
)
```

**Parameters:**
- `cv_results_path`: Path to cv_results.json file
- `output_path`: Where to save figure (optional)
- `figsize`: Figure size (width, height)

**Returns:** matplotlib Figure object

**What it shows:**
- Individual fold performance over epochs
- Mean performance across folds with standard deviation bands
- Best metrics per fold
- Aggregate statistics

### 3. `plot_grid_search_results()`

**Purpose:** Compare hyperparameter configurations.

**Usage:**
```python
from src.olfactory_modeling.visualization import plot_grid_search_results

fig = plot_grid_search_results(
    'experiments/grid_search/grid_search_results.json',
    output_path='grid_analysis.png',
    top_n=10,  # Show top 10 configs
    figsize=(16, 10)
)
```

**Parameters:**
- `grid_results_path`: Path to grid_search_results.json
- `output_path`: Where to save figure (optional)
- `top_n`: Number of top configurations to highlight
- `figsize`: Figure size (width, height)

**Returns:** matplotlib Figure object

**What it shows:**
- Ranked configurations (horizontal bar chart)
- 2D heatmap (if searching over 2 parameters)
- Best configuration details
- Score distribution across all configs
- Learning curves for top performers

### 4. `plot_prediction_scatter()`

**Purpose:** Compare predictions with ground truth.

**Usage:**
```python
from src.olfactory_modeling.visualization import plot_prediction_scatter
import numpy as np

# Get predictions
model.eval()
predictions = []
targets = []

with torch.no_grad():
    for features, target, _ in test_loader:
        pred = model(features.to(device)).cpu().numpy()
        predictions.append(pred)
        targets.append(target.numpy())

predictions = np.concatenate(predictions)
targets = np.concatenate(targets)

# Visualize
fig = plot_prediction_scatter(
    predictions,
    targets,
    output_path='predictions.png',
    title='Test Set Predictions',
    figsize=(8, 8)
)
```

**Parameters:**
- `predictions`: Predicted values (will be flattened)
- `targets`: Ground truth values (will be flattened)
- `output_path`: Where to save figure (optional)
- `title`: Plot title
- `figsize`: Figure size (width, height)

**Returns:** matplotlib Figure object

**What it shows:**
- 2D histogram with density coloring
- Perfect prediction line (y=x)
- Statistics: Pearson correlation, MSE, MAE
- Automatically samples to 10K points if too many

### 5. `plot_activity_map_comparison()`

**Purpose:** Visual comparison of predicted vs true activity maps.

**Usage:**
```python
from src.olfactory_modeling.visualization import plot_activity_map_comparison

fig = plot_activity_map_comparison(
    predictions,  # (N, H, W) array
    targets,      # (N, H, W) array
    n_samples=4,  # Show 4 random samples
    output_path='activity_maps.png',
    figsize=(16, 12)
)
```

**Parameters:**
- `predictions`: Predicted activity maps (N, H, W)
- `targets`: Ground truth activity maps (N, H, W)
- `n_samples`: Number of samples to show
- `output_path`: Where to save figure (optional)
- `figsize`: Figure size (width, height)

**Returns:** matplotlib Figure object

**What it shows:**
- Side-by-side comparison for each sample:
  - Ground truth (left)
  - Prediction (middle)
  - Difference map (right)
- Per-sample correlation and MSE
- Color-coded difference (red = overprediction, blue = underprediction)

### 6. `create_training_report()`

**Purpose:** Comprehensive training summary in one figure.

**Usage:**
```python
from src.olfactory_modeling.visualization import create_training_report

fig = create_training_report(
    metrics,
    predictions=predictions,  # Optional
    targets=targets,          # Optional
    output_path='full_report.png',
    figsize=(16, 12)
)
```

**Parameters:**
- `metrics_dict`: Training metrics from `train_nn()`
- `predictions`: Optional predictions for scatter plot
- `targets`: Optional ground truth for scatter plot
- `output_path`: Where to save figure (optional)
- `figsize`: Figure size (width, height)

**Returns:** matplotlib Figure object

**What it shows:**
- Training curves (loss, correlation, R²)
- Summary statistics table
- Prediction scatter (if provided)
- Residual plot (if provided)
- Sample activity map comparisons (if predictions are 2D)

### 7. `plot_feature_importance()`

**Purpose:** Visualize feature importance based on first-layer weights of trained neural network.

Ranks molecular descriptors by importance score (mean absolute magnitude of first-layer weights), showing which features the model relies on most heavily to predict olfactory bulb activation patterns.

**Usage:**
```python
from src.olfactory_modeling.visualization import plot_feature_importance
import pandas as pd
import torch

# Load feature names
features_df = pd.read_csv('data/02_processed/cleaned_data.csv')
feature_names = [col for col in features_df.columns if col != 'CID']

# Load trained model
from src.olfactory_modeling.models.baseline_nn import MoleculeToActivityMapMLP
model = MoleculeToActivityMapMLP(input_dim=268, output_shape=(79, 43))
model.load_state_dict(torch.load('experiments/best_model/best_model.pth'))

# Plot feature importance
fig = plot_feature_importance(
    model=model,
    feature_names=feature_names,
    top_n=20,
    output_path='feature_importance.png',
    figsize=(10, 8),
    color='#2E86AB',
    title='Top 20 Molecular Descriptors Ranked by Importance'
)
```

**Parameters:**
- `model`: Trained PyTorch model (nn.Module)
- `feature_names`: List of feature/descriptor names (optional, auto-generates if None)
- `top_n`: Number of top features to display (default: 20)
- `output_path`: Where to save figure (optional)
- `figsize`: Figure size as (width, height) tuple
- `color`: Bar color (default: '#2E86AB' - publication blue)
- `title`: Custom title (optional, auto-generates if None)

**Returns:** matplotlib Figure object

**What it shows:**
- Horizontal bar chart of top N features ranked by importance
- Importance scores displayed on bars
- Summary statistics printed to console:
  - Total number of features
  - Top 5 most important features with scores

**Key Details:**
- Importance computed as mean absolute weight magnitude across all output neurons in first layer
- Works with MLP architectures where first layer is `model.network[0]`
- Also supports encoder-decoder architectures (`model.encoder[0]`)
- Publication-quality formatting (300 DPI)
- Suitable for thesis figures and papers

**Example Output:**
```
============================================================
Feature Importance Analysis Summary
============================================================
Total features: 268
Top 20 features shown

Top 5 most important features:
  1. NtCH                           0.033904
  2. n10FaRing                      0.032740
  3. nARing                         0.032691
  4. PEOE_VSA9                      0.032689
  5. piPC8                          0.032654
============================================================
```

**Use Cases:**
- Understanding which molecular features drive predictions
- Feature interpretation and model explainability
- Identifying key axes of chemical variation
- Thesis/paper figures showing model behavior
- Comparing feature importance across different model architectures

---

## Examples

### Example 1: Basic Training with Visualization

```python
import torch
from src.olfactory_modeling.data.activity_map_dataset import get_dataloaders
from src.olfactory_modeling.models.baseline_nn import get_model
from src.olfactory_modeling.pipeline.train_nn import train_nn

# Get data
train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

# Create model
model = get_model('mlp', input_dim=268, output_shape=(79, 43), dropout=0.35)

# Train (auto-generates training_curves.png)
metrics = train_nn(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    output_dir="experiments/baseline_mlp",
    num_epochs=100,
    learning_rate=0.005,
)

print(f"✓ Training curves saved to experiments/baseline_mlp/training_curves.png")
```

### Example 2: K-Fold CV with Custom Visualization

```python
from src.olfactory_modeling.pipeline.train_nn import train_nn_kfold
from src.olfactory_modeling.visualization import plot_cv_results

# Run K-fold CV (auto-generates cv_analysis.png)
cv_results = train_nn_kfold(
    model_factory=lambda: get_model('mlp', input_dim=268, output_shape=(79, 43)),
    dataset=dataset,
    output_dir="experiments/cv_5fold",
    n_splits=5,
    num_epochs=100,
)

# Can also generate custom visualization
fig = plot_cv_results(
    'experiments/cv_5fold/cv_results.json',
    output_path='custom_cv_plot.png',
    figsize=(20, 12)  # Larger figure
)
```

### Example 3: Grid Search with Results Analysis

```python
from src.olfactory_modeling.pipeline.train_nn import grid_search
from src.olfactory_modeling.visualization import plot_grid_search_results

# Define model factory with hyperparameters
def model_factory_template(dropout=0.35, hidden_dims=[512, 256, 128]):
    return MoleculeToActivityMapMLP(
        input_dim=268,
        output_shape=(79, 43),
        dropout=dropout,
        hidden_dims=hidden_dims
    )

# Run grid search (auto-generates grid_search_analysis.png)
grid_results = grid_search(
    model_factory_template=model_factory_template,
    dataset=dataset,
    param_grid={
        'dropout': [0.2, 0.3, 0.35, 0.4, 0.5],
        'learning_rate': [0.001, 0.003, 0.005, 0.007, 0.01],
    },
    output_dir="experiments/dropout_lr_search",
    use_kfold=True,
    n_splits=5,
)

print(f"Best parameters: {grid_results['best_params']}")
print(f"Best score: {grid_results['best_score']:.4f}")
```

### Example 4: Complete Analysis Pipeline

```python
import torch
import numpy as np
from src.olfactory_modeling.visualization import (
    plot_prediction_scatter,
    plot_activity_map_comparison,
    create_training_report,
)

# After training...
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for features, targets, _ in test_loader:
        features = features.to(device)
        predictions = model(features).cpu().numpy()
        all_preds.append(predictions)
        all_targets.append(targets.numpy())

all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)

# Generate all visualizations
plot_prediction_scatter(
    all_preds, all_targets,
    output_path='experiments/analysis/predictions.png'
)

plot_activity_map_comparison(
    all_preds, all_targets,
    n_samples=6,
    output_path='experiments/analysis/activity_maps.png'
)

create_training_report(
    metrics,
    predictions=all_preds,
    targets=all_targets,
    output_path='experiments/analysis/full_report.png'
)

print("✓ Complete analysis saved to experiments/analysis/")
```

### Example 5: Feature Importance Analysis

```python
import pandas as pd
import torch
from src.olfactory_modeling.visualization import plot_feature_importance
from src.olfactory_modeling.models.baseline_nn import MoleculeToActivityMapMLP

# Load feature names from processed data
features_df = pd.read_csv('data/02_processed/cleaned_data.csv')
feature_names = [col for col in features_df.columns if col != 'CID']

print(f"Loaded {len(feature_names)} feature names")

# Load trained model
model = MoleculeToActivityMapMLP(input_dim=268, output_shape=(79, 43))
model.load_state_dict(torch.load('experiments/best_model/best_model.pth'))
model.eval()

# Generate feature importance visualization
fig = plot_feature_importance(
    model=model,
    feature_names=feature_names,
    top_n=20,
    output_path='experiments/feature_importance_top20.png',
    figsize=(10, 8),
    title='Top 20 Molecular Descriptors Ranked by Importance\n' + 
          'Based on First-Layer Weight Magnitudes'
)

# Also generate extended visualization with top 50 features
fig_extended = plot_feature_importance(
    model=model,
    feature_names=feature_names,
    top_n=50,
    output_path='experiments/feature_importance_top50.png',
    figsize=(12, 16),
    color='#A23B72'  # Different color
)

print("✓ Feature importance analysis complete!")
print("  - Top 20 features: experiments/feature_importance_top20.png")
print("  - Top 50 features: experiments/feature_importance_top50.png")
```

---

## Customization

### Change Figure Size

```python
fig = plot_training_curves(
    metrics,
    figsize=(20, 8)  # Wider figure
)
```

### Disable R² Subplot

```python
fig = plot_training_curves(
    metrics,
    show_r2=False  # Only loss and correlation
)
```

### Show More Grid Search Results

```python
fig = plot_grid_search_results(
    'grid_search_results.json',
    top_n=20  # Show top 20 instead of default 10
)
```

### Control Number of Activity Map Samples

```python
fig = plot_activity_map_comparison(
    predictions, targets,
    n_samples=8  # Show 8 samples instead of default 4
)
```

### Access Figure for Further Customization

```python
fig = plot_training_curves(metrics)

# Customize axes
axes = fig.get_axes()
axes[0].set_ylim(0, 0.5)  # Set y-axis limit for first subplot

# Add annotation
axes[1].text(50, 0.5, 'Important event', fontsize=12, color='red')

# Save with custom DPI
fig.savefig('custom_plot.png', dpi=600, bbox_inches='tight')
```

---

## Best Practices

### 1. Always Use Absolute Paths

```python
from pathlib import Path

output_dir = Path('experiments/my_model').absolute()
fig = plot_training_curves(metrics, output_path=output_dir / 'curves.png')
```

### 2. Check for Generated Files

```python
output_path = 'experiments/my_model/training_curves.png'
if Path(output_path).exists():
    print(f"✓ Visualization saved to {output_path}")
```

### 3. Use Consistent Output Directories

```bash
experiments/
  ├── baseline_mlp/
  │   ├── training_curves.png
  │   ├── best_model.pth
  │   └── metrics.json
  ├── baseline_cnn/
  │   ├── training_curves.png
  │   ├── best_model.pth
  │   └── metrics.json
  └── cv_5fold/
      ├── cv_analysis.png
      ├── cv_results.json
      └── fold_*/
```

### 4. Generate Analysis After Training

```python
# 1. Train
metrics = train_nn(...)

# 2. Evaluate
predictions, targets = evaluate_model(model, test_loader)

# 3. Visualize everything
create_training_report(metrics, predictions, targets, output_path='report.png')
plot_activity_map_comparison(predictions, targets, output_path='maps.png')
```

### 5. Use Visualizations in Notebooks

```python
# In Jupyter notebook
from IPython.display import Image, display

metrics = train_nn(...)
display(Image('experiments/my_model/training_curves.png'))
```

### 6. Save High-Resolution for Papers

All visualizations are saved at 300 DPI by default, perfect for papers. For even higher resolution:

```python
fig = plot_training_curves(metrics)
fig.savefig('paper_figure.png', dpi=600, bbox_inches='tight')
```

### 7. Combine with Profiling

```python
# Profile training
python scripts/profile_performance.py --model mlp --profile-epochs 5

# Train with visualization
python scripts/train_baseline_nn.py --model mlp --epochs 100

# Now you have both performance metrics and training visualizations!
```

---

## Troubleshooting

### Issue: "No module named 'tensorboard'"

**Solution:**
```bash
conda install tensorboard
```

### Issue: Figures not showing in terminal

**Solution:** The visualization module automatically uses a non-interactive backend (`Agg`). Figures are saved to files but won't display. This is by design for headless environments and scripts.

### Issue: "RuntimeWarning: overflow encountered in matmul"

**Solution:** This is a harmless warning from scipy when computing correlations. It doesn't affect the visualization quality. Can be suppressed:
```python
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
```

### Issue: Out of memory when plotting many samples

**Solution:** Reduce the number of samples:
```python
fig = plot_activity_map_comparison(
    predictions, targets,
    n_samples=3  # Reduce from default 4
)
```

### Issue: Plot looks crowded

**Solution:** Increase figure size:
```python
fig = plot_grid_search_results(
    'results.json',
    figsize=(20, 15)  # Larger figure
)
```

---

## Summary

**Automatic Visualizations:**
- ✅ `train_nn()` → training_curves.png
- ✅ `train_nn_kfold()` → cv_analysis.png
- ✅ `grid_search()` → grid_search_analysis.png

**Standalone Functions:**
- ✅ `plot_training_curves()` - Training progress
- ✅ `plot_cv_results()` - Cross-validation analysis
- ✅ `plot_grid_search_results()` - Hyperparameter comparison
- ✅ `plot_prediction_scatter()` - Prediction vs truth
- ✅ `plot_activity_map_comparison()` - Visual map comparison
- ✅ `create_training_report()` - Comprehensive summary

**Features:**
- 📊 Publication-quality (300 DPI)
- 🎨 Seaborn styling
- 📐 Configurable layouts
- 🔍 Detailed statistics
- 💾 Automatic saving
- 🎯 Thesis-ready

**Next Steps:**
1. Train your model with auto-visualization
2. Generate custom plots for specific analyses
3. Include visualizations in your thesis/papers
4. Iterate on hyperparameters using visual feedback

Happy visualizing! 🎨
