#!/usr/bin/env python3
"""Test visualization features for neural network training."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from src.olfactory_modeling.data.activity_map_dataset import MoleculeActivityMapDataset
from src.olfactory_modeling.models.baseline_nn import get_model
from src.olfactory_modeling.pipeline.train_nn import train_nn, train_nn_kfold, grid_search
from src.olfactory_modeling.visualization import (
    plot_training_curves,
    plot_cv_results,
    plot_grid_search_results,
    plot_prediction_scatter,
    plot_activity_map_comparison,
    create_training_report,
)

print("="*70)
print("TESTING VISUALIZATION FEATURES")
print("="*70)

# Load dataset
print("\n1. Loading dataset...")
dataset = MoleculeActivityMapDataset(
    processed_dir="data/02_processed",
    split=None,
    random_seed=42,
)
print(f"   ✓ Loaded {len(dataset)} molecules")

# Create data loaders
from torch.utils.data import DataLoader, Subset
train_subset = Subset(dataset, list(range(200)))
val_subset = Subset(dataset, list(range(200, 243)))
test_subset = Subset(dataset, list(range(243, 287)))

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

# Test 1: Basic training with visualization
print("\n2. Testing basic training with visualizations...")
model = get_model('mlp', input_dim=268, output_shape=(79, 43), dropout=0.35)

# Train the model
metrics = train_nn(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    output_dir="viz/test_training",
    num_epochs=20,
    learning_rate=0.005,
    early_stopping_patience=5,
    device=torch.device('mps'),
    verbose=True,
)

print(f"   ✓ Training complete! Best epoch: {metrics['best_epoch']}")

# Generate visualizations
print("\n3. Generating standalone visualizations...")

# Training curves
if 'train_losses' in metrics:
    fig = plot_training_curves(
        metrics,
        output_path='viz/test_training/training_curves.png'
    )
    print("   ✓ Training curves saved")

# Get predictions for test set
print("\n4. Generating prediction visualizations...")
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for features, targets, _ in test_loader:
        features = features.to(torch.device('mps'))
        predictions = model(features).cpu().numpy()
        all_preds.append(predictions)
        all_targets.append(targets.numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

# Prediction scatter plot
fig = plot_prediction_scatter(
    all_preds,
    all_targets,
    output_path='viz/test_training/prediction_scatter.png',
    title='Test Set Predictions'
)
print("   ✓ Prediction scatter saved")

# Activity map comparison
fig = plot_activity_map_comparison(
    all_preds,
    all_targets,
    n_samples=4,
    output_path='viz/test_training/activity_map_comparison.png'
)
print("   ✓ Activity map comparison saved")

# Comprehensive training report
if 'train_losses' in metrics:
    fig = create_training_report(
        metrics,
        predictions=all_preds,
        targets=all_targets,
        output_path='viz/test_training/training_report.png'
    )
    print("   ✓ Comprehensive training report saved")

# Test 2: K-fold CV with visualization
print("\n5. Testing K-fold CV with visualizations...")

def model_factory():
    return get_model('mlp', input_dim=268, output_shape=(79, 43), dropout=0.35)

cv_results = train_nn_kfold(
    model_factory=model_factory,
    dataset=dataset,
    output_dir="viz/test_cv",
    n_splits=3,
    num_epochs=10,
    batch_size=32,
    learning_rate=0.005,
    random_seed=42,
    device=torch.device('mps'),
    verbose=False,
)

mean_corr = cv_results['mean_metrics']['best_val_correlation']
std_corr = cv_results['std_metrics']['best_val_correlation']
print(f"   ✓ 3-fold CV complete: correlation = {mean_corr:.3f} ± {std_corr:.3f}")

# Visualize CV results
fig = plot_cv_results(
    'viz/test_cv/cv_results.json',
    output_path='viz/test_cv/cv_analysis.png'
)
print("   ✓ CV analysis plot saved")

# Test 3: Grid search with visualization
print("\n6. Testing grid search with visualizations...")

from src.olfactory_modeling.models.baseline_nn import MoleculeToActivityMapMLP

def model_factory_template(dropout=0.35):
    return MoleculeToActivityMapMLP(
        input_dim=268,
        output_shape=(79, 43),
        dropout=dropout,
    )

param_grid = {
    'dropout': [0.3, 0.35, 0.4],
    'learning_rate': [0.003, 0.005, 0.007],
}

grid_results = grid_search(
    model_factory_template=model_factory_template,
    dataset=dataset,
    param_grid=param_grid,
    output_dir="viz/test_grid_search",
    use_kfold=False,
    num_epochs=10,
    batch_size=32,
    random_seed=42,
    device=torch.device('mps'),
    verbose=False,
)

print(f"   ✓ Grid search complete: {len(grid_results['results'])} combinations tested")
print(f"   ✓ Best params: {grid_results['best_params']}")
print(f"   ✓ Best score: {grid_results['best_score']:.3f}")

# Visualize grid search results
fig = plot_grid_search_results(
    'viz/test_grid_search/grid_search_results.json',
    output_path='viz/test_grid_search/grid_search_analysis.png',
    top_n=9
)
print("   ✓ Grid search analysis plot saved")

print("\n" + "="*70)
print("ALL VISUALIZATION TESTS COMPLETE ✅")
print("="*70)
print("\nGenerated visualizations:")
print("  1. Training curves: viz/test_training/training_curves.png")
print("  2. Prediction scatter: viz/test_training/prediction_scatter.png")
print("  3. Activity maps: viz/test_training/activity_map_comparison.png")
print("  4. Training report: viz/test_training/training_report.png")
print("  5. CV analysis: viz/test_cv/cv_analysis.png")
print("  6. Grid search analysis: viz/test_grid_search/grid_search_analysis.png")
print("="*70)
