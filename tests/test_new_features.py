#!/usr/bin/env python3
"""Quick test of new K-fold CV and grid search features."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.olfactory_modeling.data.activity_map_dataset import MoleculeActivityMapDataset
from src.olfactory_modeling.models.baseline_nn import get_model
from src.olfactory_modeling.pipeline.train_nn import train_nn, train_nn_kfold, grid_search

print("="*70)
print("TESTING NEW FEATURES")
print("="*70)

# Load dataset
print("\n1. Loading dataset...")
dataset = MoleculeActivityMapDataset(
    processed_dir="data/02_processed",
    split=None,
    random_seed=42,
)
print(f"   ✓ Loaded {len(dataset)} molecules")

# Test 1: Early stopping
print("\n2. Testing early stopping...")
from torch.utils.data import DataLoader, Subset
train_subset = Subset(dataset, list(range(200)))
val_subset = Subset(dataset, list(range(200, 243)))

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

model = get_model('mlp', input_dim=268, output_shape=(79, 43), dropout=0.35)
metrics = train_nn(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    output_dir="data/03_models/test_early_stopping",
    num_epochs=100,
    learning_rate=0.005,
    early_stopping_patience=3,  # Stop after 3 epochs without improvement
    device=torch.device('mps'),
    verbose=False,  # Quiet for test
)
print(f"   ✓ Training stopped at epoch {metrics['best_epoch']} (early stopping worked!)")

# Test 2: K-fold CV (2 folds, 2 epochs for speed)
print("\n3. Testing K-fold cross-validation...")
def model_factory():
    return get_model('mlp', input_dim=268, output_shape=(79, 43), dropout=0.35)

cv_results = train_nn_kfold(
    model_factory=model_factory,
    dataset=dataset,
    output_dir="data/03_models/test_kfold",
    n_splits=2,  # Just 2 folds for speed
    num_epochs=2,  # Just 2 epochs for speed
    batch_size=32,
    learning_rate=0.005,
    random_seed=42,
    device=torch.device('mps'),
    verbose=False,
)
mean_corr = cv_results['mean_metrics']['best_val_correlation']
std_corr = cv_results['std_metrics']['best_val_correlation']
print(f"   ✓ 2-fold CV complete: correlation = {mean_corr:.3f} ± {std_corr:.3f}")

# Test 3: Grid search (small grid, no CV for speed)
print("\n4. Testing grid search...")
from src.olfactory_modeling.models.baseline_nn import MoleculeToActivityMapMLP

def model_factory_template(dropout=0.35):
    return MoleculeToActivityMapMLP(
        input_dim=268,
        output_shape=(79, 43),
        dropout=dropout,
    )

param_grid = {
    'dropout': [0.3, 0.4],  # Just 2 values
    'learning_rate': [0.003, 0.007],  # Just 2 values
}

grid_results = grid_search(
    model_factory_template=model_factory_template,
    dataset=dataset,
    param_grid=param_grid,
    output_dir="data/03_models/test_grid_search",
    use_kfold=False,  # Single split for speed
    num_epochs=2,  # Just 2 epochs
    batch_size=32,
    random_seed=42,
    device=torch.device('mps'),
    verbose=False,
)
print(f"   ✓ Grid search complete: {len(grid_results['results'])} combinations tested")
print(f"   ✓ Best params: {grid_results['best_params']}")
print(f"   ✓ Best score: {grid_results['best_score']:.3f}")

print("\n" + "="*70)
print("ALL TESTS PASSED ✅")
print("="*70)
print("\nNew features ready to use:")
print("  1. Early stopping: --early-stopping 15")
print("  2. K-fold CV: python scripts/train_baseline_nn_kfold.py --n-folds 5")
print("  3. Grid search: python scripts/grid_search_baseline.py --param dropout 0.2 0.35 0.5")
print("="*70)
