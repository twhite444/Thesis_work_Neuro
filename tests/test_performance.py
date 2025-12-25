#!/usr/bin/env python3
"""Quick performance test - 3 epochs to measure speed."""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from olfactory_modeling.data.activity_map_dataset import get_dataloaders
from olfactory_modeling.models.baseline_nn import get_model
from olfactory_modeling.pipeline.train_nn import train_nn

print("="*60)
print("PERFORMANCE TEST - 3 EPOCHS")
print("="*60)

# Load data
print("\nLoading data...")
train_loader, val_loader, test_loader = get_dataloaders(
    processed_dir="data/02_processed",
    batch_size=32,
)

# Create model
print("Creating MLP model with dropout=0.35...")
model = get_model(
    model_type='mlp',
    input_dim=268,
    output_shape=(79, 43),
    dropout=0.35,
)

# Train with timing
print("\nStarting training...")
start_time = time.time()

metrics = train_nn(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    output_dir="data/03_models/perf_test",
    num_epochs=3,
    learning_rate=0.005,  # New default
    weight_decay=0.0,
    device=torch.device('mps'),
    verbose=True,
)

elapsed = time.time() - start_time

print("\n" + "="*60)
print("PERFORMANCE RESULTS")
print("="*60)
print(f"Total time for 3 epochs: {elapsed:.1f}s")
print(f"Average time per epoch: {elapsed/3:.1f}s")
print(f"Final validation correlation: {metrics['best_val_correlation']:.3f}")
print(f"Final validation R²: {metrics['best_val_r2']:.3f}")
print(f"Best epoch: {metrics['best_epoch']}")
print("="*60)
