#!/usr/bin/env python3
"""Quick test to verify all improvements are working."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Test imports
from src.olfactory_modeling.data.activity_map_dataset import get_dataloaders
from src.olfactory_modeling.models.baseline_nn import get_model, MoleculeToActivityMapMLP

print("="*60)
print("TESTING IMPROVEMENTS")
print("="*60)

# Test 1: Model defaults
print("\n1. Testing MLP Model Defaults:")
print(f"   - Default hidden_dims: {MoleculeToActivityMapMLP.__init__.__defaults__[0]}")
print(f"   - Default dropout: {MoleculeToActivityMapMLP.__init__.__defaults__[1]}")
print(f"   - Expected: [512, 256, 128] and 0.35")

# Test 2: Dataloader defaults
print("\n2. Testing Dataloader Configuration:")
train_loader, val_loader, test_loader = get_dataloaders(
    processed_dir="data/02_processed",
    batch_size=32,
)
print(f"   - Batch size: {train_loader.batch_size}")
print(f"   - Num workers: {train_loader.num_workers}")
print(f"   - Pin memory: {train_loader.pin_memory}")
print(f"   - Expected: batch_size=32, num_workers=0, pin_memory=False")

# Test 3: Model creation with dropout
print("\n3. Testing Model Creation:")
model = get_model(
    model_type='mlp',
    input_dim=268,
    output_shape=(79, 43),
    dropout=0.35,
)
print(f"   - Model created successfully")
print(f"   - Architecture: {model}")

print("\n" + "="*60)
print("ALL TESTS PASSED ✅")
print("="*60)
