"""
Test brain activity training data loading.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from omegaconf import OmegaConf
from neuro_smell.datamodules.olfactory_datamodule import OlfactoryDataModule

# Load base config
config = OmegaConf.create({
    'data': {
        'data_path': 'data/02_processed/selected_features.csv',
        'brain_pca_path': 'data/02_processed/brain_pca_scores.csv',
        'target_type': 'brain_pca',
        'cid_column': 'CID',
        'check_alignment': True,
        'check_nan': True,
        'check_inf': True,
        'batch_size': 32,
        'num_workers': 0,
        'pin_memory': True,
        'train_test_split': {
            'test_size': 0.2,
            'val_size': 0.2,
            'random_state': 42,
            'shuffle': True
        }
    }
})

print("="*60)
print("TESTING BRAIN ACTIVITY DATA LOADING")
print("="*60)

# Test data loading
print('\n1. Creating datamodule...')
datamodule = OlfactoryDataModule(config)

print('\n2. Setting up data...')
datamodule.setup('fit')

info = datamodule.get_data_info()
print(f'\n✅ Data loading successful!')
print(f'\n📊 Dataset Information:')
print(f'   Input dimensions: {info["input_dim"]}')
print(f'   Output dimensions: {info["output_dim"]}')
print(f'   Train samples: {info["train_size"]}')
print(f'   Validation samples: {info["val_size"]}')
print(f'   Test samples: {info["test_size"]}')
print(f'   Total: {info["train_size"] + info["val_size"] + info["test_size"]}')

# Test dataloaders
print('\n3. Testing dataloaders...')
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()

print(f'   Train batches: {len(train_loader)}')
print(f'   Val batches: {len(val_loader)}')
print(f'   Test batches: {len(test_loader)}')

# Test a batch
print('\n4. Testing batch loading...')
x_batch, y_batch = next(iter(train_loader))
print(f'   X batch shape: {x_batch.shape}')
print(f'   y batch shape: {y_batch.shape}')
print(f'   X dtype: {x_batch.dtype}')
print(f'   y dtype: {y_batch.dtype}')

print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
print("\nReady for training! Run:")
print("  python scripts/train.py experiment=brain_activity_baseline")
print("="*60 + "\n")
