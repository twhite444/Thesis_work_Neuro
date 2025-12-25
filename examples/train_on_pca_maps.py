"""Example script to train baseline MLP on PCA-transformed activity maps.

This demonstrates training on PCA components instead of raw spatial maps,
which is faster and requires less memory.

Usage:
    # First, compute PCA if not already done:
    python scripts/run_pca_on_maps.py --n_components 20
    
    # Then train on PCA components:
    python examples/train_on_pca_maps.py
"""

from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from olfactory_modeling.data.activity_map_dataset import get_dataloaders
from olfactory_modeling.models.baseline_nn import get_model
from olfactory_modeling.pipeline.train_nn import train_nn


def main():
    # Configuration
    output_dir = "experiments/pca_mlp_baseline"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Hyperparameters
    n_components = 20  # Should match PCA computation
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-3
    weight_decay = 1e-4
    
    print("="*80)
    print("TRAINING ON PCA-TRANSFORMED ACTIVITY MAPS")
    print("="*80)
    print(f"PCA components: {n_components}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Load data (PCA-transformed)
    print("\nLoading PCA-transformed activity maps...")
    train_loader, val_loader, test_loader = get_dataloaders(
        processed_dir="data/02_processed",
        batch_size=batch_size,
        use_pca=True,  # KEY: Use PCA-transformed maps
    )
    
    # Get input dimension from dataset
    input_dim = train_loader.dataset.feature_dim
    print(f"Input features: {input_dim}")
    print(f"Output dimension: {n_components} PCA components")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model (PCA-specific MLP)
    print("\nCreating PCA-MLP model...")
    model = get_model(
        'pca_mlp',
        input_dim=input_dim,
        n_components=n_components,
        hidden_dims=[256, 128],  # Smaller than full map prediction
        dropout=0.3,
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} (trainable: {n_trainable:,})")
    
    # Train model
    print("\nStarting training...")
    print("-"*80)
    
    results = train_nn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        early_stopping_patience=15,
        device=None,  # Auto-detect
        verbose=True,
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Best validation correlation: {results['best_val_correlation']:.3f}")
    print(f"Best validation R²: {results['best_val_r2']:.3f}")
    print(f"Best epoch: {results['best_epoch']}")
    print(f"\nModel checkpoint saved to: {output_dir}/best_model.pth")
    
    # Test on held-out test set
    print("\nEvaluating on test set...")
    from olfactory_modeling.pipeline.train_nn import validate_epoch
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    
    # Load best checkpoint
    checkpoint = torch.load(Path(output_dir) / "best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_results = validate_epoch(
        model=model,
        val_loader=test_loader,
        device=device,
        verbose=True,
    )
    
    print("\nTest Set Results:")
    print(f"  Loss: {test_results['val_loss']:.4f}")
    print(f"  Correlation: {test_results['val_correlation']:.3f}")
    print(f"  R²: {test_results['val_r2']:.3f}")
    print(f"  MAE: {test_results['val_mae']:.4f}")
    
    print("\n" + "="*80)
    print("✓ EXPERIMENT COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Compare with raw map training:")
    print("     python examples/train_baseline_mlp.py")
    print("  2. Visualize predictions:")
    print("     python examples/visualize_pca_predictions.py")


if __name__ == "__main__":
    main()
