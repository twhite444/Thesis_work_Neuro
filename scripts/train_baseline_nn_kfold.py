#!/usr/bin/env python3
"""Train baseline neural network with K-fold cross-validation.

Implements the same methodology as the reference paper for robust evaluation.

Usage:
    python scripts/train_baseline_nn_kfold.py --model mlp --n-folds 5 --epochs 100
    python scripts/train_baseline_nn_kfold.py --model cnn --n-folds 3 --epochs 50
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neuro_foundation.data.activity_map_dataset import MoleculeActivityMapDataset
from src.neuro_foundation.models.baseline_nn import get_model
from src.neuro_foundation.pipeline.train_nn import train_nn_kfold


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline neural network with K-fold cross-validation"
    )
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True, choices=['mlp', 'cnn'],
                        help='Model architecture (mlp or cnn)')
    
    # Cross-validation arguments
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of cross-validation folds (default: 5, matching reference paper)')
    
    # Training arguments  
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs per fold')
    parser.add_argument('--batch-size', type=int, default=32, 
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=5e-3, 
                        help='Learning rate (default: 0.005)')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--dropout', type=float, default=0.35,
                        help='Dropout rate (default: 0.35)')
    parser.add_argument('--early-stopping', type=int, default=0,
                        help='Early stopping patience (0 = disabled)')
    
    # Data arguments
    parser.add_argument('--processed-dir', type=str, default='data/02_processed', 
                        help='Directory containing processed features and maps')
    parser.add_argument('--output-dir', type=str, default='experiments/baseline_nn_kfold',
                        help='Directory to save outputs')
    
    # Other arguments
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducible folds')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda', 'mps'],
                        help='Device to train on (auto-detected if not specified)')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress bars')
    
    args = parser.parse_args()
    
    # Load full dataset (no split - K-fold will handle splitting)
    print("Loading pre-processed features and maps...")
    full_dataset = MoleculeActivityMapDataset(
        processed_dir=args.processed_dir,
        split=None,  # Load all data
        random_seed=args.random_seed,
    )
    
    print(f"\nTotal dataset size: {len(full_dataset)} molecules")
    print(f"Feature dimension: {full_dataset.feature_dim}")
    print(f"Will be split into {args.n_folds} folds")
    
    # Create model factory function
    def model_factory():
        """Factory function to create fresh model instances."""
        return get_model(
            model_type=args.model,
            input_dim=full_dataset.feature_dim,
            output_shape=(79, 43),
            dropout=args.dropout,
        )
    
    # Set device
    device = None
    if args.device:
        device = torch.device(args.device)
    
    # Run K-fold cross-validation
    print(f"\nStarting {args.n_folds}-fold cross-validation...")
    cv_results = train_nn_kfold(
        model_factory=model_factory,
        dataset=full_dataset,
        output_dir=args.output_dir,
        n_splits=args.n_folds,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping,
        random_seed=args.random_seed,
        device=device,
        verbose=not args.quiet,
    )
    
    print("\nCross-validation complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"  - cv_results.json (detailed results)")
    print(f"  - fold_N/ (individual fold checkpoints)")


if __name__ == '__main__':
    main()
