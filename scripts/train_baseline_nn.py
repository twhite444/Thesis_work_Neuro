#!/usr/bin/env python3
"""Train baseline neural network - Unified training script with optional K-fold CV.

Calls train_nn() or train_nn_kfold() from pipeline module.

Usage:
    # Train on raw activity maps (single train/val/test split)
    python scripts/train_baseline_nn.py --model mlp --epochs 100
    python scripts/train_baseline_nn.py --model cnn --lr 0.0001
    
    # Train on PCA-transformed maps (~5x faster, auto-selects optimized architecture)
    python scripts/train_baseline_nn.py --model mlp --use-pca --epochs 100
    
    # K-fold cross-validation (more robust evaluation)
    python scripts/train_baseline_nn.py --model mlp --k-folds 5 --epochs 100
    python scripts/train_baseline_nn.py --model mlp --use-pca --k-folds 5
"""

import argparse
import sys
from pathlib import Path

import torch

from olfactory_modeling.data.activity_map_dataset import get_dataloaders, MoleculeActivityMapDataset
from olfactory_modeling.models.baseline_nn import get_model
from olfactory_modeling.pipeline.train_nn import train_nn, train_nn_kfold


def main():
    parser = argparse.ArgumentParser(description="Train baseline neural network for activity map prediction")
    
    # Model arguments
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn'],
                        help='Model architecture (mlp or cnn), default: mlp')
    
    # Training arguments  
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--k-folds', type=int, default=None,
                        help='Number of folds for K-fold cross-validation (default: None = single split)')
    parser.add_argument('--batch-size', type=int, default=32, 
                        help='Batch size (default: 32, increased for efficiency)')
    parser.add_argument('--lr', type=float, default=5e-3, 
                        help='Learning rate (default: 0.005, following reference paper)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay (L2 regularization, default: 1e-5). Set to 0 for no regularization, try 1e-4 for stronger effect.')
    parser.add_argument('--dropout', type=float, default=0.35,
                        help='Dropout rate (default: 0.35, from reference paper)')
    parser.add_argument('--early-stopping', type=int, default=15,
                        help='Early stopping patience (0 = disabled, try 15-20 for long runs), default: 15')
    
    # Data arguments
    parser.add_argument('--use-pca', action='store_true',
                        help='Use PCA-transformed maps as targets (faster training, ~170x smaller)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducible splits (K-fold or train/val/test)')
    parser.add_argument('--processed-dir', type=str, default='data/02_processed', 
                        help='Directory containing processed features and maps')
    parser.add_argument('--output-dir', type=str, default='experiments/baseline_nn',
                        help='Directory to save outputs')
    
    # Other arguments
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda', 'mps'],
                        help='Device to train on (auto-detected if not specified)')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress bars')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.use_pca and args.model == 'cnn':
        print(f"ERROR: Cannot use CNN model with --use-pca flag!")
        print(f"  CNN expects spatial 2D maps, not 1D PCA components.")
        print(f"  ")
        print(f"  Solution: Use '--model mlp' instead:")
        print(f"    python scripts/train_baseline_nn.py --model mlp --use-pca --epochs {args.epochs}")
        print(f"  ")
        print(f"  Or remove --use-pca to train on spatial maps:")
        print(f"    python scripts/train_baseline_nn.py --model cnn --epochs {args.epochs}")
        sys.exit(1)
    
    # Validate K-fold argument
    if args.k_folds is not None and args.k_folds < 2:
        print(f"ERROR: --k-folds must be at least 2 (got {args.k_folds})")
        sys.exit(1)
    
    # Set device
    device = None
    if args.device:
        device = torch.device(args.device)
    
    # Branch: K-fold cross-validation or single train/val/test split
    if args.k_folds:
        # K-FOLD CROSS-VALIDATION MODE
        print(f"\n{'='*60}")
        print(f"K-FOLD CROSS-VALIDATION MODE ({args.k_folds} folds)")
        print(f"{'='*60}\n")
        
        # Load full dataset (no split - K-fold will handle splitting)
        print(f"Loading pre-processed features and {'PCA-transformed' if args.use_pca else 'raw'} maps...")
        full_dataset = MoleculeActivityMapDataset(
            processed_dir=args.processed_dir,
            split=None,  # Load all data
            random_seed=args.random_seed,
            use_pca=args.use_pca,
        )
        
        print(f"\nDataset info:")
        print(f"  Total samples: {len(full_dataset)}")
        print(f"  Feature dimension: {full_dataset.feature_dim}")
        print(f"  Output dimension: {full_dataset.output_dim}")
        if args.use_pca:
            print(f"    (Using {full_dataset.output_dim} PCA components)")
        else:
            print(f"    (Using spatial maps {full_dataset.output_shape})")
        print(f"  Will be split into {args.k_folds} folds\n")
        
        # Create model factory function
        def model_factory():
            """Factory function to create fresh model instances for each fold."""
            model_kwargs = {'dropout': args.dropout}
            
            # Determine which model to use based on model type and use_pca flag
            if args.use_pca:
                # Use PCA-optimized MLP for PCA components
                actual_model = 'pca_mlp'
                model_kwargs['n_components'] = full_dataset.output_dim
            else:
                # Use regular model for spatial maps
                actual_model = args.model
                model_kwargs['output_shape'] = full_dataset.output_shape
            
            return get_model(
                model_type=actual_model,
                input_dim=full_dataset.feature_dim,
                **model_kwargs,
            )
        
        # Run K-fold cross-validation
        print(f"Starting {args.k_folds}-fold cross-validation...\n")
        cv_results = train_nn_kfold(
            model_factory=model_factory,
            dataset=full_dataset,
            output_dir=args.output_dir,
            n_splits=args.k_folds,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            early_stopping_patience=args.early_stopping,
            random_seed=args.random_seed,
            device=device,
            verbose=not args.quiet,
        )
        
        print(f"\n{'='*60}")
        print("K-FOLD CROSS-VALIDATION COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to: {args.output_dir}")
        print(f"  - cv_results.json (detailed results)")
        print(f"  - fold_N/ (individual fold checkpoints)\n")
        
    else:
        # SINGLE TRAIN/VAL/TEST SPLIT MODE
        print(f"\n{'='*60}")
        print("SINGLE SPLIT MODE (train/val/test)")
        print(f"{'='*60}\n")
        
        # Get dataloaders (using pre-processed data)
        print(f"Loading pre-processed features and {'PCA-transformed' if args.use_pca else 'raw'} maps...")
        train_loader, val_loader, test_loader = get_dataloaders(
            processed_dir=args.processed_dir,
            batch_size=args.batch_size,
            use_pca=args.use_pca,
            random_seed=args.random_seed,
        )
        
        # Get output dimensions from dataset
        output_dim = train_loader.dataset.output_dim
        output_shape = train_loader.dataset.output_shape
        
        print(f"\nDataset info:")
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Val samples: {len(val_loader.dataset)}")
        print(f"  Test samples: {len(test_loader.dataset)}")
        print(f"  Feature dimension: {train_loader.dataset.feature_dim}")
        print(f"  Output dimension: {output_dim}")
        if args.use_pca:
            print(f"    (Using {output_dim} PCA components)")
        else:
            print(f"    (Using spatial maps {output_shape})")
        
        # Create model
        print(f"\nCreating {args.model.upper()} model (dropout={args.dropout})...")
        
        model_kwargs = {'dropout': args.dropout}
        
        # Determine which model to use based on model type and use_pca flag
        if args.use_pca:
            # Use PCA-optimized MLP for PCA components
            actual_model = 'pca_mlp'
            model_kwargs['n_components'] = output_dim
            print(f"  Using PCA-optimized MLP")
            print(f"  Output: {output_dim} PCA components")
        else:
            # Use regular model for spatial maps
            actual_model = args.model
            model_kwargs['output_shape'] = output_shape
            print(f"  Output shape: {output_shape}")
        
        model = get_model(
            model_type=actual_model,
            input_dim=train_loader.dataset.feature_dim,
            **model_kwargs,
        )
        
        # Train using pipeline function
        print(f"\nStarting training...\n")
        metrics = train_nn(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            early_stopping_patience=args.early_stopping,
            device=device,
            verbose=not args.quiet,
        )
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best validation loss: {metrics['best_val_loss']:.4f}")
        print(f"Best validation correlation: {metrics['best_val_correlation']:.3f}")
        print(f"Best validation R²: {metrics['best_val_r2']:.3f}")
        print(f"Outputs saved to: {args.output_dir}\n")


if __name__ == '__main__':
    main()
