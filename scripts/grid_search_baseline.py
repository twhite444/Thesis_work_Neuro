#!/usr/bin/env python3
"""Grid search for baseline neural network hyperparameters.

Flexible grid search that can explore any combination of model and training hyperparameters.

Usage:
    # Search dropout and learning rate with 5-fold CV
    python scripts/grid_search_baseline.py --model mlp \
        --param dropout 0.2 0.35 0.5 \
        --param learning_rate 0.001 0.005 0.01 \
        --use-kfold --n-folds 5
    
    # Search architecture and weight decay with single split
    python scripts/grid_search_baseline.py --model mlp \
        --param hidden_dims "[512,256,128]" "[1024,512,256]" \
        --param weight_decay 0.0 1e-5 1e-4 \
        --no-kfold
    
    # Quick search with early stopping
    python scripts/grid_search_baseline.py --model mlp \
        --param dropout 0.3 0.35 0.4 \
        --param learning_rate 0.003 0.005 0.007 \
        --epochs 50 --early-stopping 10
"""

import argparse
import sys
import json
from pathlib import Path

import torch

# Add project root to path
# sys.path.insert(0, str(Path(__file__).parent.parent))  # No longer needed with proper __init__.py

from neuro_foundation.data.activity_map_dataset import MoleculeActivityMapDataset
from neuro_foundation.models.baseline_nn import MoleculeToActivityMapMLP, MoleculeToActivityMapCNN
from neuro_foundation.pipeline.train_nn import grid_search


def parse_value(value_str):
    """Parse command-line value to appropriate Python type."""
    # Try to evaluate as Python literal (for lists, numbers, etc.)
    try:
        return eval(value_str)
    except:
        # Return as string if eval fails
        return value_str


def main():
    parser = argparse.ArgumentParser(
        description="Grid search for baseline neural network hyperparameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search dropout and learning rate with CV:
  python scripts/grid_search_baseline.py --model mlp \\
      --param dropout 0.2 0.35 0.5 \\
      --param learning_rate 0.001 0.005 0.01
  
  # Search architecture variations:
  python scripts/grid_search_baseline.py --model mlp \\
      --param hidden_dims "[512,256,128]" "[1024,512,256]" "[256,128,64]"
  
  # Search regularization:
  python scripts/grid_search_baseline.py --model mlp \\
      --param dropout 0.2 0.35 0.5 \\
      --param weight_decay 0.0 1e-5 1e-4
        """
    )
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True, choices=['mlp', 'cnn'],
                        help='Model architecture (mlp or cnn)')
    
    # Grid search parameters (can specify multiple times)
    parser.add_argument('--param', action='append', nargs='+', metavar=('NAME', 'VALUE'),
                        help='Parameter to search. Format: --param name value1 value2 ...')
    
    # Cross-validation arguments
    parser.add_argument('--use-kfold', action='store_true', default=True,
                        help='Use K-fold cross-validation (default: True)')
    parser.add_argument('--no-kfold', dest='use_kfold', action='store_false',
                        help='Use single train/val split instead of K-fold')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    
    # Training arguments (defaults for parameters not in grid)
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--early-stopping', type=int, default=0,
                        help='Early stopping patience (0 = disabled, recommended: 15-20)')
    
    # Data arguments
    parser.add_argument('--processed-dir', type=str, default='data/02_processed',
                        help='Directory containing processed features and maps')
    parser.add_argument('--output-dir', type=str, default='experiments/grid_search',
                        help='Directory to save grid search results')
    
    # Other arguments
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda', 'mps'],
                        help='Device to train on (auto-detected if not specified)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress bars')
    
    args = parser.parse_args()
    
    # Parse parameter grid
    if not args.param:
        parser.error("Must specify at least one --param to search")
    
    param_grid = {}
    for param_spec in args.param:
        param_name = param_spec[0]
        param_values = [parse_value(v) for v in param_spec[1:]]
        param_grid[param_name] = param_values
    
    print("="*70)
    print("GRID SEARCH CONFIGURATION")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Parameters to search:")
    for name, values in param_grid.items():
        print(f"  {name}: {values}")
    print(f"Evaluation: {args.n_folds}-fold CV" if args.use_kfold else "Evaluation: Single split")
    print(f"Epochs per run: {args.epochs}")
    if args.early_stopping > 0:
        print(f"Early stopping: {args.early_stopping} epochs")
    print("="*70)
    
    # Load full dataset
    print("\nLoading dataset...")
    full_dataset = MoleculeActivityMapDataset(
        processed_dir=args.processed_dir,
        split=None,  # Load all data
        random_seed=args.random_seed,
    )
    
    print(f"Loaded {len(full_dataset)} molecules with {full_dataset.feature_dim} features")
    
    # Create model factory template
    if args.model == 'mlp':
        def model_factory_template(dropout=0.35, hidden_dims=[512, 256, 128]):
            return MoleculeToActivityMapMLP(
                input_dim=full_dataset.feature_dim,
                output_shape=(79, 43),
                dropout=dropout,
                hidden_dims=hidden_dims,
            )
    else:  # cnn
        def model_factory_template(dropout=0.35, latent_dim=512):
            return MoleculeToActivityMapCNN(
                input_dim=full_dataset.feature_dim,
                output_shape=(79, 43),
                dropout=dropout,
                latent_dim=latent_dim,
            )
    
    # Set device
    device = None
    if args.device:
        device = torch.device(args.device)
    
    # Run grid search
    print("\nStarting grid search...")
    results = grid_search(
        model_factory_template=model_factory_template,
        dataset=full_dataset,
        param_grid=param_grid,
        output_dir=args.output_dir,
        use_kfold=args.use_kfold,
        n_splits=args.n_folds,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.early_stopping,
        random_seed=args.random_seed,
        device=device,
        verbose=not args.quiet,
    )
    
    print("\nGrid search complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"  - grid_search_results.json (detailed results)")
    print(f"  - grid_search_summary.csv (table format)")
    print(f"  - exp_XXX/ (individual experiment outputs)")


if __name__ == '__main__':
    main()
