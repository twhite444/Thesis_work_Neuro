#!/usr/bin/env python3
"""Train baseline neural network - Thin CLI wrapper.

Calls train_nn() from pipeline module following the project architecture pattern.

Usage:
    python scripts/train_baseline_nn.py --model mlp --features ecfp --epochs 100
    python scripts/train_baseline_nn.py --model cnn --features rdkit --lr 0.0001
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neuro_foundation.data.activity_map_dataset import get_dataloaders
from src.neuro_foundation.models.baseline_nn import get_model
from src.neuro_foundation.pipeline.train_nn import train_nn


def main():
    parser = argparse.ArgumentParser(description="Train baseline neural network for activity map prediction")
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True, choices=['mlp', 'cnn'],
                        help='Model architecture (mlp or cnn)')
    
    # Training arguments  
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    # Data arguments
    parser.add_argument('--processed-dir', type=str, default='data/02_processed', 
                        help='Directory containing processed features and maps')
    parser.add_argument('--output-dir', type=str, default='experiments/baseline_nn',
                        help='Directory to save outputs')
    
    # Other arguments
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda', 'mps'],
                        help='Device to train on (auto-detected if not specified)')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress bars')
    
    args = parser.parse_args()
    
    # Get dataloaders (using pre-processed data)
    print("Loading pre-processed features and maps...")
    train_loader, val_loader, test_loader = get_dataloaders(
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
    )
    
    # Create model
    print(f"Creating {args.model.upper()} model...")
    model = get_model(
        model_type=args.model,
        input_dim=train_loader.dataset.feature_dim,
        output_shape=(79, 43),
    )
    
    # Set device
    device = None
    if args.device:
        device = torch.device(args.device)
    
    # Train using pipeline function
    metrics = train_nn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        verbose=not args.quiet,
    )
    
    print("\nTraining complete!")
    print(f"Best validation loss: {metrics['best_val_loss']:.4f}")
    print(f"Best validation correlation: {metrics['best_val_correlation']:.3f}")
    print(f"Best validation R²: {metrics['best_val_r2']:.3f}")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
