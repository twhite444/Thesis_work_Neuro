#!/usr/bin/env python3
"""Train baseline neural network to predict activity maps from molecular structure.

Usage:
    # Train MLP model with ECFP features
    python scripts/train_baseline_nn.py --model mlp --features ecfp
    
    # Train CNN decoder with RDKit descriptors
    python scripts/train_baseline_nn.py --model cnn --features rdkit --epochs 100
    
    # Resume from checkpoint
    python scripts/train_baseline_nn.py --model mlp --resume checkpoints/mlp_best.pth
"""

import argparse
import os
import sys
from pathlib import Path
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neuro_foundation.data.activity_map_dataset import get_dataloaders
from src.neuro_foundation.models.baseline_nn import get_model


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute evaluation metrics for activity map prediction.
    
    Args:
        pred: Predicted activity maps (batch_size, H, W)
        target: Target activity maps (batch_size, H, W)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # MSE (primary loss)
    mse = nn.functional.mse_loss(pred, target)
    metrics['mse'] = mse.item()
    
    # MAE
    mae = nn.functional.l1_loss(pred, target)
    metrics['mae'] = mae.item()
    
    # Spatial correlation (average over batch)
    correlations = []
    for p, t in zip(pred, target):
        p_flat = p.flatten()
        t_flat = t.flatten()
        
        # Pearson correlation
        p_mean = p_flat.mean()
        t_mean = t_flat.mean()
        
        numerator = ((p_flat - p_mean) * (t_flat - t_mean)).sum()
        denominator = torch.sqrt(((p_flat - p_mean) ** 2).sum() * ((t_flat - t_mean) ** 2).sum())
        
        if denominator > 0:
            corr = numerator / denominator
            correlations.append(corr.item())
    
    metrics['correlation'] = np.mean(correlations) if correlations else 0.0
    
    # R² score
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)
    metrics['r2'] = r2.item()
    
    return metrics


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch.
    
    Args:
        model: Neural network model
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Dictionary of average metrics over epoch
    """
    model.train()
    
    total_loss = 0.0
    all_metrics = {
        'mse': [],
        'mae': [],
        'correlation': [],
        'r2': [],
    }
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for features, activity_maps, metadata in pbar:
        # Move to device
        features = features.to(device)
        activity_maps = activity_maps.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, activity_maps)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            batch_metrics = compute_metrics(predictions, activity_maps)
        
        # Accumulate
        total_loss += loss.item()
        for key, value in batch_metrics.items():
            all_metrics[key].append(value)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'corr': f"{batch_metrics['correlation']:.3f}",
        })
    
    # Average over batches
    avg_metrics = {
        'loss': total_loss / len(dataloader),
        'mse': np.mean(all_metrics['mse']),
        'mae': np.mean(all_metrics['mae']),
        'correlation': np.mean(all_metrics['correlation']),
        'r2': np.mean(all_metrics['r2']),
    }
    
    return avg_metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Validate model.
    
    Args:
        model: Neural network model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        
    Returns:
        Dictionary of average metrics over validation set
    """
    model.eval()
    
    total_loss = 0.0
    all_metrics = {
        'mse': [],
        'mae': [],
        'correlation': [],
        'r2': [],
    }
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    for features, activity_maps, metadata in pbar:
        # Move to device
        features = features.to(device)
        activity_maps = activity_maps.to(device)
        
        # Forward pass
        predictions = model(features)
        loss = criterion(predictions, activity_maps)
        
        # Compute metrics
        batch_metrics = compute_metrics(predictions, activity_maps)
        
        # Accumulate
        total_loss += loss.item()
        for key, value in batch_metrics.items():
            all_metrics[key].append(value)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'corr': f"{batch_metrics['correlation']:.3f}",
        })
    
    # Average over batches
    avg_metrics = {
        'loss': total_loss / len(dataloader),
        'mse': np.mean(all_metrics['mse']),
        'mae': np.mean(all_metrics['mae']),
        'correlation': np.mean(all_metrics['correlation']),
        'r2': np.mean(all_metrics['r2']),
    }
    
    return avg_metrics


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "runs",
    model_name: str = "baseline",
):
    """Main training loop.
    
    Args:
        model: Neural network model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        device: Device to train on
        num_epochs: Number of epochs to train
        learning_rate: Learning rate
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for tensorboard logs
        model_name: Name for this model run
    """
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Tensorboard writer
    writer = SummaryWriter(os.path.join(log_dir, model_name))
    
    # Training loop
    best_val_loss = float('inf')
    
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*80}\n")
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Log to tensorboard
        for split, metrics in [('train', train_metrics), ('val', val_metrics)]:
            for metric_name, value in metrics.items():
                writer.add_scalar(f'{split}/{metric_name}', value, epoch)
        
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s):")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Corr: {train_metrics['correlation']:.3f}, R²: {train_metrics['r2']:.3f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Corr: {val_metrics['correlation']:.3f}, R²: {val_metrics['r2']:.3f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_metrics': val_metrics,
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_loss={val_metrics['loss']:.4f})")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_metrics': val_metrics,
            }, checkpoint_path)
    
    writer.close()
    print(f"\n{'='*80}")
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Train baseline neural network for activity map prediction")
    parser.add_argument("--model", type=str, choices=['mlp', 'cnn'], default='mlp',
                        help="Model architecture (mlp or cnn)")
    parser.add_argument("--features", type=str, choices=['ecfp', 'rdkit'], default='ecfp',
                        help="Type of molecular features")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=512,
                        help="Latent dimension (for CNN model)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--log-dir", type=str, default="runs",
                        help="Directory for tensorboard logs")
    parser.add_argument("--data-dir", type=str, default="data/01_raw",
                        help="Data directory")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/mps/cpu, auto-detected if not specified)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        feature_type=args.features,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # Get input dimension from first batch
    sample_features, _, _ = next(iter(train_loader))
    input_dim = sample_features.shape[1]
    print(f"Input dimension: {input_dim}")
    
    # Create model
    print(f"Creating {args.model.upper()} model...")
    if args.model == 'mlp':
        model = get_model('mlp', input_dim=input_dim)
    else:  # cnn
        model = get_model('cnn', input_dim=input_dim, latent_dim=args.latent_dim)
    
    model = model.to(device)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Train
    model_name = f"{args.model}_{args.features}"
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        model_name=model_name,
    )


if __name__ == "__main__":
    main()
