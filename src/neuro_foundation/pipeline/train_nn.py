"""Neural network training pipeline for activity map prediction.

Follows the same pattern as train_linear.py but for neural networks.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from tqdm import tqdm


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
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    verbose: bool = True,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    all_metrics = {
        'mse': [],
        'mae': [],
        'correlation': [],
        'r2': [],
    }
    
    iterator = tqdm(dataloader, desc=f"Epoch {epoch} [Train]") if verbose else dataloader
    for features, activity_maps, metadata in iterator:
        features = features.to(device)
        activity_maps = activity_maps.to(device)
        
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, activity_maps)
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            batch_metrics = compute_metrics(predictions, activity_maps)
        
        total_loss += loss.item()
        for key, value in batch_metrics.items():
            all_metrics[key].append(value)
        
        if verbose and isinstance(iterator, tqdm):
            iterator.set_postfix({
                'loss': f"{loss.item():.4f}",
                'corr': f"{batch_metrics['correlation']:.3f}",
            })
    
    avg_metrics = {
        'loss': total_loss / len(dataloader),
        'mse': np.mean(all_metrics['mse']),
        'mae': np.mean(all_metrics['mae']),
        'correlation': np.mean(all_metrics['correlation']),
        'r2': np.mean(all_metrics['r2']),
    }
    
    return avg_metrics


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    verbose: bool = True,
) -> Dict[str, float]:
    """Validate model for one epoch."""
    model.eval()
    
    total_loss = 0.0
    all_metrics = {
        'mse': [],
        'mae': [],
        'correlation': [],
        'r2': [],
    }
    
    iterator = tqdm(dataloader, desc=f"Epoch {epoch} [Val]") if verbose else dataloader
    for features, activity_maps, metadata in iterator:
        features = features.to(device)
        activity_maps = activity_maps.to(device)
        
        predictions = model(features)
        loss = criterion(predictions, activity_maps)
        
        batch_metrics = compute_metrics(predictions, activity_maps)
        
        total_loss += loss.item()
        for key, value in batch_metrics.items():
            all_metrics[key].append(value)
        
        if verbose and isinstance(iterator, tqdm):
            iterator.set_postfix({
                'loss': f"{loss.item():.4f}",
                'corr': f"{batch_metrics['correlation']:.3f}",
            })
    
    avg_metrics = {
        'loss': total_loss / len(dataloader),
        'mse': np.mean(all_metrics['mse']),
        'mae': np.mean(all_metrics['mae']),
        'correlation': np.mean(all_metrics['correlation']),
        'r2': np.mean(all_metrics['r2']),
    }
    
    return avg_metrics


def train_nn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: str,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Train neural network following the same pattern as train_linear_regression.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        output_dir: Directory to save checkpoints and logs
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on (auto-detected if None)
        verbose: Whether to print progress
        
    Returns:
        Dictionary of final metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    model = model.to(device)
    
    # Setup training components
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Tensorboard logging
    writer = SummaryWriter(os.path.join(output_dir, 'logs'))
    
    # Training loop
    best_val_loss = float('inf')
    best_metrics = {}
    
    if verbose:
        print(f"\nTraining on {device}")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning rate: {learning_rate}\n")
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch, verbose)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch, verbose)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Log to tensorboard
        for split, metrics in [('train', train_metrics), ('val', val_metrics)]:
            for metric_name, value in metrics.items():
                writer.add_scalar(f'{split}/{metric_name}', value, epoch)
        
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        if verbose:
            print(f"Epoch {epoch}/{num_epochs}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Corr: {train_metrics['correlation']:.3f}, R²: {train_metrics['r2']:.3f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Corr: {val_metrics['correlation']:.3f}, R²: {val_metrics['r2']:.3f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_metrics = val_metrics.copy()
            best_metrics['epoch'] = epoch
            
            checkpoint_path = os.path.join(output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, checkpoint_path)
            
            if verbose:
                print(f"  ✓ Saved best model (val_loss={val_metrics['loss']:.4f})")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, checkpoint_path)
    
    writer.close()
    
    # Save final metrics (following train_linear.py pattern)
    metrics_dict = {
        'best_val_loss': best_val_loss,
        'best_val_correlation': best_metrics.get('correlation', 0.0),
        'best_val_r2': best_metrics.get('r2', 0.0),
        'best_val_mae': best_metrics.get('mae', 0.0),
        'best_epoch': best_metrics.get('epoch', 0),
        'n_train': len(train_loader.dataset),
        'n_val': len(val_loader.dataset),
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
    }
    
    pd.Series(metrics_dict).to_json(os.path.join(output_dir, 'metrics.json'))
    
    if verbose:
        print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    
    return metrics_dict
