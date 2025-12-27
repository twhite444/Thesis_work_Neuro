"""Epoch-level training and validation runners.

This module provides functions for running single training and validation epochs.
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from ..utils.metrics import compute_metrics


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
