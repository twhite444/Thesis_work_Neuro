"""Metrics computation for neural network training.

This module provides metric calculation functions for evaluating
model predictions during training and validation.
"""
from typing import Dict

import torch
import torch.nn as nn
import numpy as np


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
