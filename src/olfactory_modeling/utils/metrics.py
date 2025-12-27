"""Utility functions for computing metrics and statistics.

This module provides unified implementations of common metrics to avoid
duplication across the codebase. All functions handle both numpy arrays
and PyTorch tensors transparently.
"""

from typing import Tuple, Union

import numpy as np
import torch
from scipy.stats import pearsonr as scipy_pearsonr


def to_numpy(arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert input to numpy array, handling torch.Tensor safely.
    
    Args:
        arr: Input array (numpy or torch.Tensor)
        
    Returns:
        numpy array (on CPU, detached if from torch)
        
    Example:
        >>> tensor = torch.tensor([1, 2, 3])
        >>> array = to_numpy(tensor)
        >>> isinstance(array, np.ndarray)
        True
    """
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    elif isinstance(arr, np.ndarray):
        return arr
    else:
        raise TypeError(f"Expected numpy array or torch.Tensor, got {type(arr)}")


def flatten_arrays(*arrays: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, ...]:
    """Flatten multiple arrays and convert to numpy.
    
    Args:
        *arrays: Variable number of arrays to flatten
        
    Returns:
        Tuple of flattened numpy arrays
        
    Example:
        >>> pred = torch.randn(10, 5, 5)
        >>> target = np.random.randn(10, 5, 5)
        >>> pred_flat, target_flat = flatten_arrays(pred, target)
        >>> pred_flat.shape
        (250,)
    """
    result = []
    for arr in arrays:
        arr_np = to_numpy(arr)
        result.append(arr_np.flatten())
    return tuple(result)


def compute_correlation(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    flatten: bool = True
) -> float:
    """Compute Pearson correlation coefficient between predictions and targets.
    
    This is the SINGLE SOURCE OF TRUTH for correlation calculation across
    the entire codebase. Uses scipy.stats.pearsonr for consistency.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        flatten: Whether to flatten arrays before computing (default: True)
        
    Returns:
        Pearson correlation coefficient (float)
        
    Raises:
        ValueError: If arrays have different shapes or contain NaN/Inf
        
    Example:
        >>> predictions = torch.randn(100)
        >>> targets = predictions + 0.1 * torch.randn(100)
        >>> corr = compute_correlation(predictions, targets)
        >>> 0.8 < corr < 1.0
        True
    """
    # Convert to numpy
    pred_np = to_numpy(predictions)
    target_np = to_numpy(targets)
    
    # Flatten if requested
    if flatten:
        pred_np = pred_np.flatten()
        target_np = target_np.flatten()
    
    # Validate shapes
    if pred_np.shape != target_np.shape:
        raise ValueError(
            f"Shape mismatch: predictions {pred_np.shape} vs targets {target_np.shape}"
        )
    
    # Check for NaN/Inf
    if not np.isfinite(pred_np).all():
        raise ValueError("Predictions contain NaN or Inf values")
    if not np.isfinite(target_np).all():
        raise ValueError("Targets contain NaN or Inf values")
    
    # Compute correlation using scipy
    corr, _ = scipy_pearsonr(pred_np, target_np)
    
    return float(corr)


def compute_mse(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    flatten: bool = True
) -> float:
    """Compute Mean Squared Error.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        flatten: Whether to flatten arrays before computing (default: True)
        
    Returns:
        MSE value (float)
        
    Example:
        >>> predictions = np.array([1, 2, 3])
        >>> targets = np.array([1, 2, 4])
        >>> compute_mse(predictions, targets)
        0.333...
    """
    pred_np = to_numpy(predictions)
    target_np = to_numpy(targets)
    
    if flatten:
        pred_np = pred_np.flatten()
        target_np = target_np.flatten()
    
    mse = np.mean((pred_np - target_np) ** 2)
    return float(mse)


def compute_mae(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    flatten: bool = True
) -> float:
    """Compute Mean Absolute Error.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        flatten: Whether to flatten arrays before computing (default: True)
        
    Returns:
        MAE value (float)
        
    Example:
        >>> predictions = np.array([1, 2, 3])
        >>> targets = np.array([1, 2, 4])
        >>> compute_mae(predictions, targets)
        0.333...
    """
    pred_np = to_numpy(predictions)
    target_np = to_numpy(targets)
    
    if flatten:
        pred_np = pred_np.flatten()
        target_np = target_np.flatten()
    
    mae = np.mean(np.abs(pred_np - target_np))
    return float(mae)


def compute_statistics(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    flatten: bool = True
) -> dict:
    """Compute all common statistics in one pass.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        flatten: Whether to flatten arrays before computing (default: True)
        
    Returns:
        Dictionary with keys: 'correlation', 'mse', 'mae'
        
    Example:
        >>> predictions = torch.randn(100)
        >>> targets = predictions + 0.1 * torch.randn(100)
        >>> stats = compute_statistics(predictions, targets)
        >>> 'correlation' in stats and 'mse' in stats and 'mae' in stats
        True
    """
    return {
        'correlation': compute_correlation(predictions, targets, flatten=flatten),
        'mse': compute_mse(predictions, targets, flatten=flatten),
        'mae': compute_mae(predictions, targets, flatten=flatten),
    }


def sample_for_plotting(
    *arrays: Union[np.ndarray, torch.Tensor],
    max_points: int = 10000,
    random_seed: int = 42
) -> Tuple[np.ndarray, ...]:
    """Sample arrays for plotting if they exceed max_points.
    
    Ensures consistent sampling across all provided arrays using the same
    random indices. Flattens and converts to numpy automatically.
    
    Args:
        *arrays: Variable number of arrays to sample
        max_points: Maximum number of points to keep (default: 10000)
        random_seed: Random seed for reproducibility (default: 42)
        
    Returns:
        Tuple of sampled numpy arrays (flattened)
        
    Example:
        >>> pred = torch.randn(100000)
        >>> target = torch.randn(100000)
        >>> pred_sample, target_sample = sample_for_plotting(pred, target, max_points=1000)
        >>> len(pred_sample)
        1000
    """
    # Flatten all arrays first
    arrays_flat = flatten_arrays(*arrays)
    
    # Check length (all should be same after flattening)
    n_points = len(arrays_flat[0])
    
    if n_points <= max_points:
        # No sampling needed
        return arrays_flat
    
    # Sample with fixed seed for reproducibility
    rng = np.random.RandomState(random_seed)
    indices = rng.choice(n_points, max_points, replace=False)
    
    # Apply same indices to all arrays
    sampled = tuple(arr[indices] for arr in arrays_flat)
    
    return sampled


# Constants for default behavior
DEFAULT_MAX_SCATTER_POINTS = 10000
"""Default maximum number of points for scatter plots.

Rationale: 10,000 points provide sufficient density for visualization while
maintaining reasonable rendering performance. Above this threshold, scatter
plots become computationally expensive and visually cluttered.
"""

DEFAULT_MAX_REPORT_POINTS = 5000
"""Default maximum number of points for report visualizations.

Rationale: Training reports include multiple subplots, so we use a more
conservative limit (5,000) to ensure fast rendering and smaller file sizes.
"""

DEFAULT_RANDOM_SEED = 42
"""Default random seed for reproducible sampling.

Using a fixed seed ensures that visualizations are reproducible across runs,
which is critical for scientific research and debugging.
"""


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute evaluation metrics for activity map prediction during training.
    
    This function computes batch-level metrics during neural network training,
    including MSE, MAE, correlation, and R² score.
    
    Args:
        pred: Predicted activity maps (batch_size, H, W)
        target: Target activity maps (batch_size, H, W)
        
    Returns:
        Dictionary containing:
            - mse: Mean squared error
            - mae: Mean absolute error
            - correlation: Mean Pearson correlation across batch
            - r2: R² score
            
    Example:
        >>> import torch
        >>> pred = torch.randn(32, 79, 43)
        >>> target = torch.randn(32, 79, 43)
        >>> metrics = compute_metrics(pred, target)
        >>> 'mse' in metrics and 'correlation' in metrics
        True
    """
    import torch.nn as nn
    
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
