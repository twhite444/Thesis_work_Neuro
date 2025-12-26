"""Cross-validation metric aggregation.

This module provides functions for aggregating metrics across
K-fold cross-validation folds.
"""
import numpy as np


def aggregate_cv_metrics(fold_metrics: list, metric_names: list) -> tuple:
    """Aggregate metrics across K-fold cross-validation folds.
    
    Args:
        fold_metrics: List of metric dictionaries from each fold
        metric_names: List of metric names to aggregate
        
    Returns:
        Tuple of (mean_metrics, std_metrics) dictionaries
    """
    mean_metrics = {}
    std_metrics = {}
    
    for metric in metric_names:
        values = [fold[metric] for fold in fold_metrics]
        mean_metrics[metric] = np.mean(values)
        std_metrics[metric] = np.std(values)
    
    return mean_metrics, std_metrics
