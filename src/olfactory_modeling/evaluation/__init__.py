"""Cross-validation utilities for model evaluation.

This package provides functions for K-fold cross-validation,
fold orchestration, and metric aggregation across folds.
"""

from .cross_validation import aggregate_cv_metrics
from .kfold_runner import run_kfold_training, log_kfold_summary

__all__ = [
    "aggregate_cv_metrics",
    "run_kfold_training",
    "log_kfold_summary",
]
