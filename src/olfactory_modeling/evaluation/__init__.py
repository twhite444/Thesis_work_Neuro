"""Cross-validation utilities for model evaluation.

This package provides functions for K-fold cross-validation and
metric aggregation across folds.
"""

from .cross_validation import aggregate_cv_metrics

__all__ = [
    "aggregate_cv_metrics",
]
