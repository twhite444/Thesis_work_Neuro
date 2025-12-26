"""Training utilities for neural network models.

This package provides modular components for training neural networks:
- Metrics computation
- Checkpointing and I/O
- Validation utilities
- Epoch runners
- Trainer classes
"""

from .metrics import compute_metrics

__all__ = [
    "compute_metrics",
]
