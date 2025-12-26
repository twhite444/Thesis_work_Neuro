"""Training utilities for neural network models.

This package provides modular components for training neural networks:
- Metrics computation
- Checkpointing and I/O
- Validation utilities
- Epoch runners
- Trainer classes
"""

from .metrics import compute_metrics
from .io_utils import save_checkpoint, generate_visualization_safe, save_json_safe
from .validation import validate_training_params

__all__ = [
    "compute_metrics",
    "save_checkpoint",
    "generate_visualization_safe",
    "save_json_safe",
    "validate_training_params",
]
