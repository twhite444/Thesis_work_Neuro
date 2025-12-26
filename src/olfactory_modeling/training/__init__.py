"""Training utilities for neural network models.

This package provides modular components for training neural networks:
- Metrics computation
- Checkpointing and I/O
- Validation utilities
- Epoch runners
- Training setup (device detection, component initialization)
- Trainer classes
"""

from .metrics import compute_metrics
from .io_utils import save_checkpoint, generate_visualization_safe, save_json_safe
from .validation import validate_training_params
from .epoch_runners import train_epoch, validate_epoch
from .setup import get_device, setup_training_components
from .trainers import Trainer, TrainerConfig

__all__ = [
    "compute_metrics",
    "save_checkpoint",
    "generate_visualization_safe",
    "save_json_safe",
    "validate_training_params",
    "train_epoch",
    "validate_epoch",
    "get_device",
    "setup_training_components",
    "Trainer",
    "TrainerConfig",
]
