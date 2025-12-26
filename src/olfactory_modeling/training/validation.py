"""Parameter validation for neural network training.

This module provides validation functions to prevent user errors
with training hyperparameters.
"""


def validate_training_params(
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> None:
    """Validate training parameters to prevent user error.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization parameter
        
    Raises:
        ValueError: If any parameter is invalid
    """
    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be > 0, got {num_epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
