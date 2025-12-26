"""I/O utilities for neural network training.

This module provides file I/O and checkpoint management functions
for training neural networks.
"""
from typing import Dict, Optional, Any, Callable
import os
import json

import torch

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    checkpoint_path: str,
    epoch: int,
    model_state_dict: Dict,
    optimizer_state_dict: Dict,
    metrics: Dict[str, float],
    verbose: bool = True,
) -> None:
    """Save model checkpoint to disk with error handling.
    
    Args:
        checkpoint_path: Path to save checkpoint
        epoch: Current epoch number
        model_state_dict: Model state dictionary
        optimizer_state_dict: Optimizer state dictionary
        metrics: Dictionary of metrics to save
        verbose: Whether to log the save operation
    """
    try:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'metrics': metrics,
        }, checkpoint_path)
        if verbose:
            logger.info(f"✓ Saved checkpoint to {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {checkpoint_path}: {e}", exc_info=True)
        raise


def generate_visualization_safe(
    viz_function: Callable,
    *args,
    verbose: bool = True,
    **kwargs
) -> None:
    """Safely generate visualization with error handling.
    
    Args:
        viz_function: Visualization function to call
        *args: Positional arguments for viz_function
        verbose: Whether to log success/failure
        **kwargs: Keyword arguments for viz_function
    """
    try:
        viz_function(*args, **kwargs)
        if verbose:
            logger.info("✓ Saved visualization")
    except ImportError as e:
        if verbose:
            logger.warning(f"Could not import visualization module: {e}")
    except Exception as e:
        if verbose:
            logger.error(f"Could not generate visualization: {e}", exc_info=True)


def save_json_safe(data: Dict, filepath: str, verbose: bool = True) -> None:
    """Safely save JSON data with error handling.
    
    Args:
        data: Dictionary to save as JSON
        filepath: Path to save JSON file
        verbose: Whether to log operations
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        if verbose:
            logger.debug(f"Saved JSON to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}", exc_info=True)
        raise
