"""Training setup utilities for neural networks.

This module provides device detection and training component initialization
functions that are reusable across different model types.
"""
from typing import Optional, Tuple
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


def get_device(device: Optional[torch.device] = None, verbose: bool = True) -> torch.device:
    """Auto-detect or validate training device.
    
    Args:
        device: Explicit device to use, or None for auto-detection
        verbose: Whether to log the selected device
        
    Returns:
        torch.device: The device to use for training
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    if verbose:
        logger.info(f"Using device: {device}")
    
    return device


def setup_training_components(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float = 0.0,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, optim.Optimizer, optim.lr_scheduler._LRScheduler, Optional[SummaryWriter]]:
    """Setup criterion, optimizer, scheduler, and TensorBoard writer.
    
    Args:
        model: PyTorch model
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization parameter
        output_dir: Directory for TensorBoard logs (None = no logging)
        verbose: Whether to log setup information
        
    Returns:
        Tuple of (criterion, optimizer, scheduler, writer)
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    writer = None
    if output_dir is not None:
        try:
            writer = SummaryWriter(os.path.join(output_dir, 'logs'))
            if verbose:
                logger.debug(f"TensorBoard logging to {os.path.join(output_dir, 'logs')}")
        except Exception as e:
            logger.error(f"Failed to create TensorBoard writer: {e}", exc_info=True)
            raise
    
    return criterion, optimizer, scheduler, writer
