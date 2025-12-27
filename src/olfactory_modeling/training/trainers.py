"""Trainer class for neural network training.

This module provides a composition-based Trainer class that encapsulates
the training loop and delegates to helper modules for specific tasks.
"""
from __future__ import annotations

import os
from typing import Dict, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..utils.logging_config import get_logger
from ..utils.metrics import compute_metrics
from .io_utils import save_checkpoint
from .epoch_runners import train_epoch, validate_epoch
from .setup import get_device, setup_training_components

logger = get_logger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for Trainer."""
    output_dir: str
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    early_stopping_patience: int = 0
    device: Optional[torch.device] = None
    verbose: bool = True


class Trainer:
    """Encapsulates training logic using composition.
    
    This class coordinates the training process by delegating to helper
    modules for specific tasks (metrics, checkpointing, epoch running).
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainerConfig,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup device
        self.device = get_device(config.device, verbose=config.verbose)
        self.model = self.model.to(self.device)
        
        # Setup training components
        self.criterion, self.optimizer, self.scheduler, self.writer = setup_training_components(
            model=self.model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            output_dir=config.output_dir,
            verbose=config.verbose,
        )
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_metrics = {}
        self.epochs_without_improvement = 0
        
        # History tracking
        self.train_losses = []
        self.val_losses = []
        self.train_correlations = []
        self.val_correlations = []
        self.train_r2 = []
        self.val_r2 = []
    
    def train(self) -> Dict[str, float]:
        """Execute the main training loop.
        
        Returns:
            Dictionary of final metrics including training history
        """
        if self.config.verbose:
            logger.info(f"Training on {self.device}")
            logger.info(f"Train samples: {len(self.train_loader.dataset)}")
            logger.info(f"Val samples: {len(self.val_loader.dataset)}")
            logger.info(f"Epochs: {self.config.num_epochs}")
            logger.info(f"Learning rate: {self.config.learning_rate}")
            if self.config.early_stopping_patience > 0:
                logger.info(f"Early stopping patience: {self.config.early_stopping_patience}")
        
        for epoch in range(1, self.config.num_epochs + 1):
            # Train and validate
            train_metrics = train_epoch(
                self.model, self.train_loader, self.optimizer,
                self.criterion, self.device, epoch, self.config.verbose
            )
            val_metrics = validate_epoch(
                self.model, self.val_loader, self.criterion,
                self.device, epoch, self.config.verbose
            )
            
            # Update history
            self._update_history(train_metrics, val_metrics)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            # TensorBoard logging
            self._log_to_tensorboard(epoch, train_metrics, val_metrics)
            
            # Log progress
            if self.config.verbose:
                self._log_epoch_progress(epoch, train_metrics, val_metrics)
            
            # Save best model and check early stopping
            if self._save_if_best(epoch, val_metrics):
                if self.config.early_stopping_patience > 0:
                    if self._check_early_stopping():
                        break
            
            # Periodic checkpointing
            if epoch % 10 == 0:
                self._save_periodic_checkpoint(epoch, val_metrics)
        
        self.writer.close()
        
        if self.config.verbose:
            logger.info(f"Training complete! Best val loss: {self.best_val_loss:.4f}")
        
        return self._build_results_dict()
    
    def _update_history(self, train_metrics: Dict, val_metrics: Dict) -> None:
        """Update training history."""
        self.train_losses.append(train_metrics['loss'])
        self.val_losses.append(val_metrics['loss'])
        self.train_correlations.append(train_metrics.get('correlation', 0.0))
        self.val_correlations.append(val_metrics.get('correlation', 0.0))
        self.train_r2.append(train_metrics.get('r2', 0.0))
        self.val_r2.append(val_metrics.get('r2', 0.0))
    
    def _log_to_tensorboard(
        self, epoch: int, train_metrics: Dict, val_metrics: Dict
    ) -> None:
        """Log metrics to TensorBoard."""
        for split, metrics in [('train', train_metrics), ('val', val_metrics)]:
            for metric_name, value in metrics.items():
                self.writer.add_scalar(f'{split}/{metric_name}', value, epoch)
        self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
    
    def _log_epoch_progress(
        self, epoch: int, train_metrics: Dict, val_metrics: Dict
    ) -> None:
        """Log epoch progress to console."""
        logger.info(f"Epoch {epoch}/{self.config.num_epochs}:")
        logger.info(
            f"  Train - Loss: {train_metrics['loss']:.4f}, "
            f"Corr: {train_metrics['correlation']:.3f}, "
            f"R²: {train_metrics['r2']:.3f}"
        )
        logger.info(
            f"  Val   - Loss: {val_metrics['loss']:.4f}, "
            f"Corr: {val_metrics['correlation']:.3f}, "
            f"R²: {val_metrics['r2']:.3f}"
        )
    
    def _save_if_best(self, epoch: int, val_metrics: Dict) -> bool:
        """Save model if it's the best so far.
        
        Returns:
            True if this is a new best model, False otherwise
        """
        if val_metrics['loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['loss']
            self.best_metrics = val_metrics.copy()
            self.best_metrics['epoch'] = epoch
            self.epochs_without_improvement = 0
            
            checkpoint_path = os.path.join(self.config.output_dir, 'best_model.pth')
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                epoch=epoch,
                model_state_dict=self.model.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                metrics=val_metrics,
                verbose=False,
            )
            
            if self.config.verbose:
                logger.info(f"  ✓ Saved best model (val_loss={val_metrics['loss']:.4f})")
            
            return True
        else:
            self.epochs_without_improvement += 1
            return False
    
    def _check_early_stopping(self) -> bool:
        """Check if early stopping criteria are met.
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.epochs_without_improvement >= self.config.early_stopping_patience:
            if self.config.verbose:
                logger.warning(
                    f"Early stopping triggered after {self.config.early_stopping_patience} "
                    f"epochs without improvement"
                )
                logger.info(
                    f"Best validation loss: {self.best_val_loss:.4f} "
                    f"at epoch {self.best_metrics['epoch']}"
                )
            return True
        return False
    
    def _save_periodic_checkpoint(self, epoch: int, val_metrics: Dict) -> None:
        """Save periodic checkpoint."""
        checkpoint_path = os.path.join(self.config.output_dir, f'checkpoint_epoch{epoch}.pth')
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            epoch=epoch,
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            metrics=val_metrics,
            verbose=self.config.verbose,
        )
    
    def _build_results_dict(self) -> Dict[str, float]:
        """Build final results dictionary.
        
        Returns:
            Dictionary containing metrics and training history
        """
        return {
            'best_val_loss': self.best_val_loss,
            'best_val_correlation': self.best_metrics.get('correlation', 0.0),
            'best_val_r2': self.best_metrics.get('r2', 0.0),
            'best_val_mae': self.best_metrics.get('mae', 0.0),
            'best_epoch': self.best_metrics.get('epoch', 0),
            'n_train': len(self.train_loader.dataset),
            'n_val': len(self.val_loader.dataset),
            'num_epochs': self.config.num_epochs,
            'learning_rate': self.config.learning_rate,
            # Training history for visualization
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_correlations': self.train_correlations,
            'val_correlations': self.val_correlations,
            'train_r2': self.train_r2,
            'val_r2': self.val_r2,
        }
