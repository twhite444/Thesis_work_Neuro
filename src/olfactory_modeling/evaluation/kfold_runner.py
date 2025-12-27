"""K-fold cross-validation runner for neural network training.

Extracted from train_nn.py to separate orchestration logic from the main API.
Handles the fold iteration loop, data loading, and per-fold training coordination.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple, Any, Optional, Callable

import torch
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import KFold

from ..utils.logging_config import get_logger
from ..training.post_training import update_fold_metadata

logger = get_logger(__name__)


def run_kfold_training(
    model_factory: Callable[[], torch.nn.Module],
    dataset: Dataset,
    train_fn: Callable,
    output_dir: str,
    n_splits: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int,
    random_seed: int,
    device: Optional[torch.device],
    verbose: bool,
) -> List[Dict[str, Any]]:
    """Run K-fold cross-validation training loop.
    
    Orchestrates the fold iteration, data splitting, and training for each fold.
    This is the core loop extracted from train_nn_kfold for better modularity.
    
    Args:
        model_factory: Function that returns a fresh model instance
        dataset: Complete dataset (will be split into folds)
        train_fn: Training function to call for each fold (e.g., train_nn)
        output_dir: Directory to save results for each fold
        n_splits: Number of CV folds
        num_epochs: Number of training epochs per fold
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization parameter
        early_stopping_patience: Stop if no improvement for N epochs
        random_seed: Random seed for reproducible fold splits
        device: Device to train on
        verbose: Whether to print progress
        
    Returns:
        List of metrics dictionaries, one per fold
    """
    # Initialize K-fold splitter
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    # Storage for results
    fold_metrics = []
    
    # Print header
    if verbose:
        logger.info("="*70)
        logger.info(f"K-FOLD CROSS-VALIDATION ({n_splits} folds)")
        logger.info("="*70)
        logger.info(f"Total samples: {len(dataset)}")
        logger.info(f"Samples per fold: ~{len(dataset) // n_splits}")
        logger.info(f"Epochs per fold: {num_epochs}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Batch size: {batch_size}")
        if early_stopping_patience > 0:
            logger.info(f"Early stopping: {early_stopping_patience} epochs")
        logger.info("="*70)
    
    # Iterate over folds
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(range(len(dataset))), 1):
        fold_results = run_single_fold(
            fold_idx=fold_idx,
            n_splits=n_splits,
            train_indices=train_indices,
            val_indices=val_indices,
            dataset=dataset,
            model_factory=model_factory,
            train_fn=train_fn,
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            random_seed=random_seed,
            device=device,
            verbose=verbose,
        )
        
        # Store results
        fold_results['fold'] = fold_idx
        fold_metrics.append(fold_results)
        
        # Log fold summary
        if verbose:
            log_fold_summary(fold_idx, fold_results)
    
    return fold_metrics


def run_single_fold(
    fold_idx: int,
    n_splits: int,
    train_indices: List[int],
    val_indices: List[int],
    dataset: Dataset,
    model_factory: Callable[[], torch.nn.Module],
    train_fn: Callable,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int,
    random_seed: int,
    device: Optional[torch.device],
    verbose: bool,
) -> Dict[str, Any]:
    """Run training for a single fold.
    
    Args:
        fold_idx: Current fold number (1-indexed)
        n_splits: Total number of folds
        train_indices: Indices for training data
        val_indices: Indices for validation data
        dataset: Complete dataset
        model_factory: Function that returns a fresh model instance
        train_fn: Training function to call (e.g., train_nn)
        output_dir: Base output directory
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization parameter
        early_stopping_patience: Early stopping patience
        random_seed: Random seed for reproducibility
        device: Device to train on
        verbose: Whether to print progress
        
    Returns:
        Metrics dictionary for this fold
    """
    if verbose:
        logger.info(f"{'='*70}")
        logger.info(f"FOLD {fold_idx}/{n_splits}")
        logger.info(f"{'='*70}")
        logger.info(f"Train samples: {len(train_indices)}")
        logger.info(f"Val samples: {len(val_indices)}")
    
    # Create data loaders for this fold
    train_loader, val_loader = create_fold_loaders(
        dataset, train_indices, val_indices, batch_size
    )
    
    # Create fresh model for this fold
    model = model_factory()
    
    # Create fold output directory
    fold_output_dir = os.path.join(output_dir, f'fold_{fold_idx}')
    
    # Train on this fold
    fold_results = train_fn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=fold_output_dir,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        device=device,
        verbose=verbose,
    )
    
    # Update fold metadata with K-fold split information
    update_fold_metadata(
        fold_output_dir=fold_output_dir,
        dataset=dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        fold_idx=fold_idx,
        n_splits=n_splits,
        random_seed=random_seed,
    )
    
    return fold_results


def create_fold_loaders(
    dataset: Dataset,
    train_indices: List[int],
    val_indices: List[int],
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for a single fold.
    
    Args:
        dataset: Complete dataset
        train_indices: Indices for training data
        val_indices: Indices for validation data
        batch_size: Batch size for both loaders
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create subsets for this fold
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    
    return train_loader, val_loader


def log_fold_summary(fold_idx: int, fold_results: Dict[str, Any]) -> None:
    """Log summary statistics for a completed fold.
    
    Args:
        fold_idx: Fold number (1-indexed)
        fold_results: Metrics dictionary for this fold
    """
    logger.info(f"\nFold {fold_idx} Results:")
    logger.info(f"  Best Val Loss: {fold_results['best_val_loss']:.4f}")
    logger.info(f"  Best Val Correlation: {fold_results['best_val_correlation']:.3f}")
    logger.info(f"  Best Val R²: {fold_results['best_val_r2']:.3f}")
    logger.info(f"  Best Epoch: {fold_results['best_epoch']}")


def log_kfold_summary(
    mean_metrics: Dict[str, float],
    std_metrics: Dict[str, float],
    best_fold: int,
    best_fold_loss: float,
    n_splits: int,
    output_dir: str,
) -> None:
    """Log final K-fold cross-validation summary.
    
    Args:
        mean_metrics: Mean metrics across folds
        std_metrics: Standard deviation of metrics across folds
        best_fold: Fold number with best validation performance
        best_fold_loss: Best validation loss value
        n_splits: Number of folds
        output_dir: Output directory where results were saved
    """
    logger.info("\n" + "="*70)
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info("="*70)
    logger.info(f"\nMean ± Std across {n_splits} folds:")
    logger.info(f"  Val Loss:        {mean_metrics['best_val_loss']:.4f} ± {std_metrics['best_val_loss']:.4f}")
    logger.info(f"  Val Correlation: {mean_metrics['best_val_correlation']:.3f} ± {std_metrics['best_val_correlation']:.3f}")
    logger.info(f"  Val R²:          {mean_metrics['best_val_r2']:.3f} ± {std_metrics['best_val_r2']:.3f}")
    logger.info(f"  Val MAE:         {mean_metrics['best_val_mae']:.4f} ± {std_metrics['best_val_mae']:.4f}")
    logger.info(f"\nBest fold: {best_fold} (val_loss={best_fold_loss:.4f})")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*70)
