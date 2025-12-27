"""Neural network training pipeline for activity map prediction.

High-level wrapper functions for neural network training following the same
pattern as train_linear.py.

Public API:
    - train_nn: Train with single train/val split
    - train_nn_kfold: Train with K-fold cross-validation
    - grid_search: Systematic hyperparameter optimization (imported from evaluation)

Features:
    - Delegates core training to Trainer class (composition-based design)
    - Automatic device detection (CUDA/MPS/CPU)
    - Comprehensive metadata and result logging
    - Error-resilient visualization generation
    - Parameter validation to prevent user errors
    
Note: Core training logic extracted to training/ and evaluation/ modules for
      better modularity and testability.
"""
from __future__ import annotations

import os
import json
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from ..utils.logging_config import get_logger
from ..utils.metadata_logger import (
    collect_pipeline_metadata,
    collect_model_metadata,
    collect_training_config,
    collect_split_info,
    collect_kfold_split_info,
    save_comprehensive_metadata,
)
from ..training.io_utils import generate_visualization_safe, save_json_safe
from ..training.validation import validate_training_params
from ..training.trainers import Trainer, TrainerConfig
from ..evaluation.cross_validation import aggregate_cv_metrics
from ..evaluation.hyperparameter_search import grid_search

logger = get_logger(__name__)


# ===== Main Training Functions =====


def train_nn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: str,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    early_stopping_patience: int = 0,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Train neural network following the same pattern as train_linear_regression.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        output_dir: Directory to save checkpoints and logs
        num_epochs: Number of training epochs (must be > 0)
        learning_rate: Learning rate for optimizer (must be > 0)
        weight_decay: L2 regularization parameter (must be >= 0)
        early_stopping_patience: Stop if no improvement for N epochs (0 = disabled)
        device: Device to train on (auto-detected if None)
        verbose: Whether to log progress
        
    Returns:
        Dictionary of final metrics including training history
        
    Raises:
        ValueError: If any parameter is invalid
    """
    # Validate parameters
    validate_training_params(
        num_epochs=num_epochs,
        batch_size=train_loader.batch_size if hasattr(train_loader, 'batch_size') else 32,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}", exc_info=True)
        raise
    
    # Create trainer config
    config = TrainerConfig(
        output_dir=output_dir,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        device=device,
        verbose=verbose,
    )
    
    # Create and run trainer
    trainer = Trainer(model, train_loader, val_loader, config)
    metrics_dict = trainer.train()
    
    # Save metrics to JSON with error handling
    try:
        pd.Series({k: v for k, v in metrics_dict.items() if not isinstance(v, list)}).to_json(
            os.path.join(output_dir, 'metrics.json')
        )
    except Exception as e:
        logger.error(f"Failed to save metrics.json: {e}", exc_info=True)
    
    # Collect and save comprehensive metadata
    try:
        # Determine if PCA was used
        use_pca = hasattr(train_loader.dataset, 'use_pca') and train_loader.dataset.use_pca
        processed_dir = getattr(train_loader.dataset, 'processed_dir', 'data/02_processed')
        
        # Collect all metadata
        pipeline_metadata = collect_pipeline_metadata(
            processed_dir=str(processed_dir),
            use_pca=use_pca,
        )
        
        model_metadata = collect_model_metadata(model)
        
        training_config = collect_training_config(
            num_epochs=num_epochs,
            batch_size=train_loader.batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            device=trainer.device,
            random_seed=getattr(train_loader.dataset, 'random_seed', None),
            optimizer_name="Adam",
        )
        
        split_info = collect_split_info(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=None,  # Test loader not provided in train_nn
            random_seed=getattr(train_loader.dataset, 'random_seed', None),
        )
        
        # Save comprehensive metadata
        save_comprehensive_metadata(
            output_dir=output_dir,
            pipeline_metadata=pipeline_metadata,
            model_metadata=model_metadata,
            training_config=training_config,
            split_info=split_info,
            metrics={k: v for k, v in metrics_dict.items() if not isinstance(v, list)},
            filename="run_metadata.json",
            verbose=verbose,
        )
    except Exception as e:
        logger.warning(f"Could not save comprehensive metadata: {e}", exc_info=True)
    
    # Generate visualization with error handling
    try:
        from olfactory_modeling.visualization import plot_training_curves
        generate_visualization_safe(
            plot_training_curves,
            metrics_dict,
            output_path=os.path.join(output_dir, 'training_curves.png'),
            show_r2=True,
            verbose=verbose,
        )
    except ImportError as e:
        if verbose:
            logger.warning(f"Could not import visualization module: {e}")
    
    return metrics_dict


def train_nn_kfold(
    model_factory: callable,
    dataset: torch.utils.data.Dataset,
    output_dir: str,
    n_splits: int = 5,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    early_stopping_patience: int = 0,
    random_seed: int = 42,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train neural network with K-fold cross-validation.
    
    Implements the same methodology as the reference paper for robust evaluation.
    
    Args:
        model_factory: Function that returns a fresh model instance (no args)
        dataset: Complete dataset (will be split into folds)
        output_dir: Directory to save results for each fold
        n_splits: Number of CV folds (default: 5, matching reference paper)
        num_epochs: Number of training epochs per fold
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization parameter
        early_stopping_patience: Stop if no improvement for N epochs (0 = disabled)
        random_seed: Random seed for reproducible fold splits
        device: Device to train on (auto-detected if None)
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing:
        - fold_metrics: List of metrics for each fold
        - mean_metrics: Mean metrics across folds
        - std_metrics: Standard deviation of metrics across folds
        - best_fold: Fold number with best validation performance
        
    Example:
        >>> def model_factory():
        ...     return MoleculeToActivityMapMLP(input_dim=268, output_shape=(79, 43))
        >>> 
        >>> results = train_nn_kfold(
        ...     model_factory=model_factory,
        ...     dataset=full_dataset,
        ...     output_dir='experiments/kfold_cv',
        ...     n_splits=5,
        ...     num_epochs=100
        ... )
        >>> print(f"Mean correlation: {results['mean_metrics']['correlation']:.3f} ± {results['std_metrics']['correlation']:.3f}")
    """
    from sklearn.model_selection import KFold
    from torch.utils.data import Subset, DataLoader
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize K-fold splitter
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    # Storage for results
    fold_metrics = []
    
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
        if verbose:
            logger.info(f"{'='*70}")
            logger.info(f"FOLD {fold_idx}/{n_splits}")
            logger.info(f"{'='*70}")
            logger.info(f"Train samples: {len(train_indices)}")
            logger.info(f"Val samples: {len(val_indices)}")
        
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
        
        # Create fresh model for this fold
        model = model_factory()
        
        # Train on this fold
        fold_output_dir = os.path.join(output_dir, f'fold_{fold_idx}')
        fold_results = train_nn(
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
        
        # Collect and save fold-specific metadata (including split indices)
        try:
            # Note: train_nn already saves comprehensive metadata via run_metadata.json
            # Here we add fold-specific split information
            use_pca = hasattr(dataset, 'use_pca') and dataset.use_pca
            
            fold_split_info = collect_kfold_split_info(
                dataset=dataset,
                fold_indices={'train': train_indices, 'val': val_indices},
                fold_number=fold_idx,
                n_splits=n_splits,
                random_seed=random_seed,
            )
            
            # Load the run_metadata.json and add fold split info
            metadata_path = os.path.join(fold_output_dir, 'run_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    run_metadata = json.load(f)
                
                # Update with K-fold specific information
                run_metadata['data_split'].update(fold_split_info)
                run_metadata['cross_validation'] = {
                    'fold_number': fold_idx,
                    'total_folds': n_splits,
                    'cv_random_seed': random_seed,
                }
                
                # Save updated metadata
                with open(metadata_path, 'w') as f:
                    json.dump(run_metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not update fold metadata: {e}", exc_info=True)
        
        # Store results
        fold_results['fold'] = fold_idx
        fold_metrics.append(fold_results)
        
        if verbose:
            logger.info(f"\nFold {fold_idx} Results:")
            logger.info(f"  Best Val Loss: {fold_results['best_val_loss']:.4f}")
            logger.info(f"  Best Val Correlation: {fold_results['best_val_correlation']:.3f}")
            logger.info(f"  Best Val R²: {fold_results['best_val_r2']:.3f}")
            logger.info(f"  Best Epoch: {fold_results['best_epoch']}")
    
    # Aggregate results across folds using helper
    metric_names = ['best_val_loss', 'best_val_correlation', 'best_val_r2', 'best_val_mae']
    mean_metrics, std_metrics = aggregate_cv_metrics(fold_metrics, metric_names)
    
    # Find best fold (lowest validation loss)
    best_fold_idx = np.argmin([fold['best_val_loss'] for fold in fold_metrics])
    best_fold = fold_metrics[best_fold_idx]['fold']
    
    # Save aggregated results
    cv_results = {
        'fold_metrics': fold_metrics,
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics,
        'best_fold': best_fold,
        'n_splits': n_splits,
        'random_seed': random_seed,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'early_stopping_patience': early_stopping_patience,
    }
    
    # Save to JSON with error handling
    json_results = {
        'mean_metrics': {k: float(v) for k, v in mean_metrics.items()},
        'std_metrics': {k: float(v) for k, v in std_metrics.items()},
        'best_fold': int(best_fold),
        'n_splits': n_splits,
        'fold_results': [
            {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v) 
             for k, v in fold.items()}
            for fold in fold_metrics
        ]
    }
    save_json_safe(json_results, os.path.join(output_dir, 'cv_results.json'), verbose=verbose)
    
    # Save comprehensive metadata for the overall CV run
    try:
        use_pca = hasattr(dataset, 'use_pca') and dataset.use_pca
        processed_dir = getattr(dataset, 'processed_dir', 'data/02_processed')
        
        # Collect pipeline and model metadata (using the first fold's model factory)
        sample_model = model_factory()
        
        pipeline_metadata = collect_pipeline_metadata(
            processed_dir=str(processed_dir),
            use_pca=use_pca,
        )
        
        model_metadata = collect_model_metadata(sample_model)
        
        training_config = collect_training_config(
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            device=device,
            random_seed=random_seed,
            optimizer_name="Adam",
        )
        
        # K-fold specific split info
        kfold_split_info = {
            'cv_method': 'KFold',
            'n_splits': n_splits,
            'shuffle': True,
            'random_seed': random_seed,
            'total_samples': len(dataset),
            'samples_per_fold': len(dataset) // n_splits,
        }
        
        # Save comprehensive metadata for CV run
        comprehensive_cv_metadata = {
            'pipeline': pipeline_metadata,
            'model': model_metadata,
            'training_config': training_config,
            'cross_validation': kfold_split_info,
            'cv_results': {
                'mean_metrics': mean_metrics,
                'std_metrics': std_metrics,
                'best_fold': int(best_fold),
                'n_folds': n_splits,
            },
        }
        
        with open(os.path.join(output_dir, 'cv_run_metadata.json'), 'w') as f:
            json.dump(comprehensive_cv_metadata, f, indent=2)
        
        if verbose:
            logger.info(f"✓ Saved comprehensive CV metadata to {output_dir}/cv_run_metadata.json")
        
        # Clean up sample model
        del sample_model
    except Exception as e:
        logger.warning(f"Could not save comprehensive CV metadata: {e}", exc_info=True)
    
    if verbose:
        logger.info("\n" + "="*70)
        logger.info("CROSS-VALIDATION SUMMARY")
        logger.info("="*70)
        logger.info(f"\nMean ± Std across {n_splits} folds:")
        logger.info(f"  Val Loss:        {mean_metrics['best_val_loss']:.4f} ± {std_metrics['best_val_loss']:.4f}")
        logger.info(f"  Val Correlation: {mean_metrics['best_val_correlation']:.3f} ± {std_metrics['best_val_correlation']:.3f}")
        logger.info(f"  Val R²:          {mean_metrics['best_val_r2']:.3f} ± {std_metrics['best_val_r2']:.3f}")
        logger.info(f"  Val MAE:         {mean_metrics['best_val_mae']:.4f} ± {std_metrics['best_val_mae']:.4f}")
        logger.info(f"\nBest fold: {best_fold} (val_loss={fold_metrics[best_fold_idx]['best_val_loss']:.4f})")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("="*70)
    
    # Generate visualization with error handling
    try:
        from olfactory_modeling.visualization import plot_cv_results
        cv_results_path = os.path.join(output_dir, 'cv_results.json')
        generate_visualization_safe(
            plot_cv_results,
            cv_results_path,
            output_path=os.path.join(output_dir, 'cv_analysis.png'),
            verbose=verbose,
        )
    except ImportError as e:
        if verbose:
            logger.warning(f"Could not import visualization module: {e}")
    
    return cv_results
