"""Post-training operations for saving results and generating visualizations.

This module handles all operations that occur after training completes:
- Saving metrics to JSON
- Saving comprehensive metadata
- Generating training visualizations
- Aggregating and saving K-fold results
"""
from __future__ import annotations

import os
import json
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

from ..utils.logging_config import get_logger
from ..utils.metadata_logger import save_comprehensive_metadata, collect_kfold_split_info
from .io_utils import save_json_safe, generate_visualization_safe

logger = get_logger(__name__)


def save_training_results(
    metrics_dict: Dict[str, Any],
    metadata: Dict[str, Any],
    output_dir: str,
    verbose: bool = True,
) -> None:
    """Save training metrics and metadata.
    
    Args:
        metrics_dict: Dictionary of training metrics
        metadata: Comprehensive metadata dictionary
        output_dir: Directory to save results
        verbose: Whether to log progress
    """
    # Save metrics to JSON
    try:
        pd.Series({k: v for k, v in metrics_dict.items() if not isinstance(v, list)}).to_json(
            os.path.join(output_dir, 'metrics.json')
        )
    except Exception as e:
        logger.error(f"Failed to save metrics.json: {e}", exc_info=True)
    
    # Save comprehensive metadata
    if metadata:
        try:
            save_comprehensive_metadata(
                output_dir=output_dir,
                pipeline_metadata=metadata.get('pipeline', {}),
                model_metadata=metadata.get('model', {}),
                training_config=metadata.get('training_config', {}),
                split_info=metadata.get('data_split', {}),
                metrics={k: v for k, v in metrics_dict.items() if not isinstance(v, list)},
                filename="run_metadata.json",
                verbose=verbose,
            )
        except Exception as e:
            logger.warning(f"Could not save comprehensive metadata: {e}", exc_info=True)


def generate_training_visualization(
    metrics_dict: Dict[str, Any],
    output_dir: str,
    verbose: bool = True,
) -> None:
    """Generate training curves visualization.
    
    Args:
        metrics_dict: Dictionary containing training history
        output_dir: Directory to save visualization
        verbose: Whether to log progress
    """
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


def save_kfold_results(
    fold_metrics: list,
    mean_metrics: Dict[str, float],
    std_metrics: Dict[str, float],
    best_fold: int,
    n_splits: int,
    output_dir: str,
    verbose: bool = True,
) -> None:
    """Save K-fold cross-validation results.
    
    Args:
        fold_metrics: List of metrics for each fold
        mean_metrics: Mean metrics across folds
        std_metrics: Standard deviation of metrics
        best_fold: Best performing fold number
        n_splits: Number of folds
        output_dir: Directory to save results
        verbose: Whether to log progress
    """
    # Prepare JSON-serializable results
    json_results = {
        'mean_metrics': {k: float(v) for k, v in mean_metrics.items()},
        'std_metrics': {k: float(v) for k, v in std_metrics.items()},
        'best_fold': int(best_fold),
        'n_splits': n_splits,
        'fold_results': [
            {k: (float(v) if isinstance(v, (np.floating, float)) 
                 else int(v) if isinstance(v, (np.integer, int)) 
                 else v) 
             for k, v in fold.items()}
            for fold in fold_metrics
        ]
    }
    
    save_json_safe(json_results, os.path.join(output_dir, 'cv_results.json'), verbose=verbose)


def save_kfold_metadata(
    metadata: Dict[str, Any],
    output_dir: str,
    verbose: bool = True,
) -> None:
    """Save comprehensive metadata for K-fold CV run.
    
    Args:
        metadata: Comprehensive CV metadata dictionary
        output_dir: Directory to save metadata
        verbose: Whether to log progress
    """
    try:
        with open(os.path.join(output_dir, 'cv_run_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if verbose:
            logger.info(f"✓ Saved comprehensive CV metadata to {output_dir}/cv_run_metadata.json")
    except Exception as e:
        logger.warning(f"Could not save comprehensive CV metadata: {e}", exc_info=True)


def generate_kfold_visualization(
    output_dir: str,
    verbose: bool = True,
) -> None:
    """Generate K-fold cross-validation visualization.
    
    Args:
        output_dir: Directory containing cv_results.json
        verbose: Whether to log progress
    """
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


def update_fold_metadata(
    fold_output_dir: str,
    dataset: Any,
    train_indices: list,
    val_indices: list,
    fold_idx: int,
    n_splits: int,
    random_seed: int,
) -> None:
    """Update fold-specific metadata with K-fold split information.
    
    Args:
        fold_output_dir: Directory for this fold's results
        dataset: Complete dataset
        train_indices: Training indices for this fold
        val_indices: Validation indices for this fold
        fold_idx: Current fold number
        n_splits: Total number of folds
        random_seed: Random seed for fold splits
    """
    try:
        fold_split_info = collect_kfold_split_info(
            dataset=dataset,
            fold_indices={'train': train_indices, 'val': val_indices},
            fold_number=fold_idx,
            n_splits=n_splits,
            random_seed=random_seed,
        )
        
        # Load existing run_metadata.json and add fold split info
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
