"""Comprehensive metadata logging for neural network training runs.

This module collects and aggregates metadata from all pipeline stages:
- Activity maps processing
- PCA transformation
- Feature preprocessing
- Data splits
- Training configuration
- Model architecture

All metadata is saved alongside training metrics for full reproducibility.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import torch

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


def load_json_safe(filepath: str) -> Dict[str, Any]:
    """Safely load JSON file with error handling.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary from JSON file, or empty dict if error
    """
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Metadata file not found: {filepath}")
            return {}
    except Exception as e:
        logger.warning(f"Could not load metadata from {filepath}: {e}")
        return {}


def load_npz_metadata(filepath: str, metadata_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load metadata from NPZ file.
    
    Args:
        filepath: Path to NPZ file
        metadata_keys: Optional list of keys to extract (None = all scalar/small arrays)
        
    Returns:
        Dictionary of metadata values
    """
    metadata = {}
    try:
        if os.path.exists(filepath):
            data = np.load(filepath)
            
            if metadata_keys is None:
                # Auto-detect metadata (small arrays and scalars)
                for key in data.keys():
                    arr = data[key]
                    # Include scalars and small arrays (< 100 elements)
                    if arr.size < 100:
                        if arr.size == 1:
                            metadata[key] = arr.item()
                        else:
                            metadata[key] = arr.tolist()
            else:
                # Extract specified keys
                for key in metadata_keys:
                    if key in data:
                        arr = data[key]
                        if arr.size == 1:
                            metadata[key] = arr.item()
                        else:
                            metadata[key] = arr.tolist()
        else:
            logger.warning(f"NPZ file not found: {filepath}")
    except Exception as e:
        logger.warning(f"Could not load metadata from {filepath}: {e}")
    
    return metadata


def collect_pipeline_metadata(
    processed_dir: str = "data/02_processed",
    use_pca: bool = False,
) -> Dict[str, Any]:
    """Collect metadata from all pipeline stages.
    
    Args:
        processed_dir: Directory containing processed data
        use_pca: Whether PCA transformation was used
        
    Returns:
        Dictionary containing all pipeline metadata
    """
    processed_path = Path(processed_dir)
    metadata = {}
    
    # 1. Activity Maps Processing Metadata
    activity_maps_meta = {}
    
    # Global mask metadata
    global_mask_meta = load_json_safe(str(processed_path / "global_mask_metadata.json"))
    if global_mask_meta:
        activity_maps_meta['global_mask'] = global_mask_meta
    
    # Processed maps metadata (from CSV)
    try:
        import pandas as pd
        maps_meta_path = processed_path / "processed_maps_metadata.csv"
        if maps_meta_path.exists():
            maps_df = pd.read_csv(maps_meta_path)
            activity_maps_meta['n_molecules'] = len(maps_df)
            activity_maps_meta['selection_strategy'] = maps_df['selection_strategy'].iloc[0] if len(maps_df) > 0 else None
            activity_maps_meta['coverage_threshold'] = maps_df['coverage_threshold'].iloc[0] if len(maps_df) > 0 else None
            activity_maps_meta['value_policy'] = maps_df['value_policy'].iloc[0] if len(maps_df) > 0 else None
            activity_maps_meta['map_shape'] = [
                int(maps_df['map_shape_h'].iloc[0]),
                int(maps_df['map_shape_w'].iloc[0])
            ] if len(maps_df) > 0 else None
    except Exception as e:
        logger.warning(f"Could not load processed maps metadata CSV: {e}")
    
    # Map statistics (if exists)
    map_stats_path = processed_path / "map_statistics.json"
    if map_stats_path.exists():
        try:
            with open(map_stats_path, 'r') as f:
                map_stats = json.load(f)
                # Add summary statistics
                if map_stats:
                    activity_maps_meta['n_maps_with_stats'] = len(map_stats)
                    # Get first map's stats as example of structure
                    first_cid = list(map_stats.keys())[0]
                    activity_maps_meta['stats_keys'] = list(map_stats[first_cid].keys())
        except Exception as e:
            logger.warning(f"Could not load map statistics: {e}")
    
    metadata['activity_maps'] = activity_maps_meta
    
    # 2. PCA Metadata (if used)
    if use_pca:
        pca_meta = {}
        
        # Load from NPZ
        pca_npz_path = processed_path / "pca_transformed_maps.npz"
        pca_npz_meta = load_npz_metadata(
            str(pca_npz_path),
            metadata_keys=['n_components', 'n_samples', 'original_shape', 
                          'explained_variance_ratio', 'cumulative_variance', 
                          'total_variance_explained']
        )
        pca_meta.update(pca_npz_meta)
        
        metadata['pca'] = pca_meta
    
    # 3. Feature Preprocessing Metadata
    preprocess_meta = load_json_safe(str(processed_path / "preprocess_metadata.json"))
    if preprocess_meta:
        metadata['preprocessing'] = preprocess_meta
    
    # 4. Feature Selection Metadata
    feature_select_meta = load_json_safe(str(processed_path / "feature_select_meta.json"))
    if feature_select_meta:
        metadata['feature_selection'] = feature_select_meta
    
    # 5. Scaler Statistics
    scaler_stats_path = processed_path / "scaler_stats.json"
    if scaler_stats_path.exists():
        try:
            with open(scaler_stats_path, 'r') as f:
                scaler_stats = json.load(f)
                # Just include summary info, not all feature means/stds
                metadata['scaler'] = {
                    'n_features': len(scaler_stats.get('feature_means', [])),
                    'method': 'StandardScaler',
                }
        except Exception as e:
            logger.warning(f"Could not load scaler stats: {e}")
    
    return metadata


def collect_model_metadata(model: torch.nn.Module) -> Dict[str, Any]:
    """Collect metadata about model architecture.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary of model metadata
    """
    metadata = {}
    
    # Model type
    metadata['model_class'] = model.__class__.__name__
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    metadata['total_parameters'] = total_params
    metadata['trainable_parameters'] = trainable_params
    
    # Layer information
    metadata['n_layers'] = len(list(model.modules()))
    
    # Model architecture string
    metadata['architecture'] = str(model)
    
    return metadata


def collect_training_config(
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int,
    device: torch.device,
    random_seed: Optional[int] = None,
    optimizer_name: str = "Adam",
    **kwargs
) -> Dict[str, Any]:
    """Collect training configuration metadata.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 regularization)
        early_stopping_patience: Early stopping patience (0 = disabled)
        device: Training device
        random_seed: Random seed for reproducibility
        optimizer_name: Name of optimizer
        **kwargs: Additional config parameters
        
    Returns:
        Dictionary of training configuration
    """
    config = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'early_stopping_patience': early_stopping_patience,
        'device': str(device),
        'random_seed': random_seed,
        'optimizer': optimizer_name,
    }
    
    # Add any additional config
    config.update(kwargs)
    
    return config


def collect_split_info(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Collect data split information.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader (optional)
        random_seed: Random seed used for splitting
        
    Returns:
        Dictionary of split information
    """
    split_info = {
        'n_train': len(train_loader.dataset),
        'n_val': len(val_loader.dataset),
        'random_seed': random_seed,
        'train_cids': train_loader.dataset.cids.tolist() if hasattr(train_loader.dataset, 'cids') else None,
        'val_cids': val_loader.dataset.cids.tolist() if hasattr(val_loader.dataset, 'cids') else None,
    }
    
    if test_loader is not None:
        split_info['n_test'] = len(test_loader.dataset)
        if hasattr(test_loader.dataset, 'cids'):
            split_info['test_cids'] = test_loader.dataset.cids.tolist()
    
    return split_info


def collect_kfold_split_info(
    dataset: torch.utils.data.Dataset,
    fold_indices: Dict[str, np.ndarray],
    fold_number: int,
    n_splits: int,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Collect K-fold split information for a specific fold.
    
    Args:
        dataset: Full dataset
        fold_indices: Dictionary with 'train' and 'val' indices for this fold
        fold_number: Current fold number (1-indexed)
        n_splits: Total number of folds
        random_seed: Random seed used for K-fold splitting
        
    Returns:
        Dictionary of K-fold split information
    """
    train_indices = fold_indices['train']
    val_indices = fold_indices['val']
    
    split_info = {
        'fold_number': fold_number,
        'total_folds': n_splits,
        'n_train': len(train_indices),
        'n_val': len(val_indices),
        'random_seed': random_seed,
    }
    
    # Add CIDs if available
    if hasattr(dataset, 'cids'):
        split_info['train_cids'] = dataset.cids[train_indices].tolist()
        split_info['val_cids'] = dataset.cids[val_indices].tolist()
    
    return split_info


def save_comprehensive_metadata(
    output_dir: str,
    pipeline_metadata: Dict[str, Any],
    model_metadata: Dict[str, Any],
    training_config: Dict[str, Any],
    split_info: Dict[str, Any],
    metrics: Dict[str, Any],
    filename: str = "run_metadata.json",
    verbose: bool = True,
) -> None:
    """Save comprehensive metadata to JSON file.
    
    Args:
        output_dir: Output directory
        pipeline_metadata: Metadata from data pipeline
        model_metadata: Metadata about model architecture
        training_config: Training configuration
        split_info: Data split information
        metrics: Training metrics
        filename: Name of metadata file
        verbose: Whether to log save operation
    """
    comprehensive_metadata = {
        'pipeline': pipeline_metadata,
        'model': model_metadata,
        'training_config': training_config,
        'data_split': split_info,
        'metrics': metrics,
    }
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(comprehensive_metadata, f, indent=2)
        
        if verbose:
            logger.info(f"✓ Saved comprehensive metadata to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save comprehensive metadata: {e}", exc_info=True)
        raise
