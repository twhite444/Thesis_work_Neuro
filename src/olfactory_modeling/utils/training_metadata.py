"""Training and split metadata collection utilities."""
from __future__ import annotations
import os
import json
from typing import Dict, Any, Optional, List
import torch
import numpy as np
from ..utils.logging_config import get_logger
logger = get_logger(__name__)

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
    config.update(kwargs)
    return config

def collect_split_info(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
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
    train_indices = fold_indices['train']
    val_indices = fold_indices['val']
    split_info = {
        'fold_number': fold_number,
        'total_folds': n_splits,
        'n_train': len(train_indices),
        'n_val': len(val_indices),
        'random_seed': random_seed,
    }
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

def collect_training_run_metadata(
    model: torch.nn.Module,
    train_loader,  # DataLoader type
    val_loader,  # DataLoader type
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int,
    device: torch.device,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    try:
        from .pipeline_metadata import collect_pipeline_metadata
        from .model_metadata import collect_model_metadata
        use_pca = hasattr(train_loader.dataset, 'use_pca') and train_loader.dataset.use_pca
        processed_dir = getattr(train_loader.dataset, 'processed_dir', 'data/02_processed')
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
            device=device,
            random_seed=random_seed or getattr(train_loader.dataset, 'random_seed', None),
            optimizer_name="Adam",
        )
        split_info = collect_split_info(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=None,
            random_seed=random_seed or getattr(train_loader.dataset, 'random_seed', None),
        )
        return {
            'pipeline': pipeline_metadata,
            'model': model_metadata,
            'training_config': training_config,
            'data_split': split_info,
        }
    except Exception as e:
        logger.warning(f"Could not collect training metadata: {e}", exc_info=True)
        return {}

def collect_kfold_run_metadata(
    model_factory: callable,
    dataset,  # Dataset type
    n_splits: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int,
    random_seed: int,
    device: Optional[torch.device],
    mean_metrics: Dict[str, float],
    std_metrics: Dict[str, float],
    best_fold: int,
) -> Dict[str, Any]:
    try:
        from .pipeline_metadata import collect_pipeline_metadata
        from .model_metadata import collect_model_metadata
        use_pca = hasattr(dataset, 'use_pca') and dataset.use_pca
        processed_dir = getattr(dataset, 'processed_dir', 'data/02_processed')
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
        kfold_split_info = {
            'cv_method': 'KFold',
            'n_splits': n_splits,
            'shuffle': True,
            'random_seed': random_seed,
            'total_samples': len(dataset),
            'samples_per_fold': len(dataset) // n_splits,
        }
        del sample_model
        return {
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
    except Exception as e:
        logger.warning(f"Could not collect K-fold CV metadata: {e}", exc_info=True)
        return {}
