"""Neural network training pipeline for activity map prediction.

Follows the same pattern as train_linear.py but for neural networks.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute evaluation metrics for activity map prediction.
    
    Args:
        pred: Predicted activity maps (batch_size, H, W)
        target: Target activity maps (batch_size, H, W)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # MSE (primary loss)
    mse = nn.functional.mse_loss(pred, target)
    metrics['mse'] = mse.item()
    
    # MAE
    mae = nn.functional.l1_loss(pred, target)
    metrics['mae'] = mae.item()
    
    # Spatial correlation (average over batch)
    correlations = []
    for p, t in zip(pred, target):
        p_flat = p.flatten()
        t_flat = t.flatten()
        
        # Pearson correlation
        p_mean = p_flat.mean()
        t_mean = t_flat.mean()
        
        numerator = ((p_flat - p_mean) * (t_flat - t_mean)).sum()
        denominator = torch.sqrt(((p_flat - p_mean) ** 2).sum() * ((t_flat - t_mean) ** 2).sum())
        
        if denominator > 0:
            corr = numerator / denominator
            correlations.append(corr.item())
    
    metrics['correlation'] = np.mean(correlations) if correlations else 0.0
    
    # R² score
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)
    metrics['r2'] = r2.item()
    
    return metrics


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    verbose: bool = True,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    all_metrics = {
        'mse': [],
        'mae': [],
        'correlation': [],
        'r2': [],
    }
    
    iterator = tqdm(dataloader, desc=f"Epoch {epoch} [Train]") if verbose else dataloader
    for features, activity_maps, metadata in iterator:
        features = features.to(device)
        activity_maps = activity_maps.to(device)
        
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, activity_maps)
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            batch_metrics = compute_metrics(predictions, activity_maps)
        
        total_loss += loss.item()
        for key, value in batch_metrics.items():
            all_metrics[key].append(value)
        
        if verbose and isinstance(iterator, tqdm):
            iterator.set_postfix({
                'loss': f"{loss.item():.4f}",
                'corr': f"{batch_metrics['correlation']:.3f}",
            })
    
    avg_metrics = {
        'loss': total_loss / len(dataloader),
        'mse': np.mean(all_metrics['mse']),
        'mae': np.mean(all_metrics['mae']),
        'correlation': np.mean(all_metrics['correlation']),
        'r2': np.mean(all_metrics['r2']),
    }
    
    return avg_metrics


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    verbose: bool = True,
) -> Dict[str, float]:
    """Validate model for one epoch."""
    model.eval()
    
    total_loss = 0.0
    all_metrics = {
        'mse': [],
        'mae': [],
        'correlation': [],
        'r2': [],
    }
    
    iterator = tqdm(dataloader, desc=f"Epoch {epoch} [Val]") if verbose else dataloader
    for features, activity_maps, metadata in iterator:
        features = features.to(device)
        activity_maps = activity_maps.to(device)
        
        predictions = model(features)
        loss = criterion(predictions, activity_maps)
        
        batch_metrics = compute_metrics(predictions, activity_maps)
        
        total_loss += loss.item()
        for key, value in batch_metrics.items():
            all_metrics[key].append(value)
        
        if verbose and isinstance(iterator, tqdm):
            iterator.set_postfix({
                'loss': f"{loss.item():.4f}",
                'corr': f"{batch_metrics['correlation']:.3f}",
            })
    
    avg_metrics = {
        'loss': total_loss / len(dataloader),
        'mse': np.mean(all_metrics['mse']),
        'mae': np.mean(all_metrics['mae']),
        'correlation': np.mean(all_metrics['correlation']),
        'r2': np.mean(all_metrics['r2']),
    }
    
    return avg_metrics


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
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization parameter (0.0 = no regularization)
        early_stopping_patience: Stop if no improvement for N epochs (0 = disabled)
        device: Device to train on (auto-detected if None)
        verbose: Whether to print progress
        
    Returns:
        Dictionary of final metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    model = model.to(device)
    
    # Setup training components
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Tensorboard logging
    writer = SummaryWriter(os.path.join(output_dir, 'logs'))
    
    # Training loop
    best_val_loss = float('inf')
    best_metrics = {}
    epochs_without_improvement = 0
    
    # Track history for visualization
    train_losses = []
    val_losses = []
    train_correlations = []
    val_correlations = []
    train_r2 = []
    val_r2 = []
    
    if verbose:
        print(f"\nTraining on {device}")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning rate: {learning_rate}")
        if early_stopping_patience > 0:
            print(f"Early stopping patience: {early_stopping_patience}")
        print()
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch, verbose)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch, verbose)
        
        # Track history
        train_losses.append(train_metrics['loss'])
        val_losses.append(val_metrics['loss'])
        train_correlations.append(train_metrics.get('correlation', 0.0))
        val_correlations.append(val_metrics.get('correlation', 0.0))
        train_r2.append(train_metrics.get('r2', 0.0))
        val_r2.append(val_metrics.get('r2', 0.0))
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Log to tensorboard
        for split, metrics in [('train', train_metrics), ('val', val_metrics)]:
            for metric_name, value in metrics.items():
                writer.add_scalar(f'{split}/{metric_name}', value, epoch)
        
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        if verbose:
            print(f"Epoch {epoch}/{num_epochs}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Corr: {train_metrics['correlation']:.3f}, R²: {train_metrics['r2']:.3f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Corr: {val_metrics['correlation']:.3f}, R²: {val_metrics['r2']:.3f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_metrics = val_metrics.copy()
            best_metrics['epoch'] = epoch
            epochs_without_improvement = 0
            
            checkpoint_path = os.path.join(output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, checkpoint_path)
            
            if verbose:
                print(f"  ✓ Saved best model (val_loss={val_metrics['loss']:.4f})")
        else:
            epochs_without_improvement += 1
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                if verbose:
                    print(f"\n⚠️  Early stopping triggered after {early_stopping_patience} epochs without improvement")
                    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_metrics['epoch']}")
                break
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, checkpoint_path)
    
    writer.close()
    
    # Save final metrics (following train_linear.py pattern)
    metrics_dict = {
        'best_val_loss': best_val_loss,
        'best_val_correlation': best_metrics.get('correlation', 0.0),
        'best_val_r2': best_metrics.get('r2', 0.0),
        'best_val_mae': best_metrics.get('mae', 0.0),
        'best_epoch': best_metrics.get('epoch', 0),
        'n_train': len(train_loader.dataset),
        'n_val': len(val_loader.dataset),
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        # Add training history for visualization
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_correlations': train_correlations,
        'val_correlations': val_correlations,
        'train_r2': train_r2,
        'val_r2': val_r2,
    }
    
    pd.Series({k: v for k, v in metrics_dict.items() if not isinstance(v, list)}).to_json(
        os.path.join(output_dir, 'metrics.json')
    )
    
    # Generate visualization
    try:
        from src.neuro_foundation.visualization import plot_training_curves
        plot_training_curves(
            metrics_dict, 
            output_path=os.path.join(output_dir, 'training_curves.png'),
            show_r2=True
        )
        if verbose:
            print(f"  ✓ Saved training curves visualization")
    except Exception as e:
        if verbose:
            print(f"  ⚠️  Could not generate visualization: {e}")
    
    if verbose:
        print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    
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
) -> Dict[str, any]:
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
        print("="*70)
        print(f"K-FOLD CROSS-VALIDATION ({n_splits} folds)")
        print("="*70)
        print(f"Total samples: {len(dataset)}")
        print(f"Samples per fold: ~{len(dataset) // n_splits}")
        print(f"Epochs per fold: {num_epochs}")
        print(f"Learning rate: {learning_rate}")
        print(f"Batch size: {batch_size}")
        if early_stopping_patience > 0:
            print(f"Early stopping: {early_stopping_patience} epochs")
        print("="*70)
    
    # Iterate over folds
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(range(len(dataset))), 1):
        if verbose:
            print(f"\n{'='*70}")
            print(f"FOLD {fold_idx}/{n_splits}")
            print(f"{'='*70}")
            print(f"Train samples: {len(train_indices)}")
            print(f"Val samples: {len(val_indices)}")
        
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
        
        # Store results
        fold_results['fold'] = fold_idx
        fold_metrics.append(fold_results)
        
        if verbose:
            print(f"\nFold {fold_idx} Results:")
            print(f"  Best Val Loss: {fold_results['best_val_loss']:.4f}")
            print(f"  Best Val Correlation: {fold_results['best_val_correlation']:.3f}")
            print(f"  Best Val R²: {fold_results['best_val_r2']:.3f}")
            print(f"  Best Epoch: {fold_results['best_epoch']}")
    
    # Aggregate results across folds
    metric_names = ['best_val_loss', 'best_val_correlation', 'best_val_r2', 'best_val_mae']
    
    mean_metrics = {}
    std_metrics = {}
    
    for metric in metric_names:
        values = [fold[metric] for fold in fold_metrics]
        mean_metrics[metric] = np.mean(values)
        std_metrics[metric] = np.std(values)
    
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
    
    # Save to JSON
    import json
    with open(os.path.join(output_dir, 'cv_results.json'), 'w') as f:
        # Convert to JSON-serializable format
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
        json.dump(json_results, f, indent=2)
    
    if verbose:
        print("\n" + "="*70)
        print("CROSS-VALIDATION SUMMARY")
        print("="*70)
        print(f"\nMean ± Std across {n_splits} folds:")
        print(f"  Val Loss:        {mean_metrics['best_val_loss']:.4f} ± {std_metrics['best_val_loss']:.4f}")
        print(f"  Val Correlation: {mean_metrics['best_val_correlation']:.3f} ± {std_metrics['best_val_correlation']:.3f}")
        print(f"  Val R²:          {mean_metrics['best_val_r2']:.3f} ± {std_metrics['best_val_r2']:.3f}")
        print(f"  Val MAE:         {mean_metrics['best_val_mae']:.4f} ± {std_metrics['best_val_mae']:.4f}")
        print(f"\nBest fold: {best_fold} (val_loss={fold_metrics[best_fold_idx]['best_val_loss']:.4f})")
        print(f"Results saved to: {output_dir}")
        print("="*70)
    
    # Generate visualization
    try:
        from src.neuro_foundation.visualization import plot_cv_results
        cv_results_path = os.path.join(output_dir, 'cv_results.json')
        plot_cv_results(
            cv_results_path,
            output_path=os.path.join(output_dir, 'cv_analysis.png')
        )
        if verbose:
            print(f"  ✓ Saved cross-validation visualization")
    except Exception as e:
        if verbose:
            print(f"  ⚠️  Could not generate CV visualization: {e}")
    
    return cv_results


def grid_search(
    model_factory_template: callable,
    dataset: torch.utils.data.Dataset,
    param_grid: Dict[str, list],
    output_dir: str,
    use_kfold: bool = True,
    n_splits: int = 5,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 5e-3,
    weight_decay: float = 0.0,
    early_stopping_patience: int = 0,
    random_seed: int = 42,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, any]:
    """Flexible grid search over hyperparameters with optional K-fold CV.
    
    Supports searching over any combination of:
    - Model hyperparameters (dropout, hidden_dims, etc.)
    - Training hyperparameters (learning_rate, weight_decay, etc.)
    
    Args:
        model_factory_template: Function that takes **model_params and returns model
        dataset: Complete dataset for training
        param_grid: Dictionary of parameter names to lists of values to try
                   Example: {
                       'dropout': [0.2, 0.35, 0.5],
                       'learning_rate': [0.001, 0.005, 0.01],
                       'weight_decay': [0.0, 1e-5, 1e-4],
                       'hidden_dims': [[512, 256, 128], [1024, 512, 256]]
                   }
        output_dir: Directory to save all results
        use_kfold: If True, use K-fold CV; if False, use single train/val split
        n_splits: Number of CV folds (only used if use_kfold=True)
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        early_stopping_patience: Early stopping patience (0 = disabled)
        random_seed: Random seed for reproducibility
        device: Device to train on
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing:
        - results: List of results for each parameter combination
        - best_params: Parameter combination with best validation performance
        - best_score: Best validation correlation achieved
        
    Example:
        >>> def model_factory(dropout=0.35, hidden_dims=[512, 256, 128]):
        ...     return MoleculeToActivityMapMLP(
        ...         input_dim=268, 
        ...         output_shape=(79, 43),
        ...         dropout=dropout,
        ...         hidden_dims=hidden_dims
        ...     )
        >>> 
        >>> param_grid = {
        ...     'dropout': [0.2, 0.35, 0.5],
        ...     'learning_rate': [0.001, 0.005],
        ...     'weight_decay': [0.0, 1e-5]
        ... }
        >>> 
        >>> results = grid_search(
        ...     model_factory_template=model_factory,
        ...     dataset=full_dataset,
        ...     param_grid=param_grid,
        ...     output_dir='experiments/grid_search',
        ...     use_kfold=True,
        ...     n_splits=5
        ... )
    """
    from itertools import product
    import json
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate model params from training params
    training_param_names = {'learning_rate', 'weight_decay', 'batch_size', 'num_epochs', 'early_stopping_patience'}
    
    model_param_grid = {k: v for k, v in param_grid.items() if k not in training_param_names}
    training_param_grid = {k: v for k, v in param_grid.items() if k in training_param_names}
    
    # Generate all combinations
    model_param_names = list(model_param_grid.keys())
    model_param_values = list(model_param_grid.values())
    model_param_combinations = list(product(*model_param_values)) if model_param_values else [()]
    
    training_param_names_list = list(training_param_grid.keys())
    training_param_values = list(training_param_grid.values())
    training_param_combinations = list(product(*training_param_values)) if training_param_values else [()]
    
    # Total combinations
    total_combinations = len(model_param_combinations) * len(training_param_combinations)
    
    if verbose:
        print("="*70)
        print(f"GRID SEARCH")
        print("="*70)
        print(f"Total parameter combinations: {total_combinations}")
        print(f"Model parameters: {list(model_param_grid.keys())}")
        print(f"Training parameters: {list(training_param_grid.keys())}")
        if use_kfold:
            print(f"Evaluation: {n_splits}-fold cross-validation")
        else:
            print(f"Evaluation: Single train/val split")
        print(f"Dataset size: {len(dataset)}")
        print("="*70)
    
    # Storage for all results
    all_results = []
    best_score = -float('inf')
    best_params = None
    
    # Iterate over all combinations
    combination_idx = 0
    for model_params in model_param_combinations:
        for training_params in training_param_combinations:
            combination_idx += 1
            
            # Build parameter dictionaries
            current_model_params = dict(zip(model_param_names, model_params)) if model_param_names else {}
            current_training_params = dict(zip(training_param_names_list, training_params)) if training_param_names_list else {}
            
            # Merge with defaults
            current_learning_rate = current_training_params.get('learning_rate', learning_rate)
            current_weight_decay = current_training_params.get('weight_decay', 0.0)
            current_batch_size = current_training_params.get('batch_size', batch_size)
            current_num_epochs = current_training_params.get('num_epochs', num_epochs)
            current_early_stopping = current_training_params.get('early_stopping_patience', early_stopping_patience)
            
            if verbose:
                print(f"\n{'='*70}")
                print(f"Combination {combination_idx}/{total_combinations}")
                print(f"{'='*70}")
                print(f"Model params: {current_model_params}")
                print(f"Training params: learning_rate={current_learning_rate}, weight_decay={current_weight_decay}, batch_size={current_batch_size}")
            
            # Create model factory with these params
            def model_factory():
                return model_factory_template(**current_model_params)
            
            # Create experiment directory
            exp_name = f"exp_{combination_idx:03d}"
            exp_dir = os.path.join(output_dir, exp_name)
            
            try:
                if use_kfold:
                    # Run K-fold CV
                    cv_results = train_nn_kfold(
                        model_factory=model_factory,
                        dataset=dataset,
                        output_dir=exp_dir,
                        n_splits=n_splits,
                        num_epochs=current_num_epochs,
                        batch_size=current_batch_size,
                        learning_rate=current_learning_rate,
                        weight_decay=current_weight_decay,
                        early_stopping_patience=current_early_stopping,
                        random_seed=random_seed,
                        device=device,
                        verbose=verbose,
                    )
                    
                    # Extract score (mean correlation)
                    score = cv_results['mean_metrics']['best_val_correlation']
                    score_std = cv_results['std_metrics']['best_val_correlation']
                    
                    result = {
                        'combination_idx': combination_idx,
                        'model_params': current_model_params,
                        'training_params': current_training_params,
                        'mean_correlation': score,
                        'std_correlation': score_std,
                        'mean_val_loss': cv_results['mean_metrics']['best_val_loss'],
                        'mean_r2': cv_results['mean_metrics']['best_val_r2'],
                        'best_fold': cv_results['best_fold'],
                    }
                else:
                    # Single train/val split
                    from sklearn.model_selection import train_test_split
                    from torch.utils.data import Subset, DataLoader
                    
                    # Split indices
                    indices = list(range(len(dataset)))
                    train_indices, val_indices = train_test_split(
                        indices, test_size=0.15, random_state=random_seed
                    )
                    
                    # Create subsets
                    train_subset = Subset(dataset, train_indices)
                    val_subset = Subset(dataset, val_indices)
                    
                    # Create dataloaders
                    train_loader = DataLoader(train_subset, batch_size=current_batch_size, shuffle=True, num_workers=0, pin_memory=False)
                    val_loader = DataLoader(val_subset, batch_size=current_batch_size, shuffle=False, num_workers=0, pin_memory=False)
                    
                    # Train model
                    model = model_factory()
                    metrics = train_nn(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        output_dir=exp_dir,
                        num_epochs=current_num_epochs,
                        learning_rate=current_learning_rate,
                        weight_decay=current_weight_decay,
                        early_stopping_patience=current_early_stopping,
                        device=device,
                        verbose=verbose,
                    )
                    
                    score = metrics['best_val_correlation']
                    
                    result = {
                        'combination_idx': combination_idx,
                        'model_params': current_model_params,
                        'training_params': current_training_params,
                        'val_correlation': score,
                        'val_loss': metrics['best_val_loss'],
                        'val_r2': metrics['best_val_r2'],
                        'best_epoch': metrics['best_epoch'],
                    }
                
                # Store result
                all_results.append(result)
                
                # Update best
                if score > best_score:
                    best_score = score
                    best_params = {**current_model_params, **current_training_params}
                
                if verbose:
                    print(f"\n✓ Score: {score:.4f}")
                    if use_kfold:
                        print(f"  (±{score_std:.4f} across folds)")
                
            except Exception as e:
                if verbose:
                    print(f"\n✗ Failed: {str(e)}")
                result = {
                    'combination_idx': combination_idx,
                    'model_params': current_model_params,
                    'training_params': current_training_params,
                    'error': str(e),
                }
                all_results.append(result)
    
    # Save all results
    grid_search_results = {
        'results': all_results,
        'best_params': best_params,
        'best_score': float(best_score),
        'total_combinations': total_combinations,
        'use_kfold': use_kfold,
        'n_splits': n_splits if use_kfold else None,
    }
    
    with open(os.path.join(output_dir, 'grid_search_results.json'), 'w') as f:
        json.dump(grid_search_results, f, indent=2)
    
    # Create summary DataFrame
    if all_results:
        summary_data = []
        for r in all_results:
            row = {**r['model_params'], **r['training_params']}
            if 'error' not in r:
                if use_kfold:
                    row['mean_correlation'] = r['mean_correlation']
                    row['std_correlation'] = r['std_correlation']
                    row['mean_val_loss'] = r['mean_val_loss']
                else:
                    row['val_correlation'] = r['val_correlation']
                    row['val_loss'] = r['val_loss']
            else:
                row['error'] = r['error']
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, 'grid_search_summary.csv'), index=False)
    
    if verbose:
        print("\n" + "="*70)
        print("GRID SEARCH COMPLETE")
        print("="*70)
        print(f"\nBest Score: {best_score:.4f}")
        print(f"Best Parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"\nResults saved to: {output_dir}")
        print(f"  - grid_search_results.json (full results)")
        print(f"  - grid_search_summary.csv (table format)")
        print("="*70)
    
    # Generate visualization
    try:
        from src.neuro_foundation.visualization import plot_grid_search_results
        grid_results_path = os.path.join(output_dir, 'grid_search_results.json')
        plot_grid_search_results(
            grid_results_path,
            output_path=os.path.join(output_dir, 'grid_search_analysis.png'),
            top_n=min(10, len(all_results))
        )
        if verbose:
            print(f"  ✓ Saved grid search visualization")
    except Exception as e:
        if verbose:
            print(f"  ⚠️  Could not generate grid search visualization: {e}")
    
    return grid_search_results
