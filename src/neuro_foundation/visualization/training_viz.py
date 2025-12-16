"""Training visualization utilities for neural networks.

This module provides comprehensive visualization tools for:
- Training/validation curves
- K-fold cross-validation results
- Grid search comparisons
- Prediction vs ground truth scatter plots
- Activity map comparisons
- Multi-panel training reports
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle


# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


def plot_training_curves(
    metrics_dict: Dict,
    output_path: Optional[Union[str, Path]] = None,
    show_r2: bool = True,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """Plot training and validation curves from metrics dictionary.
    
    Creates a 3-panel figure showing:
    - Loss over epochs
    - Correlation over epochs
    - R² score over epochs (optional)
    
    Args:
        metrics_dict: Dictionary with keys 'train_losses', 'val_losses',
                     'train_correlations', 'val_correlations', etc.
        output_path: Path to save figure. If None, figure is not saved.
        show_r2: Whether to show R² subplot (default: True)
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> metrics = train_nn(model, train_loader, val_loader, ...)
        >>> fig = plot_training_curves(metrics, 'training_curves.png')
    """
    n_panels = 3 if show_r2 else 2
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    
    epochs = np.arange(1, len(metrics_dict['train_losses']) + 1)
    
    # Loss subplot
    ax = axes[0]
    ax.plot(epochs, metrics_dict['train_losses'], 'o-', label='Train', 
            linewidth=2, markersize=4, alpha=0.8)
    ax.plot(epochs, metrics_dict['val_losses'], 's-', label='Validation',
            linewidth=2, markersize=4, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mark best epoch
    best_epoch = metrics_dict.get('best_epoch', len(epochs))
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5, 
               label=f'Best (epoch {best_epoch})')
    ax.legend()
    
    # Correlation subplot
    ax = axes[1]
    ax.plot(epochs, metrics_dict['train_correlations'], 'o-', label='Train',
            linewidth=2, markersize=4, alpha=0.8)
    ax.plot(epochs, metrics_dict['val_correlations'], 's-', label='Validation',
            linewidth=2, markersize=4, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Correlation')
    ax.set_title('Prediction Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    
    # R² subplot
    if show_r2:
        ax = axes[2]
        ax.plot(epochs, metrics_dict['train_r2'], 'o-', label='Train',
                linewidth=2, markersize=4, alpha=0.8)
        ax.plot(epochs, metrics_dict['val_r2'], 's-', label='Validation',
                linewidth=2, markersize=4, alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('R² Score')
        ax.set_title('Coefficient of Determination')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training curves to {output_path}")
    
    return fig


def plot_cv_results(
    cv_results_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """Plot K-fold cross-validation results.
    
    Creates a comprehensive visualization with:
    - Individual fold training curves
    - Mean ± std across folds
    - Final metrics comparison
    
    Args:
        cv_results_path: Path to cv_results.json file
        output_path: Path to save figure. If None, figure is not saved.
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> fig = plot_cv_results('experiments/cv_results.json', 'cv_analysis.png')
    """
    # Load results
    with open(cv_results_path, 'r') as f:
        results = json.load(f)
    
    n_folds = len(results['fold_results'])
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot individual fold losses
    ax1 = fig.add_subplot(gs[0, :2])
    for i, fold_result in enumerate(results['fold_results']):
        epochs = np.arange(1, len(fold_result['val_losses']) + 1)
        ax1.plot(epochs, fold_result['val_losses'], 'o-', alpha=0.6, 
                label=f'Fold {i+1}', markersize=3)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title(f'{n_folds}-Fold Cross-Validation: Loss per Fold')
    ax1.legend(loc='best', ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot individual fold correlations
    ax2 = fig.add_subplot(gs[1, :2])
    for i, fold_result in enumerate(results['fold_results']):
        epochs = np.arange(1, len(fold_result['val_correlations']) + 1)
        ax2.plot(epochs, fold_result['val_correlations'], 's-', alpha=0.6,
                label=f'Fold {i+1}', markersize=3)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Correlation')
    ax2.set_title(f'{n_folds}-Fold Cross-Validation: Correlation per Fold')
    ax2.legend(loc='best', ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # Bar plot: Best metrics per fold
    ax3 = fig.add_subplot(gs[0, 2])
    fold_nums = np.arange(1, n_folds + 1)
    best_corrs = [f['best_val_correlation'] for f in results['fold_results']]
    ax3.bar(fold_nums, best_corrs, alpha=0.7, color='steelblue')
    ax3.axhline(results['mean_metrics']['best_val_correlation'], 
                color='red', linestyle='--', label='Mean')
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Best Correlation')
    ax3.set_title('Best Correlation per Fold')
    ax3.set_xticks(fold_nums)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Summary statistics
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    
    mean_metrics = results['mean_metrics']
    std_metrics = results['std_metrics']
    
    summary_text = (
        f"Cross-Validation Summary\n"
        f"{'='*30}\n\n"
        f"Best Validation Metrics:\n"
        f"  Correlation: {mean_metrics['best_val_correlation']:.4f} ± {std_metrics['best_val_correlation']:.4f}\n"
        f"  Loss: {mean_metrics['best_val_loss']:.4f} ± {std_metrics['best_val_loss']:.4f}\n"
        f"  R²: {mean_metrics['best_val_r2']:.4f} ± {std_metrics['best_val_r2']:.4f}\n\n"
        f"Final Epoch Metrics:\n"
        f"  Correlation: {mean_metrics['final_val_correlation']:.4f} ± {std_metrics['final_val_correlation']:.4f}\n"
        f"  Loss: {mean_metrics['final_val_loss']:.4f} ± {std_metrics['final_val_loss']:.4f}\n"
        f"  R²: {mean_metrics['final_val_r2']:.4f} ± {std_metrics['final_val_r2']:.4f}\n\n"
        f"Training Info:\n"
        f"  Folds: {n_folds}\n"
        f"  Epochs: {len(results['fold_results'][0]['val_losses'])}\n"
    )
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Learning curves with mean and std
    ax5 = fig.add_subplot(gs[2, :])
    
    # Collect all fold data
    max_epochs = max(len(f['val_correlations']) for f in results['fold_results'])
    epochs = np.arange(1, max_epochs + 1)
    
    # Compute mean and std across folds
    all_corrs = np.array([f['val_correlations'] for f in results['fold_results']])
    mean_corr = np.mean(all_corrs, axis=0)
    std_corr = np.std(all_corrs, axis=0)
    
    ax5.plot(epochs, mean_corr, 'o-', linewidth=2, markersize=5, 
             label='Mean', color='darkblue')
    ax5.fill_between(epochs, mean_corr - std_corr, mean_corr + std_corr,
                      alpha=0.3, color='steelblue', label='± 1 std')
    
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Validation Correlation')
    ax5.set_title('Mean Learning Curve Across Folds')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved CV results to {output_path}")
    
    return fig


def plot_grid_search_results(
    grid_results_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (16, 10),
    top_n: int = 10,
) -> plt.Figure:
    """Plot grid search results with parameter comparisons.
    
    Creates visualizations showing:
    - Top N parameter combinations
    - Parameter importance heatmap (if 2D grid)
    - Learning curves for best models
    - Parameter distribution analysis
    
    Args:
        grid_results_path: Path to grid_search_results.json file
        output_path: Path to save figure. If None, figure is not saved.
        figsize: Figure size (width, height)
        top_n: Number of top configurations to show
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> fig = plot_grid_search_results('experiments/grid_search_results.json')
    """
    # Load results
    with open(grid_results_path, 'r') as f:
        results = json.load(f)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # Sort by score
    sorted_results = sorted(results['results'], 
                           key=lambda x: x['score'], reverse=True)
    
    # 1. Top N configurations bar plot
    ax1 = fig.add_subplot(gs[0, :2])
    top_configs = sorted_results[:top_n]
    config_labels = [f"Config {i+1}" for i in range(len(top_configs))]
    scores = [c['score'] for c in top_configs]
    
    bars = ax1.barh(config_labels, scores, alpha=0.7)
    bars[0].set_color('gold')  # Highlight best
    bars[1].set_color('silver') if len(bars) > 1 else None
    bars[2].set_color('#CD7F32') if len(bars) > 2 else None  # Bronze
    
    ax1.set_xlabel('Validation Correlation')
    ax1.set_title(f'Top {len(top_configs)} Configurations')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Parameter importance (if we have enough data)
    param_names = list(results['best_params'].keys())
    
    if len(param_names) == 2:
        # Create 2D heatmap
        ax2 = fig.add_subplot(gs[1, :2])
        
        # Extract unique values for each parameter
        param1_name = param_names[0]
        param2_name = param_names[1]
        
        param1_vals = sorted(set(r['params'][param1_name] for r in results['results']))
        param2_vals = sorted(set(r['params'][param2_name] for r in results['results']))
        
        # Create score matrix
        score_matrix = np.full((len(param2_vals), len(param1_vals)), np.nan)
        
        for result in results['results']:
            i = param2_vals.index(result['params'][param2_name])
            j = param1_vals.index(result['params'][param1_name])
            score_matrix[i, j] = result['score']
        
        im = ax2.imshow(score_matrix, cmap='viridis', aspect='auto')
        ax2.set_xticks(range(len(param1_vals)))
        ax2.set_yticks(range(len(param2_vals)))
        ax2.set_xticklabels([f'{v:.4f}' if isinstance(v, float) else str(v) 
                             for v in param1_vals], rotation=45, ha='right')
        ax2.set_yticklabels([f'{v:.4f}' if isinstance(v, float) else str(v) 
                             for v in param2_vals])
        ax2.set_xlabel(param1_name)
        ax2.set_ylabel(param2_name)
        ax2.set_title('Parameter Grid Heatmap')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Validation Correlation')
        
        # Annotate cells
        for i in range(len(param2_vals)):
            for j in range(len(param1_vals)):
                if not np.isnan(score_matrix[i, j]):
                    text = ax2.text(j, i, f'{score_matrix[i, j]:.3f}',
                                   ha="center", va="center", color="white" if score_matrix[i, j] < 0.5 else "black",
                                   fontsize=8)
    
    else:
        # Parameter distribution plots
        ax2 = fig.add_subplot(gs[1, :2])
        
        for param_name in param_names:
            param_vals = [r['params'][param_name] for r in results['results']]
            scores = [r['score'] for r in results['results']]
            
            ax2.scatter(param_vals, scores, alpha=0.6, s=50, label=param_name)
        
        ax2.set_xlabel('Parameter Value')
        ax2.set_ylabel('Validation Correlation')
        ax2.set_title('Parameter vs Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Best configuration summary
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    best_params_text = "Best Configuration\n" + "="*25 + "\n\n"
    for param, value in results['best_params'].items():
        if isinstance(value, float):
            best_params_text += f"{param}: {value:.4f}\n"
        else:
            best_params_text += f"{param}: {value}\n"
    
    best_params_text += f"\nBest Score: {results['best_score']:.4f}\n"
    best_params_text += f"Total Configs: {len(results['results'])}\n"
    
    ax3.text(0.1, 0.5, best_params_text, transform=ax3.transAxes,
             fontsize=10, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # 4. Distribution of scores
    ax4 = fig.add_subplot(gs[1, 2])
    all_scores = [r['score'] for r in results['results']]
    ax4.hist(all_scores, bins=min(20, len(all_scores)), alpha=0.7, color='steelblue', edgecolor='black')
    ax4.axvline(results['best_score'], color='red', linestyle='--', linewidth=2, label='Best')
    ax4.set_xlabel('Validation Correlation')
    ax4.set_ylabel('Count')
    ax4.set_title('Score Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Top 3 learning curves (if available)
    ax5 = fig.add_subplot(gs[2, :])
    
    for i, config in enumerate(top_configs[:3]):
        if 'metrics' in config and config['metrics']:
            epochs = np.arange(1, len(config['metrics']['val_correlations']) + 1)
            ax5.plot(epochs, config['metrics']['val_correlations'], 
                    'o-', label=f'Config {i+1} (score={config["score"]:.3f})',
                    linewidth=2, markersize=4, alpha=0.8)
    
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Validation Correlation')
    ax5.set_title('Learning Curves: Top 3 Configurations')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved grid search results to {output_path}")
    
    return fig


def plot_prediction_scatter(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Predictions vs Ground Truth",
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """Plot predictions vs ground truth as scatter plot with statistics.
    
    Args:
        predictions: Predicted values (flattened)
        targets: Ground truth values (flattened)
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> predictions = model(test_data)
        >>> fig = plot_prediction_scatter(predictions, targets, 'predictions.png')
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Flatten arrays
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Sample if too many points
    if len(pred_flat) > 10000:
        indices = np.random.choice(len(pred_flat), 10000, replace=False)
        pred_flat = pred_flat[indices]
        target_flat = target_flat[indices]
    
    # Compute statistics
    from scipy.stats import pearsonr
    corr, _ = pearsonr(pred_flat, target_flat)
    mse = np.mean((pred_flat - target_flat) ** 2)
    mae = np.mean(np.abs(pred_flat - target_flat))
    
    # Create scatter plot with density coloring
    h = ax.hist2d(target_flat, pred_flat, bins=50, cmap='viridis', 
                  cmin=1)  # Don't show bins with 0 counts
    
    # Add perfect prediction line
    min_val = min(target_flat.min(), pred_flat.min())
    max_val = max(target_flat.max(), pred_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
            linewidth=2, label='Perfect prediction', alpha=0.7)
    
    # Add statistics box
    stats_text = (
        f'Correlation: {corr:.4f}\n'
        f'MSE: {mse:.4f}\n'
        f'MAE: {mae:.4f}\n'
        f'N points: {len(pred_flat):,}'
    )
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Predictions')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Make square aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Add colorbar
    cbar = plt.colorbar(h[3], ax=ax)
    cbar.set_label('Count')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved prediction scatter to {output_path}")
    
    return fig


def plot_activity_map_comparison(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_samples: int = 4,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """Plot side-by-side comparison of predicted and true activity maps.
    
    Args:
        predictions: Predicted activity maps (N, H, W)
        targets: Ground truth activity maps (N, H, W)
        n_samples: Number of samples to show
        output_path: Path to save figure
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> predictions = model(test_features)
        >>> fig = plot_activity_map_comparison(predictions, targets, n_samples=6)
    """
    n_samples = min(n_samples, len(predictions))
    
    fig, axes = plt.subplots(n_samples, 3, figsize=figsize)
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Randomly select samples
    indices = np.random.choice(len(predictions), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        pred = predictions[idx]
        target = targets[idx]
        diff = pred - target
        
        # Ground truth
        im0 = axes[i, 0].imshow(target, cmap='viridis', aspect='auto')
        axes[i, 0].set_title(f'Sample {idx}: Ground Truth')
        axes[i, 0].axis('off')
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)
        
        # Prediction
        im1 = axes[i, 1].imshow(pred, cmap='viridis', aspect='auto',
                               vmin=target.min(), vmax=target.max())  # Same scale
        axes[i, 1].set_title('Prediction')
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)
        
        # Difference
        max_abs_diff = np.abs(diff).max()
        im2 = axes[i, 2].imshow(diff, cmap='RdBu_r', aspect='auto',
                               vmin=-max_abs_diff, vmax=max_abs_diff)
        axes[i, 2].set_title('Difference (Pred - True)')
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)
        
        # Compute correlation for this sample
        from scipy.stats import pearsonr
        corr, _ = pearsonr(pred.flatten(), target.flatten())
        mse = np.mean((pred - target) ** 2)
        
        # Add statistics
        axes[i, 0].text(0.5, -0.1, f'Corr: {corr:.3f}, MSE: {mse:.4f}',
                       transform=axes[i, 0].transAxes,
                       ha='center', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved activity map comparison to {output_path}")
    
    return fig


def create_training_report(
    metrics_dict: Dict,
    predictions: Optional[np.ndarray] = None,
    targets: Optional[np.ndarray] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """Create comprehensive training report with multiple visualizations.
    
    Combines:
    - Training curves
    - Final metrics summary
    - Prediction scatter (if provided)
    - Activity map samples (if provided)
    
    Args:
        metrics_dict: Training metrics from train_nn()
        predictions: Optional predictions for visualization
        targets: Optional ground truth for visualization
        output_path: Path to save figure
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> metrics = train_nn(...)
        >>> predictions = model(test_loader)
        >>> fig = create_training_report(metrics, predictions, targets, 'report.png')
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    epochs = np.arange(1, len(metrics_dict['train_losses']) + 1)
    best_epoch = metrics_dict.get('best_epoch', len(epochs))
    
    # 1. Loss curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, metrics_dict['train_losses'], 'o-', label='Train', 
            linewidth=2, markersize=4, alpha=0.8)
    ax1.plot(epochs, metrics_dict['val_losses'], 's-', label='Validation',
            linewidth=2, markersize=4, alpha=0.8)
    ax1.axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Correlation curve
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, metrics_dict['train_correlations'], 'o-', label='Train',
            linewidth=2, markersize=4, alpha=0.8)
    ax2.plot(epochs, metrics_dict['val_correlations'], 's-', label='Validation',
            linewidth=2, markersize=4, alpha=0.8)
    ax2.axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Correlation')
    ax2.set_title('Prediction Correlation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. R² curve
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, metrics_dict['train_r2'], 'o-', label='Train',
            linewidth=2, markersize=4, alpha=0.8)
    ax3.plot(epochs, metrics_dict['val_r2'], 's-', label='Validation',
            linewidth=2, markersize=4, alpha=0.8)
    ax3.axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('R² Score')
    ax3.set_title('R² Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    
    summary_text = (
        "Training Summary\n"
        "=" * 30 + "\n\n"
        f"Best Epoch: {best_epoch}\n"
        f"Total Epochs: {len(epochs)}\n\n"
        "Best Validation Metrics:\n"
        f"  Loss: {metrics_dict['best_val_loss']:.4f}\n"
        f"  Correlation: {metrics_dict['best_val_correlation']:.4f}\n"
        f"  R²: {metrics_dict['best_val_r2']:.4f}\n\n"
        "Final Validation Metrics:\n"
        f"  Loss: {metrics_dict['val_losses'][-1]:.4f}\n"
        f"  Correlation: {metrics_dict['val_correlations'][-1]:.4f}\n"
        f"  R²: {metrics_dict['val_r2'][-1]:.4f}\n"
    )
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 5-6. Prediction scatter (if provided)
    if predictions is not None and targets is not None:
        from scipy.stats import pearsonr
        
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Sample if too many points
        if len(pred_flat) > 5000:
            indices = np.random.choice(len(pred_flat), 5000, replace=False)
            pred_flat = pred_flat[indices]
            target_flat = target_flat[indices]
        
        ax5 = fig.add_subplot(gs[1, 1])
        h = ax5.hist2d(target_flat, pred_flat, bins=40, cmap='viridis', cmin=1)
        
        min_val = min(target_flat.min(), pred_flat.min())
        max_val = max(target_flat.max(), pred_flat.max())
        ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7)
        
        corr, _ = pearsonr(pred_flat, target_flat)
        ax5.set_title(f'Predictions vs Truth (r={corr:.3f})')
        ax5.set_xlabel('Ground Truth')
        ax5.set_ylabel('Predictions')
        ax5.set_aspect('equal', adjustable='box')
        plt.colorbar(h[3], ax=ax5, label='Count')
        
        # Residual plot
        ax6 = fig.add_subplot(gs[1, 2])
        residuals = pred_flat - target_flat
        ax6.scatter(target_flat, residuals, alpha=0.3, s=10)
        ax6.axhline(0, color='red', linestyle='--', linewidth=2)
        ax6.set_xlabel('Ground Truth')
        ax6.set_ylabel('Residuals')
        ax6.set_title('Residual Plot')
        ax6.grid(True, alpha=0.3)
        
        # Sample activity maps (if 2D)
        if len(predictions.shape) == 3:  # (N, H, W)
            n_samples = min(3, len(predictions))
            indices = np.random.choice(len(predictions), n_samples, replace=False)
            
            for i, idx in enumerate(indices):
                # Ground truth
                ax = fig.add_subplot(gs[2, i])
                
                # Create composite view: top half = truth, bottom half = prediction
                composite = np.vstack([targets[idx], predictions[idx]])
                
                im = ax.imshow(composite, cmap='viridis', aspect='auto')
                ax.axhline(targets.shape[1] - 0.5, color='white', linestyle='--', linewidth=2)
                ax.set_title(f'Sample {idx}\n(Top: Truth, Bottom: Pred)')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('Neural Network Training Report', fontsize=14, fontweight='bold', y=0.995)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training report to {output_path}")
    
    return fig


def plot_feature_importance(
    model,
    feature_names: Optional[List[str]] = None,
    top_n: int = 20,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 8),
    color: str = '#2E86AB',
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot feature importance based on first-layer weights from trained neural network.
    
    Computes importance scores as the absolute magnitude of first-layer weights,
    then displays the top N most important molecular descriptors in a horizontal
    bar chart suitable for publication.
    
    This visualization helps identify which molecular features (descriptors) the
    model relies on most heavily to predict olfactory bulb activation patterns.
    
    Args:
        model: Trained PyTorch neural network (nn.Module). Must have a first
               layer accessible via model.network[0] (for MLP) or similar.
        feature_names: List of feature/descriptor names. If None, uses generic
                      names like "Feature 1", "Feature 2", etc.
        top_n: Number of top features to display (default: 20)
        output_path: Path to save figure. If None, figure is not saved.
        figsize: Figure size as (width, height) tuple
        color: Bar color (default: '#2E86AB' - publication blue)
        title: Custom title. If None, uses default title.
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> from neuro_foundation.models.baseline_nn import MoleculeToActivityMapMLP
        >>> import pandas as pd
        >>> 
        >>> # Load feature names
        >>> features_df = pd.read_csv('data/02_processed/selected_features.csv')
        >>> feature_names = [col for col in features_df.columns if col != 'CID']
        >>> 
        >>> # Load trained model
        >>> model = MoleculeToActivityMapMLP(input_dim=268)
        >>> model.load_state_dict(torch.load('models/best_model.pth'))
        >>> 
        >>> # Plot feature importance
        >>> fig = plot_feature_importance(
        ...     model,
        ...     feature_names=feature_names,
        ...     top_n=20,
        ...     output_path='feature_importance.png'
        ... )
        
    Notes:
        - Importance is computed as mean absolute weight magnitude across all
          output neurons in the first layer
        - Works with MLP architectures where first layer is model.network[0]
        - For CNN architectures with encoder-decoder, extracts encoder input layer
        - Handles both Linear and Conv layers appropriately
    """
    import torch
    
    # Extract first layer weights
    first_layer = None
    
    # Try common architecture patterns
    if hasattr(model, 'network') and isinstance(model.network, torch.nn.Sequential):
        # MLP architecture (e.g., MoleculeToActivityMapMLP)
        first_layer = model.network[0]
    elif hasattr(model, 'encoder') and hasattr(model.encoder, '0'):
        # Encoder-decoder architecture
        first_layer = model.encoder[0]
    elif hasattr(model, 'fc1'):
        # Direct first layer
        first_layer = model.fc1
    else:
        # Search for first Linear layer in all children
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                first_layer = module
                break
    
    if first_layer is None or not isinstance(first_layer, torch.nn.Linear):
        raise ValueError(
            "Could not find first linear layer. Model must have accessible "
            "first layer via model.network[0], model.encoder[0], or model.fc1"
        )
    
    # Get weights: shape is (output_dim, input_dim)
    weights = first_layer.weight.data.cpu().numpy()
    
    # Compute importance: mean absolute weight magnitude across output neurons
    # This gives one importance score per input feature
    importance_scores = np.abs(weights).mean(axis=0)  # Shape: (input_dim,)
    
    # Validate dimensions
    n_features = len(importance_scores)
    if feature_names is not None and len(feature_names) != n_features:
        raise ValueError(
            f"Number of feature names ({len(feature_names)}) does not match "
            f"number of input features ({n_features})"
        )
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(n_features)]
    
    # Rank features by importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Select top N features
    top_features = importance_df.head(top_n)
    
    # Create horizontal bar chart (publication quality)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bars (reverse order so highest importance is at top)
    y_pos = np.arange(len(top_features))
    bars = ax.barh(
        y_pos,
        top_features['importance'].values,
        color=color,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Customize appearance
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'].values)
    ax.invert_yaxis()  # Highest importance at top
    
    ax.set_xlabel('Importance Score (Mean Absolute Weight)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Molecular Descriptor', fontsize=11, fontweight='bold')
    
    if title is None:
        title = (f'Top {top_n} Molecular Descriptors Ranked by Importance\n'
                 f'Based on First-Layer Weight Magnitudes')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_features['importance'].values)):
        ax.text(
            value + max(top_features['importance']) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{value:.4f}',
            va='center',
            fontsize=8,
            color='black'
        )
    
    # Add grid for readability
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved feature importance plot to {output_path}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Feature Importance Analysis Summary")
    print(f"{'='*60}")
    print(f"Total features: {n_features}")
    print(f"Top {top_n} features shown")
    print(f"\nTop 5 most important features:")
    for i, (idx, row) in enumerate(importance_df.head(5).iterrows(), 1):
        print(f"  {i}. {row['feature']:30s} {row['importance']:.6f}")
    print(f"{'='*60}\n")
    
    return fig
