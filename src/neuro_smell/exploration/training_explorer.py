"""
Training Explorer - Compare and analyze experiment results

Copyright (c) 2025 Tom White
Licensed under the MIT License

🎓 STUDENTS: Use this tool to compare your experiments!

Usage:
    python scripts/explore_stage.py stage=training

This tool helps you:
- Compare multiple experiments
- Visualize training curves
- Identify best models
- Understand what settings work best
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import json


class TrainingExplorer:
    """
    Interactive exploration of training results.
    
    Helps compare experiments and identify best configurations.
    """
    
    def __init__(self, experiments_dir: Path):
        """
        Args:
            experiments_dir: Directory containing experiment results
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments = self._load_experiments()
    
    def _load_experiments(self) -> List[Dict[str, Any]]:
        """Load all experiment results"""
        experiments = []
        
        if not self.experiments_dir.exists():
            return experiments
        
        for exp_dir in self.experiments_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            # Look for metrics file
            metrics_file = exp_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                experiments.append({
                    'name': exp_dir.name,
                    'path': exp_dir,
                    'metrics': metrics,
                })
        
        return experiments
    
    def summary(self) -> pd.DataFrame:
        """
        Get summary of all experiments as DataFrame.
        
        Returns:
            DataFrame with experiment names and metrics
        """
        if not self.experiments:
            return pd.DataFrame()
        
        rows = []
        for exp in self.experiments:
            row = {'experiment': exp['name']}
            
            # Add test metrics
            if 'test' in exp['metrics']:
                for key, val in exp['metrics']['test'].items():
                    row[f'test_{key}'] = val
            
            # Add training info
            if 'train' in exp['metrics']:
                row['final_train_loss'] = exp['metrics']['train'].get('loss', None)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df.sort_values('test_pearson_corr', ascending=False) if 'test_pearson_corr' in df.columns else df
    
    def print_summary(self):
        """Print formatted summary of all experiments"""
        df = self.summary()
        
        if df.empty:
            print("\n⚠️  No experiments found.")
            print(f"Looking in: {self.experiments_dir}")
            return
        
        print("\n" + "="*80)
        print("🏆 EXPERIMENT RESULTS SUMMARY")
        print("="*80)
        
        print(f"\nTotal experiments: {len(df)}")
        
        # Show top 5
        print(f"\n📊 Top 5 by Test Pearson Correlation:")
        if 'test_pearson_corr' in df.columns:
            top5 = df.nlargest(5, 'test_pearson_corr')
            for idx, row in top5.iterrows():
                exp_name = row['experiment']
                corr = row.get('test_pearson_corr', 'N/A')
                r2 = row.get('test_r2', 'N/A')
                mae = row.get('test_mae', 'N/A')
                
                print(f"\n  {idx+1}. {exp_name}")
                print(f"     Correlation: {corr:.4f}" if isinstance(corr, float) else f"     Correlation: {corr}")
                print(f"     R²: {r2:.4f}" if isinstance(r2, float) else f"     R²: {r2}")
                print(f"     MAE: {mae:.4f}" if isinstance(mae, float) else f"     MAE: {mae}")
        
        print("\n" + "="*80 + "\n")
    
    def compare_experiments(self, exp_names: List[str] = None) -> pd.DataFrame:
        """
        Compare specific experiments side-by-side.
        
        Args:
            exp_names: List of experiment names to compare (None = all)
        
        Returns:
            DataFrame with detailed comparison
        """
        df = self.summary()
        
        if exp_names:
            df = df[df['experiment'].isin(exp_names)]
        
        return df
    
    def plot_metric_comparison(self, metric: str = 'test_pearson_corr', save_path: Path = None):
        """
        Plot comparison of a specific metric across experiments.
        
        Args:
            metric: Metric to compare (e.g., 'test_pearson_corr')
            save_path: Optional path to save plot
        """
        df = self.summary()
        
        if df.empty or metric not in df.columns:
            print(f"⚠️  No data available for metric: {metric}")
            return
        
        # Sort by metric
        df = df.sort_values(metric, ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.3)))
        
        colors = ['green' if x == df[metric].max() else 'steelblue' for x in df[metric]]
        ax.barh(df['experiment'], df[metric], color=colors, alpha=0.7)
        
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_ylabel('Experiment')
        ax.set_title(f'Experiment Comparison: {metric}')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (exp, val) in enumerate(zip(df['experiment'], df[metric])):
            ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved plot to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_training_curves(self, exp_name: str, save_path: Path = None):
        """
        Plot training/validation curves for an experiment.
        
        Args:
            exp_name: Name of experiment
            save_path: Optional path to save plot
        """
        exp = next((e for e in self.experiments if e['name'] == exp_name), None)
        
        if exp is None:
            print(f"⚠️  Experiment not found: {exp_name}")
            return
        
        # Look for training history
        history_file = exp['path'] / "training_history.csv"
        if not history_file.exists():
            print(f"⚠️  Training history not found for: {exp_name}")
            return
        
        history = pd.read_csv(history_file)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss
        ax = axes[0, 0]
        if 'train_loss' in history.columns:
            ax.plot(history['epoch'], history['train_loss'], label='Train', linewidth=2)
        if 'val_loss' in history.columns:
            ax.plot(history['epoch'], history['val_loss'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Correlation
        ax = axes[0, 1]
        if 'train_pearson_corr' in history.columns:
            ax.plot(history['epoch'], history['train_pearson_corr'], label='Train', linewidth=2)
        if 'val_pearson_corr' in history.columns:
            ax.plot(history['epoch'], history['val_pearson_corr'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Pearson Correlation')
        ax.set_title('Correlation Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # R²
        ax = axes[1, 0]
        if 'train_r2' in history.columns:
            ax.plot(history['epoch'], history['train_r2'], label='Train', linewidth=2)
        if 'val_r2' in history.columns:
            ax.plot(history['epoch'], history['val_r2'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('R² Score')
        ax.set_title('R² Score Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # MAE
        ax = axes[1, 1]
        if 'train_mae' in history.columns:
            ax.plot(history['epoch'], history['train_mae'], label='Train', linewidth=2)
        if 'val_mae' in history.columns:
            ax.plot(history['epoch'], history['val_mae'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.set_title('Mean Absolute Error Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Training Curves: {exp_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved plot to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def find_best(self, metric: str = 'test_pearson_corr') -> Dict[str, Any]:
        """
        Find best experiment by metric.
        
        Args:
            metric: Metric to optimize
        
        Returns:
            Dictionary with best experiment info
        """
        df = self.summary()
        
        if df.empty or metric not in df.columns:
            return {}
        
        best_idx = df[metric].idxmax()
        best_row = df.loc[best_idx]
        
        return {
            'name': best_row['experiment'],
            'metric': metric,
            'value': best_row[metric],
            'all_metrics': best_row.to_dict(),
        }
