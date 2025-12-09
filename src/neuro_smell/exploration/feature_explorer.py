"""
Feature Explorer - Validate and visualize feature extraction results

Copyright (c) 2025 Tom White
Licensed under the MIT License

🎓 STUDENTS: Use this tool to validate your feature extraction before training!

Usage:
    python scripts/explore_stage.py stage=features

This tool helps you:
- Check if features were extracted correctly
- Identify missing or invalid features
- Visualize feature distributions
- Compare different feature extraction settings
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig


class FeatureExplorer:
    """
    Interactive exploration of extracted molecular features.
    
    Helps validate that feature extraction worked correctly
    and understand the feature space.
    """
    
    def __init__(self, features_df: pd.DataFrame):
        """
        Args:
            features_df: DataFrame with extracted features
        """
        self.features_df = features_df
        self.n_features = len(features_df.columns)
        self.n_samples = len(features_df)
    
    def summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the features"""
        
        summary = {
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'feature_names': list(self.features_df.columns),
        }
        
        # Check for missing values
        missing = self.features_df.isnull().sum()
        summary['missing_values'] = {
            'total': int(missing.sum()),
            'by_feature': {col: int(count) for col, count in missing.items() if count > 0}
        }
        
        # Check for constant features (no variance)
        constant_features = []
        for col in self.features_df.columns:
            if self.features_df[col].nunique() == 1:
                constant_features.append(col)
        summary['constant_features'] = constant_features
        
        # Check for infinite values
        inf_count = np.isinf(self.features_df.select_dtypes(include=[np.number])).sum().sum()
        summary['infinite_values'] = int(inf_count)
        
        # Feature value ranges
        numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns
        summary['value_ranges'] = {
            'min': float(self.features_df[numeric_cols].min().min()),
            'max': float(self.features_df[numeric_cols].max().max()),
            'mean': float(self.features_df[numeric_cols].mean().mean()),
            'std': float(self.features_df[numeric_cols].std().mean()),
        }
        
        return summary
    
    def print_summary(self):
        """Print a formatted summary"""
        summary = self.summary()
        
        print("\n" + "="*60)
        print("🔍 FEATURE EXTRACTION SUMMARY")
        print("="*60)
        
        print(f"\n📊 Dataset Size:")
        print(f"  Samples: {summary['n_samples']:,}")
        print(f"  Features: {summary['n_features']:,}")
        
        # Missing values
        if summary['missing_values']['total'] > 0:
            print(f"\n⚠️  Missing Values: {summary['missing_values']['total']}")
            for feat, count in list(summary['missing_values']['by_feature'].items())[:10]:
                pct = (count / summary['n_samples']) * 100
                print(f"  {feat}: {count} ({pct:.1f}%)")
            if len(summary['missing_values']['by_feature']) > 10:
                print(f"  ... and {len(summary['missing_values']['by_feature']) - 10} more")
        else:
            print(f"\n✅ No missing values")
        
        # Constant features
        if summary['constant_features']:
            print(f"\n⚠️  Constant Features (no variance): {len(summary['constant_features'])}")
            for feat in summary['constant_features'][:10]:
                print(f"  {feat}")
            if len(summary['constant_features']) > 10:
                print(f"  ... and {len(summary['constant_features']) - 10} more")
        else:
            print(f"\n✅ No constant features")
        
        # Infinite values
        if summary['infinite_values'] > 0:
            print(f"\n⚠️  Infinite values found: {summary['infinite_values']}")
        else:
            print(f"\n✅ No infinite values")
        
        # Value ranges
        print(f"\n📈 Feature Value Ranges:")
        print(f"  Min: {summary['value_ranges']['min']:.3f}")
        print(f"  Max: {summary['value_ranges']['max']:.3f}")
        print(f"  Mean: {summary['value_ranges']['mean']:.3f}")
        print(f"  Std Dev: {summary['value_ranges']['std']:.3f}")
        
        print("\n" + "="*60 + "\n")
    
    def plot_distributions(self, n_features: int = 20, save_path: Path = None):
        """
        Plot distributions of top features.
        
        Args:
            n_features: Number of features to plot
            save_path: Optional path to save plot
        """
        # Select features with highest variance
        numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns
        variances = self.features_df[numeric_cols].var().sort_values(ascending=False)
        top_features = variances.head(n_features).index
        
        # Create subplots
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feat in enumerate(top_features):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            data = self.features_df[feat].dropna()
            
            ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
            ax.set_title(f"{feat}\nVar: {variances[feat]:.3f}", fontsize=10)
            ax.set_xlabel('Value')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(top_features), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved plot to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_correlation_matrix(self, n_features: int = 30, save_path: Path = None):
        """
        Plot correlation matrix of top features.
        
        Args:
            n_features: Number of features to include
            save_path: Optional path to save plot
        """
        # Select features with highest variance
        numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns
        variances = self.features_df[numeric_cols].var().sort_values(ascending=False)
        top_features = variances.head(n_features).index
        
        # Compute correlation matrix
        corr_matrix = self.features_df[top_features].corr()
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation'},
            ax=ax
        )
        ax.set_title(f'Feature Correlation Matrix (Top {n_features} by Variance)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved plot to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def identify_issues(self) -> Dict[str, Any]:
        """
        Identify potential issues with the features.
        
        Returns:
            Dictionary with identified issues and recommendations
        """
        issues = {
            'critical': [],
            'warnings': [],
            'recommendations': []
        }
        
        summary = self.summary()
        
        # Critical issues
        if summary['missing_values']['total'] > 0:
            pct = (summary['missing_values']['total'] / (self.n_samples * self.n_features)) * 100
            if pct > 10:
                issues['critical'].append(
                    f"High percentage of missing values ({pct:.1f}%). "
                    "Consider checking SMILES validity or descriptor calculation."
                )
        
        if summary['infinite_values'] > 0:
            issues['critical'].append(
                f"{summary['infinite_values']} infinite values found. "
                "This will cause training failures. Check descriptor calculations."
            )
        
        # Warnings
        if summary['constant_features']:
            n_const = len(summary['constant_features'])
            pct = (n_const / self.n_features) * 100
            issues['warnings'].append(
                f"{n_const} constant features ({pct:.1f}%) have no variance. "
                "These won't help the model and should be removed."
            )
        
        if summary['value_ranges']['max'] > 1000 or summary['value_ranges']['min'] < -1000:
            issues['warnings'].append(
                "Some features have very large values. "
                "Consider using scaling in preprocessing."
            )
        
        # Recommendations
        if self.n_features > 1000:
            issues['recommendations'].append(
                f"You have {self.n_features} features. "
                "Consider using PCA (pca_default.yaml) to reduce dimensionality."
            )
        
        if self.n_features < 50:
            issues['recommendations'].append(
                f"Only {self.n_features} features extracted. "
                "You might want to use more descriptor types for better performance."
            )
        
        return issues
    
    def print_issues(self):
        """Print identified issues with formatting"""
        issues = self.identify_issues()
        
        if issues['critical']:
            print("\n" + "="*60)
            print("🚨 CRITICAL ISSUES")
            print("="*60)
            for issue in issues['critical']:
                print(f"\n❌ {issue}")
        
        if issues['warnings']:
            print("\n" + "="*60)
            print("⚠️  WARNINGS")
            print("="*60)
            for warning in issues['warnings']:
                print(f"\n⚠️  {warning}")
        
        if issues['recommendations']:
            print("\n" + "="*60)
            print("💡 RECOMMENDATIONS")
            print("="*60)
            for rec in issues['recommendations']:
                print(f"\n💡 {rec}")
        
        if not (issues['critical'] or issues['warnings'] or issues['recommendations']):
            print("\n✅ No issues found! Features look good.\n")
