"""
Preprocessing Explorer - Optimize PCA and preprocessing settings

Copyright (c) 2025 Tom White
Licensed under the MIT License

🎓 STUDENTS: Use this tool to find optimal PCA components!

Usage:
    python scripts/explore_stage.py stage=preprocessing

This tool helps you:
- Visualize explained variance by PCA components
- Compare different preprocessing strategies
- Identify optimal number of components
- Understand feature transformations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PreprocessingExplorer:
    """
    Interactive exploration of preprocessing and PCA results.
    
    Helps determine optimal preprocessing strategy and
    number of PCA components.
    """
    
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Args:
            X: Feature matrix [n_samples, n_features]
            y: Optional target values [n_samples]
        """
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        
        # Will be computed on demand
        self._pca = None
        self._scaler = None
    
    def analyze_pca(self, max_components: int = None) -> Dict[str, Any]:
        """
        Analyze PCA to determine optimal number of components.
        
        Args:
            max_components: Maximum components to test (default: all)
        
        Returns:
            Dictionary with PCA analysis results
        """
        if max_components is None:
            max_components = min(self.n_samples, self.n_features)
        
        # Scale data first
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # Fit PCA
        pca = PCA(n_components=max_components)
        pca.fit(X_scaled)
        
        self._pca = pca
        self._scaler = scaler
        
        # Calculate cumulative explained variance
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        
        # Find components needed for different thresholds
        thresholds = [0.80, 0.85, 0.90, 0.95, 0.99]
        components_for_threshold = {}
        for thresh in thresholds:
            n_comp = np.argmax(cumsum_var >= thresh) + 1
            components_for_threshold[thresh] = n_comp
        
        results = {
            'total_features': self.n_features,
            'max_components': max_components,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': cumsum_var.tolist(),
            'components_for_threshold': components_for_threshold,
        }
        
        return results
    
    def print_pca_summary(self):
        """Print formatted PCA analysis"""
        results = self.analyze_pca()
        
        print("\n" + "="*60)
        print("📊 PCA ANALYSIS")
        print("="*60)
        
        print(f"\nOriginal Features: {results['total_features']}")
        
        print(f"\n💡 Components needed for variance thresholds:")
        for thresh, n_comp in results['components_for_threshold'].items():
            pct = thresh * 100
            reduction = (1 - n_comp/results['total_features']) * 100
            print(f"  {pct:.0f}% variance: {n_comp} components ({reduction:.0f}% reduction)")
        
        # Top 10 components
        print(f"\n📈 Top 10 components explained variance:")
        for i, var in enumerate(results['explained_variance_ratio'][:10]):
            cumvar = results['cumulative_variance'][i]
            print(f"  PC{i+1}: {var*100:.2f}% (cumulative: {cumvar*100:.1f}%)")
        
        print("\n" + "="*60 + "\n")
    
    def plot_variance_explained(self, save_path: Path = None):
        """
        Plot explained variance by component.
        
        Args:
            save_path: Optional path to save plot
        """
        if self._pca is None:
            self.analyze_pca()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Individual variance
        ax = axes[0]
        n_components = len(self._pca.explained_variance_ratio_)
        components = np.arange(1, n_components + 1)
        
        ax.bar(components[:50], self._pca.explained_variance_ratio_[:50])
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('Explained Variance by Component (First 50)')
        ax.grid(True, alpha=0.3)
        
        # Cumulative variance
        ax = axes[1]
        cumsum_var = np.cumsum(self._pca.explained_variance_ratio_)
        ax.plot(components, cumsum_var, linewidth=2)
        
        # Add threshold lines
        thresholds = [0.80, 0.90, 0.95]
        colors = ['red', 'orange', 'green']
        for thresh, color in zip(thresholds, colors):
            ax.axhline(thresh, color=color, linestyle='--', alpha=0.5, 
                      label=f'{int(thresh*100)}% variance')
            n_comp = np.argmax(cumsum_var >= thresh) + 1
            ax.axvline(n_comp, color=color, linestyle='--', alpha=0.5)
            ax.text(n_comp, thresh, f'  {n_comp}', color=color, fontweight='bold')
        
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.set_title('Cumulative Explained Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved plot to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_component_loadings(self, n_components: int = 3, save_path: Path = None):
        """
        Plot loadings for top components.
        
        Args:
            n_components: Number of components to plot
            save_path: Optional path to save plot
        """
        if self._pca is None:
            self.analyze_pca()
        
        # Get loadings (components)
        loadings = self._pca.components_[:n_components]
        
        fig, axes = plt.subplots(n_components, 1, figsize=(12, 4*n_components))
        if n_components == 1:
            axes = [axes]
        
        for i, (loading, ax) in enumerate(zip(loadings, axes)):
            feature_indices = np.arange(len(loading))
            
            # Plot loadings
            colors = ['red' if x < 0 else 'blue' for x in loading]
            ax.bar(feature_indices, loading, color=colors, alpha=0.6)
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Loading')
            ax.set_title(f'PC{i+1} Loadings (Variance: {self._pca.explained_variance_ratio_[i]*100:.1f}%)')
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='black', linewidth=0.5)
            
            # Highlight top contributors
            top_indices = np.argsort(np.abs(loading))[-10:]
            for idx in top_indices:
                ax.text(idx, loading[idx], f'{idx}', fontsize=8, ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved plot to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compare_strategies(self) -> Dict[str, Dict]:
        """
        Compare different preprocessing strategies.
        
        Returns:
            Dictionary with comparison metrics
        """
        strategies = {
            'none': {'scale': False, 'pca': None},
            'scale_only': {'scale': True, 'pca': None},
            'pca_20': {'scale': True, 'pca': 20},
            'pca_50': {'scale': True, 'pca': 50},
            'pca_95': {'scale': True, 'pca': 0.95},
        }
        
        results = {}
        
        for name, strategy in strategies.items():
            X_transformed = self.X.copy()
            n_features = self.n_features
            
            # Apply scaling
            if strategy['scale']:
                scaler = StandardScaler()
                X_transformed = scaler.fit_transform(X_transformed)
            
            # Apply PCA
            if strategy['pca'] is not None:
                pca = PCA(n_components=strategy['pca'])
                X_transformed = pca.fit_transform(X_transformed)
                n_features = X_transformed.shape[1]
                variance_explained = sum(pca.explained_variance_ratio_)
            else:
                variance_explained = 1.0
            
            results[name] = {
                'n_features': n_features,
                'variance_explained': variance_explained,
                'reduction_pct': (1 - n_features/self.n_features) * 100,
            }
        
        return results
    
    def print_strategy_comparison(self):
        """Print comparison of preprocessing strategies"""
        results = self.compare_strategies()
        
        print("\n" + "="*60)
        print("⚖️  PREPROCESSING STRATEGY COMPARISON")
        print("="*60)
        
        print(f"\nOriginal: {self.n_features} features")
        
        for name, metrics in results.items():
            print(f"\n{name}:")
            print(f"  Features: {metrics['n_features']} ({metrics['reduction_pct']:.0f}% reduction)")
            print(f"  Variance: {metrics['variance_explained']*100:.1f}%")
        
        print("\n💡 Recommendations:")
        print("  - 'none': Keep all features (good for interpretability)")
        print("  - 'scale_only': Normalize values (good for neural networks)")
        print("  - 'pca_20': Fast training, some info loss")
        print("  - 'pca_50': Balanced speed/performance")
        print("  - 'pca_95': Preserve most info, slower training")
        
        print("\n" + "="*60 + "\n")
    
    def recommend_n_components(self) -> Dict[str, int]:
        """
        Recommend number of PCA components based on dataset.
        
        Returns:
            Dictionary with recommendations
        """
        if self._pca is None:
            self.analyze_pca()
        
        cumsum_var = np.cumsum(self._pca.explained_variance_ratio_)
        
        recommendations = {
            'fast_training': max(10, np.argmax(cumsum_var >= 0.80) + 1),
            'balanced': max(20, np.argmax(cumsum_var >= 0.90) + 1),
            'high_quality': max(30, np.argmax(cumsum_var >= 0.95) + 1),
        }
        
        return recommendations
