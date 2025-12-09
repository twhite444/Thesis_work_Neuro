"""
PCA-based Global Masking for Olfactory Feature Selection

This module implements the PCA masking approach from legacy/pca_copy.py.
Instead of using PCA-transformed features for training, we:
1. Apply PCA to understand feature importance
2. Create a global mask based on PCA component loadings
3. Apply the mask to ORIGINAL standardized features
4. Train on the masked original features

This preserves interpretability while using PCA for feature selection.

Copyright (c) 2025 Tom White
Licensed under the MIT License
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional


class PCAMasking:
    """
    PCA-based feature masking for olfactory predictions.
    
    Implements the approach from legacy/pca_copy.py:
    - Uses PCA loadings to identify important features
    - Creates binary mask based on mean absolute loading across components
    - Applies mask to original (standardized) features, not PCA features
    
    Key Insight:
        The mask is applied to the ORIGINAL standardized features, NOT the
        PCA-transformed features. This maintains interpretability while using
        PCA's understanding of feature importance for selection.
    
    Example:
        >>> masker = PCAMasking(n_components=50, threshold=0.1)
        >>> X_masked, mask = masker.fit_transform(X_standardized)
        >>> masker.visualize(output_dir='experiments/baseline/pca_analysis')
    """
    
    def __init__(
        self,
        n_components: int = 50,
        threshold: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize PCA masking.
        
        Args:
            n_components: Number of PCA components (or fraction for variance threshold)
            threshold: Threshold for global mask (features with mean loading > threshold are kept)
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.threshold = threshold
        self.random_state = random_state
        
        self.pca = None
        self.global_mask = None
        self.feature_importance = None
    
    def fit(self, X: np.ndarray) -> 'PCAMasking':
        """
        Fit PCA and compute global mask.
        
        Args:
            X: Standardized feature matrix (n_samples × n_features)
        
        Returns:
            self
        """
        # Apply PCA
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca.fit(X)
        
        # Compute global mask from PCA loadings
        # Mean absolute loading across all components
        # This identifies features that are important across multiple components
        self.feature_importance = np.abs(self.pca.components_).mean(axis=0)
        self.global_mask = self.feature_importance > self.threshold
        
        n_features_selected = self.global_mask.sum()
        n_features_total = len(self.global_mask)
        
        print(f"\n🎭 PCA Masking Results:")
        print(f"   PCA components: {self.pca.n_components_}")
        print(f"   Variance explained: {self.pca.explained_variance_ratio_.sum()*100:.2f}%")
        print(f"   Masking threshold: {self.threshold}")
        print(f"   Features selected: {n_features_selected} / {n_features_total} "
              f"({n_features_selected/n_features_total*100:.1f}%)")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply global mask to features.
        
        Args:
            X: Feature matrix to mask (should be the ORIGINAL standardized features)
        
        Returns:
            X_masked: Features with mask applied (n_samples × n_features_selected)
        """
        if self.global_mask is None:
            raise ValueError("Must call fit() before transform()")
        
        return X[:, self.global_mask]
    
    def fit_transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit PCA masking and apply to features.
        
        Args:
            X: Standardized feature matrix
        
        Returns:
            X_masked: Masked features
            mask: Boolean mask array
        """
        self.fit(X)
        X_masked = self.transform(X)
        return X_masked, self.global_mask
    
    def visualize(self, output_dir: str, feature_names: Optional[list] = None):
        """
        Create PCA analysis visualizations (matching legacy/pca_copy.py).
        
        Generates:
        - global_mask.png: Feature importance across all components
        - top_3_components.png: Loadings for first 3 components
        - pca_scree.png: Explained variance per component
        - pca_cumulative.png: Cumulative explained variance
        
        Args:
            output_dir: Directory to save plots
            feature_names: Optional list of feature names for labeling
        """
        if self.pca is None:
            raise ValueError("Must call fit() before visualize()")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📊 Generating PCA visualizations...")
        
        # 1. Global mask visualization (like legacy global_mask.png)
        plt.figure(figsize=(20, 6))
        x = np.arange(len(self.feature_importance))
        colors = ['green' if mask else 'red' for mask in self.global_mask]
        plt.bar(x, self.feature_importance, color=colors, alpha=0.6)
        plt.axhline(y=self.threshold, color='blue', linestyle='--', linewidth=2,
                   label=f'Threshold={self.threshold}')
        plt.xlabel('Feature Index', fontsize=12)
        plt.ylabel('Mean Absolute Loading', fontsize=12)
        plt.title(f'Global Feature Importance Mask ({self.global_mask.sum()} features selected)', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(output_path / 'global_mask.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {output_path / 'global_mask.png'}")
        
        # 2. Top 3 components (like legacy top_3_components.png)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i in range(min(3, self.pca.n_components_)):
            axes[i].bar(range(len(self.pca.components_[i])), 
                       np.abs(self.pca.components_[i]))
            axes[i].set_title(f'Component {i+1} Loadings\n'
                            f'(Var: {self.pca.explained_variance_ratio_[i]*100:.1f}%)',
                            fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Feature Index', fontsize=10)
            axes[i].set_ylabel('Absolute Loading', fontsize=10)
        plt.tight_layout()
        plt.savefig(output_path / 'top_3_components.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {output_path / 'top_3_components.png'}")
        
        # 3. Scree plot
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(self.pca.explained_variance_ratio_) + 1),
                self.pca.explained_variance_ratio_)
        plt.xlabel('Principal Component', fontsize=12)
        plt.ylabel('Explained Variance Ratio', fontsize=12)
        plt.title('PCA Scree Plot', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'pca_scree.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {output_path / 'pca_scree.png'}")
        
        # 4. Cumulative variance
        cumsum = np.cumsum(self.pca.explained_variance_ratio_)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumsum) + 1), cumsum, marker='o', linewidth=2)
        plt.xlabel('Number of Components', fontsize=12)
        plt.ylabel('Cumulative Explained Variance', fontsize=12)
        plt.title('PCA Cumulative Variance', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add horizontal lines for common thresholds
        for threshold in [0.8, 0.9, 0.95]:
            if cumsum[-1] >= threshold:
                n_comp = np.argmax(cumsum >= threshold) + 1
                plt.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5)
                plt.text(n_comp, threshold, f'{threshold:.0%} ({n_comp} comp)', 
                        verticalalignment='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path / 'pca_cumulative.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {output_path / 'pca_cumulative.png'}")
    
    def get_info(self) -> dict:
        """Get PCA masking information for logging."""
        if self.pca is None:
            return {}
        
        return {
            'n_components': self.pca.n_components_,
            'variance_explained': self.pca.explained_variance_ratio_.sum(),
            'threshold': self.threshold,
            'features_selected': self.global_mask.sum() if self.global_mask is not None else 0,
            'features_total': len(self.global_mask) if self.global_mask is not None else 0,
            'reduction_percent': (1 - self.global_mask.sum() / len(self.global_mask)) * 100 
                                 if self.global_mask is not None else 0
        }
    
    def save_mask(self, filepath: str):
        """
        Save the global mask to a file for reproducibility.
        
        Args:
            filepath: Path to save mask (as CSV)
        """
        if self.global_mask is None:
            raise ValueError("Must call fit() before save_mask()")
        
        mask_df = pd.DataFrame({
            'feature_index': np.arange(len(self.global_mask)),
            'selected': self.global_mask,
            'importance': self.feature_importance
        })
        mask_df.to_csv(filepath, index=False)
        print(f"   💾 Saved mask to: {filepath}")
    
    def load_mask(self, filepath: str):
        """
        Load a previously saved mask.
        
        Args:
            filepath: Path to mask CSV file
        """
        mask_df = pd.read_csv(filepath)
        self.global_mask = mask_df['selected'].values.astype(bool)
        self.feature_importance = mask_df['importance'].values
        print(f"   📂 Loaded mask from: {filepath}")
        print(f"      Features selected: {self.global_mask.sum()} / {len(self.global_mask)}")
