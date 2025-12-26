"""PCA dimensionality reduction for activity maps.

This module provides functionality to reduce activity maps to PCA components,
which can be used as alternative targets for neural network training.

Typical usage:
    # Fit PCA on training maps
    pca_model, pca_maps, metadata = fit_pca_on_maps(
        maps, cids, n_components=20, output_dir='data/02_processed'
    )
    
    # Later, transform new maps using fitted PCA
    new_pca_maps = transform_maps_with_pca(new_maps, pca_model)
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter


def fit_pca_on_maps(
    maps: np.ndarray,
    cids: np.ndarray,
    n_components: int = 20,
    output_dir: str = "data/02_processed",
    save_artifacts: bool = True,
    visualize: bool = True,
) -> Tuple[PCA, np.ndarray, Dict]:
    """Fit PCA on activity maps and transform them to principal components.
    
    Args:
        maps: Activity maps array with shape (n_samples, height, width)
        cids: Array of CIDs corresponding to each map
        n_components: Number of principal components to keep (default: 20)
        output_dir: Directory to save PCA model and transformed data
        save_artifacts: Whether to save PCA model and transformed data
        visualize: Whether to generate visualization plots
        
    Returns:
        Tuple of (fitted PCA model, transformed maps, metadata dict)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Flatten maps for PCA (samples x features)
    n_samples = maps.shape[0]
    flat_maps = maps.reshape(n_samples, -1)
    
    print(f"Fitting PCA on {n_samples} activity maps...")
    print(f"  Original shape: {maps.shape}")
    print(f"  Flattened shape: {flat_maps.shape}")
    
    # Standardize the data
    scaler = StandardScaler()
    standardized_maps = scaler.fit_transform(flat_maps)
    
    # Determine actual n_components (can't exceed min dimension)
    max_components = min(standardized_maps.shape)
    n_components = min(n_components, max_components)
    print(f"  Using {n_components} components (max possible: {max_components})")
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_transformed = pca.fit_transform(standardized_maps)
    
    # Calculate variance statistics
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    print(f"\nExplained variance ratio (first 5 components): {explained_var[:5]}")
    print(f"Cumulative variance (first 5): {cumulative_var[:5]}")
    print(f"Total variance explained by {n_components} components: {cumulative_var[-1]:.2%}")
    
    # Metadata
    metadata = {
        'n_components': n_components,
        'n_samples': n_samples,
        'original_shape': maps.shape,
        'explained_variance_ratio': explained_var,
        'cumulative_variance': cumulative_var,
        'total_variance_explained': cumulative_var[-1],
    }
    
    if save_artifacts:
        # Save PCA model and scaler
        pca_model_path = os.path.join(output_dir, 'pca_model.pkl')
        with open(pca_model_path, 'wb') as f:
            pickle.dump({'pca': pca, 'scaler': scaler, 'metadata': metadata}, f)
        print(f"\n✓ Saved PCA model to {pca_model_path}")
        
        # Save transformed data
        pca_data_path = os.path.join(output_dir, 'pca_transformed_maps.npz')
        np.savez(
            pca_data_path,
            pca_maps=pca_transformed,
            cids=cids,
            **metadata
        )
        print(f"✓ Saved PCA-transformed maps to {pca_data_path}")
        
        # Save as CSV for easy inspection
        pca_df = pd.DataFrame(
            pca_transformed,
            index=cids,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        pca_df.index.name = 'CID'
        csv_path = os.path.join(output_dir, 'pca_transformed_maps.csv')
        pca_df.to_csv(csv_path)
        print(f"✓ Saved PCA-transformed maps (CSV) to {csv_path}")
    
    if visualize:
        viz_dir = os.path.join('viz', 'pca')
        os.makedirs(viz_dir, exist_ok=True)
        _visualize_pca_results(pca, maps.shape[1:], explained_var, cumulative_var, viz_dir)
    
    return pca, pca_transformed, metadata


def transform_maps_with_pca(
    maps: np.ndarray,
    pca_model_path: str = "data/02_processed/pca_model.pkl"
) -> np.ndarray:
    """Transform activity maps using pre-fitted PCA model.
    
    Args:
        maps: Activity maps array with shape (n_samples, height, width)
        pca_model_path: Path to saved PCA model pickle file
        
    Returns:
        PCA-transformed maps with shape (n_samples, n_components)
    """
    # Load PCA model and scaler
    with open(pca_model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    pca = model_data['pca']
    scaler = model_data['scaler']
    
    # Flatten and standardize
    n_samples = maps.shape[0]
    flat_maps = maps.reshape(n_samples, -1)
    standardized_maps = scaler.transform(flat_maps)
    
    # Transform
    pca_transformed = pca.transform(standardized_maps)
    
    return pca_transformed


def load_pca_transformed_maps(data_dir: str = "data/02_processed") -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load pre-computed PCA-transformed activity maps.
    
    Args:
        data_dir: Directory containing pca_transformed_maps.npz
        
    Returns:
        Tuple of (pca_maps, cids, metadata)
    """
    pca_path = os.path.join(data_dir, 'pca_transformed_maps.npz')
    if not os.path.exists(pca_path):
        raise FileNotFoundError(
            f"PCA-transformed maps not found at {pca_path}. "
            "Run fit_pca_on_maps() first."
        )
    
    data = np.load(pca_path)
    pca_maps = data['pca_maps']
    cids = data['cids']
    
    # Extract metadata
    metadata = {}
    for k in data.files:
        if k not in ['pca_maps', 'cids']:
            val = data[k]
            # Handle arrays of different sizes
            if isinstance(val, np.ndarray):
                if val.size == 1:
                    metadata[k] = val.item()
                else:
                    metadata[k] = val
            else:
                metadata[k] = val
    
    return pca_maps, cids, metadata


def _visualize_pca_results(
    pca: PCA,
    original_shape: Tuple[int, int],
    explained_var: np.ndarray,
    cumulative_var: np.ndarray,
    output_dir: str
):
    """Generate visualization plots for PCA results.
    
    Args:
        pca: Fitted PCA model
        original_shape: Original map shape (height, width)
        explained_var: Explained variance ratio array
        cumulative_var: Cumulative explained variance array
        output_dir: Directory to save visualizations
    """
    # 1. Cumulative explained variance plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, marker='o', linewidth=2)
    plt.xlabel('Number of Components', fontsize=12)
    plt.ylabel('Cumulative Explained Variance', fontsize=12)
    plt.title('PCA: Cumulative Explained Variance', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    plt.axhline(y=0.90, color='orange', linestyle='--', label='90% variance')
    plt.legend()
    plt.tight_layout()
    variance_path = os.path.join(output_dir, 'pca_explained_variance.png')
    plt.savefig(variance_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved variance plot to {variance_path}")
    
    # 2. Top 3 principal components as spatial maps
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, ax in enumerate(axes):
        if i < pca.n_components_:
            component = pca.components_[i].reshape(original_shape)
            # Apply Gaussian smoothing for visualization
            smoothed = gaussian_filter(component, sigma=1)
            img = ax.imshow(smoothed, cmap='viridis')
            ax.set_title(f'PC{i+1} ({explained_var[i]:.1%} variance)', fontsize=12)
            fig.colorbar(img, ax=ax)
            ax.axis('off')
    plt.tight_layout()
    components_path = os.path.join(output_dir, 'pca_top3_components.png')
    plt.savefig(components_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved component maps to {components_path}")
    
    # 3. Spatial loadings for first 3 components
    for i in range(min(3, pca.n_components_)):
        component = pca.components_[i].reshape(original_shape)
        smoothed = gaussian_filter(component, sigma=1)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(smoothed, cmap='coolwarm')
        plt.title(f'Spatial Loadings: PC{i+1} ({explained_var[i]:.1%} variance)', fontsize=14)
        plt.colorbar(label='Loading Value')
        plt.axis('off')
        plt.tight_layout()
        
        loading_path = os.path.join(output_dir, f'pca_spatial_loadings_pc{i+1}.png')
        plt.savefig(loading_path, dpi=300)
        plt.close()
    
    print(f"  ✓ Saved spatial loadings to {output_dir}")


def visualize_pca_scatter_2d(
    pca_transformed: np.ndarray,
    cids: np.ndarray,
    output_dir: str = "viz",
    pc1: int = 0,
    pc2: int = 1,
    color_by: Optional[np.ndarray] = None,
    color_label: Optional[str] = None,
):
    """Create 2D scatter plot of PCA components.
    
    Args:
        pca_transformed: PCA-transformed data with shape (n_samples, n_components)
        cids: Array of CIDs corresponding to each sample
        output_dir: Directory to save visualization
        pc1: Index of first principal component to plot (0-indexed)
        pc2: Index of second principal component to plot (0-indexed)
        color_by: Optional array to color points by (e.g., prediction error)
        color_label: Label for color scale
    """
    if pca_transformed.shape[1] <= max(pc1, pc2):
        print(f"Error: Need at least {max(pc1, pc2) + 1} PCA components for visualization")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    
    if color_by is not None:
        scatter = plt.scatter(
            pca_transformed[:, pc1],
            pca_transformed[:, pc2],
            c=color_by,
            cmap='viridis',
            alpha=0.7,
            s=80,
            edgecolors='w'
        )
        plt.colorbar(scatter, label=color_label or 'Value')
    else:
        plt.scatter(
            pca_transformed[:, pc1],
            pca_transformed[:, pc2],
            color='blue',
            alpha=0.7,
            s=80,
            edgecolors='w'
        )
    
    plt.xlabel(f'Principal Component {pc1 + 1}', fontsize=14)
    plt.ylabel(f'Principal Component {pc2 + 1}', fontsize=14)
    plt.title('PCA Component Space: Activity Maps', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'pca_scatter_pc{pc1+1}_pc{pc2+1}.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ PCA scatter plot saved to: {output_path}")


# ============================================================================
# Pipeline Integration
# ============================================================================

def process_activity_maps_with_pca(
    data_dir: str = 'data/01_raw',
    output_dir: str = 'data/02_processed',
    n_components: int = 20,
    **process_kwargs
):
    """End-to-end pipeline: process activity maps AND compute PCA.
    
    This is a convenience function that:
    1. Processes raw activity maps (masking, selection, etc.)
    2. Fits PCA on the processed maps
    3. Saves both raw processed maps and PCA-transformed maps
    
    Args:
        data_dir: Directory containing raw activity map data
        output_dir: Directory to save processed data
        n_components: Number of PCA components to compute
        **process_kwargs: Additional arguments passed to process_activity_maps()
    """
    from olfactory_modeling.pipeline.activity_maps import process_activity_maps
    
    # Step 1: Process raw activity maps
    print("\n" + "="*80)
    print("STEP 1: Processing raw activity maps...")
    print("="*80)
    
    maps, cids, metadata = process_activity_maps(
        data_dir=data_dir,
        output_dir=output_dir,
        **process_kwargs
    )
    
    # Step 2: Fit PCA on processed maps
    print("\n" + "="*80)
    print(f"STEP 2: Fitting PCA with {n_components} components...")
    print("="*80)
    
    pca_model, pca_maps, pca_metadata = fit_pca_on_maps(
        maps=maps,
        cids=cids,
        n_components=n_components,
        output_dir=output_dir,
        save_artifacts=True,
        visualize=True
    )
    
    print("\n" + "="*80)
    print("✓ COMPLETE: Activity maps processed and PCA fitted")
    print("="*80)
    print(f"Outputs saved to: {output_dir}")
    print(f"  - processed_maps.npz (raw processed maps)")
    print(f"  - pca_model.pkl (fitted PCA model)")
    print(f"  - pca_transformed_maps.npz (PCA-transformed maps)")
    print(f"  - pca_transformed_maps.csv (PCA-transformed maps, readable)")
    
    return {
        'maps': maps,
        'pca_maps': pca_maps,
        'cids': cids,
        'pca_model': pca_model,
        'metadata': metadata,
        'pca_metadata': pca_metadata,
    }
