"""Tests for PCA transformation of activity maps."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from olfactory_modeling.pipeline.pca_transform import (
    fit_pca_on_maps,
    transform_maps_with_pca,
    load_pca_transformed_maps,
)


@pytest.fixture
def sample_activity_maps():
    """Create sample activity maps for testing.
    
    Uses structured data (linear combinations of basis patterns) rather than
    pure noise to ensure PCA stability and determinism in tests.
    """
    np.random.seed(42)
    n_samples = 50
    height, width = 79, 43
    
    # Create structured maps from a small set of basis patterns
    # This mimics real activity maps which have structure
    n_basis = 8
    basis_patterns = np.random.randn(n_basis, height, width).astype(np.float32)
    coefficients = np.random.randn(n_samples, n_basis).astype(np.float32)
    
    # Generate maps as linear combinations of basis patterns
    maps = np.zeros((n_samples, height, width), dtype=np.float32)
    for i in range(n_samples):
        for j in range(n_basis):
            maps[i] += coefficients[i, j] * basis_patterns[j]
    
    # Add small amount of noise for realism
    maps += 0.1 * np.random.randn(n_samples, height, width).astype(np.float32)
    
    cids = np.arange(1000, 1000 + n_samples)
    
    return maps, cids


def test_fit_pca_basic(sample_activity_maps):
    """Test basic PCA fitting on activity maps."""
    maps, cids = sample_activity_maps
    n_components = 10
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pca_model, pca_maps, metadata = fit_pca_on_maps(
            maps=maps,
            cids=cids,
            n_components=n_components,
            output_dir=tmpdir,
            save_artifacts=False,
            visualize=False,
        )
        
        # Check outputs
        assert pca_maps.shape == (len(maps), n_components)
        assert metadata['n_components'] == n_components
        assert metadata['n_samples'] == len(maps)
        assert len(metadata['explained_variance_ratio']) == n_components
        assert len(metadata['cumulative_variance']) == n_components


def test_fit_pca_saves_artifacts(sample_activity_maps):
    """Test that PCA artifacts are saved correctly."""
    maps, cids = sample_activity_maps
    n_components = 5
    
    with tempfile.TemporaryDirectory() as tmpdir:
        fit_pca_on_maps(
            maps=maps,
            cids=cids,
            n_components=n_components,
            output_dir=tmpdir,
            save_artifacts=True,
            visualize=False,
        )
        
        # Check files exist
        assert os.path.exists(os.path.join(tmpdir, 'pca_model.pkl'))
        assert os.path.exists(os.path.join(tmpdir, 'pca_transformed_maps.npz'))
        assert os.path.exists(os.path.join(tmpdir, 'pca_transformed_maps.csv'))


def test_transform_with_saved_pca(sample_activity_maps):
    """Test transforming new maps with saved PCA model."""
    maps, cids = sample_activity_maps
    n_components = 8
    
    # Split into train and test
    train_maps = maps[:40]
    test_maps = maps[40:]
    train_cids = cids[:40]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Fit on training data
        fit_pca_on_maps(
            maps=train_maps,
            cids=train_cids,
            n_components=n_components,
            output_dir=tmpdir,
            save_artifacts=True,
            visualize=False,
        )
        
        # Transform test data
        pca_model_path = os.path.join(tmpdir, 'pca_model.pkl')
        test_pca_maps = transform_maps_with_pca(test_maps, pca_model_path)
        
        # Check shape
        assert test_pca_maps.shape == (len(test_maps), n_components)


def test_load_pca_transformed_maps(sample_activity_maps):
    """Test loading saved PCA-transformed maps."""
    maps, cids = sample_activity_maps
    n_components = 12
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Fit and save
        fit_pca_on_maps(
            maps=maps,
            cids=cids,
            n_components=n_components,
            output_dir=tmpdir,
            save_artifacts=True,
            visualize=False,
        )
        
        # Load
        loaded_pca_maps, loaded_cids, metadata = load_pca_transformed_maps(tmpdir)
        
        # Check
        assert loaded_pca_maps.shape == (len(maps), n_components)
        assert np.array_equal(loaded_cids, cids)
        assert metadata['n_components'] == n_components


def test_pca_variance_explained(sample_activity_maps):
    """Test that explained variance increases with more components."""
    maps, cids = sample_activity_maps
    
    with tempfile.TemporaryDirectory() as tmpdir:
        _, _, metadata_5 = fit_pca_on_maps(
            maps, cids, n_components=5, output_dir=tmpdir,
            save_artifacts=False, visualize=False
        )
        
        _, _, metadata_10 = fit_pca_on_maps(
            maps, cids, n_components=10, output_dir=tmpdir,
            save_artifacts=False, visualize=False
        )
        
        # More components should explain more variance
        assert metadata_10['total_variance_explained'] > metadata_5['total_variance_explained']


def test_pca_max_components(sample_activity_maps):
    """Test that n_components is limited by data dimensions."""
    maps, cids = sample_activity_maps[:10]  # Only 10 samples
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Request more components than samples
        _, pca_maps, metadata = fit_pca_on_maps(
            maps, cids, n_components=50, output_dir=tmpdir,
            save_artifacts=False, visualize=False
        )
        
        # Should be limited to min(n_samples, n_flattened_features)
        n_samples = maps.shape[0]
        n_features = maps.shape[1] * maps.shape[2]  # Flattened dimension
        assert metadata['n_components'] <= min(n_samples, n_features)
        assert pca_maps.shape[1] == metadata['n_components']


def test_pca_deterministic(sample_activity_maps):
    """Test that PCA is deterministic (same input → same output)."""
    maps, cids = sample_activity_maps
    n_components = 6
    
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        _, pca_maps1, _ = fit_pca_on_maps(
            maps, cids, n_components=n_components, output_dir=tmpdir1,
            save_artifacts=False, visualize=False
        )
        
        _, pca_maps2, _ = fit_pca_on_maps(
            maps, cids, n_components=n_components, output_dir=tmpdir2,
            save_artifacts=False, visualize=False
        )
        
        # PCA results should be identical (sklearn PCA is deterministic with same input)
        # Note: Sign flip is allowed, so we check correlation instead of exact equality
        for i in range(n_components):
            # Each component should be perfectly correlated (either +1 or -1)
            corr = np.corrcoef(pca_maps1[:, i], pca_maps2[:, i])[0, 1]
            assert np.abs(corr) > 0.99, f"Component {i} correlation: {corr}"
