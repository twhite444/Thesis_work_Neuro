"""Test suite for visualization improvements and utility functions.

Tests the following improvements:
- Metrics utility functions (correlation, MSE, MAE, sampling)
- Input validation for plot_feature_importance
- Error handling in visualization functions
- Type conversion guards (torch.Tensor → numpy)
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile

from olfactory_modeling.utils.metrics import (
    to_numpy,
    flatten_arrays,
    compute_correlation,
    compute_mse,
    compute_mae,
    compute_statistics,
    sample_for_plotting,
    DEFAULT_MAX_SCATTER_POINTS,
)
from olfactory_modeling.visualization.training_viz import plot_feature_importance


# ==================== Test Utility Functions ====================

class TestMetricsUtilities:
    """Test suite for metrics.py utility functions."""
    
    def test_to_numpy_with_numpy_array(self):
        """Test that numpy arrays pass through unchanged."""
        arr = np.array([1, 2, 3])
        result = to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)
    
    def test_to_numpy_with_torch_tensor(self):
        """Test conversion from torch.Tensor to numpy."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = to_numpy(tensor)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])
    
    def test_to_numpy_with_cuda_tensor(self):
        """Test conversion from CUDA tensor to numpy."""
        if torch.cuda.is_available():
            tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
            result = to_numpy(tensor)
            assert isinstance(result, np.ndarray)
            np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])
    
    def test_to_numpy_with_gradient_tensor(self):
        """Test conversion from tensor with gradients."""
        tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = to_numpy(tensor)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])
    
    def test_to_numpy_with_invalid_type(self):
        """Test that invalid types raise TypeError."""
        with pytest.raises(TypeError, match="Expected numpy array or torch.Tensor"):
            to_numpy([1, 2, 3])  # list is invalid
    
    def test_flatten_arrays_single_array(self):
        """Test flattening a single array."""
        arr = np.array([[1, 2], [3, 4]])
        result = flatten_arrays(arr)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [1, 2, 3, 4])
    
    def test_flatten_arrays_multiple_arrays(self):
        """Test flattening multiple arrays simultaneously."""
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        result1, result2 = flatten_arrays(arr1, arr2)
        np.testing.assert_array_equal(result1, [1, 2, 3, 4])
        np.testing.assert_array_almost_equal(result2, [5.0, 6.0, 7.0, 8.0])
    
    def test_compute_correlation_perfect_correlation(self):
        """Test correlation with perfectly correlated data."""
        predictions = np.array([1, 2, 3, 4, 5])
        targets = np.array([1, 2, 3, 4, 5])
        corr = compute_correlation(predictions, targets)
        assert abs(corr - 1.0) < 1e-6
    
    def test_compute_correlation_no_correlation(self):
        """Test correlation with uncorrelated data."""
        np.random.seed(42)
        predictions = np.random.randn(100)
        targets = np.random.randn(100)
        corr = compute_correlation(predictions, targets)
        assert abs(corr) < 0.3  # Should be close to 0
    
    def test_compute_correlation_with_torch_tensors(self):
        """Test correlation computation with PyTorch tensors."""
        predictions = torch.randn(100)
        targets = predictions + 0.1 * torch.randn(100)
        corr = compute_correlation(predictions, targets)
        assert 0.7 < corr < 1.0  # Should be high correlation
    
    def test_compute_correlation_shape_mismatch(self):
        """Test that shape mismatch raises ValueError."""
        predictions = np.array([1, 2, 3])
        targets = np.array([1, 2])
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_correlation(predictions, targets)
    
    def test_compute_correlation_with_nan(self):
        """Test that NaN values raise ValueError."""
        predictions = np.array([1, 2, np.nan])
        targets = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="contain NaN or Inf"):
            compute_correlation(predictions, targets)
    
    def test_compute_correlation_with_inf(self):
        """Test that Inf values raise ValueError."""
        predictions = np.array([1, 2, np.inf])
        targets = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="contain NaN or Inf"):
            compute_correlation(predictions, targets)
    
    def test_compute_mse(self):
        """Test MSE computation."""
        predictions = np.array([1, 2, 3])
        targets = np.array([1, 2, 4])
        mse = compute_mse(predictions, targets)
        expected = (0**2 + 0**2 + 1**2) / 3
        assert abs(mse - expected) < 1e-6
    
    def test_compute_mae(self):
        """Test MAE computation."""
        predictions = np.array([1, 2, 3])
        targets = np.array([1, 2, 4])
        mae = compute_mae(predictions, targets)
        expected = (0 + 0 + 1) / 3
        assert abs(mae - expected) < 1e-6
    
    def test_compute_statistics(self):
        """Test that compute_statistics returns all metrics."""
        predictions = torch.randn(100)
        targets = predictions + 0.1 * torch.randn(100)
        stats = compute_statistics(predictions, targets)
        
        assert 'correlation' in stats
        assert 'mse' in stats
        assert 'mae' in stats
        assert 0.7 < stats['correlation'] < 1.0
        assert stats['mse'] > 0
        assert stats['mae'] > 0
    
    def test_sample_for_plotting_no_sampling_needed(self):
        """Test that small arrays are not sampled."""
        arr1 = np.arange(100)
        arr2 = np.arange(100) * 2
        result1, result2 = sample_for_plotting(arr1, arr2, max_points=1000)
        assert len(result1) == 100
        assert len(result2) == 100
    
    def test_sample_for_plotting_with_sampling(self):
        """Test that large arrays are sampled correctly."""
        arr1 = np.arange(100000)
        arr2 = np.arange(100000) * 2
        result1, result2 = sample_for_plotting(arr1, arr2, max_points=1000)
        assert len(result1) == 1000
        assert len(result2) == 1000
        # Check that samples come from original arrays
        assert np.all(np.isin(result1, arr1))
        assert np.all(np.isin(result2, arr2))
    
    def test_sample_for_plotting_reproducibility(self):
        """Test that sampling is reproducible with same seed."""
        arr = np.arange(100000)
        result1, = sample_for_plotting(arr, max_points=1000, random_seed=42)
        result2, = sample_for_plotting(arr, max_points=1000, random_seed=42)
        np.testing.assert_array_equal(result1, result2)


# ==================== Test Feature Importance Validation ====================

class TestFeatureImportanceValidation:
    """Test suite for plot_feature_importance input validation."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple MLP model for testing."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        # Initialize with small random weights
        with torch.no_grad():
            model[0].weight.normal_(0, 0.1)
            model[2].weight.normal_(0, 0.1)
        return model
    
    @pytest.fixture
    def feature_names(self):
        """Create feature names for testing."""
        return [f"Feature_{i}" for i in range(10)]
    
    def test_plot_feature_importance_basic(self, simple_model, feature_names):
        """Test basic functionality with valid inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "feature_importance.png"
            fig = plot_feature_importance(
                simple_model,
                feature_names=feature_names,
                top_n=5,
                output_path=output_path
            )
            assert fig is not None
            assert output_path.exists()
    
    def test_plot_feature_importance_without_feature_names(self, simple_model):
        """Test that feature names are auto-generated if not provided."""
        fig = plot_feature_importance(simple_model, top_n=5)
        assert fig is not None
    
    def test_plot_feature_importance_invalid_model_type(self, feature_names):
        """Test that non-Module types raise TypeError."""
        with pytest.raises(TypeError, match="Model must be a torch.nn.Module"):
            plot_feature_importance("not a model", feature_names=feature_names)
    
    def test_plot_feature_importance_model_without_linear_layer(self):
        """Test that models without Linear layers raise ValueError."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU()
        )
        with pytest.raises(ValueError, match="Could not find any Linear layer"):
            plot_feature_importance(model)
    
    def test_plot_feature_importance_invalid_top_n(self, simple_model):
        """Test that invalid top_n raises ValueError."""
        with pytest.raises(ValueError, match="top_n must be a positive integer"):
            plot_feature_importance(simple_model, top_n=-5)
        
        with pytest.raises(ValueError, match="top_n must be a positive integer"):
            plot_feature_importance(simple_model, top_n=0)
    
    def test_plot_feature_importance_wrong_feature_names_length(self, simple_model):
        """Test that wrong number of feature names raises ValueError."""
        wrong_names = ["Feature_1", "Feature_2"]  # Only 2 names for 10 features
        with pytest.raises(ValueError, match="Number of feature names .* does not match"):
            plot_feature_importance(simple_model, feature_names=wrong_names)
    
    def test_plot_feature_importance_invalid_feature_names_type(self, simple_model):
        """Test that invalid feature_names type raises TypeError."""
        with pytest.raises(TypeError, match="feature_names must be a list"):
            plot_feature_importance(simple_model, feature_names="not a list")
    
    def test_plot_feature_importance_top_n_exceeds_features(self, simple_model):
        """Test warning when top_n exceeds number of features."""
        # Should not raise, but should print warning and show all features
        fig = plot_feature_importance(simple_model, top_n=100)
        assert fig is not None
    
    def test_plot_feature_importance_with_nan_weights(self):
        """Test that NaN weights raise RuntimeError."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        # Set weights to NaN
        with torch.no_grad():
            model[0].weight[:] = float('nan')
        
        with pytest.raises(RuntimeError, match="contain invalid values"):
            plot_feature_importance(model)
    
    def test_plot_feature_importance_with_inf_weights(self):
        """Test that Inf weights raise RuntimeError."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        # Set weights to Inf
        with torch.no_grad():
            model[0].weight[:] = float('inf')
        
        with pytest.raises(RuntimeError, match="contain invalid values"):
            plot_feature_importance(model)


# ==================== Test MLP Architecture Pattern ====================

class MoleculeToActivityMapMLP(nn.Module):
    """Mock MLP model with network attribute (common pattern)."""
    
    def __init__(self, input_dim=268):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1680)  # 40 x 42 output
        )
    
    def forward(self, x):
        return self.network(x).view(-1, 40, 42)


class TestArchitecturePatterns:
    """Test that plot_feature_importance works with different architectures."""
    
    def test_with_mlp_network_attribute(self):
        """Test with model.network attribute (MLP pattern)."""
        model = MoleculeToActivityMapMLP(input_dim=268)
        with torch.no_grad():
            model.network[0].weight.normal_(0, 0.1)
        
        feature_names = [f"Descriptor_{i}" for i in range(268)]
        fig = plot_feature_importance(model, feature_names=feature_names, top_n=20)
        assert fig is not None
    
    def test_with_direct_linear_layer(self):
        """Test with direct nn.Linear first layer."""
        model = nn.Linear(50, 100)
        with torch.no_grad():
            model.weight.normal_(0, 0.1)
        
        fig = plot_feature_importance(model, top_n=10)
        assert fig is not None


# ==================== Integration Tests ====================

class TestVisualizationIntegration:
    """Integration tests for complete visualization workflow."""
    
    def test_end_to_end_feature_importance(self):
        """Test complete workflow from model to visualization."""
        # Create model
        model = MoleculeToActivityMapMLP(input_dim=100)
        
        # Train for a few steps (simulate training)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(5):
            x = torch.randn(8, 100)
            y = torch.randn(8, 40, 42)
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Create feature names
        feature_names = [f"Molecular_Feature_{i}" for i in range(100)]
        
        # Generate visualization
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_feature_importance.png"
            fig = plot_feature_importance(
                model,
                feature_names=feature_names,
                top_n=20,
                output_path=output_path,
                figsize=(12, 8),
                title="Test Feature Importance"
            )
            
            assert fig is not None
            assert output_path.exists()
            assert output_path.stat().st_size > 0  # File has content


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
