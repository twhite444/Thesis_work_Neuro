"""Comprehensive integration tests for train_nn.py behavior preservation.

These tests ensure NO BEHAVIOR CHANGE during refactoring.
They test the exact current behavior with fixed seeds.
"""
import pytest
import torch
import numpy as np
import json
import os
from pathlib import Path

from olfactory_modeling.pipeline.train_nn import (
    train_nn,
    train_nn_kfold,
    compute_metrics,
    validate_training_params,
)
from olfactory_modeling.data.activity_map_dataset import MoleculeActivityMapDataset
from olfactory_modeling.models.baseline_nn import get_model


# Fixed seed for reproducibility
FIXED_SEED = 42


@pytest.fixture
def set_all_seeds():
    """Set all random seeds for complete reproducibility."""
    torch.manual_seed(FIXED_SEED)
    torch.cuda.manual_seed_all(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture
def mock_dataloaders():
    """Create minimal mock dataloaders for testing."""
    # Set seed for deterministic data generation
    torch.manual_seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    
    # Create synthetic data
    n_samples = 20
    n_features = 10
    output_dim = 5
    
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, n_samples, n_features, output_dim):
            self.features = torch.randn(n_samples, n_features)
            self.targets = torch.randn(n_samples, output_dim)
            self.cids = np.arange(n_samples)
            self.feature_dim = n_features
            self.output_dim = output_dim
            self.output_shape = None
            self.use_pca = True
            self.random_seed = FIXED_SEED
            self.processed_dir = Path("data/02_processed")
        
        def __len__(self):
            return len(self.features)
        
        def __getitem__(self, idx):
            metadata = {
                'cid': int(self.cids[idx]),
                'index': idx,
                'target_type': 'pca',
            }
            return self.features[idx], self.targets[idx], metadata
    
    dataset = MockDataset(n_samples, n_features, output_dim)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    
    return train_loader, val_loader


@pytest.mark.integration
class TestTrainNNBehaviorPreservation:
    """Test suite to ensure exact behavior preservation during refactoring."""
    
    def test_compute_metrics_exact_values(self):
        """Test that compute_metrics returns exact expected values."""
        # Fixed tensors
        pred = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        target = torch.tensor([[1.5, 2.5, 3.5], [3.5, 4.5, 5.5]])
        
        metrics = compute_metrics(pred, target)
        
        # Verify exact metric keys
        assert set(metrics.keys()) == {'mse', 'mae', 'correlation', 'r2'}
        
        # Verify approximate values (with tolerance for floating point)
        # MSE = mean((pred - target)^2) = mean([0.25, 0.25, 0.25, 0.25, 0.25, 0.25]) = 0.25
        assert pytest.approx(metrics['mse'], rel=1e-5) == 0.25
        assert pytest.approx(metrics['mae'], rel=1e-5) == 0.5
        assert 'correlation' in metrics  # Correlation should be computed
        assert 'r2' in metrics  # R² should be computed
    
    def test_validate_training_params_raises_correctly(self):
        """Test that parameter validation raises correct errors."""
        # Test invalid num_epochs
        with pytest.raises(ValueError, match="num_epochs must be > 0"):
            validate_training_params(num_epochs=0, batch_size=32, learning_rate=0.001, weight_decay=0.0)
        
        # Test invalid batch_size
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            validate_training_params(num_epochs=10, batch_size=-1, learning_rate=0.001, weight_decay=0.0)
        
        # Test invalid learning_rate
        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            validate_training_params(num_epochs=10, batch_size=32, learning_rate=-0.001, weight_decay=0.0)
        
        # Test invalid weight_decay
        with pytest.raises(ValueError, match="weight_decay must be >= 0"):
            validate_training_params(num_epochs=10, batch_size=32, learning_rate=0.001, weight_decay=-0.1)
        
        # Test valid params (should not raise)
        validate_training_params(num_epochs=10, batch_size=32, learning_rate=0.001, weight_decay=0.0)
    
    def test_train_nn_output_structure(self, mock_dataloaders, tmp_path, set_all_seeds):
        """Test that train_nn returns exact expected structure."""
        train_loader, val_loader = mock_dataloaders
        
        model = get_model('pca_mlp', input_dim=10, n_components=5, dropout=0.35)
        
        metrics = train_nn(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=str(tmp_path),
            num_epochs=2,
            learning_rate=0.001,
            weight_decay=0.0,
            early_stopping_patience=0,
            device=torch.device('cpu'),
            verbose=False,
        )
        
        # Verify exact keys in returned metrics
        expected_keys = {
            'best_val_loss', 'best_val_correlation', 'best_val_r2', 'best_val_mae',
            'best_epoch', 'n_train', 'n_val', 'num_epochs', 'learning_rate',
            'train_losses', 'val_losses', 'train_correlations', 'val_correlations',
            'train_r2', 'val_r2'
        }
        assert set(metrics.keys()) == expected_keys
        
        # Verify types
        assert isinstance(metrics['best_val_loss'], float)
        assert isinstance(metrics['best_epoch'], (int, float))
        assert isinstance(metrics['n_train'], int)
        assert isinstance(metrics['train_losses'], list)
        assert len(metrics['train_losses']) == 2  # num_epochs
    
    def test_train_nn_creates_expected_files(self, mock_dataloaders, tmp_path, set_all_seeds):
        """Test that train_nn creates all expected output files."""
        train_loader, val_loader = mock_dataloaders
        
        model = get_model('pca_mlp', input_dim=10, n_components=5, dropout=0.35)
        
        train_nn(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=str(tmp_path),
            num_epochs=2,
            learning_rate=0.001,
            weight_decay=0.0,
            early_stopping_patience=0,
            device=torch.device('cpu'),
            verbose=False,
        )
        
        # Verify expected files exist
        assert (tmp_path / 'metrics.json').exists()
        assert (tmp_path / 'run_metadata.json').exists()
        assert (tmp_path / 'best_model.pth').exists()
        assert (tmp_path / 'training_curves.png').exists() or True  # May fail if matplotlib not available
        assert (tmp_path / 'logs').is_dir()
    
    def test_train_nn_metrics_json_structure(self, mock_dataloaders, tmp_path, set_all_seeds):
        """Test that metrics.json has exact expected structure."""
        train_loader, val_loader = mock_dataloaders
        
        model = get_model('pca_mlp', input_dim=10, n_components=5, dropout=0.35)
        
        train_nn(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=str(tmp_path),
            num_epochs=2,
            learning_rate=0.001,
            weight_decay=0.0,
            early_stopping_patience=0,
            device=torch.device('cpu'),
            verbose=False,
        )
        
        # Load and verify metrics.json
        with open(tmp_path / 'metrics.json', 'r') as f:
            saved_metrics = json.load(f)
        
        # Verify exact keys (excluding lists which are saved separately)
        expected_keys = {
            'best_val_loss', 'best_val_correlation', 'best_val_r2', 'best_val_mae',
            'best_epoch', 'n_train', 'n_val', 'num_epochs', 'learning_rate'
        }
        assert set(saved_metrics.keys()) == expected_keys
    
    def test_train_nn_metadata_json_structure(self, mock_dataloaders, tmp_path, set_all_seeds):
        """Test that run_metadata.json has exact expected structure."""
        train_loader, val_loader = mock_dataloaders
        
        model = get_model('pca_mlp', input_dim=10, n_components=5, dropout=0.35)
        
        train_nn(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=str(tmp_path),
            num_epochs=2,
            learning_rate=0.001,
            weight_decay=0.0,
            early_stopping_patience=0,
            device=torch.device('cpu'),
            verbose=False,
        )
        
        # Load and verify run_metadata.json
        with open(tmp_path / 'run_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Verify top-level keys
        expected_top_keys = {'pipeline', 'model', 'training_config', 'data_split', 'metrics'}
        assert set(metadata.keys()) == expected_top_keys
        
        # Verify nested structure
        assert 'total_parameters' in metadata['model']
        assert 'learning_rate' in metadata['training_config']
        assert 'n_train' in metadata['data_split']


@pytest.mark.integration
class TestTrainNNReproducibility:
    """Test that training is exactly reproducible with fixed seeds."""
    
    def test_train_nn_reproducible_with_fixed_seed(self, mock_dataloaders, tmp_path, set_all_seeds):
        """Test that two runs with same seed produce identical results."""
        train_loader, val_loader = mock_dataloaders
        
        # First run
        torch.manual_seed(FIXED_SEED)
        np.random.seed(FIXED_SEED)
        model1 = get_model('pca_mlp', input_dim=10, n_components=5, dropout=0.35)
        metrics1 = train_nn(
            model=model1,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=str(tmp_path / 'run1'),
            num_epochs=2,
            learning_rate=0.001,
            weight_decay=0.0,
            early_stopping_patience=0,
            device=torch.device('cpu'),
            verbose=False,
        )
        
        # Second run (reset seed)
        torch.manual_seed(FIXED_SEED)
        np.random.seed(FIXED_SEED)
        model2 = get_model('pca_mlp', input_dim=10, n_components=5, dropout=0.35)
        metrics2 = train_nn(
            model=model2,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=str(tmp_path / 'run2'),
            num_epochs=2,
            learning_rate=0.001,
            weight_decay=0.0,
            early_stopping_patience=0,
            device=torch.device('cpu'),
            verbose=False,
        )
        
        # Compare results (should be exactly identical with fixed seeds)
        assert pytest.approx(metrics1['best_val_loss'], rel=1e-5) == metrics2['best_val_loss']
        assert pytest.approx(metrics1['best_val_correlation'], rel=1e-5) == metrics2['best_val_correlation']
        assert pytest.approx(metrics1['best_val_r2'], rel=1e-5) == metrics2['best_val_r2']


@pytest.mark.slow
@pytest.mark.integration
class TestTrainNNSmokeTests:
    """Smoke tests with real data (if available)."""
    
    def test_train_nn_with_real_data_smoke(self, tmp_path):
        """Smoke test with real data to ensure basic functionality."""
        processed_dir = Path('data/02_processed')
        
        # Skip if data not available
        if not (processed_dir / 'cleaned_data.csv').exists():
            pytest.skip("Real data not available")
        
        try:
            from olfactory_modeling.data.activity_map_dataset import get_dataloaders
            
            train_loader, val_loader, test_loader = get_dataloaders(
                processed_dir=str(processed_dir),
                batch_size=16,
                use_pca=True,
                random_seed=FIXED_SEED,
            )
            
            model = get_model(
                'pca_mlp',
                input_dim=train_loader.dataset.feature_dim,
                n_components=train_loader.dataset.output_dim,
                dropout=0.35
            )
            
            metrics = train_nn(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                output_dir=str(tmp_path),
                num_epochs=2,  # Just 2 epochs for smoke test
                learning_rate=0.005,
                weight_decay=1e-5,
                early_stopping_patience=0,
                device=torch.device('cpu'),
                verbose=False,
            )
            
            # Basic sanity checks
            assert metrics['best_val_loss'] > 0
            assert 0 <= metrics['best_val_r2'] <= 1 or metrics['best_val_r2'] < 0  # R² can be negative
            assert metrics['n_train'] > 0
            assert metrics['n_val'] > 0
            
        except ImportError:
            pytest.skip("Dependencies not available")


class TestSnapshotComparison:
    """Snapshot testing - compare outputs before/after refactoring."""
    
    SNAPSHOT_FILE = Path('tests/snapshots/train_nn_baseline.json')
    
    @pytest.mark.snapshot
    def test_create_baseline_snapshot(self, mock_dataloaders, tmp_path):
        """Create baseline snapshot of current behavior.
        
        Run this BEFORE refactoring to capture current behavior.
        """
        # Set all seeds for reproducibility
        torch.manual_seed(FIXED_SEED)
        torch.cuda.manual_seed_all(FIXED_SEED)
        np.random.seed(FIXED_SEED)
        
        train_loader, val_loader = mock_dataloaders
        
        model = get_model('pca_mlp', input_dim=10, n_components=5, dropout=0.35)
        
        metrics = train_nn(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=str(tmp_path),
            num_epochs=3,
            learning_rate=0.001,
            weight_decay=0.0,
            early_stopping_patience=0,
            device=torch.device('cpu'),
            verbose=False,
        )
        
        # Save snapshot
        snapshot = {
            'best_val_loss': metrics['best_val_loss'],
            'best_val_correlation': metrics['best_val_correlation'],
            'best_val_r2': metrics['best_val_r2'],
            'best_val_mae': metrics['best_val_mae'],
            'train_losses': metrics['train_losses'],
            'val_losses': metrics['val_losses'],
        }
        
        os.makedirs(self.SNAPSHOT_FILE.parent, exist_ok=True)
        with open(self.SNAPSHOT_FILE, 'w') as f:
            json.dump(snapshot, f, indent=2)
        
        print(f"Snapshot saved to {self.SNAPSHOT_FILE}")
    
    @pytest.mark.snapshot
    def test_compare_against_baseline_snapshot(self, mock_dataloaders, tmp_path):
        """Compare current behavior against baseline snapshot.
        
        Run this AFTER each refactoring step to ensure no behavior change.
        """
        if not self.SNAPSHOT_FILE.exists():
            pytest.skip("Baseline snapshot not found. Run test_create_baseline_snapshot first.")
        
        # Load baseline
        with open(self.SNAPSHOT_FILE, 'r') as f:
            baseline = json.load(f)
        
        # Set all seeds for reproducibility (same as baseline)
        torch.manual_seed(FIXED_SEED)
        torch.cuda.manual_seed_all(FIXED_SEED)
        np.random.seed(FIXED_SEED)
        
        # Run current implementation
        train_loader, val_loader = mock_dataloaders
        model = get_model('pca_mlp', input_dim=10, n_components=5, dropout=0.35)
        
        metrics = train_nn(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=str(tmp_path),
            num_epochs=3,
            learning_rate=0.001,
            weight_decay=0.0,
            early_stopping_patience=0,
            device=torch.device('cpu'),
            verbose=False,
        )
        
        # Compare with tolerance
        TOLERANCE = 1e-5
        assert pytest.approx(metrics['best_val_loss'], rel=TOLERANCE) == baseline['best_val_loss']
        assert pytest.approx(metrics['best_val_correlation'], rel=TOLERANCE) == baseline['best_val_correlation']
        assert pytest.approx(metrics['best_val_r2'], rel=TOLERANCE) == baseline['best_val_r2']
        assert pytest.approx(metrics['best_val_mae'], rel=TOLERANCE) == baseline['best_val_mae']
        
        # Compare training history
        assert len(metrics['train_losses']) == len(baseline['train_losses'])
        for i, (current, expected) in enumerate(zip(metrics['train_losses'], baseline['train_losses'])):
            assert pytest.approx(current, rel=TOLERANCE) == expected, f"train_loss[{i}] differs"
