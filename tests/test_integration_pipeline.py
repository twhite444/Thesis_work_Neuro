import pandas as pd
import json
import pytest

from src.neuro_foundation.pipeline.preprocess import featurize_and_standardize
from src.neuro_foundation.pipeline.feature_select import select_features
from src.neuro_foundation.pipeline.train_linear import train_linear_regression


@pytest.mark.integration
def test_end_to_end_pipeline(tmp_path, monkeypatch):
    # Use synthetic molecules to avoid external data loading
    molecules = pd.DataFrame({
        'IsomericSMILES': ['CCO', 'CCN', 'CCC'],
        # Optional extra columns like CID can be included but aren't required by preprocess
    })

    # Monkeypatch mordred featurization to be fast/deterministic
    import src.neuro_foundation.pipeline.preprocess as pp
    def fake_smiles_to_mordred(smiles):
        # 3 samples, 4 features; include one zero-only to be dropped
        return pd.DataFrame({
            'f1': [1.0, 2.0, 3.0],
            'f2': [0.0, 0.0, 0.0],
            'f3': [10.0, 11.0, 12.0],
            'f4': [5.0, 7.0, 9.0],
        })
    monkeypatch.setattr(pp, 'smiles_to_mordred', fake_smiles_to_mordred)

    # Stage 1: Preprocess
    processed = featurize_and_standardize(molecules, output_dir=str(tmp_path))
    assert processed.shape == (3, 3)  # f2 dropped, 3 features remain
    assert (tmp_path / 'cleaned_data.csv').exists()
    assert (tmp_path / 'scaler_stats.json').exists()

    # Stage 2: Feature selection
    selected = select_features(processed, threshold=0.01, output_dir=str(tmp_path))
    # Low threshold should keep all standardized features (non-zero variance)
    assert selected.shape[1] == 3
    assert (tmp_path / 'selected_features.csv').exists()
    assert (tmp_path / 'feature_select_meta.json').exists()

    # Add synthetic target for training
    selected_with_target = selected.copy()
    selected_with_target['y'] = [0.5, 0.7, 0.9]

    # Stage 3: Train linear regression
    metrics = train_linear_regression(selected_with_target, target_column='y', output_dir=str(tmp_path))
    assert 'mse' in metrics
    assert metrics['n_features'] == 3
    assert metrics['n_samples'] == 3
    assert (tmp_path / 'model_coefficients.csv').exists()
    assert (tmp_path / 'metrics.json').exists()

    # Verify metrics.json is valid JSON and contains expected keys
    data = json.loads((tmp_path / 'metrics.json').read_text())
    assert set(['mse', 'n_features', 'n_samples', 'target']).issubset(set(data.keys()))
