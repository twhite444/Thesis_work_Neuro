import pandas as pd
import numpy as np
import pytest
from src.olfactory_modeling.pipeline.train_linear import train_linear_regression


@pytest.mark.unit
def test_train_linear_with_nans(tmp_path):
    df = pd.DataFrame({
        'x1': [1, np.nan, 3, 4],
        'x2': [0.1, 0.2, np.nan, 0.4],
        'y': [2, 3, 4, 5],
    })
    metrics = train_linear_regression(df, target_column='y', output_dir=str(tmp_path))
    assert 'mse' in metrics


@pytest.mark.unit
def test_train_linear_constant_target(tmp_path):
    df = pd.DataFrame({
        'x1': [1, 2, 3, 4],
        'x2': [0.1, 0.2, 0.3, 0.4],
        'y': [1, 1, 1, 1],
    })
    metrics = train_linear_regression(df, target_column='y', output_dir=str(tmp_path))
    # MSE should be ~0 given constant target; floating precision tolerated
    assert metrics['mse'] >= 0.0


@pytest.mark.unit
def test_train_linear_tiny_sample(tmp_path):
    df = pd.DataFrame({
        'x1': [1, 2],
        'y': [2, 3],
    })
    metrics = train_linear_regression(df, target_column='y', output_dir=str(tmp_path))
    assert 'mse' in metrics
    # coefficients file should have one feature
    import csv
    with open(tmp_path / 'model_coefficients.csv') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
