import pandas as pd
import pytest
from src.neuro_foundation.pipeline.train_linear import train_linear_regression


@pytest.mark.unit
def test_train_linear_regression_requires_target(tmp_path):
    df = pd.DataFrame({'x': [1,2,3]})
    try:
        train_linear_regression(df, target_column='y', output_dir=str(tmp_path))
        assert False, "Expected ValueError for missing target"
    except ValueError as e:
        assert "Target column" in str(e)


@pytest.mark.unit
def test_train_linear_regression_runs(tmp_path):
    df = pd.DataFrame({
        'x1': [1, 2, 3, 4],
        'x2': [0.1, 0.2, 0.3, 0.4],
        'y': [2, 3, 4, 5],
    })
    metrics = train_linear_regression(df, target_column='y', output_dir=str(tmp_path))
    assert 'mse' in metrics
    assert (tmp_path / 'model_coefficients.csv').exists()
    assert (tmp_path / 'metrics.json').exists()
