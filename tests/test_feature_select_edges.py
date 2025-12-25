import pandas as pd
import pytest
from olfactory_modeling.pipeline.feature_select import select_features


@pytest.mark.unit
def test_feature_select_zero_threshold_keeps_all(tmp_path):
    df = pd.DataFrame({
        'a': [0, 1, 0, 1],
        'b': [5, 5, 5, 5],
    })
    out = select_features(df, threshold=0.0, output_dir=str(tmp_path))
    # VarianceThreshold with 0 keeps features with any variance; 'b' has zero variance and should be dropped
    assert list(out.columns) == ['a']


@pytest.mark.unit
def test_feature_select_high_threshold_drops_all(tmp_path):
    df = pd.DataFrame({
        'a': [0, 1, 0, 1],
        'b': [2, 3, 2, 3],
    })
    # With a very high threshold, sklearn raises a ValueError indicating no features meet the threshold
    import pytest
    with pytest.raises(ValueError):
        select_features(df, threshold=10.0, output_dir=str(tmp_path))
