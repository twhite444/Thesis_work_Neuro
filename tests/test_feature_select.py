import pandas as pd
from src.neuro_foundation.pipeline.feature_select import select_features


def test_select_features_threshold(tmp_path):
    df = pd.DataFrame({
        'a': [0, 0, 0, 0],
        'b': [1, 1, 1, 1],
        'c': [0, 1, 0, 1],
    })
    out = select_features(df, threshold=0.1, output_dir=str(tmp_path))
    # 'a' and 'b' have zero variance; 'c' remains
    assert list(out.columns) == ['c']
    assert (tmp_path / 'selected_features.csv').exists()
    assert (tmp_path / 'feature_select_meta.json').exists()
