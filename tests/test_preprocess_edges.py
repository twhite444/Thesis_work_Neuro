import pandas as pd
import pytest
from src.olfactory_modeling.pipeline.preprocess import featurize_and_standardize


@pytest.mark.unit
def test_preprocess_empty_dataframe(tmp_path):
    df = pd.DataFrame({'IsomericSMILES': []})
    # Expect failure from downstream mordred due to empty input; verify graceful exception or empty output
    with pytest.raises(Exception):
        featurize_and_standardize(df, output_dir=str(tmp_path))


@pytest.mark.unit
def test_preprocess_invalid_smiles(tmp_path, monkeypatch):
    # Force invalid smiles raising behavior
    import src.olfactory_modeling.pipeline.preprocess as pp

    def fake_smiles_to_mordred(smiles):
        # Simulate invalid SMILES causing NaNs and zero-only columns
        return pd.DataFrame({
            'a': [float('nan'), float('nan')],
            'b': [0.0, 0.0],
            'c': [1.0, 2.0],
        })

    monkeypatch.setattr(pp, 'smiles_to_mordred', fake_smiles_to_mordred)

    df = pd.DataFrame({'IsomericSMILES': ['INVALID', 'ALSO_INVALID']})
    out = featurize_and_standardize(df, output_dir=str(tmp_path))
    # 'a' dropped due to NaNs, 'b' dropped due to zero-only, 'c' remains
    assert list(out.columns) == ['c']
    assert (tmp_path / 'cleaned_data.csv').exists()
    assert (tmp_path / 'scaler_stats.json').exists()


@pytest.mark.unit
def test_preprocess_scaler_stats_length(tmp_path, monkeypatch):
    import src.olfactory_modeling.pipeline.preprocess as pp

    def fake_smiles_to_mordred(smiles):
        return pd.DataFrame({'f1': [1.0, 3.0], 'f2': [2.0, 2.0]})

    monkeypatch.setattr(pp, 'smiles_to_mordred', fake_smiles_to_mordred)

    df = pd.DataFrame({'IsomericSMILES': ['CCO', 'CCN']})
    out = featurize_and_standardize(df, output_dir=str(tmp_path))
    # mean/scale length equals number of kept features (f2 zero variance is not zero-only, so kept)
    import json
    stats = json.loads((tmp_path / 'scaler_stats.json').read_text())
    assert len(stats['features']) == out.shape[1]
    assert len(stats['mean']) == out.shape[1]
    assert len(stats['scale']) == out.shape[1]
