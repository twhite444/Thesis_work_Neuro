import pandas as pd
import pytest
from src.olfactory_modeling.pipeline.preprocess import featurize_and_standardize


@pytest.mark.unit
def test_featurize_and_standardize_smiles_column_missing(tmp_path):
    df = pd.DataFrame({'SMILES': ['CCO']})
    try:
        featurize_and_standardize(df, output_dir=str(tmp_path))
        assert False, "Expected ValueError for missing IsomericSMILES"
    except ValueError as e:
        assert "IsomericSMILES" in str(e)


@pytest.mark.unit
def test_featurize_and_standardize_runs(tmp_path, monkeypatch):
    # Provide small synthetic SMILES and monkeypatch mordred function to avoid heavy compute
    def fake_smiles_to_mordred(smiles):
        # return deterministic small feature frame
        return pd.DataFrame({
            'f1': [1.0, 2.0],
            'f2': [0.0, 0.0],  # zero-only should be dropped
            'f3': [3.0, 4.0],
        })
    # Mock the pyrfume import inside the function
    import pyrfume.features
    monkeypatch.setattr(pyrfume.features, 'smiles_to_mordred', fake_smiles_to_mordred)

    df = pd.DataFrame({'IsomericSMILES': ['CCO', 'CCN'], 'CID': [1, 2]})
    out = featurize_and_standardize(df, output_dir=str(tmp_path))
    # f2 is zero-only; expect 2 features remain
    assert out.shape == (2, 2)
    # cleaned_data.csv exists
    assert (tmp_path / 'cleaned_data.csv').exists()
    # scaler_stats.json exists
    assert (tmp_path / 'scaler_stats.json').exists()
