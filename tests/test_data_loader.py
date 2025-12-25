import pandas as pd
import types
import pytest
from src.olfactory_modeling.data.pyrfume_loader import PyrfumeLoader


@pytest.mark.unit
def test_pyrfume_loader_writes_raw(tmp_path, monkeypatch):
    # Mock pyrfume functions to avoid network/file dependencies
    import src.olfactory_modeling.data.pyrfume_loader as pl

    def fake_load_manifest(name):
        return {"name": name}

    def fake_load_data(path):
        if path.endswith('molecules.csv'):
            return pd.DataFrame({
                'IsomericSMILES': ['CCO', 'CCN'],
                'CID': [1, 2],
                'MolecularWeight': [46.07, 45.08],
                'IUPACName': ['ethanol', 'ethanamine'],
                'name': ['ethanol', 'ethanamine']
            })
        elif path.endswith('1031_0.csv'):
            return pd.DataFrame({'img': [1, 2, 3]})
        raise FileNotFoundError

    monkeypatch.setattr(pl, 'pyrfume', types.SimpleNamespace(load_manifest=fake_load_manifest, load_data=fake_load_data))

    loader = PyrfumeLoader(output_dir=str(tmp_path))
    molecules = loader.load_molecules()
    images = loader.load_images()

    assert 'IsomericSMILES' in molecules.columns
    # raw files written
    assert (tmp_path / 'molecules_raw.csv').exists()
    assert (tmp_path / 'molecules_raw.npz').exists()
    # Note: load_images() returns None (images not part of refactored pipeline)
    assert images is None


@pytest.mark.unit
def test_pyrfume_loader_images_optional(tmp_path, monkeypatch):
    import src.olfactory_modeling.data.pyrfume_loader as pl

    def fake_load_manifest(name):
        return {"name": name}

    def fake_load_data(path):
        if path.endswith('molecules.csv'):
            return pd.DataFrame({
                'IsomericSMILES': ['CCO'],
                'CID': [1],
                'MolecularWeight': [46.07],
                'IUPACName': ['ethanol'],
                'name': ['ethanol']
            })
        # Simulate missing image csv
        raise FileNotFoundError

    monkeypatch.setattr(pl, 'pyrfume', types.SimpleNamespace(load_manifest=fake_load_manifest, load_data=fake_load_data))

    loader = PyrfumeLoader(output_dir=str(tmp_path))
    _ = loader.load_molecules()
    images = loader.load_images()
    assert images is None
    # molecules file exists
    assert (tmp_path / 'molecules_raw.csv').exists()
