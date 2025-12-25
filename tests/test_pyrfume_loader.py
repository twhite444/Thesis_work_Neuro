"""Comprehensive tests for pyrfume_loader module.

Tests cover:
- Data loading from Pyrfume (molecules, behavior, stimuli, activity maps)
- Helper functions (CSV and NPZ loading)
- CID-based activity map retrieval
- Error handling and edge cases
"""
import pytest
import pandas as pd
import numpy as np
from src.olfactory_modeling.data.pyrfume_loader import (
    load_molecules_csv,
    load_molecules_npz,
    load_behavior_csv,
    load_behavior_npz,
    load_stimuli_csv,
    load_stimuli_npz,
    load_activity_maps_npz,
    load_activity_maps_as_arrays,
    load_activity_maps_by_cid,
    load_activity_map_by_cid_averaged,
)


@pytest.fixture
def mock_data_dir(tmp_path):
    """Create a mock data directory with test files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create mock molecules data
    molecules_df = pd.DataFrame({
        'CID': [180, 240, 240, 58],  # Note: 240 is duplicate
        'IsomericSMILES': ['CC(=O)C', 'CC=O', 'CC=O', 'CCCCCC'],
        'MolecularWeight': [58.08, 44.05, 44.05, 86.18],
        'IUPACName': ['acetone', 'acetaldehyde', 'acetaldehyde', 'hexane'],
        'name': ['acetone', 'acetaldehyde', 'acetaldehyde', 'hexane']
    })
    molecules_df.to_csv(data_dir / 'molecules_raw.csv', index=False)
    
    # Save as NPZ (must match exact format in pyrfume_loader.py)
    np.savez(
        data_dir / 'molecules_raw.npz',
        CID=molecules_df['CID'].values,
        IsomericSMILES=molecules_df['IsomericSMILES'].values,
        MolecularWeight=molecules_df['MolecularWeight'].values,
        IUPACName=molecules_df['IUPACName'].values,
        name=molecules_df['name'].values  # Added: required by load_molecules_npz
    )
    
    # Create mock behavior data
    behavior_df = pd.DataFrame({
        'Stimulus': [180, 180, 240, 58],
        'Activity Map Path': ['csvs/180_0.csv', 'csvs/180_1.csv', 'csvs/240_0.csv', 'csvs/58_0.csv']
    })
    behavior_df.to_csv(data_dir / 'behavior_data.csv', index=False)
    
    # Save as NPZ
    np.savez(
        data_dir / 'behavior_data.npz',
        Stimulus=behavior_df['Stimulus'].values,
        ActivityMapPath=behavior_df['Activity Map Path'].values
    )
    
    # Create mock stimuli data
    stimuli_df = pd.DataFrame({
        'Stimulus': ['180_0', '180_1', '240_0', '58_0'],  # Added: required by load_stimuli_npz
        'CID': [180, 180, 240, 58],
        'Rep': [0, 1, 0, 0],
        'Name': ['acetone', 'acetone', 'acetaldehyde', 'hexane'],
        'Conditions': ['control', 'control', 'control', 'control'],
        'SourceFile': ['file1.csv', 'file1.csv', 'file2.csv', 'file3.csv']
    })
    stimuli_df.to_csv(data_dir / 'stimuli_metadata.csv', index=False)
    
    # Save as NPZ (must match exact format in pyrfume_loader.py)
    np.savez(
        data_dir / 'stimuli_metadata.npz',
        Stimulus=stimuli_df['Stimulus'].values,  # Added: required by load_stimuli_npz
        CID=stimuli_df['CID'].values,
        Rep=stimuli_df['Rep'].values,
        Name=stimuli_df['Name'].values,
        Conditions=stimuli_df['Conditions'].values,
        SourceFile=stimuli_df['SourceFile'].values
    )
    
    # Create mock activity maps
    activity_maps_csv_dir = data_dir / 'activity_maps_csv'
    activity_maps_csv_dir.mkdir()
    
    # Create simple activity maps (5x5 for testing)
    for filename in ['180_0.csv', '180_1.csv', '240_0.csv', '58_0.csv']:
        map_data = np.random.randn(5, 5) * 2  # Random values
        map_data[map_data < 0.5] = 0  # Some zeros
        pd.DataFrame(map_data).to_csv(activity_maps_csv_dir / filename)
    
    # Create NPZ with all activity maps
    maps_list = []
    cids_list = []
    filenames_list = []
    
    for cid, filename in zip([180, 180, 240, 58], 
                             ['180_0.csv', '180_1.csv', '240_0.csv', '58_0.csv']):
        map_df = pd.read_csv(activity_maps_csv_dir / filename, index_col=0)
        maps_list.append(map_df.values)
        cids_list.append(cid)
        filenames_list.append(filename)
    
    np.savez(
        data_dir / 'activity_maps.npz',
        maps=np.array(maps_list),
        cids=np.array(cids_list),
        filenames=np.array(filenames_list)
    )
    
    return data_dir


# ===== Helper Function Tests =====

@pytest.mark.unit
def test_load_molecules_csv(mock_data_dir):
    """Test loading molecules from CSV."""
    df = load_molecules_csv(str(mock_data_dir))
    assert len(df) == 4
    assert 'CID' in df.columns
    assert 'IsomericSMILES' in df.columns
    assert df['CID'].tolist() == [180, 240, 240, 58]


@pytest.mark.unit
def test_load_molecules_npz(mock_data_dir):
    """Test loading molecules from NPZ."""
    df = load_molecules_npz(str(mock_data_dir))
    assert len(df) == 4
    assert 'CID' in df.columns
    assert 'IsomericSMILES' in df.columns
    assert df['CID'].tolist() == [180, 240, 240, 58]


@pytest.mark.unit
def test_load_behavior_csv(mock_data_dir):
    """Test loading behavior from CSV."""
    df = load_behavior_csv(str(mock_data_dir))
    assert len(df) == 4
    assert 'Stimulus' in df.columns
    assert 'Activity Map Path' in df.columns


@pytest.mark.unit
def test_load_behavior_npz(mock_data_dir):
    """Test loading behavior from NPZ."""
    df = load_behavior_npz(str(mock_data_dir))
    assert len(df) == 4
    assert 'Stimulus' in df.columns
    assert 'Activity Map Path' in df.columns


@pytest.mark.unit
def test_load_stimuli_csv(mock_data_dir):
    """Test loading stimuli from CSV."""
    df = load_stimuli_csv(str(mock_data_dir))
    assert len(df) == 4
    assert 'CID' in df.columns
    assert 'Name' in df.columns
    assert 'Rep' in df.columns


@pytest.mark.unit
def test_load_stimuli_npz(mock_data_dir):
    """Test loading stimuli from NPZ."""
    df = load_stimuli_npz(str(mock_data_dir))
    assert len(df) == 4
    assert 'CID' in df.columns
    assert 'Name' in df.columns


@pytest.mark.unit
def test_load_activity_maps_npz(mock_data_dir):
    """Test loading activity maps from NPZ."""
    records = load_activity_maps_npz(str(mock_data_dir))
    assert len(records) == 4
    assert all(hasattr(r, 'cid') and hasattr(r, 'map') for r in records)
    assert all(isinstance(r.map, np.ndarray) for r in records)
    assert records[0].map.shape == (5, 5)  # 5x5 test maps
    assert [r.cid for r in records] == [180, 180, 240, 58]


@pytest.mark.unit
def test_load_activity_maps_as_arrays(mock_data_dir):
    """Test loading activity maps as arrays."""
    maps, cids = load_activity_maps_as_arrays(str(mock_data_dir))
    assert len(maps) == 4
    assert len(cids) == 4
    assert all(isinstance(m, np.ndarray) for m in maps)
    assert cids.tolist() == [180, 180, 240, 58]


# ===== CID-based Loading Tests =====

@pytest.mark.unit
def test_load_activity_maps_by_cid_single_map(mock_data_dir):
    """Test loading activity maps for CID with single map."""
    maps = load_activity_maps_by_cid(58, str(mock_data_dir))
    assert len(maps) == 1
    assert maps[0].shape == (5, 5)


@pytest.mark.unit
def test_load_activity_maps_by_cid_multiple_maps(mock_data_dir):
    """Test loading activity maps for CID with multiple maps."""
    maps = load_activity_maps_by_cid(180, str(mock_data_dir))
    assert len(maps) == 2
    assert all(m.shape == (5, 5) for m in maps)


@pytest.mark.unit
def test_load_activity_maps_by_cid_nonexistent(mock_data_dir):
    """Test loading activity maps for nonexistent CID."""
    maps = load_activity_maps_by_cid(99999, str(mock_data_dir))
    assert len(maps) == 0


@pytest.mark.unit
def test_load_activity_map_by_cid_averaged_single(mock_data_dir):
    """Test averaged map for CID with single map."""
    avg_map = load_activity_map_by_cid_averaged(58, str(mock_data_dir))
    assert avg_map is not None
    assert avg_map.shape == (5, 5)


@pytest.mark.unit
def test_load_activity_map_by_cid_averaged_multiple(mock_data_dir):
    """Test averaged map for CID with multiple maps."""
    avg_map = load_activity_map_by_cid_averaged(180, str(mock_data_dir))
    assert avg_map is not None
    assert avg_map.shape == (5, 5)
    
    # Verify it's actually averaged
    maps = load_activity_maps_by_cid(180, str(mock_data_dir))
    expected_avg = np.mean(maps, axis=0)
    np.testing.assert_array_almost_equal(avg_map, expected_avg)


@pytest.mark.unit
def test_load_activity_map_by_cid_averaged_nonexistent(mock_data_dir):
    """Test averaged map for nonexistent CID."""
    avg_map = load_activity_map_by_cid_averaged(99999, str(mock_data_dir))
    assert avg_map is None


# ===== Error Handling Tests =====

@pytest.mark.unit
def test_load_molecules_csv_missing_file(tmp_path):
    """Test error handling when molecules CSV is missing."""
    with pytest.raises(FileNotFoundError):
        load_molecules_csv(str(tmp_path))


@pytest.mark.unit
def test_load_molecules_npz_missing_file(tmp_path):
    """Test error handling when molecules NPZ is missing."""
    with pytest.raises(FileNotFoundError):
        load_molecules_npz(str(tmp_path))


@pytest.mark.unit
def test_load_activity_maps_npz_missing_file(tmp_path):
    """Test error handling when activity maps NPZ is missing."""
    with pytest.raises(FileNotFoundError):
        load_activity_maps_npz(str(tmp_path))


# ===== Data Integrity Tests =====

@pytest.mark.unit
def test_csv_npz_consistency(mock_data_dir):
    """Test that CSV and NPZ loading produce identical results."""
    csv_df = load_molecules_csv(str(mock_data_dir))
    npz_df = load_molecules_npz(str(mock_data_dir))
    
    # Should have same columns and values
    assert set(csv_df.columns) == set(npz_df.columns)
    assert len(csv_df) == len(npz_df)
    
    # Check CID values match
    assert csv_df['CID'].tolist() == npz_df['CID'].tolist()


@pytest.mark.unit
def test_activity_maps_non_zero_coverage(mock_data_dir):
    """Test that activity maps have some non-zero values."""
    maps, cids = load_activity_maps_as_arrays(str(mock_data_dir))
    
    for map_data in maps:
        # Each map should have some non-zero values
        non_zero_count = (map_data != 0).sum()
        assert non_zero_count > 0, "Activity map should have non-zero values"


@pytest.mark.unit
def test_cid_filtering(mock_data_dir):
    """Test that CID-based loading correctly filters maps."""
    all_maps, all_cids = load_activity_maps_as_arrays(str(mock_data_dir))
    
    # Check CID 180
    maps_180 = load_activity_maps_by_cid(180, str(mock_data_dir))
    expected_count = (all_cids == 180).sum()
    assert len(maps_180) == expected_count
    
    # Check CID 240
    maps_240 = load_activity_maps_by_cid(240, str(mock_data_dir))
    expected_count = (all_cids == 240).sum()
    assert len(maps_240) == expected_count


# ===== Integration Tests =====

@pytest.mark.integration
def test_full_data_loading_workflow(mock_data_dir):
    """Test complete data loading workflow."""
    # Load all data types
    molecules = load_molecules_npz(str(mock_data_dir))
    behavior = load_behavior_npz(str(mock_data_dir))
    stimuli = load_stimuli_npz(str(mock_data_dir))
    maps, cids = load_activity_maps_as_arrays(str(mock_data_dir))
    
    # Verify relationships
    assert len(behavior) == len(maps)
    assert len(behavior) == len(stimuli)
    
    # Verify CIDs are consistent
    assert set(molecules['CID'].unique()).issubset(set(behavior['Stimulus'].unique()))


@pytest.mark.integration  
def test_averaged_maps_for_all_cids(mock_data_dir):
    """Test that averaged maps can be computed for all CIDs."""
    _, cids = load_activity_maps_as_arrays(str(mock_data_dir))
    unique_cids = np.unique(cids)
    
    for cid in unique_cids:
        avg_map = load_activity_map_by_cid_averaged(int(cid), str(mock_data_dir))
        assert avg_map is not None
        assert avg_map.shape == (5, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
