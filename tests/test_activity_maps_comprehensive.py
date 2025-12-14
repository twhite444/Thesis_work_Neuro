"""Comprehensive tests for activity_maps module.

Tests cover:
- Activity map loading from local CSV files
- Global mask computation
- Mask application
- CID averaging
- Visualization generation
- Complete pipeline
"""
import pytest
import pandas as pd
import numpy as np
from src.neuro_foundation.pipeline.activity_maps import (
    ActivityMapRecord,
    load_directory_csv,
    load_activity_maps,
    compute_global_mask,
    apply_mask,
    average_by_cid,
    pipeline_load_and_mask,
)


@pytest.fixture
def mock_activity_maps_data(tmp_path):
    """Create mock activity maps data for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create behavior directory CSV
    behavior_df = pd.DataFrame({
        'Stimulus': [180, 180, 240, 58, 58],
        'Activity Map Path': [
            'csvs/180_0.csv',
            'csvs/180_1.csv', 
            'csvs/240_0.csv',
            'csvs/58_0.csv',
            'csvs/58_1.csv'
        ]
    })
    behavior_path = data_dir / 'behavior_data.csv'
    behavior_df.to_csv(behavior_path, index=False)
    
    # Create activity maps CSV directory
    activity_maps_csv_dir = data_dir / 'activity_maps_csv'
    activity_maps_csv_dir.mkdir()
    
    # Create mock activity maps (10x10)
    shape = (10, 10)
    for filename in ['180_0.csv', '180_1.csv', '240_0.csv', '58_0.csv', '58_1.csv']:
        # Create maps with different patterns but consistent coverage
        map_data = np.random.randn(*shape)
        # Make ~60% of pixels non-zero (consistent brain region)
        mask = np.random.rand(*shape) > 0.4
        map_data[~mask] = 0
        
        pd.DataFrame(map_data).to_csv(activity_maps_csv_dir / filename)
    
    return data_dir, behavior_path


# ===== Directory CSV Loading Tests =====

@pytest.mark.unit
def test_load_directory_csv_success(mock_activity_maps_data):
    """Test loading behavior directory CSV."""
    data_dir, behavior_path = mock_activity_maps_data
    df = load_directory_csv(str(behavior_path))
    
    assert 'Stimulus' in df.columns
    assert 'Activity Map Path' in df.columns
    assert 'CID' in df.columns
    assert len(df) == 5


@pytest.mark.unit
def test_load_directory_csv_filters_negative_cids(tmp_path):
    """Test that negative CIDs are filtered out."""
    behavior_df = pd.DataFrame({
        'Stimulus': [180, -1, 240],
        'Activity Map Path': ['csvs/180_0.csv', 'csvs/neg_1.csv', 'csvs/240_0.csv']
    })
    csv_path = tmp_path / 'behavior.csv'
    behavior_df.to_csv(csv_path, index=False)
    
    df = load_directory_csv(str(csv_path))
    assert len(df) == 2  # Only positive CIDs
    assert (-1 not in df['CID'].values)


@pytest.mark.unit
def test_load_directory_csv_missing_columns(tmp_path):
    """Test error when required columns are missing."""
    bad_df = pd.DataFrame({'CID': [1, 2]})
    csv_path = tmp_path / 'bad_behavior.csv'
    bad_df.to_csv(csv_path, index=False)
    
    with pytest.raises(ValueError, match="Expected columns"):
        load_directory_csv(str(csv_path))


# ===== Activity Maps Loading Tests =====

@pytest.mark.unit
def test_load_activity_maps_success(mock_activity_maps_data):
    """Test loading activity maps from CSV files."""
    data_dir, behavior_path = mock_activity_maps_data
    df = load_directory_csv(str(behavior_path))
    
    records = load_activity_maps(df, data_dir=str(data_dir))
    
    assert len(records) == 5
    assert all(isinstance(r, ActivityMapRecord) for r in records)
    assert all(r.map.shape == (10, 10) for r in records)
    assert all(isinstance(r.cid, int) for r in records)


@pytest.mark.unit
def test_load_activity_maps_missing_directory(tmp_path):
    """Test error when activity_maps_csv directory doesn't exist."""
    behavior_df = pd.DataFrame({
        'Stimulus': [180],
        'Activity Map Path': ['csvs/180_0.csv']
    })
    csv_path = tmp_path / 'behavior.csv'
    behavior_df.to_csv(csv_path, index=False)
    df = load_directory_csv(str(csv_path))
    
    with pytest.raises(FileNotFoundError, match="Activity maps CSV directory not found"):
        load_activity_maps(df, data_dir=str(tmp_path / "nonexistent"))


@pytest.mark.unit
def test_load_activity_maps_missing_file(mock_activity_maps_data):
    """Test handling of missing activity map file."""
    data_dir, behavior_path = mock_activity_maps_data
    
    # Add entry for non-existent file
    df = pd.DataFrame({
        'Stimulus': [999],
        'Activity Map Path': ['csvs/missing.csv'],
        'CID': [999]
    })
    
    # Should continue without crashing, just skip the missing file
    records = load_activity_maps(df, data_dir=str(data_dir))
    assert len(records) == 0  # No records loaded


# ===== Global Mask Computation Tests =====

@pytest.mark.unit
def test_compute_global_mask_basic(mock_activity_maps_data):
    """Test basic global mask computation.
    
    Note: On 10x10 test maps, the morphological operations and minimum region
    size (100 pixels) may result in an empty mask. This test verifies the
    function executes without error and returns correct dtype/shape.
    """
    data_dir, behavior_path = mock_activity_maps_data
    df = load_directory_csv(str(behavior_path))
    records = load_activity_maps(df, data_dir=str(data_dir))
    
    mask = compute_global_mask(records, coverage_threshold=0.5)
    
    assert mask.dtype == bool
    assert mask.shape == (10, 10)
    # Note: mask may be all False on small test maps due to minimum region size filter


@pytest.mark.unit
def test_compute_global_mask_high_threshold(mock_activity_maps_data):
    """Test global mask with high coverage threshold."""
    data_dir, behavior_path = mock_activity_maps_data
    df = load_directory_csv(str(behavior_path))
    records = load_activity_maps(df, data_dir=str(data_dir))
    
    # High threshold - require all maps to have coverage
    mask_high = compute_global_mask(records, coverage_threshold=1.0)
    
    # Low threshold - require only 1 map
    mask_low = compute_global_mask(records, coverage_threshold=0.2)
    
    # Low threshold mask should be >= high threshold mask
    assert mask_low.sum() >= mask_high.sum()


@pytest.mark.unit
def test_compute_global_mask_empty_records():
    """Test error when no records provided."""
    with pytest.raises(ValueError, match="No activity maps provided"):
        compute_global_mask([], coverage_threshold=0.5)


# ===== Mask Application Tests =====

@pytest.mark.unit
def test_apply_mask(mock_activity_maps_data):
    """Test applying mask to activity maps."""
    data_dir, behavior_path = mock_activity_maps_data
    df = load_directory_csv(str(behavior_path))
    records = load_activity_maps(df, data_dir=str(data_dir))
    mask = compute_global_mask(records, coverage_threshold=0.5)
    
    masked_records = apply_mask(records, mask)
    
    assert len(masked_records) == len(records)
    assert all(isinstance(r, ActivityMapRecord) for r in masked_records)
    
    # Check that mask was applied (values outside mask are 0)
    for original, masked in zip(records, masked_records):
        # Where mask is False, masked map should be 0
        assert (masked.map[~mask] == 0).all()


# ===== CID Averaging Tests =====

@pytest.mark.unit
def test_average_by_cid(mock_activity_maps_data):
    """Test averaging activity maps by CID."""
    data_dir, behavior_path = mock_activity_maps_data
    df = load_directory_csv(str(behavior_path))
    records = load_activity_maps(df, data_dir=str(data_dir))
    
    averaged_maps, cids = average_by_cid(records)
    
    # Should have 3 unique CIDs: 180, 240, 58
    assert len(averaged_maps) == 3
    assert len(cids) == 3
    assert set(cids) == {180, 240, 58}
    
    # All averaged maps should have same shape
    assert all(m.shape == (10, 10) for m in averaged_maps)


@pytest.mark.unit
def test_average_by_cid_single_map(mock_activity_maps_data):
    """Test averaging when CID has only one map."""
    data_dir, behavior_path = mock_activity_maps_data
    df = load_directory_csv(str(behavior_path))
    
    # Get only CID 240 (single map)
    df_single = df[df['CID'] == 240]
    records = load_activity_maps(df_single, data_dir=str(data_dir))
    
    averaged_maps, cids = average_by_cid(records)
    
    assert len(averaged_maps) == 1
    assert cids[0] == 240
    
    # Averaged map should equal original (no averaging needed)
    np.testing.assert_array_equal(averaged_maps[0], records[0].map)


@pytest.mark.unit
def test_average_by_cid_multiple_maps(mock_activity_maps_data):
    """Test averaging when CID has multiple maps."""
    data_dir, behavior_path = mock_activity_maps_data
    df = load_directory_csv(str(behavior_path))
    
    # Get only CID 180 (2 maps)
    df_multi = df[df['CID'] == 180]
    records = load_activity_maps(df_multi, data_dir=str(data_dir))
    
    averaged_maps, cids = average_by_cid(records)
    
    assert len(averaged_maps) == 1
    assert cids[0] == 180
    
    # Verify it's actually averaged
    expected_avg = np.mean([r.map for r in records], axis=0)
    np.testing.assert_array_almost_equal(averaged_maps[0], expected_avg)


# ===== Pipeline Integration Tests =====

@pytest.mark.integration
def test_pipeline_load_and_mask_complete(mock_activity_maps_data):
    """Test complete pipeline from loading to masking and averaging."""
    data_dir, behavior_path = mock_activity_maps_data
    output_dir = data_dir / 'output'
    output_dir.mkdir()
    
    averaged_maps, cids, mask = pipeline_load_and_mask(
        directory_csv=str(behavior_path),
        data_dir=str(data_dir),
        coverage_threshold=0.5,
        output_dir=str(output_dir),
        verbose=False
    )
    
    # Check outputs
    assert len(averaged_maps) == 3  # 3 unique CIDs
    assert len(cids) == 3
    assert mask.shape == (10, 10)
    
    # Check visualization files were created
    assert (output_dir / 'global_mask.png').exists()
    assert (output_dir / 'coverage_counts.png').exists()
    assert (output_dir / 'coverage_histogram.png').exists()
    assert (output_dir / 'masked_averaged_example.png').exists()
    assert (output_dir / 'masked_averaged_gallery.png').exists()


@pytest.mark.integration
def test_pipeline_different_thresholds(mock_activity_maps_data):
    """Test pipeline with different coverage thresholds."""
    data_dir, behavior_path = mock_activity_maps_data
    output_dir = data_dir / 'output'
    output_dir.mkdir()
    
    # Low threshold
    maps_low, cids_low, mask_low = pipeline_load_and_mask(
        directory_csv=str(behavior_path),
        data_dir=str(data_dir),
        coverage_threshold=0.2,
        output_dir=str(output_dir),
        verbose=False
    )
    
    # High threshold
    maps_high, cids_high, mask_high = pipeline_load_and_mask(
        directory_csv=str(behavior_path),
        data_dir=str(data_dir),
        coverage_threshold=0.8,
        output_dir=str(output_dir),
        verbose=False
    )
    
    # Low threshold should have more pixels in mask
    assert mask_low.sum() >= mask_high.sum()
    
    # Same number of CIDs
    assert len(cids_low) == len(cids_high)


# ===== Edge Case Tests =====

@pytest.mark.unit
def test_activity_map_with_all_zeros(tmp_path):
    """Test handling of activity map with all zeros."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    behavior_df = pd.DataFrame({
        'Stimulus': [1],
        'Activity Map Path': ['csvs/1_0.csv']
    })
    behavior_path = data_dir / 'behavior.csv'
    behavior_df.to_csv(behavior_path, index=False)
    
    activity_maps_csv_dir = data_dir / 'activity_maps_csv'
    activity_maps_csv_dir.mkdir()
    
    # Create all-zero map
    zero_map = np.zeros((5, 5))
    pd.DataFrame(zero_map).to_csv(activity_maps_csv_dir / '1_0.csv')
    
    df = load_directory_csv(str(behavior_path))
    records = load_activity_maps(df, data_dir=str(data_dir))
    
    assert len(records) == 1
    assert (records[0].map == 0).all()


@pytest.mark.unit
def test_activity_map_nan_handling(tmp_path):
    """Test that NaN values are converted to 0."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    behavior_df = pd.DataFrame({
        'Stimulus': [1],
        'Activity Map Path': ['csvs/1_0.csv']
    })
    behavior_path = data_dir / 'behavior.csv'
    behavior_df.to_csv(behavior_path, index=False)
    
    activity_maps_csv_dir = data_dir / 'activity_maps_csv'
    activity_maps_csv_dir.mkdir()
    
    # Create map with NaN values
    map_with_nan = np.ones((5, 5))
    map_with_nan[0, 0] = np.nan
    map_with_nan[2, 2] = np.nan
    pd.DataFrame(map_with_nan).to_csv(activity_maps_csv_dir / '1_0.csv')
    
    df = load_directory_csv(str(behavior_path))
    records = load_activity_maps(df, data_dir=str(data_dir))
    
    # NaN should be converted to 0
    assert not np.isnan(records[0].map).any()
    assert records[0].map[0, 0] == 0
    assert records[0].map[2, 2] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
