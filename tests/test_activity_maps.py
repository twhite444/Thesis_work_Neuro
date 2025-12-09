import numpy as np
import pandas as pd
import pytest

from src.neuro_foundation.pipeline.activity_maps import (
    load_directory_csv,
    load_activity_maps,
    compute_global_mask,
    apply_mask,
    average_by_cid,
    pipeline_load_and_mask,
)


@pytest.mark.unit
def test_load_directory_csv_parses_cid(tmp_path):
    csv = tmp_path / 'behavior_data.csv'
    df = pd.DataFrame({
        'Stimulus': ['123_foo', '0_bar', '456_baz'],
        'Activity Map Path': ['path/a.csv', 'path/b.csv', 'path/c.csv'],
    })
    df.to_csv(csv, index=False)
    out = load_directory_csv(str(csv))
    assert list(out['CID']) == [123, 456]


@pytest.mark.unit
def test_load_activity_maps_uses_pyrfume(tmp_path, monkeypatch):
    directory = pd.DataFrame({
        'Stimulus': ['1_a', '1_b', '2_c'],
        'Activity Map Path': ['maps/a.csv', 'maps/b.csv', 'maps/c.csv'],
        'CID': [1, 1, 2],
    })
    import src.neuro_foundation.pipeline.activity_maps as am

    class FakeData:
        def __init__(self, arr):
            self._arr = arr
        def to_numpy(self):
            return self._arr

    def fake_load_data(path):
        # Return small arrays
        if path.endswith('a.csv'):
            return FakeData(np.array([[1, np.nan],[0, 2]]))
        if path.endswith('b.csv'):
            return FakeData(np.array([[2, 1],[1, 0]]))
        if path.endswith('c.csv'):
            return FakeData(np.array([[0, 0],[np.nan, 3]]))
        raise FileNotFoundError

    monkeypatch.setattr(am, 'load_data', fake_load_data)
    recs = load_activity_maps(directory, base_path='base')
    assert len(recs) == 3
    assert recs[0].cid == 1
    assert recs[1].cid == 1
    assert recs[2].cid == 2


@pytest.mark.unit
def test_compute_global_mask_and_apply(tmp_path):
    # Two maps; require coverage of 50%
    maps = [
        np.array([[1, np.nan],[0, 2]]),
        np.array([[np.nan, 1],[1, 0]]),
    ]
    from src.neuro_foundation.pipeline.activity_maps import ActivityMapRecord
    recs = [ActivityMapRecord(cid=1, map=maps[0]), ActivityMapRecord(cid=2, map=maps[1])]
    mask = compute_global_mask(recs, coverage_threshold=0.5)
    assert mask.shape == maps[0].shape
    masked = apply_mask(recs, mask)
    # Ensure mask is binary
    assert set(np.unique(mask)).issubset({False, True})
    # Masked values should be zero where mask is False
    assert np.all((masked[0].map[~mask]) == 0)


@pytest.mark.unit
def test_average_by_cid():
    from src.neuro_foundation.pipeline.activity_maps import ActivityMapRecord
    recs = [
        ActivityMapRecord(cid=1, map=np.array([[1, 2],[3, 4]])),
        ActivityMapRecord(cid=1, map=np.array([[3, 2],[1, 0]])),
        ActivityMapRecord(cid=2, map=np.array([[0, 1],[1, 0]])),
    ]
    avg_maps, cids = average_by_cid(recs)
    assert set(cids) == {1, 2}
    # Avg for CID 1 over two maps
    m1 = [m for m,c in zip(avg_maps, cids) if c == 1][0]
    assert np.allclose(m1, np.array([[2, 2],[2, 2]]))


@pytest.mark.integration
def test_pipeline_load_and_mask_end_to_end(tmp_path, monkeypatch):
    # Build directory CSV
    csv = tmp_path / 'behavior_data.csv'
    pd.DataFrame({
        'Stimulus': ['10_a', '10_b', '20_c'],
        'Activity Map Path': ['maps/a.csv', 'maps/b.csv', 'maps/c.csv'],
    }).to_csv(csv, index=False)

    # Fake pyrfume load_data
    import src.neuro_foundation.pipeline.activity_maps as am

    class FakeData:
        def __init__(self, arr):
            self._arr = arr
        def to_numpy(self):
            return self._arr

    def fake_load_data(path):
        if path.endswith('a.csv'):
            return FakeData(np.array([[1, 0],[0, 1]]))
        if path.endswith('b.csv'):
            return FakeData(np.array([[2, 0],[0, 2]]))
        if path.endswith('c.csv'):
            return FakeData(np.array([[0, 1],[1, 0]]))
        raise FileNotFoundError

    monkeypatch.setattr(am, 'load_data', fake_load_data)

    maps, cids, mask = pipeline_load_and_mask(str(csv), base_path='base', coverage_threshold=0.5, output_dir=str(tmp_path))
    assert len(maps) == 2  # averaged per CID (10 has two maps)
    assert set(cids) == {10, 20}
    # Visualizations exist
    assert (tmp_path / 'global_mask.png').exists()
    assert (tmp_path / 'masked_averaged_example.png').exists()
    assert (tmp_path / 'masked_averaged_gallery.png').exists()
    assert (tmp_path / 'coverage_counts.png').exists()
    assert (tmp_path / 'coverage_histogram.png').exists()
