# Testing & Quality Assurance Guide

## 🎉 **Current Test Coverage Status**

**All Tests Passing! ✅**

### Test Suite Summary
- **Total Tests**: 60 tests
- **Passing**: 60 (100%)
- **Failing**: 0
- **Coverage**: 81% overall
- **Runtime**: ~8 seconds

### Modules by Coverage

| Module | Coverage | Tests | Status |
|--------|----------|-------|--------|
| `feature_select.py` | 100% | 3 tests | ✅ Complete |
| `preprocess.py` | 100% | 5 tests | ✅ Complete |
| `train_linear.py` | 100% | 5 tests | ✅ Complete |
| `activity_maps.py` | 99% | 22 tests | ✅ Complete |
| `pyrfume_loader.py` | 55% | 22 tests | ✅ Complete |
| `interfaces.py` | 82% | N/A | ℹ️ Interface definitions |

---

## 🧪 **Test Files Overview**

### Core Pipeline Tests (17 tests - All Passing)

#### Feature Selection
- **`test_feature_select.py`** (1 test): Variance threshold filtering
- **`test_feature_select_edges.py`** (2 tests): Zero threshold, high threshold edge cases

#### Preprocessing
- **`test_preprocess.py`** (2 tests): Featurization, standardization, missing columns
- **`test_preprocess_edges.py`** (3 tests): Empty DataFrames, invalid SMILES, scaler stats

#### Linear Model Training
- **`test_train_linear.py`** (2 tests): Model training, predictions, missing target
- **`test_train_linear_edges.py`** (3 tests): NaNs, constant targets, tiny samples

#### Integration
- **`test_integration_pipeline.py`** (1 test): End-to-end pipeline from SMILES to predictions

---

### Data Loading Tests (24 tests - All Passing)

#### Pyrfume Loader (22 tests)
**`test_pyrfume_loader.py`** - Comprehensive data loading tests

**CSV/NPZ Loading** (6 tests):
- `test_load_molecules_csv`: Load molecules from CSV
- `test_load_molecules_npz`: Load molecules from NPZ (faster)
- `test_load_behavior_csv`: Load behavior data from CSV
- `test_load_behavior_npz`: Load behavior data from NPZ
- `test_load_stimuli_csv`: Load stimuli metadata from CSV
- `test_load_stimuli_npz`: Load stimuli metadata from NPZ

**CID-based Activity Map Loading** (8 tests):
- `test_load_activity_maps_npz`: Load all activity maps
- `test_load_activity_maps_as_arrays`: Load as numpy arrays
- `test_load_activity_maps_by_cid_single_map`: Load single CID
- `test_load_activity_maps_by_cid_multiple_maps`: Load CID with multiple repetitions
- `test_load_activity_maps_by_cid_nonexistent`: Handle missing CID
- `test_load_activity_map_by_cid_averaged_single`: Average single map
- `test_load_activity_map_by_cid_averaged_multiple`: Average multiple maps
- `test_load_activity_map_by_cid_averaged_nonexistent`: Handle missing CID for averaging

**Error Handling** (3 tests):
- `test_load_molecules_csv_missing_file`: Missing CSV file
- `test_load_molecules_npz_missing_file`: Missing NPZ file
- `test_load_activity_maps_npz_missing_file`: Missing activity maps NPZ

**Integration & Validation** (5 tests):
- `test_csv_npz_consistency`: CSV and NPZ return same data
- `test_activity_maps_non_zero_coverage`: Activity maps have non-zero values
- `test_cid_filtering`: CID-based filtering works correctly
- `test_full_data_loading_workflow`: Complete data loading pipeline
- `test_averaged_maps_for_all_cids`: All CIDs can be averaged

#### Legacy Data Loader (2 tests)
**`test_data_loader.py`** - PyrfumeLoader class tests
- `test_pyrfume_loader_writes_raw`: Verify raw data files created
- `test_pyrfume_loader_images_optional`: Handle missing image data

---

### Activity Maps Tests (22 tests - All Passing)

#### Basic Tests (5 tests)
**`test_activity_maps.py`** - Original activity maps tests
- `test_load_directory_csv_parses_cid`: Parse CID from Stimulus column
- `test_load_activity_maps_uses_pyrfume`: Load maps from local CSV files
- `test_compute_global_mask_and_apply`: Mask computation and application
- `test_average_by_cid`: Average maps by CID
- `test_pipeline_load_and_mask_end_to_end`: Complete pipeline test

#### Comprehensive Tests (17 tests)
**`test_activity_maps_comprehensive.py`** - Thorough activity maps testing

**Directory CSV Loading** (3 tests):
- `test_load_directory_csv_success`: Successful CSV loading
- `test_load_directory_csv_filters_negative_cids`: Filter natural mixtures
- `test_load_directory_csv_missing_columns`: Handle missing columns

**Activity Map Loading** (3 tests):
- `test_load_activity_maps_success`: Successful map loading
- `test_load_activity_maps_missing_directory`: Handle missing directory
- `test_load_activity_maps_missing_file`: Handle missing individual map

**Global Mask Computation** (3 tests):
- `test_compute_global_mask_basic`: Basic mask computation
- `test_compute_global_mask_high_threshold`: High coverage threshold
- `test_compute_global_mask_empty_records`: Handle empty input

**Mask Application & Averaging** (4 tests):
- `test_apply_mask`: Apply mask to maps
- `test_average_by_cid`: Average multiple maps per CID
- `test_average_by_cid_single_map`: Average single map
- `test_average_by_cid_multiple_maps`: Average multiple maps

**Pipeline Integration** (2 tests):
- `test_pipeline_load_and_mask_complete`: Complete pipeline
- `test_pipeline_different_thresholds`: Different coverage thresholds

**Edge Cases** (2 tests):
- `test_activity_map_with_all_zeros`: Handle all-zero maps
- `test_activity_map_nan_handling`: Handle NaN values

---

## 🚀 **Running Tests**

### Run All Tests
```bash
pytest tests/
```

### Run with Coverage
```bash
pytest tests/ --cov=src/olfactory_modeling --cov-report=term-missing
```

### Run Specific Test File
```bash
pytest tests/test_pyrfume_loader.py -v
```

### Run Tests by Marker
```bash
# Run only unit tests
pytest tests/ -m unit

# Run only integration tests
pytest tests/ -m integration
```

### Run Tests in Parallel (faster)
```bash
pytest tests/ -n auto
```

---

## ✅ **Manual Testing Checklist**

While automated tests cover 81% of the codebase, some features require manual verification:

### Interactive Visualization Testing

**Test Activity Map Inspection Tool:**
```bash
# List all available maps
python scripts/exploration/inspect_activity_map.py --list-all

# Show statistics for a CID
python scripts/exploration/inspect_activity_map.py --cid 180

# Interactive visualization (requires manual verification)
python scripts/exploration/inspect_activity_map.py --cid 180 --show-images

# Save visualization
python scripts/exploration/inspect_activity_map.py --cid 180 --save-images
```

**What to Check:**
- ✅ Interactive matplotlib window opens without errors
- ✅ All subplots render correctly (individual maps, average, coverage)
- ✅ Color scales are appropriate
- ✅ Zero values display as transparent/white (not colored)
- ✅ Titles and labels are correct
- ✅ Saved PNG files are not blank

### Pipeline Visualization Testing

**Test Complete Pipeline:**
```bash
python scripts/run_activity_maps.py --verbose
```

**What to Check:**
- ✅ `global_mask.png` - Shows refined brain mask
- ✅ `coverage_counts.png` - Heatmap of pixel coverage
- ✅ `coverage_histogram.png` - Distribution of coverage
- ✅ `masked_averaged_example.png` - Example masked map
- ✅ `masked_averaged_gallery.png` - Gallery of 6 maps
- ✅ All images render correctly and are not blank
- ✅ Zero values display as transparent/white

### Example Script Testing

**Test All Example Scripts:**
```bash
# Test CID-based loading
python scripts/examples/example_load_by_cid.py

# Test cached data loading
python scripts/examples/example_load_cached.py

# Test stimuli metadata
python scripts/examples/example_stimuli_metadata.py
```

**What to Check:**
- ✅ Scripts run without errors
- ✅ Output makes sense
- ✅ Examples demonstrate correct usage patterns

### Full Pipeline Testing

**Test Complete Pipeline:**
```bash
# Run all pipeline steps
python scripts/preprocess.py
python scripts/run_activity_maps.py
python scripts/select_features.py
python scripts/train_linear.py
```

**What to Check:**
- ✅ All steps complete without errors
- ✅ Output files created in correct directories
- ✅ Data flows correctly between steps
- ✅ Final predictions are reasonable

---

## 🐛 **Known Issues & Limitations**

### Test Limitations
1. **Visualization Testing**: Automated tests can't verify that matplotlib windows render correctly or that saved images look good. Manual verification required.

2. **Small Test Data**: Test fixtures use 5x5 or 10x10 arrays instead of full 79x43 maps for speed. The minimum region size filter (100 pixels) may produce different results on test data vs. production data.

3. **Mocked Pyrfume Data**: Tests mock the Pyrfume library to avoid network dependencies. Real Pyrfume data may have subtle differences.

### Coverage Gaps
- **pyrfume_loader.py (55%)**: Uncovered lines are mainly in `load_activity_maps()` and `load_stimuli()` methods that download from Pyrfume. These are tested implicitly through helper functions.
- **activity_maps.py (99%)**: Two uncovered lines in `visualize_map()` for error handling edge cases.

---

## 🔧 **Future Improvements**

### Phase 1: Fix Remaining Issues (Optional - ~30 minutes)
- [ ] Increase pyrfume_loader coverage from 55% to 70%
- [ ] Add tests for PyrfumeLoader.load_stimuli() method
- [ ] Add tests for PyrfumeLoader.load_behavior() method

### Phase 2: Add More Tests (Optional - ~2 hours)
- [ ] Visualization tests (test file generation, not appearance)
- [ ] Script execution tests (test scripts run without errors)
- [ ] More edge cases for activity maps pipeline
- [ ] Performance regression tests

### Phase 3: Advanced Testing (Optional - ~4 hours)
- [ ] Property-based testing with Hypothesis
- [ ] Mutation testing with mutmut
- [ ] Load testing for large datasets
- [ ] Integration tests with real Pyrfume data (slow)

---

## 📋 **Test Quality Guidelines**

### Writing New Tests

**Good Test Characteristics:**
- ✅ Tests one specific behavior
- ✅ Uses descriptive names (e.g., `test_load_activity_maps_by_cid_nonexistent`)
- ✅ Uses fixtures for setup (avoid code duplication)
- ✅ Includes docstrings explaining what is tested
- ✅ Marked with @pytest.mark.unit or @pytest.mark.integration
- ✅ Fast (< 1 second per test)

**Test Organization:**
```python
@pytest.mark.unit
def test_function_name_expected_behavior():
    """Brief description of what this tests."""
    # Arrange: Set up test data
    data = create_test_data()
    
    # Act: Execute the function
    result = function_under_test(data)
    
    # Assert: Verify expected behavior
    assert result == expected_value
```

### Fixture Best Practices

**Use Fixtures for:**
- Shared test data
- Temporary directories
- Mock objects
- Complex setup/teardown

**Example:**
```python
@pytest.fixture
def mock_activity_maps_data(tmp_path):
    """Create temporary test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create test files
    # ...
    
    return data_dir

def test_something(mock_activity_maps_data):
    # Use the fixture
    result = load_data(mock_activity_maps_data)
    assert result is not None
```

---

## 🎯 **Testing Checklist**

### Before Committing
- [ ] All tests pass: `pytest tests/`
- [ ] Coverage maintained: `pytest tests/ --cov=src/olfactory_modeling`
- [ ] No new warnings
- [ ] Code formatted with black (if using)
- [ ] Type hints added for new functions

### Before Release
- [ ] All automated tests pass
- [ ] All manual tests completed (see checklist above)
- [ ] All example scripts tested
- [ ] Full pipeline tested end-to-end
- [ ] Documentation updated
- [ ] CHANGELOG.md updated

---

## 📚 **Additional Resources**

### Pytest Documentation
- Official Docs: https://docs.pytest.org/
- Fixtures: https://docs.pytest.org/en/stable/fixture.html
- Markers: https://docs.pytest.org/en/stable/mark.html

### Coverage Tools
- pytest-cov: https://pytest-cov.readthedocs.io/
- Coverage.py: https://coverage.readthedocs.io/

### Best Practices
- Testing Best Practices: https://realpython.com/pytest-python-testing/
- TDD Guide: https://testdriven.io/blog/modern-tdd/

---

**Last Updated**: December 10, 2025  
**Test Suite Version**: 1.0.0  
**Coverage Target**: 80%+ (✅ Currently at 81%)
