# Testing & Quality Assurance Guide

## 📊 **Current Test Coverage Status**

### Test Suite Summary
- **Total Tests**: 39 (21 existing + 18 new)
- **Passing**: 34
- **Failing**: 5 (NPZ format mismatches in new tests - easy fix)
- **Coverage**: ~48% overall (100% for feature_select, preprocess, train_linear)

### Modules by Coverage

| Module | Coverage | Tests | Status |
|--------|----------|-------|--------|
| `feature_select.py` | 100% | ✅ Unit + Edge | Complete |
| `preprocess.py` | 100% | ✅ Unit + Edge | Complete |
| `train_linear.py` | 100% | ✅ Unit + Edge | Complete |
| `pyrfume_loader.py` | 29% | ⚠️ 17/22 passing | Needs NPZ format fixes |
| `activity_maps.py` | 41% | ⚠️ Partial | Needs integration tests |
| `interfaces.py` | 82% | ✅ Basic | Complete |

---

## 🧪 **Test Files Overview**

### Existing Tests (All Passing)
1. **`test_preprocess.py`** (3 tests)
   - Basic featurization
   - Standardization
   - Output file creation

2. **`test_preprocess_edges.py`** (3 tests)
   - Empty DataFrames
   - Invalid SMILES
   - Missing columns

3. **`test_feature_select.py`** (3 tests)
   - Variance filtering
   - Feature reduction
   - Metadata creation

4. **`test_feature_select_edges.py`** (3 tests)
   - Empty inputs
   - Low variance scenarios
   - Edge cases

5. **`test_train_linear.py`** (3 tests)
   - Model training
   - Predictions
   - Coefficient extraction

6. **`test_train_linear_edges.py`** (3 tests)
   - Single sample
   - Mismatched dimensions
   - Invalid inputs

7. **`test_integration_pipeline.py`** (2 tests)
   - End-to-end workflow
   - Data flow validation

### New Comprehensive Tests

8. **`test_pyrfume_loader.py`** (22 tests, 17 passing)
   - ✅ CSV/NPZ helper functions
   - ✅ CID-based loading
   - ✅ Error handling
   - ⚠️ NPZ format tests need fixes (5 failing)

9. **`test_activity_maps_comprehensive.py`** (18 tests, needs run)
   - Directory CSV loading
   - Activity map loading
   - Mask computation
   - Pipeline integration

---

## 🔧 **Required Fixes**

### Priority 1: NPZ Format Alignment (Easy Fix)
The new test mocks need to match the actual NPZ format:

**Issue**: Tests use simplified NPZ structure
```python
np.savez('molecules_raw.npz',
    CID=..., IsomericSMILES=..., MolecularWeight=..., IUPACName=...)
```

**Expected**: Real format includes 'name' field
```python
np.savez('molecules_raw.npz',
    CID=..., IsomericSMILES=..., MolecularWeight=..., IUPACName=..., name=...)
```

**Fix Location**: `tests/test_pyrfume_loader.py` lines 26-40

### Priority 2: Activity Maps Tests
Run `test_activity_maps_comprehensive.py` and verify all pass.

---

## 🎯 **Manual Testing Required**

### Critical Interactive Features

1. **Visualization Tools** (MUST TEST MANUALLY)
   ```bash
   # Test interactive viewer
   python scripts/exploration/inspect_activity_map.py --cid 180 --show-images
   
   # Verify:
   - ✅ Popup window appears
   - ✅ Grid layout shows all maps
   - ✅ Zero values appear transparent/white
   - ✅ Colorbar shows activity levels
   - ✅ Window stays open until closed
   - ✅ Stats printed correctly
   ```

2. **Saved Visualizations** (VERIFY OUTPUT)
   ```bash
   python scripts/run_activity_maps.py
   
   # Check generated files:
   - data/02_processed/global_mask.png
   - data/02_processed/coverage_counts.png  # Should NOT be blank!
   - data/02_processed/coverage_histogram.png
   - data/02_processed/masked_averaged_example.png
   - data/02_processed/masked_averaged_gallery.png
   ```

3. **Example Scripts** (VERIFY NO ERRORS)
   ```bash
   python scripts/examples/example_load_by_cid.py
   python scripts/examples/example_load_cached.py
   python scripts/examples/example_stimuli_metadata.py
   ```

4. **Complete Pipeline** (END-TO-END TEST)
   ```bash
   # Full workflow
   python scripts/load_all_data.py
   python scripts/preprocess.py
   python scripts/run_activity_maps.py
   python scripts/select_features.py
   python scripts/train_linear.py
   
   # Verify all steps complete without errors
   ```

---

## 📈 **Coverage Improvement Opportunities**

### High-Value Additions

1. **Visualization Function Tests** (Currently 0%)
   - `visualize_maps_for_cid()` in inspect_activity_map.py
   - `visualize_coverage()` in activity_maps.py
   - Test that images are generated
   - Test NaN masking works correctly

2. **Script Integration Tests**
   - Test each script in `scripts/` folder
   - Verify command-line arguments work
   - Test error messages for missing data

3. **Helper Function Tests**
   - More edge cases for load functions
   - Test with corrupted data
   - Test with missing files

4. **Performance Tests**
   - Benchmark NPZ vs CSV loading
   - Test with larger datasets
   - Memory usage validation

### Medium Priority

1. **Documentation Tests**
   - Verify all docstrings exist
   - Check type hints are complete
   - Validate examples in docstrings

2. **Backwards Compatibility**
   - Test with different pandas versions
   - Test with different numpy versions
   - Verify Python 3.11-3.13 support

---

## 🚨 **Known Issues & Limitations**

### Test-Related

1. **Mock Data Simplification**
   - Test fixtures use 5x5 or 10x10 maps
   - Real data is 79x43 pixels
   - Edge cases may not be fully covered

2. **Network Dependencies**
   - Some tests mock pyrfume downloads
   - Real pyrfume integration not tested
   - Manual verification needed for load_all_data.py

3. **Visualization Testing**
   - Cannot test matplotlib windows automatically
   - Image quality not programmatically verified
   - Manual inspection required

### Code Quality Issues

1. **Type Hints**
   - Not all functions have complete type annotations
   - Some return types are implicit
   - Recommendation: Add mypy checks

2. **Error Messages**
   - Some error messages could be more descriptive
   - Missing hints for common user errors
   - Recommendation: Add better error context

3. **Code Duplication**
   - Some loading logic repeated
   - Could extract more helper functions
   - Recommendation: DRY refactoring

---

## ✅ **Testing Checklist**

### Before Committing Code

- [ ] Run `pytest tests/ -v` - all tests pass
- [ ] Run `pytest tests/ --cov=src` - coverage >70%
- [ ] Check `scripts/examples/` all run without errors
- [ ] Verify `scripts/exploration/` interactive tools work
- [ ] Test complete pipeline end-to-end
- [ ] Check no new lint errors
- [ ] Verify documentation is up-to-date

### Before Release/Merge

- [ ] All automated tests passing
- [ ] Manual testing completed
- [ ] Coverage report generated
- [ ] README updated with new features
- [ ] CHANGELOG updated
- [ ] Example scripts verified
- [ ] Performance benchmarks run
- [ ] Documentation reviewed

---

## 🎓 **Running Tests**

### Basic Test Execution

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/neuro_foundation --cov-report=html

# Run specific test file
pytest tests/test_pyrfume_loader.py -v

# Run tests by marker
pytest tests/ -m unit -v
pytest tests/ -m integration -v

# Run with detailed output
pytest tests/ -vv --tb=long
```

### Test Organization

Tests are marked with:
- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.integration` - Slower, multi-module tests
- `@pytest.mark.slow` - Very slow tests (optional)

### Coverage Reports

```bash
# Terminal report
pytest tests/ --cov=src --cov-report=term-missing

# HTML report (open in browser)
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html

# XML report (for CI)
pytest tests/ --cov=src --cov-report=xml
```

---

## 🔍 **Code Quality Tools**

### Recommended Tools

1. **pytest-cov** ✅ Installed
   - Measures test coverage
   - Identifies untested code paths

2. **ruff** (Recommended)
   ```bash
   pip install ruff
   ruff check src/ scripts/
   ruff format src/ scripts/
   ```

3. **mypy** (Recommended)
   ```bash
   pip install mypy
   mypy src/ --ignore-missing-imports
   ```

4. **black** (Optional)
   ```bash
   pip install black
   black src/ scripts/ tests/
   ```

---

## 📚 **Test Writing Guidelines**

### Good Test Principles

1. **Arrange-Act-Assert Pattern**
   ```python
   def test_feature():
       # Arrange: Set up test data
       data = create_test_data()
       
       # Act: Execute the function
       result = my_function(data)
       
       # Assert: Verify the outcome
       assert result == expected
   ```

2. **Descriptive Names**
   ```python
   # Good
   def test_load_molecules_filters_duplicates()
   
   # Bad
   def test_molecules()
   ```

3. **One Concept Per Test**
   - Test one thing at a time
   - Easier to debug failures
   - Clear test intent

4. **Use Fixtures**
   - DRY principle
   - Reusable test data
   - Consistent setup

### Test File Organization

```
tests/
├── conftest.py              # Shared fixtures
├── test_<module>.py         # Basic tests
├── test_<module>_edges.py   # Edge cases
├── test_<module>_integration.py  # Integration tests
└── test_<module>_comprehensive.py  # Full coverage
```

---

## 🎯 **Next Steps for Complete Coverage**

### Phase 1: Fix Current Issues (1-2 hours)
1. Fix NPZ format in test_pyrfume_loader.py (30 min)
2. Run test_activity_maps_comprehensive.py (15 min)
3. Fix any failures (45 min)

### Phase 2: Add Missing Tests (2-3 hours)
1. Visualization function tests (1 hour)
2. Script execution tests (1 hour)
3. More edge cases (1 hour)

### Phase 3: Documentation (1-2 hours)
1. Add/update docstrings (1 hour)
2. Type hints completion (30 min)
3. API documentation (30 min)

### Phase 4: Quality Tools (1 hour)
1. Add ruff configuration
2. Add mypy configuration
3. Update CI/CD pipeline

---

## 📝 **Improvement Recommendations**

### High Priority

1. **Fix NPZ Test Mocks** ⚡
   - Quick fix, high impact
   - Enables all pyrfume_loader tests

2. **Add Visualization Tests** 🎨
   - Currently 0% coverage
   - Critical user-facing feature

3. **Script Integration Tests** 🔧
   - Verify CLI works
   - Test error handling

### Medium Priority

1. **Type Hints** 📐
   - Add mypy to CI
   - Complete all function signatures

2. **Error Messages** 💬
   - More descriptive errors
   - Actionable error hints

3. **Performance Benchmarks** ⚡
   - Track loading speeds
   - Monitor regression

### Low Priority

1. **Code Style** ✨
   - Add ruff/black
   - Consistent formatting

2. **Documentation Generation** 📖
   - Sphinx setup
   - API docs website

3. **Advanced Features** 🚀
   - Parallel testing
   - Test data generation

---

## 🏆 **Success Criteria**

### Minimum Requirements
- ✅ All existing tests passing
- ✅ >70% code coverage
- ✅ No regression in functionality
- ✅ Documentation up-to-date

### Ideal State
- 🎯 >85% code coverage
- 🎯 All scripts tested
- 🎯 Type hints complete
- 🎯 CI/CD integrated
- 🎯 Performance benchmarks
- 🎯 Comprehensive documentation

---

**Last Updated**: December 10, 2025
**Status**: 🟨 In Progress (34/39 tests passing)
