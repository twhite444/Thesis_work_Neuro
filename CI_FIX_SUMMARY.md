# CI Fix Summary - Pytest Collection Error

## Problem
**Both** GitHub Actions CI jobs (unit and integration) were failing with the following error:
```
ERROR legacy/neural_network_test.py
E   ModuleNotFoundError: No module named 'torch'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
```

### Affected CI Jobs
- ✅ `test-unit` job - **FIXED** (was running `pytest -q -m unit`)
- ✅ `test-integration` job - **FIXED** (was running `pytest -q -m integration`)

## Root Cause
Pytest was collecting test files from the `legacy/` directory, which contains old experimental code (`neural_network_test.py`) that requires PyTorch. PyTorch is not in `requirements.txt` and is not needed for the current project.

The issue occurred because `pytest.ini` didn't specify which directories to collect tests from, so pytest defaulted to collecting from all directories in the workspace.

## Solution
Updated `pytest.ini` to:
1. **Restrict test collection to `tests/` directory only** using `testpaths = tests`
2. **Explicitly exclude legacy directory** using `norecursedirs = legacy .git __pycache__ *.egg-info`

### Changes Made

**File: `pytest.ini`**
```ini
[pytest]
# Only collect tests from the tests/ directory, not legacy/
testpaths = tests

markers =
    unit: marks tests as unit (fast, isolated)
    integration: marks tests as integration (end-to-end, multi-module)
    slow: marks tests as slow (optional to run)

# Explicitly ignore legacy directory to prevent collection errors
norecursedirs = legacy .git __pycache__ *.egg-info
```

## Verification

### Before Fix (Both CI Jobs Failed)
```bash
# Unit test job
$ pytest -q -m unit
ERROR legacy/neural_network_test.py
E   ModuleNotFoundError: No module named 'torch'
37 deselected, 1 error in 3.42s

# Integration test job  
$ pytest -q -m integration
ERROR legacy/neural_network_test.py
E   ModuleNotFoundError: No module named 'torch'
85 deselected, 1 error in 5.97s
```

### After Fix (Both CI Jobs Pass Collection)
```bash
# Unit test job
$ pytest -q -m unit
......................................................                  [100%]
54 passed, 46 deselected, 1 warning in 3.51s ✅

# Integration test job (AFTER ADDITIONAL PIPELINE FIX)
$ pytest -q -m integration
......                                                              [100%]
6 passed, 94 deselected in 6.38s ✅

# Full test suite
$ pytest tests/ -v
======================== 100 passed, 1 warning in 7.05s ========================= ✅
```

**Note:** The integration tests required an additional fix to `pipeline_load_and_mask()` to properly use the `output_dir` parameter (see PIPELINE_FIX_SUMMARY.md for details).

## Impact
- ✅ **Both CI jobs will now pass** - Complete success!
- ✅ Unit tests: 54 passing, 46 deselected
- ✅ Integration tests: 6 passing, 94 deselected
- ✅ **100% test pass rate** (100/100 tests passing)
- ✅ No changes needed to test files
- ✅ Legacy code remains in place but is excluded from test collection
- ✅ No additional dependencies required

## Next Steps
Both CI pipeline jobs should now complete successfully with 100% pass rate! 

### Fixes Applied
1. **pytest.ini** - Restricted test collection to `tests/` directory only
2. **activity_maps.py** - Fixed `pipeline_load_and_mask()` to use `output_dir` parameter

The fixes:
- Use pytest best practices for test organization
- Prevent accidental collection of non-test code
- Make the test suite more maintainable and explicit
- Ensure visualization outputs respect function parameters

See **PIPELINE_FIX_SUMMARY.md** for details on the additional pipeline visualization fix.

## Related Files
- `pytest.ini` - Updated configuration
- `legacy/neural_network_test.py` - Legacy file that was causing collection errors (unchanged)
- `tests/` - Primary test directory (unchanged)
