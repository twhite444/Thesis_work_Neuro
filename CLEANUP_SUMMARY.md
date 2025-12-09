# Project Cleanup Summary

**Date:** December 9, 2025

## Overview
Cleaned up the project to keep only standard/legacy-based configurations and scripts, removing all experimental and test files.

## Files Removed

### Config Files
- `configs/data/brain_activity.yaml` - Experimental brain activity data config
- `configs/data/test_data.yaml` - Test data config
- `configs/experiment/brain_activity_baseline.yaml` - Experimental brain training config
- `configs/experiment/initial_test.yaml` - Initial test config
- `configs/experiment/test_custom.yaml` - Custom test config
- `configs/experiments/` - Entire duplicate experiments folder

### Script Files
- `scripts/check_invalid_stimuli.py` - Validation script for stimuli
- `scripts/explore_complete_pipeline.py` - Comprehensive exploration script
- `scripts/test_alignment.py` - Data alignment test script
- `scripts/test_brain_training.py` - Brain training validation script
- `scripts/verify_pca_masking.py` - PCA masking verification script

### Documentation Files
- `docs/CONTROL_STIMULI_FIX.md` - Control stimuli fix documentation
- `docs/PCA_MASKING.md` - PCA masking documentation
- `docs/SUCCESS_REPORT.md` - Success report
- `docs/TRAINING_SETUP_COMPLETE.md` - Training setup guide
- `docs/UPDATED_BUILD_CHANGES.md` - Build changes documentation
- `docs/testing.md` - Testing documentation

### Output Directories
- `exploration_outputs/` - All exploration outputs and visualizations
- `test_output/` - All test outputs
- `experiments/baseline/` - Baseline experiment runs
- `experiments/brain_activity_baseline/` - Brain activity training runs
- `experiments/initial_test/` - Initial test runs
- `experiments/test_error/` - Test error runs
- `experiments/test_large/` - Large model test runs
- `experiments/test_medium/` - Medium model test runs
- `experiments/test_overrides/` - Override test runs
- `experiments/test_pca/` - PCA test runs

### Other
- `venv/` - Duplicate virtual environment (keeping `.venv/`)

## Files Kept (Standard/Legacy-Based)

### Config Files (16 files)
```
configs/config.yaml                          # Main config
configs/data/olfactory_default.yaml          # Standard data config
configs/experiment/example_baseline.yaml     # Example baseline
configs/experiment/example_no_pca.yaml       # Example without PCA
configs/experiment/template.yaml             # Template for new experiments
configs/model/large_net.yaml                 # Large network
configs/model/medium_net.yaml                # Medium network
configs/model/small_net.yaml                 # Small network
configs/preprocessing/legacy_pca.yaml        # Legacy PCA preprocessing
configs/preprocessing/none.yaml              # No preprocessing
configs/preprocessing/pca_aggressive.yaml    # Aggressive PCA
configs/preprocessing/pca_default.yaml       # Default PCA
configs/preprocessing/variance_only.yaml     # Variance filtering only
configs/training/default.yaml                # Default training
configs/training/full_training.yaml          # Full training
configs/training/quick_test.yaml             # Quick test training
```

### Script Files (6 files)
```
scripts/cleanup.py                  # Cleanup utility
scripts/download_pyrfume_data.py    # Download Pyrfume data
scripts/process_brain_maps.py       # Process brain activity maps
scripts/run_legacy_pipeline.py      # Run legacy pipeline
scripts/run_pipeline.py             # Run new pipeline
scripts/train.py                    # Training entry point
```

### Documentation (1 file)
```
docs/READY_TO_USE.md               # User guide
```

### Core Directories Preserved
- `data/` - All processed data files
- `legacy/` - Legacy code and scripts
- `src/` - Source code modules
- `tests/` - Unit tests
- `notebooks/` - Jupyter notebooks
- `experiments/` - Empty folder for future training runs (with .gitkeep)

## Project Structure After Cleanup

```
Thesis_work_Neuro/
├── .venv/                    # Virtual environment
├── configs/                  # Configuration files (16 YAML files)
│   ├── config.yaml
│   ├── data/
│   ├── experiment/
│   ├── model/
│   ├── preprocessing/
│   └── training/
├── data/                     # Data files
│   ├── 00_raw/
│   ├── 01_features/
│   ├── 02_processed/
│   └── 03_splits/
├── docs/                     # Documentation
│   └── READY_TO_USE.md
├── experiments/              # Training outputs (empty)
├── legacy/                   # Legacy code
├── notebooks/                # Jupyter notebooks
├── scripts/                  # Utility scripts (6 scripts)
├── src/                      # Source code
│   └── neuro_smell/
├── tests/                    # Unit tests
├── LICENSE
├── README.md
├── README_STUDENTS.md
├── requirements.txt
└── setup.py
```

## Summary

**Removed:**
- 6 experimental config files
- 5 test/exploration script files
- 6 experimental documentation files
- Multiple experimental output directories
- 1 duplicate virtual environment folder

**Total space freed:** Significant reduction in clutter and experimental files

**Kept:**
- 16 standard configuration files
- 6 essential scripts
- All data files and processed outputs
- Complete source code
- Unit tests
- Legacy code for reference

## Next Steps

The project is now cleaned and standardized with only essential files. To run experiments:

1. Use existing experiment configs: `example_baseline.yaml`, `example_no_pca.yaml`
2. Create new configs using `template.yaml` as a base
3. Run training: `python scripts/train.py experiment=example_baseline`
4. Process brain maps: `python scripts/process_brain_maps.py`
5. Run legacy pipeline: `python scripts/run_legacy_pipeline.py`

All experimental features have been removed. The project follows the standard structure based on the legacy codebase.
