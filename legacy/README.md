# 📜 Legacy Code - Original Thesis Implementation

This folder contains the **original research code** from the thesis project that helped secure grant funding.

## ⚠️ Important

**This code is preserved for:**
- Reproducing exact thesis results
- Understanding original methodology
- Historical reference
- Comparing with new implementation

**Do NOT edit these files!** They are frozen snapshots of the working thesis code.

## 📁 Files

### `neural_network_test.py`
- **Purpose**: Original neural network training script
- **Contains**: Manual training loops, grid search, model evaluation
- **Maps to new system**: `src/neuro_smell/models/base_predictor.py` + `scripts/train.py`

### `build.py`
- **Purpose**: Feature extraction from SMILES strings
- **Contains**: RDKit molecular descriptor calculation
- **Maps to new system**: `src/neuro_smell/stages/feature_extraction.py`

### `pca.py`
- **Purpose**: PCA dimensionality reduction
- **Contains**: Variance thresholding, PCA transformation
- **Maps to new system**: `src/neuro_smell/stages/preprocessing.py`

### `grid_search_models.py`
- **Purpose**: Hyperparameter tuning across multiple models
- **Contains**: Lasso, ElasticNet, SVR, KNN, Voting Regressor
- **Maps to new system**: Config-driven experimentation in `configs/experiment/`

### `install_packages.py`
- **Purpose**: Dependency installation script
- **Maps to new system**: `requirements.txt`

## 🔄 Migration Map

| Legacy File | New Location | Notes |
|-------------|--------------|-------|
| `neural_network_test.py` | `src/neuro_smell/models/` | Now uses PyTorch Lightning |
| `build.py` | `src/neuro_smell/stages/feature_extraction.py` | Added caching |
| `pca.py` | `src/neuro_smell/stages/preprocessing.py` | Now optional/configurable |
| `grid_search_models.py` | `configs/experiment/*.yaml` | Config-driven instead of code |
| `install_packages.py` | `requirements.txt` | Standard Python packaging |

## 🚀 Using New System

Instead of running these legacy files, use the new system:

```bash
# Old way
python neural_network_test.py

# New way
python scripts/train.py experiment=baseline
```

See `README_STUDENTS.md` for complete guide.

## 📊 Original Results

These files produced results that:
- ✅ Achieved significant correlation between predicted and actual brain maps
- ✅ Identified key molecular descriptors (e.g., RNCG)
- ✅ Helped secure grant funding for continued research
- ✅ Form the baseline for all future experiments

## 🏷️ Version

Tag: `v1.0-thesis-original`
Branch: `backup/pre-refactor`

To run original code exactly as it was:
```bash
git checkout v1.0-thesis-original
```

---

**Last Updated**: December 2025
**Status**: Archived, Read-Only
