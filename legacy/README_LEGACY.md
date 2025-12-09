# Legacy Code Reference

This folder contains the **complete original implementation** from the master branch (formerly prefactor). These files serve as the reference implementation for the refactored architecture.

**Last Updated:** December 8, 2024  
**Source:** master branch (complete legacy codebase)

---

## 📂 Directory Structure

### Core Pipeline Scripts

#### ✅ **build.py** - VERIFIED MATCH
The main data preprocessing pipeline. **Our implementation matches this exactly!**

- **Purpose:** Load Pyrfume data, extract Mordred features, preprocess, and select features
- **Key Functions:**
  - `load_data()` - Load leon dataset, remove 175 duplicate CIDs
  - `preprocess_data()` - Batch Mordred extraction (1,826 features), NaN/zero removal, StandardScaler
  - `select_features()` - VarianceThreshold(1.0) selection
- **Output:** 287 molecules × 149 features
- **Refactored Version:** `scripts/run_legacy_pipeline.py` (adds smart caching)

**Verification Status:** ✅ Line-by-line comparison confirmed on 2024-12-08

#### **model_comparison_pytorch.py** - Model Training & Evaluation
The main model training script with multiple architectures.

- **Purpose:** Train neural networks on molecular features to predict perceptual ratings
- **Models:** Small, Medium, Large architectures (configurable hidden layers)
- **Features:**
  - Behavioral data integration (perceptual ratings as targets)
  - Train/test split (80/20)
  - Early stopping
  - Learning rate scheduler (ReduceLROnPlateau)
  - Correlation metric evaluation
  - Model checkpointing
- **Refactored Equivalent:** `scripts/train.py` + model configs

### Visualization & Analysis Scripts

#### **PCA_visualization.py**
PCA analysis and visualization of molecular features.
- Dimensionality reduction
- Variance explained plots
- Component visualization

#### **manual_training_visualisation.py**
Training metrics visualization from saved logs.
- Loss curves
- Metric plots
- Model performance analysis

#### **visualise_losses.py**
Simple loss visualization utility.

#### **visualize_model_architecture.py**
Neural network architecture diagrams.

### Experimental Scripts

#### **GNN.py**
Graph Neural Network approach for molecular property prediction.
- Uses molecular graph structure
- Alternative to descriptor-based approach

#### **grid_search_models.py** / **grid_search_multi_target.py**
Hyperparameter search utilities.
- Architecture search
- Learning rate tuning
- Batch size optimization

#### **neural_network_test.py**
Quick testing script for model validation.

### Data Processing Scripts

#### **pca.py**
Original PCA implementation.
- Feature reduction
- Variance analysis

#### **pca_copy.py** / **pca_trial.py**
Alternative PCA experiments.

#### **install_packages.py**
Legacy package installation script.

### Old Implementations

#### **basic_neural_pipeline copy.py**
Earlier version of the pipeline (for reference).

---

## 📊 Data Files

### **pca_transformed_data.csv**
PCA-transformed features from legacy experiments.

---

## 🖼️ Visualization Artifacts

### **global_mask.png**
Feature importance mask visualization.

### **top_3_components.png**
Top 3 PCA components visualization.

### **training_loss.png**
Training loss curves from legacy experiments.

### **training_metrics.png**
Model performance metrics visualization.

### **molecular_graph.png**
Example molecular graph structure (for GNN).

---

## 🧠 Model Files

### **best_model.pth** (379 KB)
Best performing model checkpoint from legacy training.

### **molecular_gnn.pth** (379 KB)
Trained Graph Neural Network weights.

### **model.onnx** (782 KB)
ONNX export of trained model (for deployment).

**Note:** These model files are excluded from git tracking (see `.gitignore`) but kept in legacy folder for local reference.

---

## 🔄 Mapping to Refactored Architecture

| Legacy File | Refactored Equivalent | Status |
|-------------|----------------------|--------|
| `build.py` | `scripts/run_legacy_pipeline.py` | ✅ **VERIFIED MATCH** |
| `model_comparison_pytorch.py` | `scripts/train.py` + configs | 🔄 In Progress |
| `pca.py` | `src/neuro_smell/stages/preprocessing.py` | ✅ Integrated |
| `grid_search_*.py` | Hydra configs + sweeps | 📋 Planned |
| `GNN.py` | Future enhancement | 📋 Planned |
| Visualization scripts | `src/neuro_smell/visualization/` | 📋 Planned |

---

## ✅ Verification History

### December 8, 2024 - Build Pipeline Verification
**Verified:** `build.py` preprocessing pipeline  
**Method:** Line-by-line comparison with `scripts/run_legacy_pipeline.py`  
**Result:** ✅ **EXACT MATCH**

**Key Findings:**
1. ✅ Duplicate CID handling - Identical (175 removed)
2. ✅ Batch Mordred extraction - Identical (1,826 features)
3. ✅ SMILES validation - Identical (`is_valid_smiles()`)
4. ✅ CID index preservation - Identical
5. ✅ NaN removal - Identical (`dropna(axis=1, how='any')`)
6. ✅ Zero removal - Identical (`~(df.eq(0).any(axis=0))`)
7. ✅ StandardScaler - Identical
8. ✅ VarianceThreshold(1.0) - Identical

**Output Verification:**
- Input: 462 molecules → 287 unique (after deduplication)
- Mordred: 1,826 descriptors
- After cleaning: 703 features
- After selection: 149 features
- Runtime: 40 seconds (first run)

**Improvements in Refactored Version:**
- ✨ Smart caching (97% faster reruns)
- ✨ CLI interface (`--variance-threshold`, `--no-cache`, `--force`)
- ✨ Better file organization (`data/00_raw/`, `data/02_processed/`)
- ✨ Enhanced logging

---

## 📋 Next Steps

### Immediate
- [ ] Verify `model_comparison_pytorch.py` against refactored training
- [ ] Compare model architectures (Small, Medium, Large)
- [ ] Verify behavioral data integration

### Future
- [ ] Integrate visualization scripts into refactored structure
- [ ] Add hyperparameter search using Hydra
- [ ] Investigate GNN approach for molecular property prediction

---

## 🔍 Usage Notes

**DO NOT MODIFY** files in this folder - they are reference implementations only.

To use the verified legacy pipeline:
```bash
# Legacy (reference only - use this folder for comparison)
# python legacy/build.py  # DON'T RUN THIS

# Refactored (use this instead)
python scripts/run_legacy_pipeline.py
```

To run legacy model training (for comparison):
```bash
# Legacy (reference only)
# python legacy/model_comparison_pytorch.py  # DON'T RUN THIS

# Refactored (use this instead)
python scripts/train.py +experiment=baseline
```

---

## 📚 Additional Documentation

- **Refactored Architecture:** See `README.md` in project root
- **Build Verification:** See `docs/SUCCESS_REPORT.md`
- **Quick Start:** See `docs/READY_TO_USE.md`

---

**Maintainer Notes:**
- Legacy files pulled from master branch on 2024-12-08
- All files are **read-only references**
- Use refactored implementations for actual work
- Update verification status as components are validated
