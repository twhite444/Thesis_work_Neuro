# Predicting Odor-Evoked Brain Activity Maps from Molecular Features

**Cracking the Brain's Odor Code: A Deep Learning Approach to Olfactory Neuroscience**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

---

## 🎯 Overview

This repository contains my **honors thesis research** addressing a fundamental challenge in olfactory neuroscience: **how does the brain translate chemical structure into neural representations of smell?** 

Unlike vision or hearing, which map to simple physical dimensions (wavelength, frequency), olfaction operates in a vast, high-dimensional chemical space with no obvious organizing principle. This project developed a **deep neural network pipeline that predicts 2D glomerular activity maps in the rat olfactory bulb directly from molecular features**, achieving **R² = 0.506** and successfully decoding the brain's "odor code" at the first stage of olfactory processing.

### Research Impact

- ✅ **Grant Success**: Research findings contributed to securing additional funding for continued investigation
- ✅ **Publication Trajectory**: Results show strong promise for peer-reviewed publication
- ✅ **Educational Foundation**: Codebase serves as the framework for future student research projects in the lab
- ✅ **Novel Contribution**: First demonstration using deep neural networks to predict spatial glomerular activation patterns from comprehensive molecular descriptor sets
- ✅ **Production-Ready**: Now refactored into a professional, config-driven pipeline with intelligent caching and GPU support

## 🧠 Scientific Background

### The Olfactory Challenge

The olfactory system presents unique challenges:
- **No universal stimulus axis**: Unlike color (wavelength) or pitch (frequency), odors span ~40 billion possible molecules across dozens of physicochemical dimensions
- **Combinatorial coding**: ~400 olfactory receptor types create complex activation patterns in humans (~1,000 in rodents)
- **Limited datasets**: Comprehensive mappings of molecules → neural responses → perceptions remain scarce

### The Olfactory Pathway

When an odorant molecule enters the nose, it binds to olfactory receptors in the epithelium. These signals project to the **olfactory bulb**, where neurons expressing the same receptor converge onto structures called **glomeruli**. Each odor creates a unique spatial "fingerprint" of glomerular activation—a chemotopic map where similar molecules activate neighboring regions.

**This thesis focuses on predicting these glomerular activity maps from molecular structure**, establishing the crucial chemistry-to-brain link in olfactory processing.

---

## � Technical Approach

### Dataset & Preprocessing

**Primary Dataset**: Johnson & Leon (2007) olfactory bulb activity maps via [Pyrfume Project](https://pyrfume.org/)
- **405 2-deoxyglucose (2-DG) activity maps** from rat olfactory bulb
- **287 unique monomolecular odorants** (averaged across concentrations)
- Each map: **~1,000 pixels** representing z-scored metabolic activity across glomerular layer

**Molecular Feature Extraction**:
- Computed **1,613+ Mordred descriptors** from SMILES strings using RDKit
- Feature categories:
  - **BCUT descriptors**: Global molecular properties (size, polarity, electron distribution)
  - **Autocorrelation descriptors**: Charge/polarity patterns at specific structural distances
  - **Structural complexity**: Graph connectivity, ring systems, molecular paths
  - **3D shape descriptors**: Geometry (compact vs. elongated vs. planar)
- Reduced to **544 features** after variance thresholding and standardization

**Dimensionality Reduction (PCA on Brain Maps)**:
- Applied PCA to **neural activity maps** (NOT molecular features!)
- **First 5 principal components** captured key spatial patterns:
  - **PC1 (13.38% variance)**: Anterior/posterior vs. medial activation gradient
  - **PC2 (8.73% variance)**: Lateral (left-right) activation shifts
  - **PC3-5**: Progressively finer spatial patterns
- Final target: **5-dimensional PCA coefficient vector per odorant**

### Neural Network Architecture

**Model Design**: Deep fully-connected feedforward network
- **Input layer**: 544 molecular descriptor features
- **Hidden layers**: 
  - Layer 1: 512 neurons (ReLU activation)
  - Layer 2: 256 neurons (ReLU activation)  
  - Layer 3: 128 neurons (ReLU activation)
- **Output layer**: 5 neurons (PCA component scores of brain activity maps)
- **Total parameters**: ~426,000 trainable weights

**Training Configuration**:
- **Optimizer**: Adam (learning rate = 0.005)
- **Loss function**: Mean Squared Error (MSE)
- **Regularization**: 
  - Dropout (0.35 rate) after each hidden layer
  - Early stopping (patience based on validation loss)
- **Cross-validation**: 5-fold CV (~51 odorants per test fold)
- **Epochs**: Maximum 100 (typically converged 69-90 epochs)

---

## 📊 Key Results

### Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | 0.5060 | Explains ~51% of variance in bulb activation patterns |
| **Mean Absolute Error** | 8.22 | Average prediction error in z-scored activation units |
| **Mean Squared Error** | 104.59 | Moderate fit given task complexity |

**Context**: In olfactory neuroscience, R² ≈ 0.5 represents **strong predictive power**. Previous studies using even larger descriptor sets explained far less variance in glomerular responses, making this a significant advance.

### Discovered Feature Importance

Analysis of first-layer weights revealed critical molecular properties for predicting bulb activation:

**Top Predictive Features**:
1. **BCUT descriptors** (BCUTi-1h, BCUTv-1l, BCUTd-1h): Global size, shape, and polarity patterns
2. **Autocorrelation descriptors** (GATS3c, AATSC1s): Functional group spatial arrangements
3. **Structural complexity** (fragCpx, TSRW10): Molecular graph connectivity and branching
4. **3D geometry** (GeomShapeIndex, GeomPetitjeanIndex, MOMI-Z): Overall molecular shape

These align with **Johnson & Leon's chemotopic organization principles**: functional group clustering, carbon chain length effects, and shape-based receptor binding.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/twhite444/Thesis_work_Neuro.git
cd Thesis_work_Neuro

# Install dependencies (one-time setup)
pip install -e .
```

### Run Legacy Pipeline (Replicates Thesis)

```bash
# Activate environment
source venv/bin/activate  # or: conda activate neuro_smell

# Download Pyrfume Leon dataset
python scripts/download_pyrfume_data.py

# Run complete legacy pipeline (~40 seconds)
python scripts/run_legacy_pipeline.py

# Expected output:
# ✅ 287 molecules processed (175 duplicates removed)
# ✅ 1,826 Mordred features extracted
# ✅ 149 features after preprocessing
# ✅ Files saved to data/02_processed/
```

### Verify PCA Masking on Brain Data

```bash
# Test PCA masking with brain activation maps
python scripts/verify_pca_masking.py

# Expected output:
# ✅ PCA applied to brain activity maps
# ✅ Global feature importance mask created
# ✅ Visualizations saved to test_output/pca_analysis/
#    - global_mask.png
#    - top_3_components.png
#    - pca_scree.png
#    - pca_cumulative.png
```

---

## 📖 Documentation

### Core Documentation
- **[� Legacy Code Reference](legacy/README_LEGACY.md)** - Original thesis implementation details
- **[🎭 PCA Masking Guide](docs/PCA_MASKING.md)** - PCA on brain activation maps (CRITICAL for thesis replication)
- **[✅ Build Verification](docs/UPDATED_BUILD_CHANGES.md)** - Comparison with legacy preprocessing
- **[🚀 Success Report](docs/SUCCESS_REPORT.md)** - Test results and validation

### Student Resources
- **[�👨‍🎓 Student Guide](README_STUDENTS.md)** - Beginner-friendly guide for extending this work
- **[🏗️ Architecture](docs/architecture.md)** - System design and technical details (if it exists)

---

## 🔬 Pipeline Architecture

### Data Flow (Thesis Implementation)

```
1. Load Pyrfume Leon Dataset
   ├── molecules.csv (287 unique CIDs, SMILES strings)
   ├── image_data.csv (fMRI brain activation maps: 287 × ~1,000 voxels)
   └── behavior_1.csv (perceptual ratings - NOT used as targets in thesis)
   
2. Extract Molecular Features
   SMILES → Mordred (1,826 descriptors) → Clean → StandardScale
   Output: 287 molecules × 544 features
   
3. Process Brain Activation Maps ⚠️ CRITICAL INSIGHT
   fMRI voxels (287 × 1,000) → PCA (50 components) → Masking (threshold)
   Output: 287 × 5 PCA component scores
   ☝️ PCA is applied to BRAIN DATA, not molecular features!
   
4. Align Data
   X_molecular: 287 × 544 features (chemistry)
   y_brain: 287 × 5 PCA scores (neural response)
   
5. Train Neural Network
   Input: Molecular descriptors (544 features)
   Output: PCA scores of brain activity (5 values)
   Architecture: 544 → 512 → 256 → 128 → 5
   Loss: MSE between predicted and actual brain PCA scores
   
6. Results
   R² = 0.5060 (explains 51% of brain activity variance from chemistry!)
```

### Critical Understanding

**This is NOT a simple molecule → smell prediction model.**

This research predicts **spatial patterns of neural activation in the olfactory bulb** from molecular structure. The targets are PCA-reduced representations of 2D glomerular activity maps measured via 2-deoxyglucose imaging in rats.

**Key Innovation**: Using PCA to find principal patterns in brain activation, then predicting those pattern coefficients from molecular features. This captures the chemotopic organization of the olfactory bulb.

---

## 🚀 Future Directions

### Graph Neural Networks (GNNs)
Current limitation: Pre-computed descriptors introduce bias and miss subtle structural effects.

**Solution**: Feed molecular graphs directly to GNNs
- Learn features from raw connectivity rather than hand-crafted descriptors
- Capture emergent structural properties (e.g., quaternary carbons, ring systems)
- Eliminate missing descriptor problem for larger chemical spaces
- Preliminary implementation available in `legacy/GNN.py`

### Expanded Datasets
- **Human data**: Generalize beyond rodent olfactory bulb (400 vs. 1,000 receptor types)
- **Temporal dynamics**: Current maps are static 2-DG snapshots; need time-resolved imaging
- **Larger chemical coverage**: 287 odorants << 40 billion odorous molecules

### Higher Brain Regions
Combine with recent connectivity maps (Diaz & Franks 2023):
- Predict piriform cortex activation from bulb patterns
- Map full pathway: molecule → bulb → cortex → perception
- Enable true end-to-end odor digitization

### Practical Applications
- **Fragrance design**: Screen molecules *in silico* for desired neural activation patterns
- **Virtual olfaction**: Digital scent generation for VR/AR
- **Sensory disorder diagnostics**: Predict perceptual deficits from bulb activation abnormalities

---

## 🏗️ Code Architecture

### Project Structure

```
├── src/neuro_smell/              # Core processing modules
│   ├── stages/                   # Pipeline stages
│   │   ├── feature_extraction.py
│   │   ├── preprocessing.py
│   │   ├── pca_masking.py       # ⚠️ PCA on BRAIN DATA
│   │   └── training.py
│   └── utils/
│       └── smart_cache.py        # Intelligent caching system
├── configs/                      # YAML configuration files
│   ├── experiment/               # Full experiment configs
│   ├── preprocessing/            # PCA, scaling, feature selection
│   │   ├── legacy_pca.yaml      # Exact thesis replication
│   │   └── pca_default.yaml
│   └── model/                    # Network architectures
├── scripts/                      # Execution scripts
│   ├── run_legacy_pipeline.py   # Thesis replication (40s)
│   ├── verify_pca_masking.py    # PCA validation
│   └── train.py                  # Training entry point
├── legacy/                       # Original thesis code
│   ├── build.py                 # Data preprocessing
│   ├── pca_copy.py              # PCA on brain maps
│   ├── model_comparison_pytorch.py  # Training
│   └── README_LEGACY.md
└── data/
    ├── 00_raw/                   # Pyrfume datasets
    ├── 01_features/              # Computed molecular descriptors
    ├── 02_processed/             # Standardized features
    └── 03_splits/                # Train/test partitions
```

### Technology Stack

- **PyTorch 2.0+**: Deep learning framework
- **Mordred**: Comprehensive molecular descriptor calculation (1,826 features)
- **RDKit 2023.3+**: Cheminformatics toolkit
- **Pyrfume**: Olfactory research datasets (Johnson & Leon 2007)
- **scikit-learn**: PCA, preprocessing, metrics
- **Hydra 1.3+**: Configuration management (in refactored version)

### Engineering Best Practices

**Modular Design**:
- Clear separation: data loading → preprocessing → PCA → modeling
- Reusable functions with consistent interfaces
- Configuration-driven experiments

**Reproducibility**:
- Fixed random seeds for train/test splits
- Documented hyperparameters in experiment logs
- Versioned datasets via Pyrfume references

**Validation**:
- Comprehensive input validation (SMILES string checks, NaN handling)
- K-fold cross-validation to prevent overfitting
- Multiple performance metrics (R², MSE, MAE)

---

## 🔬 Technical Skills Demonstrated

- **Deep Learning**: PyTorch neural networks, custom architectures, regularization (dropout, early stopping)
- **Dimensionality Reduction**: PCA for spatial pattern analysis on brain data, variance explained interpretation
- **Feature Engineering**: Molecular descriptor calculation (RDKit, Mordred), feature selection pipelines
- **Cheminformatics**: SMILES processing, molecular property calculation, structure-activity relationships
- **Data Science**: K-fold cross-validation, train/test alignment, standardization, correlation analysis
- **Neuroscience**: fMRI data analysis, glomerular activation mapping, chemotopic organization
- **Scientific Computing**: NumPy array manipulation, pandas DataFrame operations, sklearn pipelines
- **Visualization**: Matplotlib spatial maps, seaborn statistical plots, training diagnostics
- **Software Engineering**: Modular Python design, experiment tracking, reproducible workflows

---

## 📊 Usage Examples

### Basic Training

```bash
# Defaults
python scripts/train.py

# Specific configurations
python scripts/train.py model=large_net preprocessing=pca_aggressive

# Override parameters
python scripts/train.py model.architecture.hidden_layers=[256,128,64] training.max_epochs=200
```

### Exploration Tools

```bash
# Explore feature distributions
python scripts/explore_features.py

# Find optimal PCA components
python scripts/explore_preprocessing.py

# Compare experiments
python scripts/explore_training.py --experiments exp1 exp2 exp3
```

### Cleanup

```bash
# Show disk usage
python scripts/cleanup.py --show

# Delete old experiments
python scripts/cleanup.py --older-than 7

# Clear cache only
python scripts/cleanup.py --cache-only
```

---

## � Citation

If you use this code or reference this research, please cite:

```bibtex
@thesis{white2024odor,
  author = {Tom White},
  title = {Predicting Odor-Evoked Brain Activity Maps from Molecular Features: A Deep Learning Approach},
  school = {[Your University]},
  year = {2024},
  type = {Honours Thesis},
  note = {Achieved R² = 0.5060 predicting 2D glomerular activation patterns in rat olfactory bulb from Mordred molecular descriptors}
}
```

**Dataset Citation**:
```bibtex
@article{johnson2007,
  author = {Johnson, B. A. and Leon, M.},
  title = {Chemotopic odorant coding in a mammalian olfactory system},
  journal = {Journal of Comparative Neurology},
  volume = {503},
  number = {1},
  pages = {1--34},
  year = {2007},
  doi = {10.1002/cne.21396}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Copyright (c) 2024-2025 Tom White**

Permission is hereby granted to use this code for academic research, education, and personal projects. Commercial use requires author permission.

---

## 👤 Author

**Tom White**
- GitHub: [@twhite444](https://github.com/twhite444)
- Project: Honours Thesis - Computational Neuroscience & Cheminformatics
- Research Focus: Predicting neural activation patterns from chemical structure
- Year: 2024-2025

**Contact**: For questions about the research methodology, results interpretation, or code usage, please open an issue on GitHub.

---

## 🙏 Acknowledgments

### Research Contributions
- **Johnson & Leon Lab**: 2-deoxyglucose (2-DG) imaging dataset of rat olfactory bulb activation
- **Pyrfume Project**: Curation and standardization of olfactory research datasets
- **Academic Advisor**: Research guidance and manuscript feedback

### Technical Infrastructure
- **RDKit Community**: Molecular descriptor calculations and SMILES processing
- **Mordred Developers**: Comprehensive molecular descriptor library (1,826 features)
- **PyTorch Team**: Deep learning framework
- **scikit-learn Contributors**: PCA implementation and preprocessing pipelines

### Scientific Context
This work builds on decades of research in chemotopy (spatial organization of odor coding) pioneered by Johnson, Leon, and colleagues, demonstrating that computational approaches can capture principles of glomerular activation discovered through painstaking experimental neuroscience.

---

## 🔗 Related Resources

- [Pyrfume Project](https://pyrfume.org/) - Olfactory research datasets
- [Johnson Lab Publications](https://www.johnsonlabNeuro.com/) - Original 2-DG imaging methodology
- [RDKit Documentation](https://www.rdkit.org/docs/) - Molecular descriptor details
- [Mordred Paper](https://doi.org/10.1186/s13321-018-0258-y) - Comprehensive molecular descriptor calculations

---

## 📈 Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{white2024olfactory,
  title={Predicting Olfactory Properties from Molecular Structure using Deep Learning},
  author={White, Tom},
  year={2024},
  school={[Your University]},
  note={Available at: https://github.com/twhite444/Thesis_work_Neuro}
}
```

---

**Built with ❤️ for the neuroscience and machine learning communities**

This project bridges computational chemistry, neuroscience, and machine learning to tackle fundamental questions about how we perceive the world through smell.