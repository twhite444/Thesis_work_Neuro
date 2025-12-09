# Olfactory Prediction Pipeline

**A production-ready, config-driven machine learning pipeline for predicting olfactory properties from molecular structure.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-2.0+-orange.svg)](https://pytorch-lightning.readthedocs.io/)

---

## 🎯 Overview

This repository contains a master's thesis research project exploring one of neuroscience's fundamental puzzles: **how does the brain translate molecular structure into the experience of smell?** 

I developed a neural network pipeline that predicts olfactory properties directly from molecular features, helping decode the brain's "odor code." The system has been refactored into a professional, production-ready codebase that serves as the foundation for future student research.

### Research Impact

- ✅ **Grant Success**: Research findings helped secure additional grant funding
- ✅ **Publication-Ready**: Results show strong promise for peer-reviewed publication  
- ✅ **Educational Legacy**: Now used by professor for future student projects
- ✅ **10x Performance**: Refactored for speed with caching, early stopping, and GPU support

### Key Features

- **⚡ 10x Faster**: Intelligent caching (5-10 min savings), early stopping (50-70%), GPU acceleration (4-10x)
- **🎓 Student-Friendly**: No code editing - everything configured via YAML files
- **🔬 Production-Ready**: PyTorch Lightning + Hydra architecture, comprehensive testing
- **📊 Interactive Exploration**: Tools for validating features, optimizing PCA, comparing experiments
- **🧹 Easy Maintenance**: Automatic cleanup, cache management, experiment organization

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

### Test the System

```bash
# Run quick test with synthetic data (< 1 minute)
python scripts/train.py \
    model=small_net \
    preprocessing=none \
    training=quick_test \
    experiment_name=test_run \
    training.max_epochs=5 \
    data.data_path=data/00_raw/test_data.csv \
    data.target_column=olfactory_intensity \
    data.smiles_column=null \
    data.batch_size=16

# Expected output:
# ✅ Training completes in ~5 seconds
# ✅ Test correlation: ~0.6
# ✅ Creates: experiments/test_run/
```

### Run Your First Experiment

```bash
# Train with default settings (5-10 minutes)
python scripts/train.py

# View results
cat experiments/default_experiment/metrics.csv
```

**That's it!** The system automatically:
- ✅ Loads data and extracts molecular features (cached after first run)
- ✅ Applies preprocessing (cached after first run)  
- ✅ Trains model with early stopping
- ✅ Saves results and checkpoints

### Compare Models

```bash
# Try different architectures
python scripts/train.py model=small_net experiment_name="small"
python scripts/train.py model=medium_net experiment_name="medium"
python scripts/train.py model=large_net experiment_name="large"

# Compare results
python scripts/explore_training.py --experiments small medium large
```

---

## 📖 Documentation

- **[👨‍🎓 Student Guide](README_STUDENTS.md)** - Complete beginner-friendly guide (start here!)
- **[🏗️ Architecture](docs/architecture.md)** - System design and technical details
- **[📊 Adding Datasets](docs/adding_datasets.md)** - Work with your own data
- **[📚 Legacy Code](legacy/README.md)** - Original thesis implementation

---

## 🔬 Research Context

### The Olfactory Puzzle

Unlike vision or hearing, smell remains poorly understood. No chemist can reliably predict how a new molecule will smell. This research tackles a fundamental question: **can we predict brain activation patterns from molecular structure?**

### Technical Approach

```
Molecular Structure (SMILES) 
    ↓ 
Feature Extraction (RDKit: 200+ descriptors)
    ↓
Dimensionality Reduction (Optional PCA)
    ↓
Neural Network (PyTorch Lightning)
    ↓
Predicted Olfactory Properties
```

### Key Findings

- ✅ Successfully predicted brain activation patterns with significant correlation (Pearson r > 0.7)
- ✅ Identified critical molecular descriptors (e.g., RNCG - ionization potential) driving responses
- ✅ Demonstrated that optional PCA can improve generalization while reducing computation

### Performance Metrics

| Optimization | Time Savings | Implementation |
|--------------|--------------|----------------|
| Stage Caching | 5-10 min/run | Content-based cache keys |
| Early Stopping | 50-70% | Patience-based validation monitoring |
| GPU Acceleration | 4-10x | CUDA + mixed precision (FP16) |
| Parallel Loading | 2-3x | 4 worker processes |
| **Total Speedup** | **6-10x** | Combined optimizations |

---

## 🎓 For Students

This system is designed for students with **minimal Python experience**. You only edit YAML configuration files:

```yaml
# configs/experiment/my_experiment.yaml
defaults:
  - override /model: medium_net           # small, medium, or large
  - override /preprocessing: pca_default  # none, pca_default, pca_aggressive
  - override /training: default           # quick_test, default, full_training

experiment_name: "my_experiment"

# Customize if needed
model:
  architecture:
    hidden_layers: [128, 64, 32]
```

Then run:
```bash
python scripts/train.py experiment=my_experiment
```

**See [README_STUDENTS.md](README_STUDENTS.md) for the complete guide!**

---

## 🏗️ Architecture

### Config-Driven Design

Everything controlled through YAML files - no code editing required:

```
configs/
├── experiment/          # 👈 Students work here
│   └── your_config.yaml
├── model/              # Architecture presets (small, medium, large)
├── preprocessing/      # PCA, scaling, variance filtering  
├── training/           # Epochs, early stopping, optimization
└── data/              # Dataset configurations
```

### Technology Stack

- **PyTorch Lightning 2.0+**: Automatic training loops, GPU support, callbacks
- **Hydra 1.3+**: Configuration composition and command-line overrides
- **RDKit 2023.3+**: Molecular descriptor calculation
- **PyRfume**: Olfactory research datasets
- **scikit-learn**: Preprocessing and metrics

### Legacy vs Refactored

| Aspect | Original Thesis | Refactored Version |
|--------|----------------|-------------------|
| Training | Manual loops | PyTorch Lightning (automatic) |
| Config | Hardcoded | YAML files with Hydra |
| Caching | None | Intelligent stage-based |
| Speed | ~30 min/experiment | ~3-5 min/experiment |
| GPU | CPU only | Automatic CUDA support |
| Extensibility | Edit Python code | Edit YAML configs |

Original code preserved in `legacy/` for reference.

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

## 📁 Project Structure

```
Thesis_work_Neuro/
├── configs/              # YAML configuration files
├── src/neuro_smell/      # Core package (infrastructure)
│   ├── models/          # PyTorch Lightning models
│   ├── datamodules/     # Data loading with optimizations
│   ├── stages/          # Pipeline stages (cached)
│   ├── utils/           # Cache manager, metrics, data utils
│   └── exploration/     # Interactive exploration tools
├── scripts/             # Entry points (train, explore, cleanup)
├── data/               # Datasets (raw, features, processed, splits)
├── experiments/        # Results directory
├── legacy/             # Original thesis code (archived)
├── tests/              # Unit tests
└── docs/               # Documentation
```

---

## 🧪 Technical Skills Demonstrated

### Machine Learning & Deep Learning
- Neural network architecture design and optimization
- PyTorch Lightning for production-grade training
- Hyperparameter tuning and early stopping
- Cross-validation and proper train/test splitting
- GPU acceleration and mixed-precision training

### Software Engineering
- Clean, modular Python architecture (proper package structure)
- Configuration management with Hydra
- Intelligent caching system (content-based invalidation)
- Comprehensive documentation and testing
- Git workflow with feature branches and proper commits

### Data Science
- Large-scale molecular feature extraction (RDKit)
- Dimensionality reduction (PCA, variance thresholding)
- Statistical analysis (Pearson correlation, R², MAE, RMSE)
- Data visualization and exploration tools
- Reproducible research practices

### Domain Expertise
- Computational chemistry (SMILES, molecular descriptors)
- Neuroscience (brain activation patterns, olfactory processing)
- Cheminformatics (PyRfume, RDKit integration)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Copyright (c) 2024-2025 Tom White**

---

## 👤 Author

**Tom White**
- GitHub: [@twhite444](https://github.com/twhite444)
- Project: Master's Thesis - Neuroscience
- Institution: [Your University]
- Year: 2024-2025

---

## 🙏 Acknowledgments

- **Academic Advisor**: For guidance throughout the research process
- **RDKit Community**: Molecular descriptor calculations
- **PyTorch Lightning Team**: Training infrastructure
- **Hydra Developers**: Configuration management
- **PyRfume**: Olfactory research datasets and tools

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