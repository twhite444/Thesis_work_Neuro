# Student Guide: Olfactory Prediction Pipeline

**Copyright (c) 2024-2025 Tom White | Licensed under the MIT License**

Welcome! This guide will help you run experiments and understand this codebase, even if you're new to Python or machine learning.

---

## 🎯 Quick Start (5 Minutes)

### 1. Installation

```bash
# Clone the repository
git clone <repo-url>
cd Thesis_work_Neuro

# Install dependencies (this may take a few minutes)
pip install -e .
```

### 2. Run Your First Experiment

```bash
# Train a model with default settings
python scripts/train.py

# That's it! The system will:
# - Load your data
# - Extract features (if needed)
# - Train a model
# - Save results
```

### 3. View Results

Results are saved in `experiments/<experiment-name>/`

---

## 📖 Table of Contents

1. [How This System Works](#how-this-system-works)
2. [Running Experiments](#running-experiments)
3. [Configuration Guide](#configuration-guide)
4. [Understanding Results](#understanding-results)
5. [Common Tasks](#common-tasks)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

---

## 🔧 How This System Works

### The Pipeline

This system runs experiments in **three stages**:

```
📄 CSV Data → 🧪 Feature Extraction → 🔬 Preprocessing → 🤖 Training → 📊 Results
```

1. **Feature Extraction**: Convert SMILES to molecular descriptors (cached!)
2. **Preprocessing**: Optional PCA dimensionality reduction (cached!)
3. **Training**: Train neural network with early stopping

**💡 The magic**: Once a stage is computed, it's **cached**. If you change training settings but not data settings, feature extraction is skipped (saves 5-10 minutes!).

### Configuration-Driven Design

**You never edit Python code.** Everything is controlled by YAML config files:

```
configs/
├── config.yaml              # Main config (don't edit)
├── experiment/              # 👈 YOU WORK HERE
│   ├── my_experiment.yaml   # Your experiment configs
│   └── ...
├── model/                   # Model architectures (small, medium, large)
├── preprocessing/           # Preprocessing options (PCA, scaling)
├── training/                # Training settings (epochs, early stopping)
└── data/                    # Dataset configurations
```

---

## 🚀 Running Experiments

### Basic Usage

```bash
# Use default settings
python scripts/train.py

# Use a pre-made experiment config
python scripts/train.py experiment=example_baseline

# Try without PCA
python scripts/train.py experiment=example_no_pca
```

### Override from Command Line

You can change any setting without editing files:

```bash
# Use a larger model
python scripts/train.py model=large_net

# Train for longer
python scripts/train.py training.max_epochs=200

# Change multiple things at once
python scripts/train.py model=large_net training=full_training preprocessing=pca_aggressive
```

### Full Pipeline with Caching

For maximum efficiency, use the pipeline runner:

```bash
# Run full pipeline (uses cache when possible)
python scripts/run_pipeline.py experiment=my_experiment
```

---

## ⚙️ Configuration Guide

### Creating Your Own Experiment

1. **Copy a template:**
   ```bash
   cp configs/experiment/example_baseline.yaml configs/experiment/my_experiment.yaml
   ```

2. **Edit your config:**
   ```yaml
   # @package _global_
   
   defaults:
     - override /model: medium_net          # small_net, medium_net, or large_net
     - override /preprocessing: pca_default  # none, pca_default, pca_aggressive
     - override /training: default           # quick_test, default, full_training
   
   experiment_name: "my_experiment"
   
   # Override specific settings
   model:
     architecture:
       hidden_layers: [128, 64, 32]  # Custom architecture
   
   training:
     max_epochs: 100
     early_stopping:
       patience: 15
   ```

3. **Run your experiment:**
   ```bash
   python scripts/train.py experiment=my_experiment
   ```

### Available Configurations

#### Models (`configs/model/`)

- **`small_net`**: [64, 32] layers - Fast, for quick tests (1-2 min)
- **`medium_net`**: [128, 64, 32] layers - Balanced, recommended default (5-10 min)
- **`large_net`**: [256, 128, 64, 32] layers - Maximum capacity (15-30 min)

#### Preprocessing (`configs/preprocessing/`)

- **`none`**: No preprocessing - Use all features as-is
- **`pca_default`**: 50 components (~95% variance) - Standard
- **`pca_aggressive`**: 20 components (~85% variance) - Faster training
- **`variance_only`**: Remove low-variance features, no PCA

#### Training (`configs/training/`)

- **`quick_test`**: 10 epochs, patience=3 - Just test if it works (1-2 min)
- **`default`**: 100 epochs, patience=15 - Standard training (5-10 min)
- **`full_training`**: 200 epochs, patience=25, FP16 - Maximum performance (15-30 min)

---

## 📊 Understanding Results

### Where Are My Results?

After training, find results in:
```
experiments/<experiment-name>/
├── config.yaml              # Full config used
├── checkpoints/             # Model checkpoints
│   └── best_model.ckpt     # Best model (highest validation correlation)
├── logs/                    # TensorBoard logs
└── metrics.csv              # Training metrics
```

### Key Metrics

The system tracks several metrics:

- **Pearson Correlation** ⭐ - Primary metric (range: -1 to 1, higher is better)
  - Measures linear relationship between predictions and true values
  - 0.7+ is good, 0.8+ is excellent, 0.9+ is outstanding
  
- **R² Score** - Variance explained (range: -∞ to 1, higher is better)
  - How much variance in data the model captures
  
- **MAE** (Mean Absolute Error) - Average prediction error (lower is better)
  
- **RMSE** (Root Mean Squared Error) - Penalizes large errors (lower is better)

### Viewing Training Progress

```bash
# View metrics in terminal
cat experiments/my_experiment/metrics.csv

# Or use TensorBoard for interactive plots
tensorboard --logdir experiments/my_experiment/logs
```

---

## 🛠️ Common Tasks

### Task 1: Compare Different Models

```bash
# Train with each model size
python scripts/train.py model=small_net experiment_name="model_comparison_small"
python scripts/train.py model=medium_net experiment_name="model_comparison_medium"
python scripts/train.py model=large_net experiment_name="model_comparison_large"

# Compare results
python scripts/explore_training.py --experiments model_comparison_*
```

### Task 2: Test If PCA Helps

```bash
# With PCA
python scripts/train.py preprocessing=pca_default experiment_name="with_pca"

# Without PCA
python scripts/train.py preprocessing=none experiment_name="without_pca"

# Compare
python scripts/explore_training.py --experiments with_pca without_pca
```

### Task 3: Find Best PCA Components

```bash
# Interactive exploration
python scripts/explore_preprocessing.py

# This will:
# - Show variance explained by each component
# - Recommend optimal number of components
# - Visualize component importance
```

### Task 4: Use New Data

1. **Place your CSV in `data/00_raw/`**
   ```
   data/00_raw/my_new_data.csv
   ```

2. **Create a data config:**
   ```bash
   cp configs/data/olfactory_default.yaml configs/data/my_data.yaml
   ```
   
   Edit `my_data.yaml`:
   ```yaml
   data_path: "data/00_raw/my_new_data.csv"
   target_column: "your_target_column"
   smiles_column: "your_smiles_column"  # If extracting features
   ```

3. **Run with your data:**
   ```bash
   python scripts/train.py data=my_data experiment_name="new_data_test"
   ```

### Task 5: Clean Up Old Experiments

```bash
# See what's taking up space
python scripts/cleanup.py --show

# Delete experiments older than 7 days
python scripts/cleanup.py --older-than 7

# Delete specific experiments
python scripts/cleanup.py --experiments old_experiment_1 old_experiment_2

# Clean cache but keep experiments
python scripts/cleanup.py --cache-only
```

---

## ❓ Troubleshooting

### Problem: "Module not found" errors

**Solution**: Install the package properly
```bash
pip install -e .
```

### Problem: "CUDA out of memory"

**Solution**: Use smaller batch size or model
```bash
python scripts/train.py data.batch_size=16 model=small_net
```

Or disable GPU:
```bash
python scripts/train.py training.accelerator=cpu
```

### Problem: Training is too slow

**Solutions:**

1. Use smaller model: `model=small_net`
2. Enable mixed precision: `training.precision=16`
3. Use quick test settings: `training=quick_test`
4. Reduce data: Subset your CSV before loading

### Problem: Model not learning (loss not decreasing)

**Solutions:**

1. Check your data: `python scripts/explore_features.py`
2. Try different learning rate: `model.optimizer.lr=0.001`
3. Use simpler model first: `model=small_net`
4. Check for missing values in data

### Problem: Validation loss increasing (overfitting)

**Solutions:**

1. Enable dropout: `model.architecture.dropout=0.3`
2. Reduce model size: `model=small_net`
3. More aggressive early stopping: `training.early_stopping.patience=5`
4. Add regularization: `model.optimizer.weight_decay=0.01`

### Problem: "Cache key mismatch" warnings

**Solution**: Cache detected config change. This is normal! The stage will recompute.

To force recomputation:
```bash
python scripts/cleanup.py --cache-only
```

---

## 🚀 Advanced Usage

### Custom Architectures

Edit your experiment config to try different architectures:

```yaml
model:
  architecture:
    hidden_layers: [512, 256, 128, 64]  # Deeper network
    activation: "relu"                   # or "tanh", "gelu"
    dropout: 0.3                         # Regularization
    batch_norm: true                     # Normalize activations
```

### Hyperparameter Sweeps

Want to try many settings? Use Hydra multirun:

```bash
# Try multiple learning rates
python scripts/train.py -m model.optimizer.lr=0.001,0.0001,0.00001

# Try multiple architectures
python scripts/train.py -m 'model.architecture.hidden_layers="[64,32]","[128,64,32]","[256,128,64]"'
```

Each run gets its own directory in `multirun/`.

### Using GPU

If you have a CUDA-compatible GPU:

```yaml
training:
  accelerator: "gpu"
  devices: 1
  precision: 16  # Enable mixed precision for 2x speedup
```

### Custom Metrics

The system computes Pearson correlation by default. To add custom metrics, see `src/neuro_smell/utils/metrics.py`.

### Callbacks

PyTorch Lightning callbacks are configured in `training` settings:

```yaml
training:
  callbacks:
    early_stopping:
      monitor: "val_correlation"
      patience: 15
      mode: "max"
    model_checkpoint:
      monitor: "val_correlation"
      mode: "max"
      save_top_k: 3
```

---

## 📚 Additional Resources

### Project Structure

```
Thesis_work_Neuro/
├── configs/              # All configuration files (YOU EDIT HERE)
├── src/neuro_smell/      # Core code (DON'T EDIT - it's infrastructure)
├── scripts/              # Entry points (run these)
├── data/                 # Your datasets
├── experiments/          # Results go here
├── legacy/               # Original thesis code (archived)
├── docs/                 # Additional documentation
└── notebooks/            # Jupyter notebooks for exploration
```

### Important Files

- **`README.md`**: Main project README
- **`README_STUDENTS.md`**: This file
- **`requirements.txt`**: Python dependencies
- **`setup.py`**: Package installation
- **`LICENSE`**: MIT License

### Learning More

- **PyTorch Lightning**: https://pytorch-lightning.readthedocs.io/
- **Hydra**: https://hydra.cc/docs/intro/
- **RDKit**: https://www.rdkit.org/docs/
- **Neural Networks**: https://www.deeplearningbook.org/

### Getting Help

1. Check this guide thoroughly
2. Look at `configs/experiment/example_*.yaml` for examples
3. Run exploration tools to understand your data
4. Check the error message carefully - they're usually informative

---

## 🎓 Understanding the Thesis Work

This system was built for research on predicting olfactory properties from molecular structure. The original work (in `legacy/`) used Keras and manual training loops. This refactored version:

- **10x faster** (caching + GPU + early stopping)
- **Easier to use** (config-driven, no code editing)
- **More maintainable** (proper package structure)
- **Better documented** (you're reading it!)

The core scientific approach remains the same:
1. Represent molecules as descriptor vectors
2. Optionally reduce dimensionality with PCA
3. Train neural network to predict olfactory properties
4. Evaluate with Pearson correlation

**Key Innovation**: Making this accessible to students with minimal coding experience through configuration files.

---

## ✅ Checklist for First Experiment

- [ ] Installed dependencies (`pip install -e .`)
- [ ] Placed data CSV in `data/00_raw/`
- [ ] Created data config in `configs/data/`
- [ ] Created experiment config in `configs/experiment/`
- [ ] Ran quick test: `python scripts/train.py training=quick_test`
- [ ] Checked results in `experiments/<name>/`
- [ ] Viewed metrics: `cat experiments/<name>/metrics.csv`
- [ ] Ready to iterate!

---

**Good luck with your experiments! 🚀**

*Questions? Check the troubleshooting section or examine the example configs.*
