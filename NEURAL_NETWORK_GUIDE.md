# Molecular Structure → Activity Map Prediction

## 🎯 Task Definition

**Goal:** Predict brain activity maps (79×43 spatial patterns) from molecular structure

**Input:** Molecular structure (SMILES string)  
**Output:** Activity map representing spatial brain response pattern  
**Type:** Multi-output regression (3,397 continuous values per molecule)

This is a **structure-to-function** prediction problem relevant for:
- Understanding odor perception mechanisms
- Drug discovery and screening
- Neuroscience research on olfactory processing

---

## 📊 Dataset

### Data Statistics
- **287 molecules** with activity maps (after selection from 405 total maps)
- **Activity maps:** 79×43 pixels (3,397 features per map)
- **Coverage:** 20-36% active pixels per map (mean: 29.5%)
- **Value range:** -7.9 to 11.3 (standardized activity)

### Train/Val/Test Split
- **Train:** 70% (200 molecules)
- **Validation:** 15% (43 molecules)
- **Test:** 15% (44 molecules)

Random seed: 42 (for reproducibility)

---

## 🏗️ Implemented Models

### 1. MLP Baseline (`MoleculeToActivityMapMLP`)

**Architecture:**
```
Input: Molecular features (ECFP-2048 or RDKit descriptors)
    ↓
Dense(512) → ReLU → Dropout(0.2)
    ↓
Dense(1024) → ReLU → Dropout(0.2)
    ↓
Dense(512) → ReLU → Dropout(0.2)
    ↓
Dense(3397) → Reshape(79, 43)
```

**Pros:**
- Simple and fast to train
- Good baseline for comparison
- Works with any molecular features

**Cons:**
- Doesn't leverage spatial structure of output
- May struggle to capture spatial patterns

---

### 2. CNN Decoder (`MoleculeToActivityMapCNN`)

**Architecture:**
```
Input: Molecular features (ECFP-2048 or RDKit descriptors)
    ↓
Encoder: Dense(1024) → Dense(512) → latent representation
    ↓
Reshape to spatial: (64, 5, 3)
    ↓
ConvTranspose2d: (64,5,3) → (32,10,6) → (16,20,11) → (8,40,22) → (1,79,43)
```

**Pros:**
- Generates spatial patterns naturally via transposed convolutions
- Respects 2D structure of brain activity maps
- Better suited for spatial prediction tasks

**Cons:**
- More parameters than MLP
- Requires tuning upsampling architecture

---

## 🔧 Molecular Features

### ECFP (Extended-Connectivity Fingerprints)
- **Default:** ECFP-2 (radius=2, 2048 bits)
- Circular fingerprints encoding local molecular structure
- Standard in cheminformatics
- Fixed-size vector representation

### RDKit Descriptors
- **~200 descriptors** including:
  - Molecular weight, LogP, TPSA
  - Atom/bond counts
  - Aromatic rings
  - Hydrogen bond donors/acceptors
  - And many more physicochemical properties

---

## 📈 Evaluation Metrics

### Primary Metrics
1. **MSE (Mean Squared Error)** - Primary loss function
2. **Spatial Correlation** - Pearson correlation between predicted and true maps
3. **R² Score** - Coefficient of determination
4. **MAE (Mean Absolute Error)** - Average absolute pixel difference

### Interpretation
- **Correlation > 0.5:** Moderate spatial pattern similarity
- **Correlation > 0.7:** Strong spatial pattern similarity
- **R² > 0.3:** Model explains >30% of variance (decent for this complex task)

---

## 🚀 Quick Start

### 1. Test the Pipeline (Quick Validation)

```bash
# Test with MLP model, small run to verify everything works
python scripts/train_baseline_nn.py \
    --model mlp \
    --features ecfp \
    --batch-size 16 \
    --epochs 5 \
    --lr 0.001
```

### 2. Train MLP Baseline

```bash
# Full training run with ECFP features
python scripts/train_baseline_nn.py \
    --model mlp \
    --features ecfp \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.001 \
    --checkpoint-dir checkpoints/mlp_ecfp \
    --log-dir runs/mlp_ecfp
```

### 3. Train CNN Decoder

```bash
# Full training with CNN decoder
python scripts/train_baseline_nn.py \
    --model cnn \
    --features ecfp \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.001 \
    --latent-dim 512 \
    --checkpoint-dir checkpoints/cnn_ecfp \
    --log-dir runs/cnn_ecfp
```

### 4. Monitor Training

```bash
# Launch tensorboard
tensorboard --logdir runs/

# Open browser to http://localhost:6006
```

### 5. Compare Feature Types

```bash
# Try RDKit descriptors instead of ECFP
python scripts/train_baseline_nn.py \
    --model cnn \
    --features rdkit \
    --batch-size 32 \
    --epochs 100
```

---

## 📁 Directory Structure

```
Thesis_work_Neuro/
├── data/
│   ├── 01_raw/
│   │   ├── molecules.csv              # Molecular structures (SMILES)
│   │   ├── selected_maps.csv          # Best map per CID (from selection)
│   │   ├── behavior_data.csv          # Activity map metadata
│   │   └── activity_maps_csv/         # Raw activity map CSV files
│   └── selected_maps.csv              # Selected maps for training
├── src/neuro_foundation/
│   ├── data/
│   │   └── activity_map_dataset.py    # PyTorch Dataset class
│   └── models/
│       ├── __init__.py
│       └── baseline_nn.py             # MLP and CNN models
├── scripts/
│   └── train_baseline_nn.py           # Training script
├── checkpoints/                        # Saved model weights
└── runs/                              # Tensorboard logs
```

---

## 🎓 Expected Performance (Initial Estimates)

### MLP Baseline
- **Correlation:** 0.3-0.5 (moderate)
- **R²:** 0.15-0.30
- **Training time:** ~5-10 min per epoch (CPU)

### CNN Decoder
- **Correlation:** 0.4-0.6 (good)
- **R²:** 0.25-0.40
- **Training time:** ~10-15 min per epoch (CPU)

*Note: These are estimates. Actual performance depends on data quality, hyperparameters, and model capacity.*

---

## 🔮 Next Steps After Baseline

### Phase 2: Graph Neural Networks
1. Implement GNN encoder (GCN/GAT) to process molecular graphs
2. Use existing graph generation code from `graph_viz.py`
3. Combine GNN encoder with CNN decoder
4. **Expected improvement:** +10-20% correlation over baseline

### Phase 3: Advanced Architectures
- Attention mechanisms for interpretability
- Multi-task learning (predict multiple properties)
- Transfer learning from pre-trained molecular models
- Ensemble methods

### Phase 4: Analysis & Publication
- Visualize predicted vs actual activity maps
- Analyze which molecular features drive predictions
- Identify failure modes and edge cases
- Write methods section for thesis/paper

---

## 🐛 Troubleshooting

### Issue: CUDA out of memory
**Solution:** Reduce batch size: `--batch-size 16` or `--batch-size 8`

### Issue: Training too slow on CPU
**Solution:** 
- Reduce number of workers: `--num-workers 0`
- Use smaller model: reduce latent_dim or hidden_dims
- Use GPU if available (auto-detected)

### Issue: Model not improving
**Solutions:**
- Lower learning rate: `--lr 0.0001`
- Increase model capacity (more layers/neurons)
- Try different feature type (ECFP vs RDKit)
- Check for data quality issues

### Issue: Overfitting (train loss << val loss)
**Solutions:**
- Increase dropout in model code
- Use L2 regularization
- Reduce model capacity
- Get more training data (use data augmentation)

---

## 📊 Model Checkpoints

Training saves two types of checkpoints:

1. **Best model:** `{model_name}_best.pth`
   - Saved whenever validation loss improves
   - Use this for final evaluation

2. **Periodic checkpoints:** `{model_name}_epoch{N}.pth`
   - Saved every 10 epochs
   - Useful for analyzing training dynamics

### Resume Training
```bash
python scripts/train_baseline_nn.py \
    --model mlp \
    --resume checkpoints/mlp_ecfp_best.pth \
    --epochs 200
```

---

## 📚 References

### Molecular Fingerprints
- Rogers, D., & Hahn, M. (2010). Extended-connectivity fingerprints. Journal of chemical information and modeling, 50(5), 742-754.

### Olfactory Neuroscience
- Wilson, D. A., & Stevenson, R. J. (2006). Learning to smell: olfactory perception from neurobiology to behavior. JHU Press.

### Related Work
- Keller, A., et al. (2017). Predicting human olfactory perception from chemical features of odor molecules. Science, 355(6327), 820-826.

---

## ✅ Current Status

**Completed:**
- ✅ Dataset class with train/val/test splits
- ✅ MLP baseline model
- ✅ CNN decoder model
- ✅ Training pipeline with metrics
- ✅ Tensorboard logging
- ✅ Checkpoint saving/loading

**Ready to Run:**
- ⚡ Quick test run (5 epochs)
- ⚡ Full MLP training (100 epochs)
- ⚡ Full CNN training (100 epochs)

**Next:**
- 🔄 Validate pipeline with test run
- 🔄 Full training for both models
- 🔄 Visualize predictions
- 🔄 Implement GNN models

---

**Good luck with training! 🚀**

Start with a quick test run to make sure everything works, then launch the full training jobs. Monitor progress with tensorboard and adjust hyperparameters as needed.
