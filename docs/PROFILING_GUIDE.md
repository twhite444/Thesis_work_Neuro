# Performance Profiling Guide

Complete guide to profiling and optimizing training performance using the built-in profiling utilities.

## Table of Contents

- [Quick Start](#quick-start)
- [Profiling Tools](#profiling-tools)
  - [Timer](#timer---simple-profiling)
  - [EpochTimer](#epochtimer---training-breakdown)
  - [profile_dataloader()](#profile_dataloader---dataloader-analysis)
  - [compare_device_performance()](#compare_device_performance---device-benchmarking)
- [Automated Profiling Script](#automated-profiling-script)
- [Interpreting Results](#interpreting-results)
- [Common Issues & Solutions](#common-issues--solutions)

---

## Quick Start

**Run the automated profiling script:**

```bash
# Complete profiling (recommended first run)
python scripts/profile_performance.py --model mlp --profile-epochs 3 --compare-devices

# Quick dataloader check
python scripts/profile_performance.py --model mlp --profile-batches 20

# Just compare CPU vs MPS
python scripts/profile_performance.py --model mlp --compare-devices
```

**Use profiling utilities in your code:**

```python
from olfactory_modeling.utils.profiling import Timer, EpochTimer

# Simple timing
timer = Timer()
with timer.time('my_operation'):
    expensive_function()
timer.print_summary()

# Detailed epoch profiling
epoch_timer = EpochTimer()
# ... (see examples below)
```

---

## Profiling Tools

### Timer - Simple Profiling

**Use Case:** Time any code section and get summary statistics.

**Basic Usage:**

```python
from olfactory_modeling.utils.profiling import Timer

timer = Timer()

# Time different sections
with timer.time('data_loading'):
    molecules = load_molecules()
    features = compute_descriptors(molecules)

with timer.time('model_creation'):
    model = create_model()

with timer.time('data_preprocessing'):
    X_train, X_val = preprocess_data(features)

# Print summary
timer.print_summary()
```

**Output:**

```
Timer Summary:
================================
  data_loading:
    Count:   1
    Total:   2.345s (67.2%)
    Mean:    2.345s
    
  model_creation:
    Count:   1
    Total:   0.123s (3.5%)
    Mean:    0.123s
    
  data_preprocessing:
    Count:   1
    Total:   1.024s (29.3%)
    Mean:    1.024s
    
Total time: 3.492s
================================
```

**Advanced Usage - Multiple Calls:**

```python
timer = Timer()

for epoch in range(10):
    with timer.time('forward_pass'):
        predictions = model(data)
    
    with timer.time('loss_computation'):
        loss = criterion(predictions, targets)
    
    with timer.time('backward_pass'):
        loss.backward()

timer.print_summary()
```

**Output:**

```
Timer Summary:
================================
  forward_pass:
    Count:   10
    Total:   0.450s (30.0%)
    Mean:    0.045s
    Std:     0.002s
    Min:     0.043s
    Max:     0.048s
    
  loss_computation:
    Count:   10
    Total:   0.150s (10.0%)
    Mean:    0.015s
    Std:     0.001s
    
  backward_pass:
    Count:   10
    Total:   0.900s (60.0%)
    Mean:    0.090s
    Std:     0.003s
    
Total time: 1.500s
================================
```

---

### EpochTimer - Training Breakdown

**Use Case:** Detailed profiling of training loops to identify bottlenecks.

**Basic Usage:**

```python
from olfactory_modeling.utils.profiling import EpochTimer

epoch_timer = EpochTimer()

for epoch in range(num_epochs):
    epoch_timer.start_epoch()
    model.train()
    
    for batch_idx, (features, targets, metadata) in enumerate(train_loader):
        epoch_timer.start_batch()
        
        # Profile data transfer
        with epoch_timer.time_section('data_to_device'):
            features = features.to(device)
            targets = targets.to(device)
        
        # Profile forward pass
        with epoch_timer.time_section('forward'):
            predictions = model(features)
            loss = criterion(predictions, targets)
        
        # Profile backward pass
        with epoch_timer.time_section('backward'):
            optimizer.zero_grad()
            loss.backward()
        
        # Profile optimizer step
        with epoch_timer.time_section('optimizer'):
            optimizer.step()
        
        epoch_timer.end_batch()
    
    epoch_timer.end_epoch()

# Print detailed breakdown
epoch_timer.print_epoch_summary()
```

**Output:**

```
======================================================================
EPOCH TIMING BREAKDOWN
======================================================================

Batches:
  Count:        7
  Mean time:    0.127s
  Std:          0.254s
  Throughput:   7.85 batches/s

Section breakdown:

  data_to_device:
    Total:   0.034s (3.8%)
    Mean:    0.0049s
    Per batch: 0.0049s

  forward:
    Total:   0.336s (37.7%)
    Mean:    0.0480s
    Per batch: 0.0480s

  backward:
    Total:   0.068s (7.7%)
    Mean:    0.0098s
    Per batch: 0.0098s

  optimizer:
    Total:   0.453s (50.8%)
    Mean:    0.0648s
    Per batch: 0.0648s

Epoch statistics:
  Completed epochs: 1
  Mean epoch time:  0.90s
  Last epoch time:  0.90s
======================================================================
```

**What to Look For:**

- **data_to_device > 10%**: Dataloader might be slow
- **forward > 50%**: Model might be too complex
- **backward > 30%**: Large model or complex loss
- **optimizer > 50%**: Normal for Adam/AdamW (our case)

---

### profile_dataloader() - DataLoader Analysis

**Use Case:** Measure dataloader performance and identify data loading bottlenecks.

**Basic Usage:**

```python
from olfactory_modeling.utils.profiling import profile_dataloader
from olfactory_modeling.data.activity_map_dataset import get_dataloaders

# Get dataloader
train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=32,
    processed_dir='data/02_processed'
)

# Profile it
stats = profile_dataloader(
    train_loader,
    num_batches=20,
    device='mps'
)
```

**Output:**

```
Profiling dataloader (20 batches)...

DataLoader Profile:
  Batches: 20
  Mean batch size: 32.0
  Data loading: 0.0000s/batch
  Device transfer: 0.0052s/batch
  Total: 0.0052s/batch
  Throughput: 193.73 batches/s
```

**Returned Statistics:**

```python
stats = {
    'batch_count': 20,
    'mean_batch_size': 32.0,
    'data_loading': {'mean': 0.0000, 'std': 0.0001, ...},
    'device_transfer': {'mean': 0.0052, 'std': 0.0003, ...},
    'total_time': {'mean': 0.0052, 'std': 0.0003, ...},
    'throughput': 193.73
}
```

**When to Profile:**

- After changing `num_workers`
- After changing `batch_size`
- After modifying dataset `__getitem__`
- When training seems slow despite fast model

**Good vs Bad:**

- ✅ **Good**: < 10ms/batch, throughput > 100 batches/s
- ⚠️ **Slow**: 10-100ms/batch, throughput 10-100 batches/s
- ❌ **Very Slow**: > 100ms/batch, throughput < 10 batches/s

---

### compare_device_performance() - Device Benchmarking

**Use Case:** Compare model inference speed across different devices (CPU, MPS, CUDA).

**Basic Usage:**

```python
from olfactory_modeling.utils.profiling import compare_device_performance
from olfactory_modeling.models.baseline_nn import get_model
import torch

# Create model
model = get_model(
    model_type='mlp',
    input_dim=268,
    output_shape=(79, 43),
    dropout=0.35
)

# Get sample input
sample_input = torch.randn(1, 268)

# Compare devices
stats = compare_device_performance(
    model,
    sample_input,
    devices=['cpu', 'mps'],  # or ['cpu', 'cuda'] on Linux/Windows
    num_iterations=100
)
```

**Output:**

```
Comparing device performance (100 iterations)...

cpu:
  Mean: 0.09ms
  Std:  0.03ms
  Throughput: 11387.2 inferences/s

mps:
  Mean: 0.39ms
  Std:  0.04ms
  Throughput: 2560.5 inferences/s

Best device: cpu

Speedup comparison (vs cpu):
  mps: 0.22x slower
```

**Interpretation:**

For our small MLP model on MacBook:
- **CPU is 4.5x faster** for single inferences
- **MPS has overhead** that dominates for small models
- **For batch inference**: MPS might be faster with larger batches

**When to Use Each Device:**

| Scenario | Best Device | Reason |
|----------|-------------|--------|
| Small model (<1M params) | CPU | Kernel launch overhead |
| Large model (>10M params) | MPS/CUDA | Parallel computation |
| Small batch (<32) | CPU | Overhead dominates |
| Large batch (>128) | MPS/CUDA | Better utilization |
| Development/debugging | CPU | Easier debugging |
| Production inference | MPS/CUDA | Usually faster overall |

---

## Automated Profiling Script

**Location:** `scripts/profile_performance.py`

**Full Usage:**

```bash
python scripts/profile_performance.py \
    --model mlp \                    # or 'cnn'
    --batch-size 32 \                # batch size for testing
    --profile-batches 20 \           # batches to profile for dataloader
    --profile-epochs 3 \             # epochs to profile (0 = skip)
    --compare-devices \              # compare CPU vs MPS
    --device mps                     # device for main profiling
```

**Common Use Cases:**

**1. Initial Setup - Full Profiling:**
```bash
python scripts/profile_performance.py --model mlp --profile-epochs 3 --compare-devices
```

**2. Quick Dataloader Check:**
```bash
python scripts/profile_performance.py --model mlp --profile-batches 20
```

**3. Device Comparison Only:**
```bash
python scripts/profile_performance.py --model mlp --compare-devices
```

**4. After Making Changes:**
```bash
# Profile before
python scripts/profile_performance.py --model mlp --profile-epochs 3 > before.txt

# Make changes to model/dataloader/etc.

# Profile after
python scripts/profile_performance.py --model mlp --profile-epochs 3 > after.txt

# Compare
diff before.txt after.txt
```

**5. Test Different Batch Sizes:**
```bash
for bs in 16 32 64 128; do
    echo "Batch size: $bs"
    python scripts/profile_performance.py --model mlp --batch-size $bs --profile-epochs 1
done
```

---

## Interpreting Results

### Dataloader Performance

**Symptoms of slow dataloader:**
- `data_to_device` > 10% in EpochTimer
- Dataloader throughput < 100 batches/s
- Training shows GPU/MPS idle time

**Common fixes:**
- Decrease `num_workers` (especially on macOS)
- Increase `batch_size`
- Simplify `__getitem__` in dataset
- Use faster data formats (HDF5, Parquet)
- Reduce data augmentation complexity

### Training Loop Performance

**Normal breakdown for our model:**
- Forward: 30-40% (model computation)
- Backward: 5-10% (gradient computation)
- Optimizer: 40-60% (Adam has momentum updates)
- Data transfer: <5% (small tensors)

**Red flags:**
- Forward > 60%: Model too complex or inefficient
- Backward > 30%: Complex loss or large model
- Optimizer > 70%: Consider SGD instead of Adam
- Data transfer > 10%: Dataloader bottleneck

### Device Comparison

**When CPU is faster:**
- Very small models (<100K params)
- Batch size = 1
- Simple operations (linear, ReLU)

**When MPS/CUDA is faster:**
- Large models (>1M params)
- Large batches (>64)
- Complex operations (convolutions)

**For our baseline:**
- MLP (268→512→256→128→3397): ~500K params
- CPU faster for single inference (small model)
- MPS faster for batch training (parallel batches)

---

## Common Issues & Solutions

### Issue 1: Slow Training Despite Fast Model

**Symptom:**
```
DataLoader Profile:
  Total: 0.150s/batch  ❌ SLOW
  Throughput: 6.7 batches/s
```

**Solution:**
1. Reduce `num_workers` to 0 (macOS)
2. Increase batch size
3. Profile `__getitem__` in dataset

### Issue 2: Data Transfer Bottleneck

**Symptom:**
```
Section breakdown:
  data_to_device: 0.234s (45%)  ❌ TOO HIGH
```

**Solution:**
1. Check dataloader is fast (see Issue 1)
2. Use `persistent_workers=True` if `num_workers > 0`
3. Move preprocessing to GPU if possible
4. Reduce data copy operations

### Issue 3: MPS Slower Than Expected

**Symptom:**
```
cpu:  0.09ms
mps:  0.39ms  ❌ SLOWER
Best device: cpu
```

**Explanation:**
- MPS has kernel launch overhead
- Small models don't saturate GPU
- This is normal and expected

**When to still use MPS:**
- Training with batches (parallel across batch)
- Despite slower per-inference, total training might be faster
- Profile full training loop, not just inference

### Issue 4: Optimizer Taking Too Long

**Symptom:**
```
Section breakdown:
  optimizer: 0.650s (75%)  ❌ TOO HIGH
```

**Solutions:**
1. Use SGD instead of Adam (simpler updates)
2. Reduce model complexity
3. This is often normal for Adam (momentum + variance)

### Issue 5: Inconsistent Timing

**Symptom:**
```
Batches:
  Mean time: 0.127s
  Std:       0.254s  ❌ HIGH VARIANCE
```

**Causes:**
- First batch slower (initialization)
- Garbage collection during training
- Background processes

**Solutions:**
1. Run more iterations to average out variance
2. Use warmup (exclude first few batches)
3. Close other applications

---

## Example Workflows

### Workflow 1: Optimize New Dataset

```bash
# 1. Initial profiling
python scripts/profile_performance.py --model mlp --profile-batches 50 --profile-epochs 3

# 2. If dataloader is slow (>10ms/batch)
# Modify: reduce num_workers, increase batch_size
# in src/olfactory_modeling/data/activity_map_dataset.py

# 3. Re-profile
python scripts/profile_performance.py --model mlp --profile-batches 50

# 4. If still slow, profile dataset __getitem__
# Add timing in dataset code

# 5. Optimize based on findings
```

### Workflow 2: Choose Best Device

```bash
# 1. Compare devices
python scripts/profile_performance.py --model mlp --compare-devices

# 2. Test with actual training
python scripts/profile_performance.py --model mlp --profile-epochs 5 --device cpu > cpu_profile.txt
python scripts/profile_performance.py --model mlp --profile-epochs 5 --device mps > mps_profile.txt

# 3. Compare total times
grep "Time per epoch" cpu_profile.txt
grep "Time per epoch" mps_profile.txt

# 4. Use faster device in training scripts
python scripts/train_baseline_nn.py --model mlp --epochs 100 --device [cpu/mps]
```

### Workflow 3: Profile After Model Changes

```bash
# 1. Baseline before changes
python scripts/profile_performance.py --model mlp --profile-epochs 5 > baseline.txt

# 2. Make model changes (add layers, change activation, etc.)

# 3. Profile after changes
python scripts/profile_performance.py --model mlp --profile-epochs 5 > modified.txt

# 4. Compare
echo "=== BASELINE ==="
grep "Time per epoch" baseline.txt
grep "Section breakdown" -A 10 baseline.txt

echo "=== MODIFIED ==="
grep "Time per epoch" modified.txt
grep "Section breakdown" -A 10 modified.txt
```

---

## Advanced Tips

### 1. Profile Specific Code Sections

```python
from olfactory_modeling.utils.profiling import Timer

timer = Timer()

# Profile feature engineering
with timer.time('descriptor_computation'):
    morgan_fps = compute_morgan(molecules)

with timer.time('fingerprint_concatenation'):
    all_features = np.concatenate([morgan_fps, other_features])

# Profile preprocessing
with timer.time('standardization'):
    X_scaled = scaler.fit_transform(X)

with timer.time('feature_selection'):
    X_selected = selector.fit_transform(X_scaled, y)

timer.print_summary()
```

### 2. Memory Profiling (Manual)

```python
import torch

# Before operation
torch.cuda.empty_cache()  # or torch.mps.empty_cache()
mem_before = torch.cuda.memory_allocated()

# Operation
with timer.time('memory_intensive_op'):
    large_tensor = model(huge_batch)

# After operation
mem_after = torch.cuda.memory_allocated()
print(f"Memory used: {(mem_after - mem_before) / 1e6:.2f} MB")
```

### 3. Continuous Profiling During Training

```python
from olfactory_modeling.utils.profiling import EpochTimer

epoch_timer = EpochTimer()

# Profile every 10 epochs
for epoch in range(100):
    if epoch % 10 == 0:
        epoch_timer.start_epoch()
        profile_this_epoch = True
    else:
        profile_this_epoch = False
    
    for batch in train_loader:
        if profile_this_epoch:
            epoch_timer.start_batch()
            with epoch_timer.time_section('forward'):
                # ... training code
                pass
            epoch_timer.end_batch()
        else:
            # Normal training without profiling overhead
            pass
    
    if profile_this_epoch:
        epoch_timer.end_epoch()
        print(f"\n=== Profiling at epoch {epoch} ===")
        epoch_timer.print_batch_summary()
```

---

## Summary

**Quick Reference:**

| Tool | Use Case | When to Use |
|------|----------|-------------|
| `Timer` | General profiling | Any code section |
| `EpochTimer` | Training breakdown | Training loops |
| `profile_dataloader()` | DataLoader analysis | Slow data loading |
| `compare_device_performance()` | Device selection | Choose CPU/MPS/CUDA |
| `profile_performance.py` | Complete diagnosis | Initial setup, troubleshooting |

**Profiling Checklist:**

- [ ] Run `profile_performance.py --profile-epochs 3 --compare-devices`
- [ ] Check dataloader throughput (should be >100 batches/s)
- [ ] Verify section breakdown (forward ~30-40%, optimizer ~40-60%)
- [ ] Compare device performance (use faster one)
- [ ] Profile after any significant changes
- [ ] Monitor for performance regressions

**Remember:**

- Profile early and often
- Small changes can have big impacts
- Device choice depends on model size and batch size
- Dataloader optimization often yields biggest speedups
- Normal variance is okay, but watch for outliers

---

**Next Steps:**

1. Run initial profiling: `python scripts/profile_performance.py --model mlp --profile-epochs 3 --compare-devices`
2. Address any bottlenecks identified
3. Choose optimal device based on results
4. Set up continuous monitoring for long experiments
5. Document any custom optimizations you discover

Happy profiling! 🚀
