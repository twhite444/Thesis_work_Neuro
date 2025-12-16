# Profiling Tools - Quick Reference

## 🚀 Quick Start

**Run complete profiling:**
```bash
python scripts/profile_performance.py --model mlp --profile-epochs 3 --compare-devices
```

## 📊 What You Get

### 1. DataLoader Performance
```
DataLoader Profile:
  Total: 0.0052s/batch
  Throughput: 193.73 batches/s
  ✓ Data loading is fast
```

### 2. Device Comparison
```
cpu:  0.09ms/inference (11,387 inferences/s)
mps:  0.39ms/inference (2,561 inferences/s)
Best device: cpu (4.5x faster)
```

### 3. Training Breakdown
```
Section breakdown:
  forward:     37.7% (model computation)
  backward:     7.7% (gradient computation)
  optimizer:   50.8% (Adam updates - normal)
  data_to_device: 3.8% (data transfer)
```

## 🛠️ Available Tools

### In Python Code

```python
# Simple timing
from src.neuro_foundation.utils.profiling import Timer

timer = Timer()
with timer.time('my_operation'):
    expensive_function()
timer.print_summary()
```

```python
# Detailed training profiling
from src.neuro_foundation.utils.profiling import EpochTimer

epoch_timer = EpochTimer()
for epoch in range(num_epochs):
    epoch_timer.start_epoch()
    for batch in dataloader:
        epoch_timer.start_batch()
        with epoch_timer.time_section('forward'):
            predictions = model(features)
        epoch_timer.end_batch()
    epoch_timer.end_epoch()
epoch_timer.print_epoch_summary()
```

```python
# Profile dataloader
from src.neuro_foundation.utils.profiling import profile_dataloader

stats = profile_dataloader(train_loader, num_batches=20, device='mps')
```

```python
# Compare devices
from src.neuro_foundation.utils.profiling import compare_device_performance

stats = compare_device_performance(model, sample_input, devices=['cpu', 'mps'])
```

### CLI Script

```bash
# Full profiling
python scripts/profile_performance.py --model mlp --profile-epochs 3 --compare-devices

# Just dataloader
python scripts/profile_performance.py --model mlp --profile-batches 20

# Just device comparison
python scripts/profile_performance.py --model mlp --compare-devices
```

## ✅ Key Findings

**Our Current Performance:**
- ✅ DataLoader: **Very fast** (5-8ms/batch, 120-190 batches/s)
- ✅ Training: **Optimized** (0.05-0.12s/batch, 8-21 it/s)
- ✅ No bottlenecks identified

**Device Performance:**
- For our small models: **CPU is 4.5x faster** than MPS for single inference
- For batch training: **MPS might still be beneficial** (parallel across batch)
- Recommendation: Use CPU for development, test MPS for long training runs

**Training Breakdown:**
- Forward pass: 30-40% (normal)
- Backward pass: 5-10% (normal)
- Optimizer: 40-60% (normal for Adam)
- Data transfer: <5% (excellent)

## 📚 Documentation

- **Complete Guide:** `docs/PROFILING_GUIDE.md` (700+ lines)
  - Detailed examples for all tools
  - Troubleshooting common issues
  - Interpretation guidelines
  - Advanced workflows

- **Performance Improvements:** `docs/PERFORMANCE_IMPROVEMENTS.md`
  - 15-60x speedup details
  - Architecture alignment with reference paper
  - Profiling section with examples

## 🎯 When to Use

| Scenario | Tool | Command |
|----------|------|---------|
| General profiling | Timer | In Python code |
| Training analysis | EpochTimer | In Python code |
| DataLoader slow | profile_dataloader() | In Python or CLI script |
| Choose device | compare_device_performance() | CLI script |
| Initial setup | profile_performance.py | CLI script |
| After changes | profile_performance.py | CLI script |

## 🔍 What to Look For

**Good:**
- ✅ DataLoader < 10ms/batch
- ✅ Throughput > 100 batches/s
- ✅ Data transfer < 10% of training time
- ✅ Forward/backward/optimizer breakdown looks normal

**Needs Attention:**
- ⚠️ DataLoader > 100ms/batch
- ⚠️ Throughput < 10 batches/s
- ⚠️ Data transfer > 10% of training time
- ⚠️ Any section > 70% of training time

## 📝 Example Workflow

```bash
# 1. Initial profiling
python scripts/profile_performance.py --model mlp --profile-epochs 3 --compare-devices

# 2. Check results - look for bottlenecks
# If dataloader slow: reduce num_workers, increase batch_size
# If model slow: simplify architecture
# If optimizer slow: consider SGD instead of Adam

# 3. After making changes, re-profile
python scripts/profile_performance.py --model mlp --profile-epochs 3

# 4. Compare before/after
# Verify improvements were effective
```

## 🎉 Summary

You now have **comprehensive profiling tools** to:

1. ✅ Identify performance bottlenecks
2. ✅ Compare device performance (CPU vs MPS)
3. ✅ Optimize dataloader settings
4. ✅ Measure training efficiency
5. ✅ Validate performance improvements

**Current Status:** Training is well-optimized with no major bottlenecks! 🚀

For detailed documentation, see: `docs/PROFILING_GUIDE.md`
