# Scripts Consolidation Summary

**Date:** December 9, 2025

## Overview
Cleaned up scripts directory by removing duplicate and legacy-specific scripts, keeping only essential scripts for daily use.

## Changes

**Before:** 6 scripts (41.2K total)
**After:** 3 scripts (15.8K total)
**Reduction:** 50% fewer scripts, 62% less code

## Final Structure

```
scripts/
├── train.py              # Main training script (2.0K)
├── cleanup.py            # Utility script (8.7K)
└── download_pyrfume_data.py  # Setup script (5.1K)
```

## Removed Scripts

### 1. run_pipeline.py (5.4K)
**Reason:** Duplicate functionality - everything covered by train.py

### 2. run_legacy_pipeline.py (12K)
**Reason:** Legacy validation complete - no longer needed

### 3. process_brain_maps.py (7.9K)
**Reason:** Brain activity configs removed - not in standard workflow

## Simple Workflow

```bash
# 1. Setup (once)
python scripts/download_pyrfume_data.py

# 2. Train (daily)
python scripts/train.py
python scripts/train.py experiment=my_test

# 3. Clean (as needed)
python scripts/cleanup.py --cache all
```

## Benefits

✅ **50% fewer scripts** - Less confusion
✅ **Clear purpose** - Each script has one job
✅ **Easier maintenance** - Fewer files to manage
✅ **Simple workflow** - 3 steps instead of 6 options

## Summary

Clean, focused, and easy to understand. No functionality lost - everything available through train.py with config overrides.
