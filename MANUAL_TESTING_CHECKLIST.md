# Manual Testing Required

## ⚠️ **Important: Manual Testing Checklist**

While we have 81% automated test coverage with 60 passing tests, some features require manual verification because automated tests cannot verify visual appearance or interactive behavior.

---

## 🎨 **Visualization Testing**

### 1. Interactive Activity Map Inspection

**Command:**
```bash
python scripts/exploration/inspect_activity_map.py --cid 180 --show-images
```

**What to manually verify:**
- [ ] Matplotlib window opens without errors
- [ ] Window displays 4 subplots:
  - [ ] Individual activity maps (should show 3 maps for CID 180)
  - [ ] Averaged activity map
  - [ ] Coverage map
  - [ ] Zero-filtered average
- [ ] Color scales are appropriate (blues/viridis)
- [ ] Zero values display as white/transparent (not colored)
- [ ] Titles and labels are correct
- [ ] Window can be closed cleanly

**Additional tests:**
```bash
# Test with different CIDs
python scripts/exploration/inspect_activity_map.py --cid 240 --show-images
python scripts/exploration/inspect_activity_map.py --cid 58 --show-images

# Test list-all functionality
python scripts/exploration/inspect_activity_map.py --list-all

# Test save-images functionality
python scripts/exploration/inspect_activity_map.py --cid 180 --save-images
# Check that activity_map_180_inspection.png is created and not blank
```

---

### 2. Pipeline Visualization Output

**Command:**
```bash
python scripts/run_activity_maps.py --verbose
```

**What to manually verify:**
- [ ] Script completes without errors
- [ ] Following PNG files created in `data/02_processed/`:
  - [ ] `global_mask.png` - Shows refined brain mask
  - [ ] `coverage_counts.png` - Heatmap of pixel coverage across maps
  - [ ] `coverage_histogram.png` - Distribution of coverage counts
  - [ ] `masked_averaged_example.png` - Example of one masked/averaged map
  - [ ] `masked_averaged_gallery.png` - Gallery of 6 maps in grid

**Visual verification for each image:**
- [ ] Image is not blank
- [ ] Image has correct dimensions
- [ ] Zero values display as white/transparent
- [ ] Color scale is appropriate
- [ ] Title/labels are correct

**Test with different thresholds:**
```bash
# Low threshold (more brain regions)
python scripts/run_activity_maps.py --coverage-threshold 0.3

# High threshold (fewer brain regions)
python scripts/run_activity_maps.py --coverage-threshold 0.7
```

---

## 📜 **Example Scripts Testing**

### 1. CID-based Loading Example

**Command:**
```bash
python scripts/examples/example_load_by_cid.py
```

**What to manually verify:**
- [ ] Script runs without errors
- [ ] Output shows:
  - [ ] Number of maps for CID 180
  - [ ] Average map coverage percentage
  - [ ] Examples of single vs multiple maps
  - [ ] Batch loading of multiple CIDs
  - [ ] Top CIDs with most maps
- [ ] All output values are reasonable

---

### 2. Cached Data Loading Example

**Command:**
```bash
python scripts/examples/example_load_cached.py
```

**What to manually verify:**
- [ ] Script runs without errors
- [ ] Output shows:
  - [ ] CSV loading times
  - [ ] NPZ loading times
  - [ ] Speedup comparison (should show 1.3-1.6x)
  - [ ] Data shape information
  - [ ] Examples of each data type
- [ ] NPZ loading is faster than CSV
- [ ] All data types load successfully

---

### 3. Stimuli Metadata Example

**Command:**
```bash
python scripts/examples/example_stimuli_metadata.py
```

**What to manually verify:**
- [ ] Script runs without errors
- [ ] Output shows:
  - [ ] Total number of stimuli (432)
  - [ ] Example stimuli records
  - [ ] Search by molecule name
  - [ ] Filter by CID
  - [ ] Link to activity maps
- [ ] All queries return sensible results

---

## 🔄 **Complete Pipeline Testing**

### Run Full Pipeline End-to-End

**Commands (in order):**
```bash
# Step 1: Download data (if not already cached)
python scripts/load_all_data.py

# Step 2: Preprocess and extract features
python scripts/preprocess.py

# Step 3: Process activity maps
python scripts/run_activity_maps.py

# Step 4: Select features
python scripts/select_features.py

# Step 5: Train model
python scripts/train_linear.py
```

**What to manually verify:**

**After Step 1 (load_all_data.py):**
- [ ] `data/01_raw/` directory exists
- [ ] Files created:
  - [ ] `molecules_raw.csv` and `molecules_raw.npz`
  - [ ] `behavior_data.csv` and `behavior_data.npz`
  - [ ] `stimuli_metadata.csv` and `stimuli_metadata.npz`
  - [ ] `activity_maps.npz`
  - [ ] `activity_maps_csv/` directory with 405 CSV files

**After Step 2 (preprocess.py):**
- [ ] `data/02_processed/` directory exists
- [ ] Files created:
  - [ ] `cleaned_data.csv` (287 rows, ~1187 columns)
  - [ ] `scaler_stats.json`
- [ ] No errors during Mordred feature calculation

**After Step 3 (run_activity_maps.py):**
- [ ] Visualization PNG files created (see section 2 above)
- [ ] All images render correctly

**After Step 4 (select_features.py):**
- [ ] Files created:
  - [ ] `selected_features.csv` (reduced number of columns)
  - [ ] `feature_select_meta.json`
- [ ] Number of features reduced (variance threshold applied)

**After Step 5 (train_linear.py):**
- [ ] Files created:
  - [ ] `model_coefficients.csv`
  - [ ] `predictions.csv`
- [ ] No errors during model training

---

## 🐛 **Known Issues to Watch For**

### Issue 1: Matplotlib Backend
**Symptom:** Interactive plots don't display or throw backend errors

**Fix:**
```bash
# Try different backend
export MPLBACKEND=TkAgg
python scripts/exploration/inspect_activity_map.py --cid 180 --show-images
```

---

### Issue 2: Missing Data Files
**Symptom:** FileNotFoundError when running scripts

**Fix:**
```bash
# Ensure data is downloaded
python scripts/load_all_data.py

# Or use cached data option
python scripts/preprocess.py  # Will use cached NPZ files if available
```

---

### Issue 3: Blank PNG Files
**Symptom:** PNG files are created but appear blank

**Possible causes:**
- Zero values throughout the entire map
- Mask filtering out all pixels
- Incorrect color scale

**Debug:**
```bash
# Check with verbose output
python scripts/run_activity_maps.py --verbose

# Try lower coverage threshold
python scripts/run_activity_maps.py --coverage-threshold 0.3
```

---

## ✅ **Manual Testing Completion Checklist**

### Basic Tests (Required - 15 minutes)
- [ ] Interactive visualization opens (`inspect_activity_map.py --show-images`)
- [ ] Pipeline visualizations created (`run_activity_maps.py`)
- [ ] All 5 PNG files are not blank
- [ ] At least one example script runs successfully

### Comprehensive Tests (Recommended - 30 minutes)
- [ ] All 3 example scripts run successfully
- [ ] Multiple CIDs tested with interactive visualization
- [ ] Different coverage thresholds tested
- [ ] Saved images verified manually
- [ ] Complete pipeline runs successfully (all 5 steps)

### Thorough Tests (Optional - 1 hour)
- [ ] All visualization files inspected visually
- [ ] Edge cases tested (CID with 1 map, CID with 11 maps)
- [ ] Different matplotlib backends tested
- [ ] All output files have sensible values
- [ ] No console errors or warnings (except expected Mordred warning)

---

## 📝 **Reporting Issues**

If you find issues during manual testing:

1. **Note the exact command that failed**
2. **Copy the full error message**
3. **Note your environment:**
   - Python version: `python --version`
   - OS: macOS/Linux/Windows
   - Matplotlib backend: `python -c "import matplotlib; print(matplotlib.get_backend())"`
4. **Check if issue is reproducible**
5. **Document workaround if found**

---

## 🎯 **Expected Results Summary**

### Visualizations
- ✅ All matplotlib windows open without errors
- ✅ All subplots render with appropriate content
- ✅ Zero values display as white/transparent
- ✅ Color scales are meaningful (not all one color)
- ✅ Titles and labels are correct

### Example Scripts
- ✅ All 3 scripts run without errors
- ✅ Output values are reasonable
- ✅ Examples demonstrate correct API usage

### Complete Pipeline
- ✅ All 5 steps complete successfully
- ✅ All expected output files created
- ✅ File contents are reasonable
- ✅ No unexpected errors or warnings

---

**Manual Testing Status:** ⚠️ **PENDING**

Please complete at least the **Basic Tests** (15 minutes) to verify interactive features work correctly.

Once manual testing is complete, the project will be 100% verified and production-ready!

---

**Automated Testing:** ✅ **COMPLETE** (60/60 tests passing, 81% coverage)  
**Manual Testing:** ⚠️ **REQUIRED** (See checklist above)  

**Last Updated:** December 10, 2025
