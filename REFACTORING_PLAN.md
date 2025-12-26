# Train NN Refactoring Execution Plan

**Mission**: Refactor `train_nn.py` (1199 lines) into modular components with **ZERO BEHAVIOR CHANGE**.

## Safety First Principles

1. ✅ **Comprehensive test suite created** (`test_train_nn_refactor_safety.py`)
2. ✅ **Baseline snapshot captured** (`tests/snapshots/train_nn_baseline.json`)
3. ✅ **All 9 safety tests passing** before starting refactoring
4. 🎯 **After each extraction**: Run tests + snapshot comparison
5. 🎯 **No behavior change**: Exact same outputs, JSON keys, metrics, file names

## Current Status: Phase 0 Complete ✅

- [x] Created comprehensive test suite with 9 tests
- [x] Established baseline snapshot of current behavior
- [x] All tests passing (100% green)
- [x] Reproducibility verified with fixed seeds
- [x] Registered snapshot marker in pytest.ini

## Refactoring Phases

### Phase 1: Extract Metrics Module (LOW RISK)
**Target**: `src/olfactory_modeling/training/metrics.py`  
**Lines to extract**: ~60 lines from `train_nn.py`

**What to extract**:
- `compute_metrics()` function (currently at line ~330)
- Any metric-related utilities

**Success criteria**:
- [  ] `metrics.py` created with `compute_metrics()`
- [  ] Import updated in `train_nn.py`: `from olfactory_modeling.training.metrics import compute_metrics`
- [  ] All 9 safety tests pass
- [  ] Snapshot comparison passes (identical outputs)
- [  ] `train_nn.py` reduced by ~60 lines

**Migration steps**:
1. Create `src/olfactory_modeling/training/__init__.py`
2. Create `src/olfactory_modeling/training/metrics.py`
3. Copy `compute_metrics()` to `metrics.py` (exact copy)
4. Add imports to `metrics.py`
5. Update import in `train_nn.py`
6. Run tests: `pytest tests/test_train_nn_refactor_safety.py -v`
7. Run snapshot: `pytest tests/test_train_nn_refactor_safety.py::TestSnapshotComparison::test_compare_against_baseline_snapshot -v`
8. Commit if all pass

---

### Phase 2: Extract IO Utilities (LOW RISK)
**Target**: `src/olfactory_modeling/training/io_utils.py`  
**Lines to extract**: ~40 lines

**What to extract**:
- `save_checkpoint()` function
- `save_json_safe()` function
- `generate_visualization_safe()` function

**Success criteria**:
- [  ] `io_utils.py` created
- [  ] Functions imported in `train_nn.py`
- [  ] All tests pass
- [  ] Snapshot comparison passes
- [  ] `train_nn.py` reduced by ~40 lines

---

### Phase 3: Extract Validation Logic (MEDIUM RISK)
**Target**: `src/olfactory_modeling/training/validation.py`  
**Lines to extract**: ~20 lines

**What to extract**:
- `validate_training_params()` function

**Success criteria**:
- [  ] `validation.py` created
- [  ] Function imported in `train_nn.py`
- [  ] All tests pass (especially `test_validate_training_params_raises_correctly`)
- [  ] Snapshot comparison passes

---

### Phase 4: Extract Epoch Runners (MEDIUM RISK)
**Target**: `src/olfactory_modeling/training/epoch_runners.py`  
**Lines to extract**: ~150 lines

**What to extract**:
- `train_epoch()` function (~80 lines)
- `validate_epoch()` function (~70 lines)

**Success criteria**:
- [  ] `epoch_runners.py` created
- [  ] Functions imported in `train_nn.py`
- [  ] All tests pass
- [  ] Snapshot comparison passes (critical - training loop behavior must be identical)
- [  ] `train_nn.py` reduced by ~150 lines

---

### Phase 5: Extract Cross-Validation (MEDIUM-HIGH RISK)
**Target**: `src/olfactory_modeling/evaluation/cross_validation.py`  
**Lines to extract**: ~100 lines

**What to extract**:
- `aggregate_cv_metrics()` function
- Helper functions for K-fold aggregation

**Success criteria**:
- [  ] `src/olfactory_modeling/evaluation/__init__.py` created
- [  ] `cross_validation.py` created
- [  ] Functions imported in `train_nn.py`
- [  ] All tests pass
- [  ] Snapshot comparison passes

---

### Phase 6: Create Trainer Class (HIGH RISK - COMPOSITION PREFERRED)
**Target**: `src/olfactory_modeling/training/trainers.py`  
**Lines to extract**: Refactor `train_nn()` into class

**What to create**:
```python
class Trainer:
    """Encapsulates training logic using composition, not deep inheritance."""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Composition: inject dependencies
        self.metrics_calculator = MetricsCalculator()
        self.checkpoint_saver = CheckpointSaver(config.output_dir)
        self.epoch_runner = EpochRunner(self.metrics_calculator)
    
    def train(self):
        """Main training loop - delegates to helpers."""
        for epoch in range(self.config.num_epochs):
            train_metrics = self.epoch_runner.train_epoch(
                self.model, self.train_loader, self.optimizer, ...
            )
            val_metrics = self.epoch_runner.validate_epoch(
                self.model, self.val_loader, ...
            )
            self.checkpoint_saver.save_if_best(self.model, val_metrics)
        return self.aggregate_results()
```

**Success criteria**:
- [  ] `trainers.py` created with `Trainer` class
- [  ] `train_nn()` wrapper function delegates to `Trainer.train()`
- [  ] **CRITICAL**: All tests pass
- [  ] **CRITICAL**: Snapshot comparison passes
- [  ] Public API unchanged (`train_nn()` still works exactly the same)

---

### Phase 7: Extract Grid Search (MEDIUM RISK)
**Target**: `src/olfactory_modeling/evaluation/hyperparameter_search.py`  
**Lines to extract**: ~400 lines

**What to extract**:
- `grid_search()` function

**Success criteria**:
- [  ] `hyperparameter_search.py` created
- [  ] `grid_search()` imported in `train_nn.py`
- [  ] All tests pass
- [  ] Snapshot comparison passes

---

### Phase 8: Final Cleanup & Verification
**Target**: Ensure `train_nn.py` is ~150 lines

**Tasks**:
- [  ] Remove all extracted code from `train_nn.py`
- [  ] Keep only thin wrapper functions (`train_nn()`, `train_nn_kfold()`, `grid_search()`)
- [  ] Verify all imports are clean
- [  ] Run full test suite
- [  ] Run snapshot comparison
- [  ] Verify line count: `wc -l src/olfactory_modeling/pipeline/train_nn.py` ≈ 150 lines

---

## Testing Protocol (After Each Phase)

```bash
# 1. Run unit tests
pytest tests/test_train_nn_refactor_safety.py::TestTrainNNBehaviorPreservation -v

# 2. Run snapshot comparison (CRITICAL)
pytest tests/test_train_nn_refactor_safety.py::TestSnapshotComparison::test_compare_against_baseline_snapshot -v

# 3. Run reproducibility test
pytest tests/test_train_nn_refactor_safety.py::TestTrainNNReproducibility -v

# 4. If all pass, commit
git add -A
git commit -m "refactor(train_nn): Extract [MODULE_NAME] - no behavior change"
```

## Risk Levels & Mitigation

**LOW RISK** (Phases 1-2):
- Pure utility functions
- No state, no side effects
- Easy to test in isolation

**MEDIUM RISK** (Phases 3-5):
- Functions with some coupling
- May have hidden dependencies
- Careful import management needed

**HIGH RISK** (Phase 6):
- Large refactoring of main training loop
- **Mitigation**: Use composition over inheritance
- **Mitigation**: Keep `train_nn()` wrapper unchanged
- **Mitigation**: Test extensively after this phase

## Success Metrics

### Code Quality
- [  ] `train_nn.py` reduced from 1199 to ~150 lines (88% reduction)
- [  ] Zero code duplication
- [  ] Each module has single responsibility
- [  ] Clear separation of concerns

### Behavior Preservation
- [  ] All 9 safety tests pass
- [  ] Snapshot comparison passes (identical outputs)
- [  ] Reproducibility maintained (fixed seed = fixed results)
- [  ] All existing scripts/notebooks still work

### Future Extensibility
- [  ] Can add GNN trainer in ~60 lines by inheriting/composing from `Trainer`
- [  ] New model types trivial to add
- [  ] Easy to test individual components

## Timeline

**Conservative Estimate**: 3-4 days
- Day 1: Phases 1-3 (LOW + MEDIUM RISK)
- Day 2: Phases 4-5 (MEDIUM + MEDIUM-HIGH RISK)
- Day 3: Phase 6 (HIGH RISK - go slow)
- Day 4: Phases 7-8 (Cleanup & verification)

**Aggressive Estimate**: 1-2 days (if user is comfortable with faster pace)

---

## Current File Structure (Target)

```
src/olfactory_modeling/
├── training/
│   ├── __init__.py
│   ├── metrics.py           # Phase 1 ← compute_metrics()
│   ├── io_utils.py          # Phase 2 ← save_checkpoint(), save_json_safe(), etc.
│   ├── validation.py        # Phase 3 ← validate_training_params()
│   ├── epoch_runners.py     # Phase 4 ← train_epoch(), validate_epoch()
│   └── trainers.py          # Phase 6 ← Trainer class (composition-based)
├── evaluation/
│   ├── __init__.py
│   ├── cross_validation.py  # Phase 5 ← aggregate_cv_metrics()
│   └── hyperparameter_search.py  # Phase 7 ← grid_search()
└── pipeline/
    └── train_nn.py          # Phase 8 ← Thin wrappers only (~150 lines)
```

---

## Next Step: Start Phase 1

Ready to begin? We'll start with the safest extraction: **metrics.py** (LOW RISK).

**Command to run**:
```bash
# Create the structure
mkdir -p src/olfactory_modeling/training
```

Then we'll extract `compute_metrics()` and run all tests to verify no behavior change.
