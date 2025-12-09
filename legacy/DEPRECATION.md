# Legacy Components Deprecated

The legacy brain processing flow and monolithic scripts are deprecated in favor of the staged pipeline.

## Do Not Use
- `scripts/process_brain_maps.py`
- `src/neuro_smell/stages/brain_activity.py`

## Use Instead
- Pipeline runner: `scripts/run_brain_pca_pipeline.py`
- PCA stage: `src/neuro_smell/stages/brain_targets.py`
- Dataset assembly: `src/neuro_smell/datasets/assembler.py`
- Alignment: `src/neuro_smell/alignment/aligner.py`

## Validation First
Before any training, validate your canonical dataset:

```zsh
python scripts/validate_dataset.py data/02_processed/features_and_targets.csv
```

This enforces:
- Presence of `CID`
- Target columns (PC*) exist and have no NaNs
- No duplicate CIDs
- No zero-variance targets

If validation fails, fix upstream generation and re-run the pipeline.
