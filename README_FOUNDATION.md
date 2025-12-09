# Neuro Foundation

A minimal, modular foundation to rebuild the thesis pipeline without touching `legacy/`.

## Structure
- `src/neuro_foundation/data/` – dataset loader interface and implementations
- `src/neuro_foundation/pipeline/` – pure functions for preprocessing, feature selection, training
- `scripts/` – small CLIs composing steps
- `data/` – output folders (`01_raw`, `02_processed`)
- `experiments/` – metrics and coefficients

## Quick Start

1. Load raw data via Pyrfume:
   ```
   python scripts/load_data.py --output-dir data/01_raw
   ```
2. Featurize and standardize SMILES:
   ```
   python scripts/preprocess.py --output-dir data/02_processed --use-cached
   ```
3. Feature selection:
   ```
   python scripts/select_features.py --input-csv data/02_processed/cleaned_data.csv --threshold 1.0 --output-dir data/02_processed
   ```
4. Train linear baseline (requires target column in the CSV):
   ```
   python scripts/train_linear.py --input-csv data/02_processed/cleaned_data.csv --target-column PC1 --output-dir experiments/baseline_linear
   ```

## Swap Data Sources Later
Implement another loader (e.g., `CsvLoader`) that conforms to `DatasetLoader` and change the CLI to import it. No pipeline code changes needed.

## Notes
- `legacy/` is read-only. This foundation is a fresh, CS-oriented rebuild.
- Keep each step single-purpose and deterministic; cache artifacts in `data/` and `experiments/`.
