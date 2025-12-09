from __future__ import annotations
import argparse
import pandas as pd
from src.neuro_foundation.pipeline.preprocess import featurize_and_standardize
from src.neuro_foundation.data.pyrfume_loader import PyrfumeLoader


def main(output_dir: str, use_cached: bool):
    if use_cached:
        molecules = pd.read_csv('data/01_raw/molecules_raw.csv')
    else:
        molecules = PyrfumeLoader(output_dir='data/01_raw').load_molecules()
    df = featurize_and_standardize(molecules, output_dir=output_dir)
    print(f"Saved cleaned data to {output_dir}/cleaned_data.csv ({df.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Featurize SMILES and standardize")
    parser.add_argument("--output-dir", default="data/02_processed", help="Directory to save cleaned data")
    parser.add_argument("--use-cached", action="store_true", help="Use cached raw molecules from data/01_raw")
    args = parser.parse_args()
    main(args.output_dir, args.use_cached)
