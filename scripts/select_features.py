from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neuro_foundation.pipeline.feature_select import select_features


def main(input_csv: str, threshold: float, output_dir: str):
    df = pd.read_csv(input_csv)
    selected_df = select_features(df, threshold=threshold, output_dir=output_dir)
    print(f"Saved selected features to {output_dir}/selected_features.csv ({selected_df.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variance-based feature selection")
    parser.add_argument("--input-csv", default="data/02_processed/cleaned_data.csv")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--output-dir", default="data/02_processed")
    args = parser.parse_args()
    main(args.input_csv, args.threshold, args.output_dir)
