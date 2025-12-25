from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from olfactory_modeling.pipeline.train_linear import train_linear_regression


def main(input_csv: str, target_column: str, output_dir: str):
    df = pd.read_csv(input_csv)
    metrics = train_linear_regression(df, target_column=target_column, output_dir=output_dir)
    print(f"Linear baseline MSE: {metrics['mse']:.6f} | samples={metrics['n_samples']} features={metrics['n_features']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train linear regression baseline")
    parser.add_argument("--input-csv", default="data/02_processed/cleaned_data.csv")
    parser.add_argument("--target-column", required=False, help="Target column name (if present in CSV)")
    parser.add_argument("--output-dir", default="experiments/baseline_linear")
    args = parser.parse_args()

    if not args.target_column:
        print("No target column provided; training will fail unless the CSV includes a target. Provide --target-column to proceed.")
    main(args.input_csv, args.target_column, args.output_dir)
