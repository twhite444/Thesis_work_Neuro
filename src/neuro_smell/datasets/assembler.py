"""
Dataset Assembler - Merge features with brain PCA scores.
"""
from pathlib import Path
import pandas as pd


def merge_features_and_targets(features_csv: str, scores_csv: str, out_csv: str):
    f = pd.read_csv(features_csv)
    s = pd.read_csv(scores_csv)
    merged = f.merge(s, on="CID", how="inner")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return merged
