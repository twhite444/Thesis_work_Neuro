"""
Molecule Loader - Read descriptor or SMILES datasets with validation.

Contract:
- Input: CSV/Parquet with at least `CID` and numeric descriptor columns, optional `SMILES`.
- Output: pandas DataFrame with columns [CID, f1..fn] and optional `SMILES` (not used by tabular).
"""
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional


def load_molecules(path: str, smiles_col: Optional[str] = None, cid_col: str = "CID") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Molecule file not found: {path}")
    df = pd.read_csv(p)
    if cid_col not in df.columns:
        raise ValueError(f"CID column '{cid_col}' missing in {path}")
    # Basic validations
    if df[cid_col].isnull().any():
        raise ValueError("CID contains NaN values")
    if df[cid_col].duplicated().any():
        raise ValueError("Duplicate CIDs found; ensure uniqueness")
    # Replace infs
    num_cols = df.select_dtypes(include=[np.number]).columns
    if np.isinf(df[num_cols]).any().any():
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
        df.dropna(subset=num_cols, inplace=True)
    return df
