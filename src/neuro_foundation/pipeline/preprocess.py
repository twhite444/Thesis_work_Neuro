from __future__ import annotations
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pyrfume.features import smiles_to_mordred


def featurize_and_standardize(molecules: pd.DataFrame, output_dir: str = "data/02_processed") -> pd.DataFrame:
    """Featurize SMILES to Mordred, drop NaNs and zero-only columns, and standardize.

    Saves cleaned_data.csv with CID as index and returns the standardized DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)

    if 'IsomericSMILES' not in molecules.columns:
        raise ValueError("Expected 'IsomericSMILES' column in molecules")
    if 'CID' not in molecules.columns:
        raise ValueError("Expected 'CID' column in molecules")

    smiles = molecules['IsomericSMILES'].tolist()
    cids = molecules['CID'].values
    
    mordred_features = smiles_to_mordred(smiles)

    # Drop columns with any NaN
    filtered = mordred_features.dropna(axis=1, how='any')
    # Drop columns that are entirely zero
    zero_only = filtered.eq(0).all(axis=0)
    filtered = filtered.loc[:, ~zero_only]

    scaler = StandardScaler()
    standardized = scaler.fit_transform(filtered)
    standardized_df = pd.DataFrame(standardized, columns=filtered.columns, index=cids)
    standardized_df.index.name = 'CID'

    cleaned_path = os.path.join(output_dir, 'cleaned_data.csv')
    standardized_df.to_csv(cleaned_path, index=True)

    # Persist scaler stats for reproducibility
    stats = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'features': filtered.columns.tolist(),
    }
    pd.Series(stats).to_json(os.path.join(output_dir, 'scaler_stats.json'))

    return standardized_df
