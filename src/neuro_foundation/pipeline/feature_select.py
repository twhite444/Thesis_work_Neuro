from __future__ import annotations
import os
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def select_features(df: pd.DataFrame, threshold: float = 1.0, output_dir: str = "data/02_processed") -> pd.DataFrame:
    """Select features using variance threshold while preserving CID index.
    
    Args:
        df: DataFrame with CID as index
        threshold: Variance threshold for feature selection
        output_dir: Directory to save selected features
        
    Returns:
        DataFrame with selected features and CID as index
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Preserve the index (CID)
    cid_index = df.index
    
    selector = VarianceThreshold(threshold=threshold)
    selected = selector.fit_transform(df)
    selected_cols = df.columns[selector.get_support()]
    selected_df = pd.DataFrame(selected, columns=selected_cols, index=cid_index)
    selected_df.index.name = 'CID'
    
    selected_df.to_csv(os.path.join(output_dir, 'selected_features.csv'), index=True)

    meta = {
        'threshold': threshold,
        'n_features_in': int(df.shape[1]),
        'n_features_out': int(selected_df.shape[1]),
    }
    pd.Series(meta).to_json(os.path.join(output_dir, 'feature_select_meta.json'))
    return selected_df
