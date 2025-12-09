from __future__ import annotations
import os
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def select_features(df: pd.DataFrame, threshold: float = 1.0, output_dir: str = "data/02_processed") -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    selector = VarianceThreshold(threshold=threshold)
    selected = selector.fit_transform(df)
    selected_cols = df.columns[selector.get_support()]
    selected_df = pd.DataFrame(selected, columns=selected_cols)
    selected_df.to_csv(os.path.join(output_dir, 'selected_features.csv'), index=False)

    meta = {
        'threshold': threshold,
        'n_features_in': int(df.shape[1]),
        'n_features_out': int(selected_df.shape[1]),
    }
    pd.Series(meta).to_json(os.path.join(output_dir, 'feature_select_meta.json'))
    return selected_df
