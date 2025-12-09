"""
Feature Preprocessor - VarianceThreshold + StandardScaler with state persistence.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any
from sklearn.preprocessing import StandardScaler


@dataclass
class FeaturePreprocState:
    scaler: StandardScaler
    kept_columns: List[str]


def fit(X_df: pd.DataFrame, threshold: float = 1.0) -> FeaturePreprocState:
    # drop zero-variance columns by threshold
    variances = X_df.var(axis=0)
    kept_cols = [c for c in X_df.columns if variances[c] >= threshold]
    X_kept = X_df[kept_cols]
    scaler = StandardScaler().fit(X_kept.values.astype(np.float32))
    return FeaturePreprocState(scaler=scaler, kept_columns=kept_cols)


def transform(X_df: pd.DataFrame, state: FeaturePreprocState) -> np.ndarray:
    X_kept = X_df[state.kept_columns].values.astype(np.float32)
    return state.scaler.transform(X_kept)


from typing import Tuple

def fit_transform(X_df: pd.DataFrame, threshold: float = 1.0) -> Tuple[np.ndarray, FeaturePreprocState]:
    st = fit(X_df, threshold)
    X = transform(X_df, st)
    return X, st
