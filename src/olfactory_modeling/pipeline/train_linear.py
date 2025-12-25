from __future__ import annotations
import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def train_linear_regression(df: pd.DataFrame, target_column: str, output_dir: str = "experiments/baseline_linear") -> dict:
    os.makedirs(output_dir, exist_ok=True)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)

    model = LinearRegression()
    model.fit(X_imp, y)

    y_pred = model.predict(X_imp)
    mse = float(mean_squared_error(y, y_pred))

    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_,
    })
    coef_df.to_csv(os.path.join(output_dir, 'model_coefficients.csv'), index=False)

    metrics = {
        'mse': mse,
        'n_features': int(X.shape[1]),
        'n_samples': int(X.shape[0]),
        'target': target_column,
    }
    pd.Series(metrics).to_json(os.path.join(output_dir, 'metrics.json'))
    return metrics
