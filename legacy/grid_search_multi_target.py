import os
import numpy as np
import pandas as pd
import logging
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Lasso, ElasticNet, SGDRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data(file_path, n_components=2):
    """Load dataset and split into train-test sets."""
    # Load the feature dataset (X)
    data = pd.read_csv(file_path)
    
    # Load the PCA-transformed data (y) and select the first n_components
    y = pd.read_csv('output_data/pca_transformed_data.csv', index_col=0).iloc[:, :n_components]  # Select first n_components

    # Ensure alignment by CID
    if 'CID' in data.columns:
        data.set_index('CID', inplace=True)
    if y.index.name != 'CID':
        raise ValueError("PCA-transformed data must have 'CID' as the index.")

    # Align X and y by their shared CID index
    common_cids = data.index.intersection(y.index)
    if len(common_cids) == 0:
        raise ValueError("No common CIDs found between X and y.")
    X_aligned = data.loc[common_cids]
    y_aligned = y.loc[common_cids]

    # Debugging: Check alignment
    logging.info(f"Aligned X shape: {X_aligned.shape}")
    logging.info(f"Aligned y shape: {y_aligned.shape}")
    assert (X_aligned.index == y_aligned.index).all(), "X and y are not aligned by CID!"

    # Split into train-test sets
    return train_test_split(X_aligned, y_aligned, test_size=0.2, random_state=42)

def define_models():
    """Define models and their respective hyperparameters for grid search."""
    model_configs = {
        'lasso': (
            MultiOutputRegressor(Lasso(max_iter=5000, tol=1e-5)),  # Wrap in MultiOutputRegressor
            {'estimator__alpha': [0.1, 1, 10]}
        ),
        'elastic_net': (
            MultiOutputRegressor(ElasticNet(max_iter=5000, tol=1e-5)),  # Wrap in MultiOutputRegressor
            {'estimator__alpha': [0.1, 1, 10], 'estimator__l1_ratio': [0.2, 0.5, 0.8]}
        ),
        'sgd': (
            MultiOutputRegressor(SGDRegressor(max_iter=10000, tol=1e-5)),  # Wrap in MultiOutputRegressor
            {'estimator__alpha': [0.01, 0.1], 'estimator__penalty': ['l2', 'l1', 'elasticnet']}
        ),
        'svr': (
            MultiOutputRegressor(SVR()),  # Wrap in MultiOutputRegressor
            {'estimator__C': [0.1, 1, 10], 'estimator__kernel': ['linear', 'rbf']}
        ),
        'knn': (
            MultiOutputRegressor(KNeighborsRegressor()),  # Wrap in MultiOutputRegressor
            {'estimator__n_neighbors': [3, 5, 7, 10], 'estimator__weights': ['uniform', 'distance'], 'estimator__metric': ['euclidean', 'manhattan']}
        ),
    }

    models = {
        name: {'model': model, 'params': {f"model__{key}": values for key, values in params.items()}}
        for name, (model, params) in model_configs.items()
    }

    return models

def run_grid_search(models, X_train, y_train, scoring_metric='neg_mean_squared_error'):
    """Perform hyperparameter tuning using GridSearchCV."""
    results = {}
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)  # Ensure the directory exists

    for name, spec in tqdm(models.items(), desc="Grid Search Progress"):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Normalize features before training
            ('model', spec['model'])
        ])

        gs = GridSearchCV(
            pipeline,
            spec['params'],
            cv=5,
            scoring=scoring_metric,
            n_jobs=-1,
            verbose=0  # Disable verbose output
        )

        try:
            gs.fit(X_train, y_train)
            results[name] = gs
            logging.info(f"Best score for {name}: {gs.best_score_}, Best params: {gs.best_params_}")

            # Save the best model
            joblib.dump(gs.best_estimator_, os.path.join(model_dir, f"{name}_best_model.pkl"))
        except Exception as e:
            logging.error(f"Grid search failed for {name}: {e}")

    return results

if __name__ == "__main__":
    file_path = 'output_data/selected_features.csv'  # Path to dataset
    n_components = 3  # Number of PCA components to use as target variables

    # Load dataset
    X_train, X_test, y_train, y_test = load_data(file_path, n_components=n_components)

    # Define models
    models = define_models()

    # Run grid search
    results = run_grid_search(models, X_train, y_train)

    # Evaluate model performance
    best_model = joblib.load("models/lasso_best_model.pkl")  # Example: Load the best model for lasso
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")
    