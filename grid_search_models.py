import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Lasso, ElasticNet, SGDRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data
    y = pd.read_csv('output_data/pca_transformed_data.csv').iloc[:, 0]
    return train_test_split(X, y, test_size=0.2, random_state=42)

#optional function for if you want to use a column from the features as the target
# def load_data(file_path, target_column):
#     data = pd.read_csv(file_path)
#     X = data.drop(columns=[target_column])
#     y = data[target_column]
#     return train_test_split(X, y, test_size=0.2, random_state=42)

def define_models():
    models = {}

    # Lasso Regression
    models['lasso'] = {
        'model': Lasso(),
        'params': {
            'alpha': [0.001, 0.01, 0.1, 1, 10]
        }
    }

    # ElasticNet
    models['elastic_net'] = {
        'model': ElasticNet(),
        'params': {
            'alpha': [0.001, 0.01, 0.1, 1, 10],
            'l1_ratio': [0.2, 0.5, 0.8]
        }
    }

    # SGD Regressor
    models['sgd'] = {
        'model': SGDRegressor(max_iter=1000, tol=1e-3),
        'params': {
            'alpha': [0.0001, 0.001, 0.01],
            'penalty': ['l2', 'l1', 'elasticnet']
        }
    }

    # Support Vector Regression
    models['svr'] = {
        'model': SVR(),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
    }

    # K-Nearest Neighbors
    models['knn'] = {
        'model': KNeighborsRegressor(),
        'params': {
            'n_neighbors': [3, 5, 7, 10],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    }

    # Voting Regressor
    lasso = Lasso(alpha=0.01)
    sgd = SGDRegressor(max_iter=1000, tol=1e-3)
    knn = KNeighborsRegressor(n_neighbors=5)
    models['voting'] = {
        'model': VotingRegressor([('lasso', lasso), ('sgd', sgd), ('knn', knn)]),
        'params': {
            'voting__lasso__alpha': [0.001, 0.01],
            'voting__sgd__alpha': [0.0001, 0.001],
            'voting__knn__n_neighbors': [3, 5]
        }
    }

    return models

def run_grid_search(models, X_train, y_train):
    results = {}
    for name, spec in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Input scaling for models
            ('model', spec['model'])
        ])
        gs = GridSearchCV(pipeline, spec['params'], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        gs.fit(X_train, y_train)
        results[name] = gs
        print(f"Best score for {name}: {gs.best_score_}, Best params: {gs.best_params_}")
    return results

if __name__ == "__main__":
    file_path = 'output_data/selected_features.csv'  # Path to your CSV file
    X_train, X_test, y_train, y_test = load_data(file_path)
    models = define_models()
    results = run_grid_search(models, X_train, y_train)
