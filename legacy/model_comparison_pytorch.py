import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

# Define a base neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[128, 64], dropout_rate=0.5):
        super(NeuralNetwork, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def load_data(file_path, n_components=5):
    """Load dataset and split into train-test sets."""
    # Load the feature dataset (X)
    data = pd.read_csv(file_path)
    # Print the shape of the data
    print(f"Shape of the feature dataset(X): {data.shape}")
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

    # Split into train-test sets
    return train_test_split(X_aligned, y_aligned, test_size=0.2, random_state=42)

def train_model(model, criterion, optimizer, train_loader, val_loader, max_epochs=50, scheduler=None, patience=7):
    """Train the model and log training/validation losses with early stopping."""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Step the scheduler if provided
        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return train_losses, val_losses

def plot_losses(train_losses, val_losses, model_name):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training vs Validation Loss ({model_name})')
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_models(models, X_train, y_train, X_test, y_test, max_epochs=50):
    """Evaluate multiple models and compare their performance."""
    results = []
    for model_name, model_config in models.items():
        print(f"Training model: {model_name}")
        # Initialize the model
        model = NeuralNetwork(
            input_size=X_train.shape[1],
            output_size=y_train.shape[1],
            hidden_sizes=model_config['hidden_sizes'],
            dropout_rate=model_config['dropout_rate']
        )
        criterion = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=model_config['learning_rate'], weight_decay=1e-4)

        # Initialize scheduler if specified
        scheduler = None
        if 'scheduler' in model_config:
            scheduler_config = model_config['scheduler']
            if scheduler_config['type'] == 'ReduceLROnPlateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, factor=scheduler_config['factor'], patience=scheduler_config['patience']
                )

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

        # Create DataLoader for batching
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Train the model
        train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader, max_epochs=max_epochs, scheduler=scheduler)

        # Plot the losses
        plot_losses(train_losses, val_losses, model_name)

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor).detach().numpy()
            y_true = y_test.values
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)

        # Scatter plot of predicted vs actual values
        plt.scatter(y_test.values, y_pred, alpha=0.7)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Predicted vs Actual')
        plt.show()

        # Save the final validation loss and metrics
        results.append((model_name, train_losses[-1], val_losses[-1], r2, mae))

    # Print the results
    print("\nModel Comparison Results:")
    for model_name, train_loss, val_loss, r2, mae in results:
        print(f"{model_name}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, R² = {r2:.4f}, MAE = {mae:.4f}")

    # Identify the best model
    best_model = min(results, key=lambda x: x[2])  # Sort by validation loss
    print(f"\nBest Model: {best_model[0]} with Final Validation Loss = {best_model[2]:.4f}")

def evaluate_models_kfold(models, X, y, k=5, max_epochs=50, show_graphs=True, analyze_features=False, feature_names=None):
    """Evaluate multiple models using k-fold cross-validation with options to show graphs and analyze features."""
    results = []
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for model_name, model_config in models.items():
        print(f"Evaluating model: {model_name} with {k}-fold cross-validation")
        fold_results = []
        all_train_losses = []
        all_val_losses = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"Fold {fold + 1}/{k}")
            
            # Split data into train and validation sets for this fold
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Initialize the model
            model = NeuralNetwork(
                input_size=X_train.shape[1],
                output_size=y_train.shape[1],
                hidden_sizes=model_config["hidden_sizes"],
                dropout_rate=model_config["dropout_rate"]
            )
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=model_config["learning_rate"], weight_decay=model_config.get("weight_decay", 0))

            # Initialize scheduler if specified
            scheduler = None
            if "scheduler" in model_config:
                scheduler_config = model_config["scheduler"]
                if scheduler_config["type"] == "ReduceLROnPlateau":
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, factor=scheduler_config["factor"], patience=scheduler_config["patience"]
                    )

            # Convert data to PyTorch tensors
            X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
            X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

            # Create DataLoader for batching
            train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

            # Train the model
            train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader, max_epochs=max_epochs, scheduler=scheduler)

            # Save losses for plotting
            all_train_losses.append(train_losses)
            all_val_losses.append(val_losses)

            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                y_pred = model(X_val_tensor).detach().numpy()
                y_true = y_val.values
                r2 = r2_score(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)

            # Save fold results
            fold_results.append((val_losses[-1], r2, mae))

            # Plot predicted vs actual for this fold if show_graphs is True
            if show_graphs:
                plt.figure(figsize=(8, 6))
                plt.scatter(y_true, y_pred, alpha=0.7)
                plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color="red", linestyle="--", label="Ideal Line (y=x)")
                plt.xlabel("Actual Values")
                plt.ylabel("Predicted Values")
                plt.title(f"Predicted vs Actual (Fold {fold + 1}) - {model_name}")
                plt.legend()
                plt.grid(True)
                plt.show()

        # Aggregate results across folds
        avg_val_loss = np.mean([result[0] for result in fold_results])
        avg_r2 = np.mean([result[1] for result in fold_results])
        avg_mae = np.mean([result[2] for result in fold_results])

        results.append((model_name, avg_val_loss, avg_r2, avg_mae))

        # Plot training vs validation losses across all folds if show_graphs is True
        if show_graphs:
            plt.figure(figsize=(10, 6))
            for fold_idx, (train_loss, val_loss) in enumerate(zip(all_train_losses, all_val_losses)):
                plt.plot(range(1, len(train_loss) + 1), train_loss, label=f"Fold {fold_idx + 1} - Train Loss", linestyle="--", alpha=0.7)
                plt.plot(range(1, len(val_loss) + 1), val_loss, label=f"Fold {fold_idx + 1} - Val Loss", alpha=0.7)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Training vs Validation Loss ({model_name})")
            plt.legend()
            plt.grid(True)
            plt.show()

        # Analyze feature importance if enabled
        if analyze_features and feature_names is not None:
            print(f"\nAnalyzing feature importance for model: {model_name}")
            feature_importance = analyze_feature_importance(model, feature_names)


            # Visualize feature importance
            features, scores = zip(*feature_importance)
            plt.figure(figsize=(10, 6))
            plt.barh(features[:20], scores[:20], color="skyblue")  # Plot top 20 features
            plt.xlabel("Importance Score")
            plt.ylabel("Features")
            plt.title("Top 20 Feature Importance")
            plt.gca().invert_yaxis()
            plt.show()

    # Print the results
    print("\nModel Comparison Results (K-Fold):")
    for model_name, val_loss, r2, mae in results:
        print(f"{model_name}: Avg Val Loss = {val_loss:.4f}, Avg R² = {r2:.4f}, Avg MAE = {mae:.4f}")

    # Identify the best model
    best_model = min(results, key=lambda x: x[1])  # Sort by average validation loss
    print(f"\nBest Model: {best_model[0]} with Avg Validation Loss = {best_model[1]:.4f}")

    return results

def analyze_results(results):
    """Analyze training vs validation loss to check for overfitting."""
    print("\nModel Analysis:")
    for model_name, train_loss, val_loss in results:
        generalization_gap = val_loss - train_loss
        print(f"{model_name}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Generalization Gap = {generalization_gap:.4f}")

def analyze_feature_importance(model, feature_names):
    """Analyze feature importance based on the weights of the first layer."""
    first_layer_weights = model.model[0].weight.detach().cpu().numpy()  # Extract weights of the first layer
    importance = np.mean(np.abs(first_layer_weights), axis=0)  # Compute mean absolute weights
    feature_importance = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)

    print("\nFeature Importance:")
    for feature, score in feature_importance:
        print(f"{feature}: {score:.4f}")

    return feature_importance

def main():
    # File paths
    data_file_path = 'output_data/selected_features.csv'

    # Load dataset
    X_train, X_test, y_train, y_test = load_data(data_file_path, n_components=3)

    # Define models to compare
    models = {  "model_combined": {
        "hidden_sizes": [512, 256, 128],  # Deep architecture with 4 layers
        "dropout_rate": 0.4,  # Optimal dropout rate for regularization
        "learning_rate": 0.0005,  # Balanced learning rate for fine adjustments
        "weight_decay": 1e-4,  # Regularization to prevent overfitting
        "activation": "ReLU",  # ReLU activation for non-linearity
        "optimizer": "Adam",  # Adam optimizer for efficient training
        "input_size": 5,  # Best-performing PCA input size
        
    }
    }

    """ models = {
    # Baseline Models
    "Model_Baseline_1": {"hidden_sizes": [128, 64], "dropout_rate": 0.5, "learning_rate": 0.001},
    "Model_Baseline_2": {"hidden_sizes": [256, 128], "dropout_rate": 0.4, "learning_rate": 0.001},

    # Variations in Hidden Layer Sizes
    "Model_Small": {"hidden_sizes": [64, 32], "dropout_rate": 0.5, "learning_rate": 0.001},
    "Model_Medium": {"hidden_sizes": [256, 128, 64], "dropout_rate": 0.5, "learning_rate": 0.001},
    "Model_Large": {"hidden_sizes": [512, 256, 128, 64], "dropout_rate": 0.5, "learning_rate": 0.001},

    # Variations in Dropout Rates
    "Model_Dropout_0.3": {"hidden_sizes": [512, 256, 128], "dropout_rate": 0.3, "learning_rate": 0.001},
    "Model_Dropout_0.4": {"hidden_sizes": [512, 256, 128], "dropout_rate": 0.4, "learning_rate": 0.001},
    "Model_Dropout_0.6": {"hidden_sizes": [512, 256, 128], "dropout_rate": 0.6, "learning_rate": 0.001},

    # Variations in Learning Rates
    "Model_LR_0.0001": {"hidden_sizes": [512, 256, 128], "dropout_rate": 0.5, "learning_rate": 0.0001},
    "Model_LR_0.0005": {"hidden_sizes": [512, 256, 128], "dropout_rate": 0.5, "learning_rate": 0.0005},
    "Model_LR_0.001": {"hidden_sizes": [512, 256, 128], "dropout_rate": 0.5, "learning_rate": 0.001},

    # Variations in Weight Decay
    "Model_WeightDecay_1e-4": {"hidden_sizes": [512, 256, 128], "dropout_rate": 0.5, "learning_rate": 0.001, "weight_decay": 1e-4},
    "Model_WeightDecay_1e-5": {"hidden_sizes": [512, 256, 128], "dropout_rate": 0.5, "learning_rate": 0.001, "weight_decay": 1e-5},
    "Model_WeightDecay_0": {"hidden_sizes": [512, 256, 128], "dropout_rate": 0.5, "learning_rate": 0.001, "weight_decay": 0},

    # Variations in Learning Rate Schedulers
    "Model_Scheduler_ReduceLROnPlateau": {
        "hidden_sizes": [512, 256, 128],
        "dropout_rate": 0.5,
        "learning_rate": 0.001,
        "scheduler": {
            "type": "ReduceLROnPlateau",
            "factor": 0.5,
            "patience": 5
        }
    },
    "Model_Scheduler_StepLR": {
        "hidden_sizes": [512, 256, 128],
        "dropout_rate": 0.5,
        "learning_rate": 0.001,
        "scheduler": {
            "type": "StepLR",
            "step_size": 10,
            "gamma": 0.5
        }
    },
    "Model_Scheduler_CosineAnnealing": {
        "hidden_sizes": [512, 256, 128],
        "dropout_rate": 0.5,
        "learning_rate": 0.001,
        "scheduler": {
            "type": "CosineAnnealingLR",
            "T_max": 50
        }
    },

    # Variations in Depth (Number of Layers)
    "Model_Shallow": {"hidden_sizes": [512], "dropout_rate": 0.5, "learning_rate": 0.001},
    "Model_Deep_4": {"hidden_sizes": [512, 256, 128, 64], "dropout_rate": 0.5, "learning_rate": 0.001},
    "Model_Deep_5": {"hidden_sizes": [512, 256, 128, 64, 32], "dropout_rate": 0.5, "learning_rate": 0.001},

    # Variations in Batch Normalization
    "Model_BatchNorm_Enabled": {"hidden_sizes": [512, 256, 128], "dropout_rate": 0.5, "learning_rate": 0.001, "batch_norm": True},
    "Model_BatchNorm_Disabled": {"hidden_sizes": [512, 256, 128], "dropout_rate": 0.5, "learning_rate": 0.001, "batch_norm": False},

    # Variations in Activation Functions
    "Model_Activation_ReLU": {"hidden_sizes": [512, 256, 128], "dropout_rate": 0.5, "learning_rate": 0.001, "activation": "ReLU"},
    "Model_Activation_LeakyReLU": {"hidden_sizes": [512, 256, 128], "dropout_rate": 0.5, "learning_rate": 0.001, "activation": "LeakyReLU"},
    "Model_Activation_Tanh": {"hidden_sizes": [512, 256, 128], "dropout_rate": 0.5, "learning_rate": 0.001, "activation": "Tanh"},

    # Variations in Optimizers
    "Model_Optimizer_Adam": {"hidden_sizes": [512, 256, 128], "dropout_rate": 0.5, "learning_rate": 0.001, "optimizer": "Adam"},
    "Model_Optimizer_SGD": {"hidden_sizes": [512, 256, 128], "dropout_rate": 0.5, "learning_rate": 0.001, "optimizer": "SGD"},
    "Model_Optimizer_RMSprop": {"hidden_sizes": [512, 256, 128], "dropout_rate": 0.5, "learning_rate": 0.001, "optimizer": "RMSprop"},

    # Variations in Input Size (for PCA Components)
    "Model_Input_3_PCA": {"hidden_sizes": [512, 256, 128], "dropout_rate": 0.5, "learning_rate": 0.001, "input_size": 3},
    "Model_Input_5_PCA": {"hidden_sizes": [512, 256, 128], "dropout_rate": 0.5, "learning_rate": 0.001, "input_size": 5},
    "Model_Input_10_PCA": {"hidden_sizes": [512, 256, 128], "dropout_rate": 0.5, "learning_rate": 0.001, "input_size": 10}
}
 """
    # Evaluate models
    feature_names = X_train.columns.tolist()  # List of feature names
    results = evaluate_models_kfold(
        models, 
        pd.concat([X_train, X_test]), 
        pd.concat([y_train, y_test]), 
        k=5, 
        max_epochs=100, 
        show_graphs=False, 
        analyze_features=True, 
        feature_names=feature_names
    )

    """ evaluate_models(
        models, 
        X_train,
        y_train,
        X_test, 
        y_test, 
        max_epochs=100

    ) """

    

if __name__ == "__main__":
    main()




