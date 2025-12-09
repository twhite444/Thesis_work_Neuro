import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)  # Add this in the NeuralNetwork class
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Add this in the NeuralNetwork class

    def forward(self, x):
        x = self.bn1(self.relu(self.fc1(x)))  # Apply batch normalization after the first layer
        x = self.dropout(x)  # Apply dropout after batch normalization
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_data(file_path, n_components=3):
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

    # Split into train-test sets
    return train_test_split(X_aligned, y_aligned, test_size=0.2, random_state=42)

def train_model(model, criterion, optimizer, train_loader, val_loader, max_epochs=50):
    """Train the model and log training/validation losses."""
    train_losses = []
    val_losses = []

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

        print(f"Epoch {epoch + 1}/{max_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

def plot_losses(train_losses, val_losses):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # File paths
    data_file_path = 'output_data/selected_features.csv'

    # Load dataset
    X_train, X_test, y_train, y_test = load_data(data_file_path, n_components=3)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    # Create DataLoader for batching
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize the model, loss function, and optimizer
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    model = NeuralNetwork(input_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    max_epochs = 50
    train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader, max_epochs=max_epochs)

    # Plot the losses
    plot_losses(train_losses, val_losses)

if __name__ == "__main__":
    main()