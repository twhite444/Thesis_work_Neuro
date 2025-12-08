import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_data(file_path):
    """ Load processed data from CSV """
    selected_features = pd.read_csv(f"output_data/selected_features.csv")
    return selected_features

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, layer_sizes):
        super(NeuralNetwork, self).__init__()
        layers = []
        last_size = input_dim
        for size in layer_sizes:
            layers.append(nn.Linear(last_size, size))
            layers.append(nn.ReLU())
            last_size = size
        layers.append(nn.Linear(last_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_model(model, criterion, optimizer, data_loader, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

def evaluate_model(model, data_loader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            predictions.extend(outputs.numpy())
            actuals.extend(targets.numpy())
    mse = mean_squared_error(actuals, predictions)
    return mse

def run_experiment(data_path, target_column, test_size=0.2, random_state=42, epochs=50, batch_size=32, learning_rates=[0.01, 0.001], layer_configs=[[32], [64, 32], [128, 64, 32]]):
    data = load_data(data_path)
    X = data.drop(columns=[target_column]).values.astype(np.float32)
    y = data[target_column].values.astype(np.float32).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    results = []
    for learning_rate in learning_rates:
        for layers in layer_configs:
            model = NeuralNetwork(input_dim=X_train.shape[1], layer_sizes=layers)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            train_model(model, criterion, optimizer, train_loader, epochs)
            mse = evaluate_model(model, test_loader)
            results.append({'learning_rate': learning_rate, 'layers': layers, 'mse': mse})
            print(f"Tested {layers} with learning rate {learning_rate}: MSE = {mse}")

    return results

if __name__ == "__main__":
    data_path = 'output_data/selected_features.csv'  # Path to your CSV file
    target_column = 'Target'  # Replace 'Target' with the actual name of your target column
    results = run_experiment(data_path, target_column)
    print(results)
