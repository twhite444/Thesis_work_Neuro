import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
import os
from tqdm import tqdm

class MolecularGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MolecularGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, x, edge_index, batch):
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Global mean pooling
        x = global_mean_pool(x, batch)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc2(x)
        return x

def load_behavior_data(behavior_df):
    # Extract CIDs and values from Stimulus column
    behavior_df['CID'] = behavior_df['Stimulus'].str.extract(r'(-?\d+)').astype(int)
    behavior_df['Value'] = behavior_df['Stimulus'].str.extract(r'(-?\d+)_\d+').astype(float)
    
    # Group by CID and calculate mean value
    behavior_values = behavior_df.groupby('CID')['Value'].mean().values
    
    # Normalize behavior values
    scaler = StandardScaler()
    behavior_values = scaler.fit_transform(behavior_values.reshape(-1, 1)).flatten()
    return behavior_values, scaler

def create_molecular_graph(molecules_df, behavior_df):
    # Load selected features
    selected_features = pd.read_csv('output_data/selected_features.csv', index_col='CID')
    
    # Find common CIDs between molecules and behavior data
    molecule_cids = molecules_df['CID'].values.tolist()
    behavior_cids = behavior_df['Stimulus'].str.extract(r'(-?\d+)').astype(int).values.tolist()
    common_cids = list(set(molecule_cids) & set(behavior_cids))
    print(f"Number of common CIDs: {len(common_cids)}")
    
    # Filter data to only include common CIDs
    molecules_df = molecules_df[molecules_df['CID'].isin(common_cids)]
    behavior_df = behavior_df[behavior_df['Stimulus'].str.extract(r'(-?\d+)').astype(int).isin(common_cids)]
    
    # Align features with molecules
    features = selected_features.loc[common_cids].values
    
    # Convert to tensor
    x = torch.tensor(features, dtype=torch.float)
    
    # Create edges based on molecular similarity
    edge_list = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            similarity = np.sum(features[i] * features[j]) / (np.linalg.norm(features[i]) * np.linalg.norm(features[j]))
            if similarity > 0.5:  # Higher threshold for more meaningful connections
                edge_list.extend([[i, j], [j, i]])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Create labels from behavior data
    behavior_values, scaler = load_behavior_data(behavior_df)
    y = torch.tensor(behavior_values, dtype=torch.float)
    
    # Create batch indices - each molecule is its own batch
    batch = torch.arange(x.size(0), dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, y=y, batch=batch), scaler

def train_model(model, data, optimizer, criterion, epochs=100, patience=10):
    model.train()
    losses = []
    r2_scores = []
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        losses.append(loss.item())
        
        # Calculate R² score
        with torch.no_grad():
            y_true = data.y.numpy()
            y_pred = out.numpy()
            r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
            r2_scores.append(r2)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}, R²: {r2:.4f}')
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    return losses, r2_scores

def visualize_results(model, data, losses, r2_scores):
    # Plot training losses
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(r2_scores)
    plt.title('R² Score Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()
    
    # Visualize molecular graph
    G = nx.Graph()
    edge_list = data.edge_index.t().numpy()
    for edge in edge_list:
        G.add_edge(edge[0], edge[1])
    
    plt.figure(figsize=(12, 8))
    nx.draw(G, with_labels=False, node_size=50, alpha=0.6)
    plt.title('Molecular Graph Structure')
    plt.savefig('molecular_graph.png')
    plt.close()

def main():
    # Load data
    molecules_df = pd.read_csv('output_data/molecules_raw.csv')
    behavior_df = pd.read_csv('output_data/behavior_data.csv')
    
    # Create graph data
    data, scaler = create_molecular_graph(molecules_df, behavior_df)
    
    # Print data shapes for debugging
    print(f"Number of molecules: {data.x.size(0)}")
    print(f"Number of features: {data.x.size(1)}")
    print(f"Number of behavior values: {data.y.size(0)}")
    print(f"Number of edges: {data.edge_index.size(1)}")
    
    # Initialize model
    input_dim = data.x.size(1)  # Use the number of selected features
    hidden_dim = 128  # Increased hidden dimension for more complex features
    output_dim = 1  # Single output dimension for behavior prediction
    model = MolecularGNN(input_dim, hidden_dim, output_dim)
    
    # Training setup with reduced learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # Train model
    losses, r2_scores = train_model(model, data, optimizer, criterion)
    
    # Visualize results
    visualize_results(model, data, losses, r2_scores)
    
    # Save model
    torch.save(model.state_dict(), 'molecular_gnn.pth')

if __name__ == "__main__":
    main()
