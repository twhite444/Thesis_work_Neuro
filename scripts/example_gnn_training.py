#!/usr/bin/env python3
"""Example GNN model for molecular property prediction.

This script demonstrates how to build and train a simple Graph Neural Network
for molecular property prediction using the molecular graph dataset.

Note: This is a minimal example to demonstrate the data pipeline integration.
For production use, you would want to:
1. Add proper hyperparameter tuning
2. Implement early stopping
3. Add model checkpointing
4. Use more sophisticated GNN architectures
5. Add validation metrics and visualization
"""

import sys
import os
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, Dropout
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.neuro_foundation.data.gnn_utils import (
    create_pyg_dataset,
    create_train_val_test_split
)


class SimpleGNN(torch.nn.Module):
    """Simple Graph Convolutional Network for molecular property prediction.
    
    Architecture:
    - 3 GCN layers with ReLU activation
    - Global mean pooling to aggregate node features
    - 2 fully connected layers for prediction
    
    Args:
        num_node_features: Dimension of input node features (137)
        hidden_dim: Hidden layer dimension (default: 64)
        num_classes: Number of output classes (default: 1 for regression)
        dropout: Dropout probability (default: 0.2)
    """
    
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 64,
        num_classes: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # GCN layers
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # MLP for prediction
        self.mlp = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, data):
        """Forward pass.
        
        Args:
            data: PyG Data/Batch object with attributes:
                - x: Node features [num_nodes, num_node_features]
                - edge_index: Edge connectivity [2, num_edges]
                - batch: Batch assignment [num_nodes]
                
        Returns:
            Predictions [batch_size, num_classes]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GCN layers with ReLU activation
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # MLP for prediction
        x = self.mlp(x)
        
        return x


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch.
    
    Args:
        model: GNN model
        loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch)
        
        # Compute loss (assuming target is stored in batch.y)
        # For this example, we'll create dummy targets
        # In practice, you would load real target values
        target = torch.randn(batch.num_graphs, 1).to(device)
        loss = criterion(out, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on validation/test set.
    
    Args:
        model: GNN model
        loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    for batch in loader:
        batch = batch.to(device)
        
        # Forward pass
        out = model(batch)
        
        # Compute loss with dummy targets
        target = torch.randn(batch.num_graphs, 1).to(device)
        loss = criterion(out, target)
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def main():
    print("=" * 70)
    print("GNN Model Training Example")
    print("=" * 70)
    print()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    # Load dataset
    print("Loading molecular graph dataset...")
    dataset = create_pyg_dataset(data_dir='data/01_raw')
    print(f"✓ Loaded {len(dataset)} molecules")
    print()
    
    # Split dataset
    print("Splitting dataset...")
    train_ds, val_ds, test_ds = create_train_val_test_split(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    print(f"  Training: {len(train_ds)} molecules")
    print(f"  Validation: {len(val_ds)} molecules")
    print(f"  Test: {len(test_ds)} molecules")
    print()
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print(f"Created DataLoaders with batch_size={batch_size}")
    print()
    
    # Initialize model
    print("Initializing GNN model...")
    num_node_features = dataset[0].x.shape[1]
    model = SimpleGNN(
        num_node_features=num_node_features,
        hidden_dim=64,
        num_classes=1,
        dropout=0.2
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model initialized with {num_params:,} trainable parameters")
    print()
    print("Model architecture:")
    print(model)
    print()
    
    # Initialize optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    print("=" * 70)
    print("Training (Demo - 5 epochs)")
    print("=" * 70)
    print()
    
    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        val_loss = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
    
    print()
    
    # Final test evaluation
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Loss: {test_loss:.4f}")
    print()
    
    # Model analysis
    print("=" * 70)
    print("Model Analysis")
    print("=" * 70)
    print()
    
    # Get a sample batch
    sample_batch = next(iter(train_loader)).to(device)
    print(f"Sample batch:")
    print(f"  Graphs: {sample_batch.num_graphs}")
    print(f"  Total nodes: {sample_batch.x.shape[0]}")
    print(f"  Total edges: {sample_batch.edge_index.shape[1]}")
    print()
    
    # Forward pass
    with torch.no_grad():
        output = model(sample_batch)
    print(f"Model output shape: {output.shape}")
    print(f"Sample predictions (first 5):")
    print(output[:5].cpu().numpy())
    print()
    
    print("=" * 70)
    print("✓ Training complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Add real target values (e.g., molecular properties)")
    print("  2. Implement proper evaluation metrics (MAE, R², etc.)")
    print("  3. Add model checkpointing and early stopping")
    print("  4. Experiment with different GNN architectures (GAT, GraphSAGE, etc.)")
    print("  5. Perform hyperparameter tuning")
    print()


if __name__ == '__main__':
    main()
