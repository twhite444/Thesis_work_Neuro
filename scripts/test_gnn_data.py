#!/usr/bin/env python3
"""Test script for GNN data loading utilities.

This script demonstrates how to load molecular graphs and use them with
PyTorch Geometric DataLoader for GNN training.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.neuro_foundation.data.gnn_utils import (
    create_pyg_dataset,
    create_train_val_test_split,
    print_dataset_info,
    get_node_feature_names,
    get_edge_feature_names
)

try:
    from torch_geometric.loader import DataLoader
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: PyTorch Geometric not available for DataLoader demo")


def main():
    print("=" * 70)
    print("GNN Data Loading Test")
    print("=" * 70)
    print()
    
    # Create dataset
    print("Loading molecular graph dataset...")
    dataset = create_pyg_dataset(data_dir='data/01_raw')
    print(f"✓ Loaded {len(dataset)} molecules")
    print()
    
    # Print dataset info
    print_dataset_info(dataset)
    print()
    
    # Test individual graph access
    print("=" * 70)
    print("Testing Individual Graph Access")
    print("=" * 70)
    print()
    
    # Access by index
    data = dataset[0]
    print(f"Graph 0:")
    print(f"  CID: {data.cid}")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.edge_index.shape[1]}")
    print(f"  Node features shape: {data.x.shape}")
    if hasattr(data, 'edge_attr'):
        print(f"  Edge features shape: {data.edge_attr.shape}")
    print()
    
    # Access by CID
    print("Accessing graph by CID...")
    data_180 = dataset.get_by_cid(180)
    if data_180 is not None:
        print(f"✓ Found CID 180:")
        print(f"  Nodes: {data_180.num_nodes}")
        print(f"  Edges: {data_180.edge_index.shape[1]}")
    print()
    
    # Test train/val/test split
    print("=" * 70)
    print("Testing Dataset Split")
    print("=" * 70)
    print()
    
    train_ds, val_ds, test_ds = create_train_val_test_split(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    print(f"Split results:")
    print(f"  Training set: {len(train_ds)} molecules ({len(train_ds)/len(dataset)*100:.1f}%)")
    print(f"  Validation set: {len(val_ds)} molecules ({len(val_ds)/len(dataset)*100:.1f}%)")
    print(f"  Test set: {len(test_ds)} molecules ({len(test_ds)/len(dataset)*100:.1f}%)")
    print()
    
    # Test DataLoader
    if TORCH_GEOMETRIC_AVAILABLE:
        print("=" * 70)
        print("Testing PyTorch Geometric DataLoader")
        print("=" * 70)
        print()
        
        # Create DataLoader
        batch_size = 32
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        print(f"Created DataLoader with batch_size={batch_size}")
        print(f"Number of batches: {len(train_loader)}")
        print()
        
        # Test first batch
        print("Testing first batch:")
        for batch in train_loader:
            print(f"  Batch contains {batch.num_graphs} graphs")
            print(f"  Total nodes in batch: {batch.x.shape[0]}")
            print(f"  Total edges in batch: {batch.edge_index.shape[1]}")
            print(f"  Node features shape: {batch.x.shape}")
            if hasattr(batch, 'edge_attr'):
                print(f"  Edge features shape: {batch.edge_attr.shape}")
            print(f"  Batch index shape: {batch.batch.shape}")
            break
        print()
    
    # Display feature names
    print("=" * 70)
    print("Feature Definitions")
    print("=" * 70)
    print()
    
    node_features = get_node_feature_names()
    print(f"Node features ({len(node_features)} total):")
    print(f"  First 10: {node_features[:10]}")
    print(f"  Last 10: {node_features[-10:]}")
    print()
    
    edge_features = get_edge_feature_names()
    print(f"Edge features ({len(edge_features)} total):")
    for i, name in enumerate(edge_features):
        print(f"  {i}: {name}")
    print()
    
    # Summary
    print("=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
    print()
    print("GNN data pipeline is ready to use!")
    print()
    print("Example usage:")
    print("  from src.neuro_foundation.data.gnn_utils import create_pyg_dataset")
    print("  from torch_geometric.loader import DataLoader")
    print("  ")
    print("  dataset = create_pyg_dataset()")
    print("  loader = DataLoader(dataset, batch_size=32, shuffle=True)")
    print("  ")
    print("  for batch in loader:")
    print("      # Train your GNN model on batch")
    print("      pass")
    print()


if __name__ == '__main__':
    main()
