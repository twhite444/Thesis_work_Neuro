"""Test script for feature importance visualization.

This script:
1. Loads feature names from processed data
2. Trains a simple neural network (or loads existing model)
3. Generates feature importance visualization
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from neuro_foundation.data.activity_map_dataset import MoleculeActivityMapDataset
from neuro_foundation.models.baseline_nn import MoleculeToActivityMapMLP
from neuro_foundation.pipeline.train_nn import train_nn
from neuro_foundation.visualization import plot_feature_importance


def load_feature_names(processed_dir: str = "data/02_processed") -> list[str]:
    """Load feature names from selected_features.csv.
    
    Args:
        processed_dir: Directory containing processed data
        
    Returns:
        List of feature names (excludes 'CID' column)
    """
    features_path = Path(processed_dir) / "selected_features.csv"
    
    if not features_path.exists():
        raise FileNotFoundError(
            f"Selected features not found at {features_path}. "
            "Run preprocessing pipeline first."
        )
    
    # Load just the header to get column names
    features_df = pd.read_csv(features_path, nrows=0)
    
    # Get all column names except 'CID'
    feature_names = [col for col in features_df.columns if col != 'CID']
    
    print(f"✓ Loaded {len(feature_names)} feature names")
    return feature_names


def main():
    """Run feature importance visualization test."""
    
    print("="*70)
    print("Feature Importance Visualization Test")
    print("="*70)
    
    # Set device
    device = torch.device('cpu')  # Use CPU for quick test
    print(f"\nUsing device: {device}")
    
    # Load feature names
    print("\n1. Loading feature names...")
    feature_names = load_feature_names()
    print(f"   First 5 features: {feature_names[:5]}")
    
    # Load dataset
    print("\n2. Loading dataset...")
    dataset = MoleculeActivityMapDataset(
        processed_dir="data/02_processed",
        split='train',
        random_seed=42
    )
    
    # Check if we have a trained model
    model_path = Path("models/best_model.pth")
    
    if model_path.exists():
        print("\n3. Loading existing trained model...")
        model = MoleculeToActivityMapMLP(
            input_dim=dataset.feature_dim,
            hidden_dims=[512, 256, 128],
            output_shape=(79, 43),
            dropout=0.35
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"   ✓ Loaded model from {model_path}")
        
    else:
        print("\n3. Training a quick model (5 epochs)...")
        print("   (For real analysis, use a fully trained model)")
        
        # Create dataloaders
        train_loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0
        )
        
        val_dataset = MoleculeActivityMapDataset(
            processed_dir="data/02_processed",
            split='val',
            random_seed=42
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0
        )
        
        # Create model
        model = MoleculeToActivityMapMLP(
            input_dim=dataset.feature_dim,
            hidden_dims=[512, 256, 128],
            output_shape=(79, 43),
            dropout=0.35
        )
        model.to(device)
        
        # Train quickly
        metrics = train_nn(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir="models_test",
            num_epochs=5,  # Quick training
            learning_rate=1e-3,
            early_stopping_patience=3,
            device=device,
            verbose=True
        )
        
        print(f"   ✓ Training complete. Best val correlation: {metrics['best_val_correlation']:.4f}")
    
    # Generate feature importance visualization
    print("\n4. Generating feature importance visualization...")
    
    output_dir = Path("viz/feature_importance")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "top_20_features.png"
    
    fig = plot_feature_importance(
        model=model,
        feature_names=feature_names,
        top_n=20,
        output_path=output_path,
        figsize=(10, 8),
        color='#2E86AB',
        title='Top 20 Molecular Descriptors Ranked by Importance\nBased on First-Layer Weight Magnitudes'
    )
    
    print(f"\n✓ Feature importance visualization saved to: {output_path}")
    
    # Also generate top 50 for comprehensive analysis
    output_path_50 = output_dir / "top_50_features.png"
    fig_50 = plot_feature_importance(
        model=model,
        feature_names=feature_names,
        top_n=50,
        output_path=output_path_50,
        figsize=(12, 16),
        color='#A23B72',
        title='Top 50 Molecular Descriptors Ranked by Importance'
    )
    
    print(f"✓ Extended visualization (top 50) saved to: {output_path_50}")
    
    print("\n" + "="*70)
    print("Feature importance analysis complete! 🎉")
    print("="*70)
    print(f"\nVisualization files:")
    print(f"  - {output_path}")
    print(f"  - {output_path_50}")
    print("\nThese visualizations show which molecular features the neural")
    print("network relies on most heavily to predict olfactory bulb activation.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
