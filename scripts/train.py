#!/usr/bin/env python3
"""
Train Script - Simple entry point for training models

Usage:
    # Train with default config
    python scripts/train.py
    
    # Use a specific experiment config
    python scripts/train.py experiment=example_baseline
    
    # Override specific parameters
    python scripts/train.py model=large_net training=full_training
    
    # Override from command line
    python scripts/train.py model.architecture.hidden_layers=[256,128,64] training.max_epochs=50

This script uses Hydra for configuration management.
All configs are in configs/ directory.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import hydra
from omegaconf import DictConfig, OmegaConf

from neuro_smell.datamodules.olfactory_datamodule import OlfactoryDataModule
from neuro_smell.models.base_predictor import OdorPredictor
from neuro_smell.stages.training import train_model


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    """
    Main training function.
    
    Args:
        config: Hydra configuration loaded from configs/
    """
    print("\n" + "="*60)
    print("🧠 Olfactory Prediction Training")
    print("="*60)
    
    # Print configuration
    print("\n📋 Configuration:")
    print(OmegaConf.to_yaml(config))
    
    # Train model
    results = train_model(config)
    
    # Print final results
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"\nExperiment: {results['experiment_name']}")
    print(f"Output directory: {results['output_dir']}")
    
    if results['test_results']:
        print(f"\n🎯 Test Results:")
        for key, value in results['test_results'].items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
