"""Neuro Foundation - Modular neuroscience molecular data analysis.

A comprehensive toolkit for analyzing molecular odorant data and neural activity maps
from the Pyrfume database, supporting both classical machine learning and graph neural
network approaches.

Main Features:
- Classical ML Pipeline: Mordred descriptors with configurable preprocessing
- Graph Neural Networks: Molecular graph representations with PyTorch Geometric
- Activity Maps: Pre-processed brain activity patterns with masking strategies
- Modular Design: Clean separation of data loading, preprocessing, training, and visualization

Quick Start:
    from neuro_foundation import Config, load_data, preprocess_data, train_model

    # Load and preprocess data
    config = Config()
    molecules, behavior, activity_maps = load_data()
    features = preprocess_data(molecules)

    # Train a model
    model = train_model(features, behavior['intensity'])
"""

from .config import Config, default_config
from .data import load_molecules, load_behavior, load_activity_maps
from .pipeline import preprocess_data, train_linear_model, train_nn_model

__version__ = "0.1.0"
__all__ = [
    "Config",
    "default_config",
    "load_molecules",
    "load_behavior",
    "load_activity_maps",
    "preprocess_data",
    "train_linear_model",
    "train_nn_model",
]