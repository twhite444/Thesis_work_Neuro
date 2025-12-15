"""Neural network models for predicting activity maps from molecular structure."""

from .baseline_nn import MoleculeToActivityMapMLP, MoleculeToActivityMapCNN

__all__ = [
    'MoleculeToActivityMapMLP',
    'MoleculeToActivityMapCNN',
]
