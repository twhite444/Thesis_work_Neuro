"""
Neural network models for olfactory prediction.

Available models:
- OdorPredictor: PyTorch Lightning module for odor prediction
"""

from .base_predictor import OdorPredictor

__all__ = ['OdorPredictor']
