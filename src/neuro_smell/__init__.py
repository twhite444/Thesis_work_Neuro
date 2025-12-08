"""
Neuro-Smell: Neural Network Pipeline for Olfactory Prediction

This package provides tools for predicting brain activation patterns
from molecular features using deep learning.
"""

__version__ = "2.0.0"
__author__ = "Tom White"

# Make key classes easily importable
from neuro_smell.models.base_predictor import OdorPredictor
from neuro_smell.datamodules.olfactory_datamodule import OlfactoryDataModule

__all__ = [
    "OdorPredictor",
    "OlfactoryDataModule",
]
