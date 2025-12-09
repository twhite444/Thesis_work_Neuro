"""
Pipeline stages for the olfactory prediction pipeline.

Available stages:
- feature_extraction: Extract molecular descriptors from SMILES
- preprocessing: Optional PCA, scaling, feature selection
- training: Train models with PyTorch Lightning
"""

from .feature_extraction import (
    FeatureExtractor,
    extract_features,
    get_feature_summary,
)
from .preprocessing import (
    Preprocessor,
    preprocess_data,
)
from .training import (
    TrainingStage,
    train_model,
)

__all__ = [
    'FeatureExtractor',
    'extract_features',
    'get_feature_summary',
    'Preprocessor',
    'preprocess_data',
    'TrainingStage',
    'train_model',
]
