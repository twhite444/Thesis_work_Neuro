"""Olfactory Modeling - Modular neuroscience molecular data analysis.

A comprehensive toolkit for analyzing molecular odorant data and neural activity maps
from the Pyrfume database, supporting both classical machine learning and graph neural
network approaches.

Main Features:
- Classical ML Pipeline: Mordred descriptors with configurable preprocessing
- Graph Neural Networks: Molecular graph representations with PyTorch Geometric
- Activity Maps: Pre-processed brain activity patterns with masking strategies
- Modular Design: Clean separation of data loading, preprocessing, training, and visualization

Quick Start:
    from olfactory_modeling.data import PyrfumeLoader
    from olfactory_modeling.pipeline import featurize_and_standardize
    from olfactory_modeling.models import MoleculeToActivityMapMLP

    # Load data
    loader = PyrfumeLoader()
    molecules = loader.load_molecules()

    # Process features
    features = featurize_and_standardize(molecules)
"""

from . import config, data, models, pipeline, utils, visualization

__version__ = "0.1.0"