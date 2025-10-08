# Predicting Odor-Evoked Brain Activities from Molecular Features

A machine learning approach to understanding how the brain processes smell by predicting neural activation patterns from molecular structure.

## About This Project

This repository contains my honors thesis research exploring one of neuroscience's biggest puzzles: how does the brain translate chemical structure into the experience of smell? Unlike vision or hearing, smell remains poorly understood, and no chemist can predict how a new molecule will smell.

I developed a neural network pipeline that learns to predict brain activation maps directly from molecular features, helping decode the brain's "odor code."

## Research Impact

- **Grant Success**: My research findings helped my professor secure additional funding for continued investigation
- **Publication Potential**: Results show strong promise for peer-reviewed publication
- **Educational Legacy**: This codebase now serves as the foundation for future student research projects

## Technical Approach

### Machine Learning Pipeline
- Custom PyTorch neural networks with optimized architectures
- Grid search across multiple model types (Lasso, ElasticNet, SVR, KNN, Ensemble methods)
- Comprehensive feature selection using PCA and variance thresholding
- Cross-validation with proper train/test splitting

### Data Processing
- Molecular feature extraction from SMILES strings using RDKit
- Automated data cleaning and validation
- Standardized preprocessing pipeline
- Brain activation map processing and visualization

### Code Quality
- Modular Python architecture with clear separation of concerns
- Reproducible experiments with documented parameters
- Extensible design for future research iterations
- Proper error handling and data validation throughout

## Key Findings

Successfully demonstrated that molecular features can predict brain activation patterns with significant correlation. The model identified critical molecular descriptors (like RNCG - ionization potential) that drive olfactory neural responses.

## Technical Skills Demonstrated

- **Machine Learning**: Neural networks, ensemble methods, hyperparameter optimization
- **Data Science**: Large-scale preprocessing, dimensionality reduction, statistical analysis  
- **Python Development**: Object-oriented design, modular architecture


This project bridges computational chemistry, neuroscience, and machine learning to tackle fundamental questions about how we perceive the world through smell.