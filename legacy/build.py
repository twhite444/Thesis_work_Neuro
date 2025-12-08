# -*- coding: utf-8 -*-
"""Modularized Neural Network Pipeline"""

# Import necessary libraries
import pyrfume
from pyrfume.odorants import display_molecules, embed_molecules
from pyrfume.features import smiles_to_mordred
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from rdkit import Chem

# Set the environment variable
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Ensure output folder exists
output_dir = "output_data"
os.makedirs(output_dir, exist_ok=True)

############################################
# 1. Initialization: Load data
############################################

def load_data():
    arshamian_manifest = pyrfume.load_manifest('arshamian_2022')
    leon_manifest = pyrfume.load_manifest('leon')
    
    molecules = pyrfume.load_data('leon/molecules.csv')
    molecules.reset_index(inplace=True)
    molecules.rename(columns={'index': 'CID'}, inplace=True)
    behavior_data = pyrfume.load_data('leon/behavior_1.csv')
    image_data = pyrfume.load_data('leon/csvs/1031_0.csv')

    # Save raw data to CSV for later use
    molecules.to_csv(f"{output_dir}/molecules_raw.csv", index=True)
    behavior_data.to_csv(f"{output_dir}/behavior_data.csv", index=True)
    image_data.to_csv(f"{output_dir}/image_data.csv", index=True)

    return molecules
def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # Check if mol is None (invalid SMILES)
    if mol is None:
        return False
    return True

############################################
# 2. Data Cleaning and Preprocessing
############################################

def featurize_smiles(molecules):
    smiles = molecules["IsomericSMILES"].tolist()
    valid_smiles = [s for s in smiles if is_valid_smiles(s)]
    successful_cids = []
    mordred_features = []

    for cid, smile in zip(molecules['CID'], valid_smiles):
        try:
            features = smiles_to_mordred([smile])
            if not features.empty:
                successful_cids.append(cid)
                mordred_features.append(features)
        except Exception as e:
            print(f"Failed to featurize CID {cid} with SMILES {smile}: {e}")

    mordred_features = pd.concat(mordred_features, ignore_index=True)
    return successful_cids, mordred_features

def preprocess_data(molecules):
    # Featurize molecules
    smiles = molecules["IsomericSMILES"].tolist()
    print(f"Number of SMILES strings: {len(smiles)}")
    # Check validity of SMILES strings
    valid_smiles = [s for s in smiles if is_valid_smiles(s)]
    print(f"Number of valid SMILES strings: {len(valid_smiles)}")
    #mordred_features = smiles_to_mordred(smiles)
    successful_cids, mordred_features = featurize_smiles(molecules)
    print(f"Number of successfully featurized molecules: {len(successful_cids)}")
    print(mordred_features.head())

    
    # Remove rows with NaN values and columns with zero variance
    filtered_data = mordred_features.dropna(axis=1, how='any')
    initial_columns = mordred_features.shape[1]
    filtered_data = mordred_features.dropna(axis=1, how='any')
    after_nan_removal_columns = filtered_data.shape[1]
    filtered_data = filtered_data.loc[:, ~(filtered_data.eq(0).any(axis=0))]
    after_zero_variance_removal_columns = filtered_data.shape[1]

    print(f"Initial columns: {initial_columns}")
    print(f"Columns after NaN removal: {after_nan_removal_columns}")
    print(f"Columns after zero variance removal: {after_zero_variance_removal_columns}")
    filtered_data = filtered_data.loc[:, ~(filtered_data.eq(0).any(axis=0))]

    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(filtered_data)

    standardized_df = pd.DataFrame(standardized_data, columns=filtered_data.columns)

    # Save cleaned data to CSV
    standardized_df.to_csv(f"{output_dir}/cleaned_data.csv", index=False)
    return standardized_df

# cleaned_data = preprocess_data(molecules)

############################################
# 3. Feature Selection
############################################

def select_features(data, variance_threshold=1.0):
    # Select features with variance above threshold
    selector = VarianceThreshold(threshold=variance_threshold)
    selected_data = selector.fit_transform(data)
    selected_features = data.columns[selector.get_support()]

    selected_df = pd.DataFrame(selected_data, columns=selected_features)

    # Save selected features to CSV
    selected_df.to_csv(f"{output_dir}/selected_features.csv", index=False)
    return selected_df

#selected_features = select_features(cleaned_data)


def process_all(variance_threshold=1.0):
    molecules = load_data()
    cleaned_data = preprocess_data(molecules)
    selected_features = select_features(cleaned_data, variance_threshold)
    return selected_features

if __name__ == "__main__":
    output_dir = "output_data"
    variance_threshold = 1.0  # Change this value based on your variance threshold needs
    selected_features = process_all(variance_threshold)
    print("Data loading, preprocessing, and feature selection completed.")

