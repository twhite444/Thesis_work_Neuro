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

    # Check for duplicate CIDs
    duplicate_cids = molecules[molecules.duplicated(subset='CID', keep=False)]
    print(f"Duplicate CIDs in molecules before removal:\n{duplicate_cids}")

    # Remove duplicate CIDs by keeping the first occurrence
    molecules = molecules.drop_duplicates(subset='CID', keep='first')

    # Debug: Check for duplicates after removal
    duplicate_cids_after = molecules[molecules.duplicated(subset='CID', keep=False)]
    print(f"Duplicate CIDs in molecules after removal:\n{duplicate_cids_after}")

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
    smiles = molecules["IsomericSMILES"].unique().tolist()
    print(f"Number of SMILES strings: {len(smiles)}")
    # Check validity of SMILES strings
    valid_smiles = [s for s in smiles if is_valid_smiles(s)]
    print(f"Number of valid SMILES strings: {len(valid_smiles)}")
    mordred_features = smiles_to_mordred(smiles)
    print("Mordred features shape:", mordred_features.shape)

    # Add CID to mordred features for alignment
    print("Molecules CID values:", molecules["CID"].head())
    mordred_features["CID"] = molecules["CID"].values[:mordred_features.shape[0]]  # Ensure alignment
    mordred_features.set_index("CID", inplace=True)
    print("Mordred features index after setting CID:")
    print(mordred_features.index)

    # Remove rows with NaN values and columns with zero variance
    initial_columns = mordred_features.shape[1]
    filtered_data = mordred_features.dropna(axis=1, how='any')
    after_nan_removal_columns = filtered_data.shape[1]
    filtered_data = filtered_data.loc[:, ~(filtered_data.eq(0).any(axis=0))]
    after_zero_variance_removal_columns = filtered_data.shape[1]

    print(f"Initial columns: {initial_columns}")
    print(f"Columns after NaN removal: {after_nan_removal_columns}")
    print(f"Columns after zero variance removal: {after_zero_variance_removal_columns}")

    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(filtered_data)

    standardized_df = pd.DataFrame(standardized_data, columns=filtered_data.columns, index=filtered_data.index)

    # Save cleaned data to CSV
    standardized_df.to_csv(f"{output_dir}/cleaned_data.csv", index=True)

    # Debug: Print index information
    print("Cleaned data index:", standardized_df.index.name)
    print("First few rows of cleaned data:")
    print(standardized_df.head())

    return standardized_df

############################################
# 3. Feature Selection
############################################

def select_features(data, variance_threshold=1.0):
    # Select features with variance above threshold
    selector = VarianceThreshold(threshold=variance_threshold)
    selected_data = selector.fit_transform(data)
    selected_features = data.columns[selector.get_support()]

    selected_df = pd.DataFrame(selected_data, columns=selected_features, index=data.index)

    # Save selected features to CSV
    data.to_csv(f"{output_dir}/selected_features.csv", index=True)
    
    
    # Debug: Print index information
    print("Selected features index:", selected_df.index.name)
    print("First few rows of selected features:")
    print(selected_df.head())
    
    return selected_df

def remove_correlated_features(data):
    # Remove highly correlated features
    correlation_matrix = data.corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
    data_reduced = data.drop(columns=to_drop)

    print(f"Reduced data shape after removing correlated features: {data_reduced.shape}")
    return data_reduced



def process_all(variance_threshold=1.0):
    molecules = load_data()
    print("Molecules index after loading:", molecules.index.name)
    #analyze_feature_variances(molecules)

    cleaned_data = preprocess_data(molecules)
    print("Cleaned data index after preprocessing:", cleaned_data.index.name)

    selected_features = select_features(cleaned_data, variance_threshold)
    print("Selected features index after feature selection:", selected_features.index.name)

    #reduced_data = remove_correlated_features(selected_features)
    print("Reduced data index after removing correlated features:", reduced_data.index.name)


    return reduced_data

if __name__ == "__main__":
    output_dir = "output_data"
    variance_threshold = 1.0  # Change this value based on your variance threshold needs
    reduced_data = process_all(variance_threshold)
    print("Data loading, preprocessing, feature selection, and correlation removal completed.")

