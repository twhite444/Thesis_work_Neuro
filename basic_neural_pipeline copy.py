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
    image_data = pyrfume.load_data('leon/csvs/1031_0.csv')

    # Save raw data to CSV for later use
    molecules.to_csv(f"{output_dir}/molecules_raw.csv", index=False)
    image_data.to_csv(f"{output_dir}/image_data.csv", index=False)

    return molecules

# molecules = load_data()

############################################
# 2. Data Cleaning and Preprocessing
############################################

def preprocess_data(molecules):
    # Featurize molecules
    smiles = molecules["IsomericSMILES"].tolist()
    mordred_features = smiles_to_mordred(smiles)
    
    # Remove rows with NaN values and columns with zero variance
    filtered_data = mordred_features.dropna(axis=1, how='any')
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




############################################
# 4. Model Building and Training
############################################

def train_linear_regression(data, target_column):
    # Define target and features
    target = data[target_column]
    features = data.drop(columns=[target_column])

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    features_imputed = imputer.fit_transform(features)

    # Train linear regression model
    model = LinearRegression()
    model.fit(features_imputed, target)

    # Predict and calculate MSE
    predictions = model.predict(features_imputed)
    mse = mean_squared_error(target, predictions)

    # Save coefficients
    coefficients = pd.DataFrame({"Feature": features.columns, "Coefficient": model.coef_})
    coefficients.to_csv(f"{output_dir}/model_coefficients.csv", index=False)

    return model, mse





# Placeholder: Ensure you replace 'aromatic_group_column' with the actual target column
# trained_model, model_mse = train_linear_regression(selected_features, "aromatic_group_column")

############################################
# 5. Visualization
############################################

def plot_coefficients(model, feature_names):
    plt.figure(figsize=(10, 6))
    sorted_coefficients = np.sort(model.coef_)[::-1]
    sns.barplot(x=sorted_coefficients, y=feature_names)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature Index')
    plt.title('Rank-ordered Coefficients')
    plt.show()
    # Save plot to file
    plt.savefig(f"{output_dir}/coefficients_plot.png")


############################################
# 6. Neural Network (Optional Step)
############################################
from keras.models import Sequential
from keras.layers import Dense, Activation

def build_nn(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Regression output

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model

if __name__ == "__main__":
    # Load data
    molecules = load_data()

    # Preprocess data
    cleaned_data = preprocess_data(molecules)

    # Perform feature selection
    selected_features = select_features(cleaned_data)

    # Uncomment and modify the following lines if you need to train models
    trained_model, model_mse = train_linear_regression(selected_features, "VE2_A")
    print(f"Model MSE: {model_mse}")

    plot_coefficients(trained_model, selected_features.columns)
    nn_model = build_nn(input_dim=cleaned_data.shape[1])

    def main_from_csv():
        # Load preprocessed data from CSV
        cleaned_data = pd.read_csv(f"{output_dir}/cleaned_data.csv")
        selected_features = pd.read_csv(f"{output_dir}/selected_features.csv")

        # Train linear regression model
        trained_model, model_mse = train_linear_regression(selected_features, "VE2_A")
        print(f"Model MSE: {model_mse}")

        # Plot coefficients
        plot_coefficients(trained_model, selected_features.columns)

    if __name__ == "__main__":
        main_from_csv()


