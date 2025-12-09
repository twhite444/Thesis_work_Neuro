import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pyrfume import load_data
import pyrfume
from scipy.ndimage import label, binary_dilation, binary_erosion, gaussian_filter
from tqdm import tqdm

def load_maps_and_apply_mask(coverage_threshold=1.0, verbose=False):
    """Load activity maps and apply a global mask based on coverage threshold."""
    # Load behavior data and extract CIDs
    directory = pd.read_csv('output_data/behavior_data.csv')
    directory['CID'] = directory['Stimulus'].map(lambda x: int(str(x).split('_')[0]))
    directory = directory[directory['CID'] > 0]  # Keep only valid CIDs

    # Debugging: Check the directory
    print(f"Number of rows in directory: {len(directory)}")
    print("First few rows of directory:")
    print(directory.head())

    # Initialize variables
    activity_maps_df = pd.DataFrame(columns=['CID', 'Activity Map'])
    valid_counts = None

    # Process activity maps
    for _, row in tqdm(directory.iterrows(), total=directory.shape[0], desc="Processing activity maps"):
        map_path = os.path.join('leon/', row['Activity Map Path'])
        activity_map = np.nan_to_num(load_data(map_path).to_numpy(), nan=0)
        activity_maps_df = pd.concat([activity_maps_df, pd.DataFrame({'CID': [row['CID']], 'Activity Map': [activity_map]})], ignore_index=True)

        if valid_counts is None:
            valid_counts = np.zeros_like(activity_map, dtype=int)
        valid_counts += ~np.isnan(activity_map)

    # Create and refine global mask
    global_mask = valid_counts >= int(coverage_threshold * len(activity_maps_df))
    refined_global_mask = binary_erosion(binary_dilation(global_mask))
    labeled_mask, _ = label(refined_global_mask)
    valid_regions = np.isin(labeled_mask, np.where(np.bincount(labeled_mask.ravel()) >= 100)[0])
    refined_global_mask &= valid_regions

    # Apply the global mask to each map
    activity_maps_df['Masked Activity Map'] = activity_maps_df['Activity Map'].apply(lambda map_: map_ * refined_global_mask)

    # Group by CID and calculate the mean of masked activity maps
    grouped = activity_maps_df.groupby('CID')['Masked Activity Map'].apply(lambda maps: np.nanmean(np.stack(maps.to_list()), axis=0))
    masked_maps = grouped.tolist()
    cids = grouped.index.tolist()

    if verbose:
        print(f"Number of maps before groupby: {len(activity_maps_df)}")
        print(f"Number of maps after groupby: {len(masked_maps)}")

    return masked_maps, cids, refined_global_mask

def perform_pca(masked_maps, n_components=20):
    """Perform PCA on flattened and masked activity maps."""
    # Flatten maps for PCA
    flat_maps = np.array([map_.flatten() for map_ in masked_maps])

    # Standardize the data
    standardized_data = StandardScaler().fit_transform(flat_maps)

    # Perform PCA
    n_components = min(n_components, standardized_data.shape[0], standardized_data.shape[1])
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(standardized_data)

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    print("Cumulative explained variance:", cumulative_variance)
    
    # Plot cumulative explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(cumulative_variance, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by PCA Components')
    plt.grid()
    plt.show()
    
    return principal_components, pca

def visualize_results(pca, global_mask, output_dir='output_data'):
    """Visualize PCA components and the global mask."""
    # Visualize PCA components
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, ax in enumerate(axes):
        component = pca.components_[i].reshape(global_mask.shape)
        img = ax.imshow(gaussian_filter(component, sigma=1), cmap='viridis')
        ax.set_title(f'PCA Component {i+1}')
        fig.colorbar(img, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_3_components.png'))
    plt.show()

    # Visualize global mask
    plt.figure(figsize=(10, 8))
    plt.imshow(global_mask, cmap='gray')
    plt.title('Refined Global Mask')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'global_mask.png'))
    plt.show()

def visualize_spatial_loadings(pca, global_mask, output_dir='output_data'):
    """Visualize PCA loadings as spatial maps."""
    for i in range(3):  # Visualize the first 3 components
        component = pca.components_[i].reshape(global_mask.shape)
        plt.figure(figsize=(8, 6))
        plt.imshow(gaussian_filter(component, sigma=1), cmap='coolwarm')
        plt.title(f'Spatial Loadings for PC{i + 1}')
        plt.colorbar(label='Loading Value')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'spatial_loadings_pc{i + 1}.png'))
        plt.show()

def visualize_pca_components_scatter(principal_components, cids, output_dir='output_data'):
    """
    Create a 2D scatter plot of the first two PCA components for each odor map.
    All points are shown as blue dots without labels.
    
    Parameters:
    -----------
    principal_components : numpy.ndarray
        The PCA-transformed data with shape (n_samples, n_components)
    cids : list
        List of CIDs corresponding to each data point
    output_dir : str
        Directory to save the output visualization
    """
    if principal_components.shape[1] < 2:
        print("Error: Need at least 2 PCA components for visualization")
        return
    
    # Create figure with appropriate size
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot with all blue dots
    plt.scatter(
        principal_components[:, 0], 
        principal_components[:, 1], 
        color='blue',  # All points are blue
        alpha=0.7, 
        s=80,  # Marker size
        edgecolors='w'
    )
    
    # Add labels and title
    plt.xlabel('Principal Component 1', fontsize=14)
    plt.ylabel('Principal Component 2', fontsize=14)
    plt.title('PCA Component Space: Odor Maps', fontsize=16)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'pca_component_scatter.png')
    plt.savefig(output_path, dpi=300)
    print(f"PCA scatter plot saved to: {output_path}")
    
    # Show the plot
    plt.show()

def main():
    # Load activity maps and their corresponding CIDs
    masked_maps, cids, global_mask = load_maps_and_apply_mask()

    # Load molecular attributes (x) from build.py
    molecular_attributes = pd.read_csv('output_data/selected_features.csv', index_col=0)

    # Create a DataFrame for activity maps (y) with CIDs as the index
    activity_maps_df = pd.DataFrame({'Activity Map': masked_maps}, index=cids)
    print(f"Activity maps DataFrame shape: {activity_maps_df.shape}")
    print(f"Molecular attributes DataFrame shape: {molecular_attributes.shape}")

    # Align x and y based on CID
    common_cids = activity_maps_df.index.intersection(molecular_attributes.index)
    aligned_x = molecular_attributes.loc[common_cids]
    # Resave the molecular attributes CSV to ensure alignment
    aligned_x.to_csv('output_data/aligned_molecular_attributes.csv', index=True)
    aligned_y = activity_maps_df.loc[common_cids, 'Activity Map'].tolist()

    # Perform PCA on the aligned activity maps
    principal_components, pca = perform_pca(aligned_y)
    
    # Visualize the PCA components in a scatter plot
    visualize_pca_components_scatter(principal_components, common_cids)

    # Extract PCA loadings (for activity maps, not molecular attributes)
    flattened_feature_count = pca.components_.shape[1]
    print(f"Flattened feature count: {flattened_feature_count}")
    print(f"shape of PCA components: {pca.components_.shape}")
    loadings = pd.DataFrame(
        pca.components_.T,
        index=[f'Feature_{i+1}' for i in range(flattened_feature_count)],
        columns=[f'PC{i+1}' for i in range(pca.n_components_)]
    )
    print(loadings.head())

    # Visualize results
    visualize_results(pca, global_mask)
    visualize_spatial_loadings(pca, global_mask)

    # Save the PCA-transformed data to a CSV file
    pca_df = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])], index=common_cids)
    pca_df.index.name = 'CID'  # Add 'CID' as the index name
    pca_df.to_csv('output_data/pca_transformed_data.csv', index=True)

def temporary_main():
    masked_maps, cids, global_mask = load_maps_and_apply_mask()
    visualize_results(None, global_mask)

if __name__ == "__main__":
    main()
#temporary_main()
#temporary_main()
