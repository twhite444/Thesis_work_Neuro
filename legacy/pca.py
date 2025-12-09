import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pyrfume import load_data
from scipy.ndimage import gaussian_filter, binary_fill_holes, label, binary_dilation, binary_erosion

def load_maps_and_apply_mask(coverage_threshold=1.0):
    """Load activity maps, apply a global mask based on coverage threshold."""
    # Load directory data
    directory = pd.read_csv('output_data/behavior_data.csv')
    directory['CID'] = directory.index.map(lambda x: str(x).split('_')[0]).astype('int64')
    directory = directory[directory['CID'] > 0]
    selected_features = pd.read_csv('output_data/selected_features.csv')

    molecules_raw = pd.read_csv('output_data/molecules_raw.csv')
    selected_features['CID'] = molecules_raw['CID']
    selected_features.set_index('CID', inplace=True, drop=True)

    selected_cids = selected_features.index.unique()

    directory = directory[directory['CID'].isin(selected_cids)]
    
    all_maps = []
    valid_counts = None
    
    for idx, row in directory.iterrows():
        map_path = os.path.join('leon/', row['Activity Map Path'])
        activity_map = load_data(map_path).to_numpy()

        if valid_counts is None:
            valid_counts = np.zeros_like(activity_map, dtype=int)
        
        valid_counts += ~np.isnan(activity_map)
        all_maps.append(np.nan_to_num(activity_map, nan=0))  # Replace NaN with zero for PCA
    
    # Create global mask
    global_mask = valid_counts >= int(coverage_threshold * len(all_maps))
    smoothed_global_mask = binary_dilation(global_mask)
    smoothed_global_mask = binary_erosion(smoothed_global_mask)
    labeled_mask, num_features = label(smoothed_global_mask)
    region_sizes = np.bincount(labeled_mask.ravel())
    min_region_size = 100  # Minimum size threshold for regions
    valid_regions = np.isin(labeled_mask, np.where(region_sizes >= min_region_size)[0])
    refined_global_mask = smoothed_global_mask & valid_regions
    #refined_global_mask = gaussian_filter(global_mask.astype(float), sigma=1) > 0.5
    #refined_global_mask = binary_fill_holes(refined_global_mask)  # Fill holes to ensure continuity in the mask

    # Apply the global mask to each map
    masked_maps = [map * refined_global_mask for map in all_maps]
    
    return masked_maps, refined_global_mask

def perform_pca(masked_maps):
    """Perform PCA on flattened and masked activity maps."""
    # Flatten maps for PCA
    flat_maps = np.array([map_.flatten() for map_ in masked_maps])
    
    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(flat_maps)
    
    # Initialize PCA, choose min(n_samples, n_features)
    n_components = min(standardized_data.shape[0], standardized_data.shape[1], 3)
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(standardized_data)
    
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    return principal_components, pca

def visualize_pca_components(principal_components, pca, global_mask):
    """Visualize the first few principal components as images."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, ax in enumerate(axes):
        component = pca.components_[i].reshape(global_mask.shape)
        filtered_component = gaussian_filter(component, sigma=1)  # Applying a Gaussian filter for smoothing
        img = ax.imshow(filtered_component, cmap='viridis')
        ax.set_title(f'PCA Component {i+1}')
        fig.colorbar(img, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join('output_data', 'global_mask.png'))

    plt.show()

def visualize_and_save_mask(mask):
    """ Visualize and save the global mask as an image. """
    plt.figure(figsize=(10, 8))
    plt.imshow(mask, cmap='gray')
    plt.title('Refined Global Mask')
    plt.axis('off')
    plt.savefig(os.path.join('output_data', 'global_mask.png'))
    plt.show()

def main():
    masked_maps, global_mask = load_maps_and_apply_mask()
    principal_components, pca = perform_pca(masked_maps)
    visualize_pca_components(principal_components, pca, global_mask)
    # Save the PCA transformed data to a CSV file
    pca_df = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])
    pca_df.to_csv('pca_transformed_data.csv', index=False)

def temporary_main():
    masked_maps, global_mask = load_maps_and_apply_mask()
    visualize_and_save_mask(global_mask)

if __name__ == "__main__":
    main()
    #temporary_main()
