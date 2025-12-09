import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

# Set the output directory
output_dir = 'output_data'

def visualize_pca_components(dimensions=2, components=None, label_points=10):
    """
    Create a visualization of principal components in component space
    based on the PCA transformed data.
    
    Parameters:
    -----------
    dimensions : int
        Number of dimensions for visualization (1, 2, or 3)
    components : list of int
        Which components to visualize (e.g., [0, 1, 2] for first three components)
        If None, uses the first N components where N=dimensions
    label_points : int
        Number of points to label in the visualization
    """
    # Load the PCA transformed data
    pca_data_path = os.path.join(output_dir, 'pca_transformed_data.csv')
    pca_data = pd.read_csv(pca_data_path, index_col=0)
    
    # Validate dimensions
    if dimensions not in [1, 2, 3]:
        print("Dimensions must be 1, 2, or 3. Defaulting to 2.")
        dimensions = 2
    
    # Set up components if not provided
    if components is None:
        components = list(range(dimensions))
    elif len(components) != dimensions:
        print(f"Number of components ({len(components)}) doesn't match dimensions ({dimensions}). Using first {dimensions} components.")
        components = list(range(dimensions))
    
    # Extract the components
    pc_data = []
    for comp in components:
        if comp < pca_data.shape[1]:
            pc_data.append(pca_data.iloc[:, comp])
        else:
            print(f"Component {comp} is out of range. Using component 0 instead.")
            pc_data.append(pca_data.iloc[:, 0])
    
    # Create visualization based on dimensions
    if dimensions == 1:
        create_1d_visualization(pc_data, pca_data.index, components, label_points)
    elif dimensions == 2:
        create_2d_visualization(pc_data, pca_data.index, components, label_points)
    else:  # dimensions == 3
        create_3d_visualization(pc_data, pca_data.index, components, label_points)

def create_1d_visualization(pc_data, indices, components, label_points):
    """Create a 1D visualization of a PCA component"""
    plt.figure(figsize=(12, 6))
    
    # Create vertical line plot
    y = np.random.normal(0, 0.1, len(pc_data[0]))  # Small random y offset for visibility
    plt.scatter(pc_data[0], y, alpha=0.7, edgecolors='w', linewidth=0.5)
    
    # Label points
    for i in range(min(label_points, len(indices))):
        plt.annotate(str(indices[i]), (pc_data[0][i], y[i]), fontsize=8)
    
    # Set labels and title
    plt.xlabel(f'Principal Component {components[0]+1}', fontsize=12)
    plt.yticks([])  # Hide y-axis ticks
    plt.title(f'PCA Component {components[0]+1} Distribution', fontsize=14)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f'pca_component_{components[0]+1}_1d.png')
    plt.savefig(output_path, dpi=300)
    print(f"1D PCA visualization saved to: {output_path}")
    
    # Show the plot
    plt.show()

def create_2d_visualization(pc_data, indices, components, label_points):
    """Create a 2D visualization of PCA components"""
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    scatter = plt.scatter(pc_data[0], pc_data[1], alpha=0.7, edgecolors='w', linewidth=0.5)
    
    # Add labels for some points
    for i in range(min(label_points, len(indices))):
        plt.annotate(str(indices[i]), (pc_data[0][i], pc_data[1][i]), fontsize=8)
    
    # Set labels and title
    plt.xlabel(f'Principal Component {components[0]+1}', fontsize=12)
    plt.ylabel(f'Principal Component {components[1]+1}', fontsize=12)
    plt.title(f'PCA Components: PC{components[0]+1} vs PC{components[1]+1}', fontsize=14)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f'pca_components_{components[0]+1}_{components[1]+1}_2d.png')
    plt.savefig(output_path, dpi=300)
    print(f"2D PCA visualization saved to: {output_path}")
    
    # Show the plot
    plt.show()

def create_3d_visualization(pc_data, indices, components, label_points):
    """Create a 3D visualization of PCA components"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create 3D scatter plot
    scatter = ax.scatter(pc_data[0], pc_data[1], pc_data[2], alpha=0.7, edgecolors='w', linewidth=0.5)
    
    # Add labels for some points
    for i in range(min(label_points, len(indices))):
        ax.text(pc_data[0][i], pc_data[1][i], pc_data[2][i], str(indices[i]), fontsize=8)
    
    # Set labels and title
    ax.set_xlabel(f'Principal Component {components[0]+1}', fontsize=12)
    ax.set_ylabel(f'Principal Component {components[1]+1}', fontsize=12)
    ax.set_zlabel(f'Principal Component {components[2]+1}', fontsize=12)
    plt.title(f'PCA Components: PC{components[0]+1} vs PC{components[1]+1} vs PC{components[2]+1}', fontsize=14)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f'pca_components_{components[0]+1}_{components[1]+1}_{components[2]+1}_3d.png')
    plt.savefig(output_path, dpi=300)
    print(f"3D PCA visualization saved to: {output_path}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize PCA components')
    parser.add_argument('--dimensions', type=int, default=2, choices=[1, 2, 3],
                        help='Number of dimensions for visualization (1, 2, or 3)')
    parser.add_argument('--components', type=int, nargs='+',
                        help='Which components to visualize (e.g., 0 1 2 for first three components, zero-indexed)')
    parser.add_argument('--label_points', type=int, default=10,
                        help='Number of points to label in the visualization')
    
    args = parser.parse_args()
    
    # Convert components to 0-indexed if provided
    components = args.components
    if components:
        components = [max(0, c) for c in components]  # Ensure all components are >= 0
    
    visualize_pca_components(args.dimensions, components, args.label_points)