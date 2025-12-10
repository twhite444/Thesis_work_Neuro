"""Molecular graph visualization utilities.

This module provides functions to visualize molecular graphs as images,
similar to the activity map visualizations. Useful for inspecting graph
structures and understanding the featurization.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, List
import os

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Install with: conda install -c conda-forge rdkit")

from .molecular_graphs import load_graph_data, get_graph_by_cid


def draw_molecule_from_smiles(
    smiles: str,
    title: Optional[str] = None,
    size: tuple = (400, 400),
    show_atom_indices: bool = False
) -> Optional[object]:
    """Draw a molecule from SMILES string.
    
    Args:
        smiles: SMILES string
        title: Optional title for the image
        size: Image size (width, height)
        show_atom_indices: Whether to show atom indices
        
    Returns:
        PIL Image or None if invalid SMILES
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for molecule visualization")
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Add explicit hydrogens for complete structure
    mol = Chem.AddHs(mol)
    
    # Generate 2D coordinates
    AllChem.Compute2DCoords(mol)
    
    # Draw molecule
    if show_atom_indices:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
    
    img = Draw.MolToImage(mol, size=size)
    
    return img


def visualize_molecular_graph(
    cid: int,
    graph_data: dict,
    molecules_df=None,
    output_path: Optional[Union[str, Path]] = None,
    show_atom_indices: bool = False,
    figsize: tuple = (12, 8)
) -> None:
    """Visualize a molecular graph with structure and statistics.
    
    Creates a visualization showing:
    - 2D molecular structure
    - Graph statistics (nodes, edges, features)
    - Feature summary
    
    Args:
        cid: Compound ID
        graph_data: Dictionary containing graph data from load_graph_data
        molecules_df: Optional DataFrame with molecule info (SMILES, name, etc.)
        output_path: Path to save the visualization (PNG)
        show_atom_indices: Whether to show atom indices on structure
        figsize: Figure size (width, height)
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for molecule visualization")
    
    # Get graph for this CID
    graph = get_graph_by_cid(cid, graph_data)
    if graph is None:
        print(f"No graph found for CID {cid}")
        return
    
    # Get SMILES if molecules_df provided
    smiles = None
    if molecules_df is not None:
        mol_row = molecules_df[molecules_df['CID'] == cid]
        if len(mol_row) > 0:
            smiles = mol_row.iloc[0].get('IsomericSMILES', mol_row.iloc[0].get('SMILES', None))
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Layout: 2 columns
    # Left: molecule structure (2/3 width)
    # Right: statistics (1/3 width)
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[3, 1])
    
    # Molecule structure
    ax_mol = fig.add_subplot(gs[0, 0])
    if smiles:
        img = draw_molecule_from_smiles(smiles, size=(600, 600), show_atom_indices=show_atom_indices)
        if img:
            ax_mol.imshow(img)
            ax_mol.axis('off')
            ax_mol.set_title(f'Molecular Structure (CID {cid})', fontsize=14, fontweight='bold')
        else:
            ax_mol.text(0.5, 0.5, 'Invalid SMILES', ha='center', va='center')
            ax_mol.axis('off')
    else:
        ax_mol.text(0.5, 0.5, 'No SMILES available', ha='center', va='center')
        ax_mol.axis('off')
    
    # Graph statistics
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_stats.axis('off')
    
    stats_text = f"""
Graph Statistics
================

Nodes: {graph['num_nodes']}
Edges: {graph['num_edges']}

Node Features:
  Dimensions: {graph['node_features'].shape[1]}
  Shape: {graph['node_features'].shape}

Edge Index:
  Shape: {graph['edge_index'].shape}
  (bidirectional)
"""
    
    if 'edge_attr' in graph:
        stats_text += f"""
Edge Features:
  Dimensions: {graph['edge_attr'].shape[1]}
  Shape: {graph['edge_attr'].shape}
"""
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Node feature heatmap (sample)
    ax_features = fig.add_subplot(gs[1, :])
    
    # Show first few atoms' features as heatmap
    num_atoms_to_show = min(10, graph['num_nodes'])
    sample_features = graph['node_features'][:num_atoms_to_show, :]
    
    im = ax_features.imshow(sample_features.T, aspect='auto', cmap='viridis')
    ax_features.set_xlabel(f'Atoms (showing first {num_atoms_to_show})', fontsize=10)
    ax_features.set_ylabel('Feature Dims (137)', fontsize=10)
    ax_features.set_title('Node Feature Matrix (sample)', fontsize=12)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax_features, orientation='horizontal', pad=0.1)
    cbar.set_label('Feature Value', fontsize=9)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_multiple_graphs(
    cids: List[int],
    graph_data: dict,
    molecules_df=None,
    output_dir: Optional[Union[str, Path]] = None,
    max_per_row: int = 3
) -> None:
    """Visualize multiple molecular graphs in a grid layout.
    
    Args:
        cids: List of compound IDs to visualize
        graph_data: Dictionary containing graph data from load_graph_data
        molecules_df: Optional DataFrame with molecule info
        output_dir: Directory to save individual visualizations
        max_per_row: Maximum molecules per row in grid
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for molecule visualization")
    
    # Calculate grid dimensions
    num_mols = len(cids)
    n_cols = min(max_per_row, num_mols)
    n_rows = (num_mols + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if num_mols == 1:
        axes = np.array([axes])
    axes = axes.flatten() if num_mols > 1 else axes
    
    for idx, cid in enumerate(cids):
        ax = axes[idx] if num_mols > 1 else axes
        
        # Get SMILES
        smiles = None
        if molecules_df is not None:
            mol_row = molecules_df[molecules_df['CID'] == cid]
            if len(mol_row) > 0:
                smiles = mol_row.iloc[0].get('IsomericSMILES', mol_row.iloc[0].get('SMILES', None))
        
        # Get graph stats
        graph = get_graph_by_cid(cid, graph_data)
        
        if smiles and graph:
            img = draw_molecule_from_smiles(smiles, size=(300, 300))
            if img:
                ax.imshow(img)
                ax.set_title(f'CID {cid}\n{graph["num_nodes"]} atoms, {graph["num_edges"]} bonds',
                           fontsize=10)
            else:
                ax.text(0.5, 0.5, f'CID {cid}\nInvalid SMILES', ha='center', va='center')
        else:
            ax.text(0.5, 0.5, f'CID {cid}\nNo data', ha='center', va='center')
        
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_mols, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save or show
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = Path(output_dir) / 'molecular_graphs_gallery.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved gallery to {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Also save individual visualizations if output_dir provided
    if output_dir:
        for cid in cids:
            output_path = Path(output_dir) / f'molecular_graph_CID_{cid}.png'
            visualize_molecular_graph(cid, graph_data, molecules_df, output_path)


def compare_molecule_and_graph(
    cid: int,
    graph_data: dict,
    molecules_df=None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (14, 6)
) -> None:
    """Create side-by-side comparison of molecule structure and graph representation.
    
    Args:
        cid: Compound ID
        graph_data: Dictionary containing graph data
        molecules_df: Optional DataFrame with molecule info
        output_path: Path to save the visualization
        figsize: Figure size
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for molecule visualization")
    
    # Get data
    graph = get_graph_by_cid(cid, graph_data)
    if graph is None:
        print(f"No graph found for CID {cid}")
        return
    
    smiles = None
    if molecules_df is not None:
        mol_row = molecules_df[molecules_df['CID'] == cid]
        if len(mol_row) > 0:
            smiles = mol_row.iloc[0].get('IsomericSMILES', mol_row.iloc[0].get('SMILES', None))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left: molecule with atom indices
    if smiles:
        img = draw_molecule_from_smiles(smiles, size=(500, 500), show_atom_indices=True)
        if img:
            ax1.imshow(img)
            ax1.set_title(f'Molecular Structure with Atom Indices\nCID {cid}', fontsize=12, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'Invalid SMILES', ha='center', va='center')
    else:
        ax1.text(0.5, 0.5, 'No SMILES available', ha='center', va='center')
    ax1.axis('off')
    
    # Right: graph connectivity visualization
    # Create adjacency matrix from edge_index
    edge_index = graph['edge_index']
    num_nodes = graph['num_nodes']
    
    # Build adjacency matrix
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        adj_matrix[src, dst] = 1
    
    # Plot adjacency matrix
    im = ax2.imshow(adj_matrix, cmap='Blues', interpolation='nearest')
    ax2.set_title(f'Graph Connectivity Matrix\n{num_nodes} nodes, {graph["num_edges"]} edges', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Atom Index', fontsize=10)
    ax2.set_ylabel('Atom Index', fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Connected', fontsize=9)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_graph_summary(graph_data: dict) -> None:
    """Print summary statistics for all graphs in dataset.
    
    Args:
        graph_data: Dictionary containing graph data from load_graph_data
    """
    print("=" * 70)
    print("Molecular Graph Dataset Summary")
    print("=" * 70)
    print()
    
    num_molecules = len(graph_data['cids'])
    print(f"Total molecules: {num_molecules}")
    print(f"Valid molecules: {graph_data['valid_mask'].sum()}")
    print()
    
    # Node statistics
    num_nodes = graph_data['num_nodes']
    print(f"Nodes (atoms) per molecule:")
    print(f"  Mean: {num_nodes.mean():.1f} ± {num_nodes.std():.1f}")
    print(f"  Range: [{num_nodes.min()}, {num_nodes.max()}]")
    print(f"  Total: {num_nodes.sum():,}")
    print()
    
    # Edge statistics
    num_edges = graph_data['num_edges']
    print(f"Edges (bonds) per molecule:")
    print(f"  Mean: {num_edges.mean():.1f} ± {num_edges.std():.1f}")
    print(f"  Range: [{num_edges.min()}, {num_edges.max()}]")
    print(f"  Total: {num_edges.sum():,}")
    print()
    
    # Feature dimensions
    sample_idx = 0
    sample_features = graph_data['node_features_list'][sample_idx]
    print(f"Node feature dimensions: {sample_features.shape[1]}")
    
    if 'edge_attr_list' in graph_data:
        sample_edge_features = graph_data['edge_attr_list'][sample_idx]
        print(f"Edge feature dimensions: {sample_edge_features.shape[1]}")
    else:
        print(f"Edge features: Not included")
    
    print()
    print("=" * 70)
