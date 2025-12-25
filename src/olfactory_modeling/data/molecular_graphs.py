"""Molecular graph featurization for GNN models.

This module converts SMILES strings to PyTorch Geometric compatible graph data
with node features, edge indices, and optional edge features using RDKit.

Features:
- Atom-level node features (atomic number, degree, hybridization, etc.)
- Bond-level edge features (bond type, conjugation, ring membership)
- Compatible with PyTorch Geometric Data format
- Batch processing for entire molecule datasets
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from rdkit import Chem
import warnings


# ============================================================================
# Atom Feature Extraction
# ============================================================================

def get_atom_features(atom: Chem.Atom) -> np.ndarray:
    """Extract feature vector for a single atom.
    
    Features (total: 137 dimensions):
    - Atomic number (one-hot, 100 elements): 100 dims
    - Degree (one-hot, 0-10): 11 dims
    - Formal charge (one-hot, -2 to +2): 5 dims
    - Hybridization (one-hot, SP/SP2/SP3/SP3D/SP3D2): 5 dims
    - Aromaticity (binary): 1 dim
    - Total number of Hs (one-hot, 0-4): 5 dims
    - Radical electrons (one-hot, 0-4): 5 dims
    - In ring (binary): 1 dim
    - Chirality (one-hot, 4 types): 4 dims
    
    Args:
        atom: RDKit Atom object
        
    Returns:
        Feature vector as numpy array (137,)
    """
    features = []
    
    # Atomic number (one-hot encoding for elements 1-100)
    atomic_num = atom.GetAtomicNum()
    atomic_features = [0] * 100
    if 1 <= atomic_num <= 100:
        atomic_features[atomic_num - 1] = 1
    features.extend(atomic_features)
    
    # Degree (number of bonded neighbors)
    degree = atom.GetDegree()
    degree_features = [0] * 11  # 0 to 10
    if degree <= 10:
        degree_features[degree] = 1
    features.extend(degree_features)
    
    # Formal charge
    charge = atom.GetFormalCharge()
    charge_features = [0] * 5  # -2, -1, 0, +1, +2
    charge_idx = charge + 2  # Map to 0-4
    if 0 <= charge_idx <= 4:
        charge_features[charge_idx] = 1
    features.extend(charge_features)
    
    # Hybridization
    hybridization = atom.GetHybridization()
    hyb_features = [0] * 5
    hyb_types = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]
    if hybridization in hyb_types:
        hyb_features[hyb_types.index(hybridization)] = 1
    features.extend(hyb_features)
    
    # Aromaticity
    features.append(int(atom.GetIsAromatic()))
    
    # Total number of Hs
    num_hs = atom.GetTotalNumHs()
    hs_features = [0] * 5  # 0 to 4+
    if num_hs <= 4:
        hs_features[num_hs] = 1
    else:
        hs_features[4] = 1
    features.extend(hs_features)
    
    # Radical electrons
    num_radical = atom.GetNumRadicalElectrons()
    radical_features = [0] * 5  # 0 to 4+
    if num_radical <= 4:
        radical_features[num_radical] = 1
    else:
        radical_features[4] = 1
    features.extend(radical_features)
    
    # In ring
    features.append(int(atom.IsInRing()))
    
    # Chirality
    chirality = atom.GetChiralTag()
    chiral_features = [0] * 4
    chiral_types = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ]
    if chirality in chiral_types:
        chiral_features[chiral_types.index(chirality)] = 1
    features.extend(chiral_features)
    
    return np.array(features, dtype=np.float32)


def get_bond_features(bond: Chem.Bond) -> np.ndarray:
    """Extract feature vector for a single bond.
    
    Features (total: 10 dimensions):
    - Bond type (one-hot, SINGLE/DOUBLE/TRIPLE/AROMATIC): 4 dims
    - Conjugation (binary): 1 dim
    - In ring (binary): 1 dim
    - Stereo (one-hot, 4 types): 4 dims
    
    Args:
        bond: RDKit Bond object
        
    Returns:
        Feature vector as numpy array (10,)
    """
    features = []
    
    # Bond type
    bond_type = bond.GetBondType()
    type_features = [0] * 4
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]
    if bond_type in bond_types:
        type_features[bond_types.index(bond_type)] = 1
    features.extend(type_features)
    
    # Conjugation
    features.append(int(bond.GetIsConjugated()))
    
    # In ring
    features.append(int(bond.IsInRing()))
    
    # Stereo
    stereo = bond.GetStereo()
    stereo_features = [0] * 4
    stereo_types = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOANY
    ]
    if stereo in stereo_types:
        stereo_features[stereo_types.index(stereo)] = 1
    features.extend(stereo_features)
    
    return np.array(features, dtype=np.float32)


# ============================================================================
# Molecule to Graph Conversion
# ============================================================================

def smiles_to_graph(smiles: str, include_edge_features: bool = True) -> Optional[Dict[str, np.ndarray]]:
    """Convert SMILES string to graph representation.
    
    Args:
        smiles: SMILES string representation of molecule
        include_edge_features: If True, include edge feature matrix
        
    Returns:
        Dictionary with keys:
        - 'node_features': Node feature matrix (num_atoms, 137)
        - 'edge_index': Edge indices (2, num_bonds*2) - undirected graph
        - 'edge_attr': Edge feature matrix (num_bonds*2, 10) [if include_edge_features=True]
        - 'num_nodes': Number of atoms
        - 'num_edges': Number of bonds (undirected)
        Returns None if SMILES is invalid
    """
    try:
        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Add explicit hydrogens for better feature representation
        mol = Chem.AddHs(mol)
        
        # Extract node features (atom features)
        node_features = []
        for atom in mol.GetAtoms():
            atom_features = get_atom_features(atom)
            node_features.append(atom_features)
        node_features = np.array(node_features, dtype=np.float32)
        
        # Extract edge indices and features
        edge_index = []
        edge_attr = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Add both directions for undirected graph
            edge_index.append([i, j])
            edge_index.append([j, i])
            
            if include_edge_features:
                bond_features = get_bond_features(bond)
                edge_attr.append(bond_features)
                edge_attr.append(bond_features)  # Same features for both directions
        
        if edge_index.size == 0:  # NumPy: explicit .size check for clarity
            # Molecule has no bonds (single atom)
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, 10), dtype=np.float32)
        else:
            edge_index = np.array(edge_index, dtype=np.int64).T  # Shape: (2, num_edges)
            if include_edge_features:
                edge_attr = np.array(edge_attr, dtype=np.float32)
        
        result = {
            'node_features': node_features,
            'edge_index': edge_index,
            'num_nodes': node_features.shape[0],
            'num_edges': edge_index.shape[1] // 2 if edge_index.shape[1] > 0 else 0
        }
        
        if include_edge_features:
            result['edge_attr'] = edge_attr
        
        return result
        
    except Exception as e:
        warnings.warn(f"Failed to process SMILES '{smiles}': {str(e)}")
        return None


# ============================================================================
# Batch Processing
# ============================================================================

def molecules_to_graphs(
    molecules_df: pd.DataFrame,
    smiles_col: str = 'IsomericSMILES',
    cid_col: str = 'CID',
    include_edge_features: bool = True,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """Convert multiple molecules to graph format.
    
    Args:
        molecules_df: DataFrame with SMILES strings
        smiles_col: Column name for SMILES strings
        cid_col: Column name for compound IDs
        include_edge_features: If True, include edge features
        verbose: If True, print progress
        
    Returns:
        Dictionary suitable for saving as NPZ with keys:
        - 'node_features_list': List of node feature matrices
        - 'edge_index_list': List of edge index arrays
        - 'edge_attr_list': List of edge attribute arrays [if include_edge_features=True]
        - 'cids': Array of CIDs
        - 'num_nodes': Array of node counts per molecule
        - 'num_edges': Array of edge counts per molecule
        - 'valid_mask': Boolean mask of successfully converted molecules
    """
    if verbose:
        print(f"Converting {len(molecules_df)} molecules to graphs...")
    
    node_features_list = []
    edge_index_list = []
    edge_attr_list = []
    cids = []
    num_nodes_list = []
    num_edges_list = []
    valid_mask = []
    
    for idx, row in molecules_df.iterrows():
        smiles = row[smiles_col]
        cid = row[cid_col]
        
        graph = smiles_to_graph(smiles, include_edge_features=include_edge_features)
        
        if graph is not None:
            node_features_list.append(graph['node_features'])
            edge_index_list.append(graph['edge_index'])
            if include_edge_features:
                edge_attr_list.append(graph['edge_attr'])
            cids.append(cid)
            num_nodes_list.append(graph['num_nodes'])
            num_edges_list.append(graph['num_edges'])
            valid_mask.append(True)
        else:
            valid_mask.append(False)
            if verbose:
                print(f"  Warning: Failed to convert CID {cid}")
    
    if verbose:
        success_rate = sum(valid_mask) / len(valid_mask) * 100
        print(f"Successfully converted {sum(valid_mask)}/{len(valid_mask)} molecules ({success_rate:.1f}%)")
        print(f"Average nodes per molecule: {np.mean(num_nodes_list):.1f}")
        print(f"Average edges per molecule: {np.mean(num_edges_list):.1f}")
    
    # Create object arrays properly by pre-allocating
    node_features_arr = np.empty(len(node_features_list), dtype=object)
    for i, arr in enumerate(node_features_list):
        node_features_arr[i] = arr
    
    edge_index_arr = np.empty(len(edge_index_list), dtype=object)
    for i, arr in enumerate(edge_index_list):
        edge_index_arr[i] = arr
    
    result = {
        'cids': np.array(cids, dtype=np.int64),
        'num_nodes': np.array(num_nodes_list, dtype=np.int64),
        'num_edges': np.array(num_edges_list, dtype=np.int64),
        'valid_mask': np.array(valid_mask, dtype=bool),
        'node_features_list': node_features_arr,
        'edge_index_list': edge_index_arr,
    }
    
    if include_edge_features:
        edge_attr_arr = np.empty(len(edge_attr_list), dtype=object)
        for i, arr in enumerate(edge_attr_list):
            edge_attr_arr[i] = arr
        result['edge_attr_list'] = edge_attr_arr
    
    return result


# ============================================================================
# Helper Functions for Loading
# ============================================================================

def load_graph_data(data_dir: str = "data/01_raw") -> Dict[str, Any]:
    """Load pre-computed molecular graph data from NPZ file.
    
    Args:
        data_dir: Directory containing molecular_graphs.npz
        
    Returns:
        Dictionary with graph data for all molecules
    """
    import os
    npz_path = os.path.join(data_dir, 'molecular_graphs.npz')
    data = np.load(npz_path, allow_pickle=True)
    
    return {
        'cids': data['cids'],
        'num_nodes': data['num_nodes'],
        'num_edges': data['num_edges'],
        'valid_mask': data['valid_mask'],
        'node_features_list': data['node_features_list'],
        'edge_index_list': data['edge_index_list'],
        'edge_attr_list': data.get('edge_attr_list', None)
    }


def get_graph_by_cid(cid: int, graph_data: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
    """Retrieve graph data for a specific CID.
    
    Args:
        cid: Compound ID
        graph_data: Dictionary from load_graph_data()
        
    Returns:
        Dictionary with node_features, edge_index, edge_attr (if available)
        Returns None if CID not found
    """
    cids = graph_data['cids']
    mask = cids == cid
    
    if not mask.any():
        return None
    
    idx = np.where(mask)[0][0]
    
    result = {
        'node_features': graph_data['node_features_list'][idx],
        'edge_index': graph_data['edge_index_list'][idx],
        'num_nodes': graph_data['num_nodes'][idx],
        'num_edges': graph_data['num_edges'][idx],
    }
    
    if graph_data['edge_attr_list'] is not None:
        result['edge_attr'] = graph_data['edge_attr_list'][idx]
    
    return result


def graph_statistics(graph_data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute statistics about the molecular graphs.
    
    Args:
        graph_data: Dictionary from load_graph_data()
        
    Returns:
        Dictionary with statistics
    """
    num_mols = len(graph_data['cids'])
    num_nodes = graph_data['num_nodes']
    num_edges = graph_data['num_edges']
    
    return {
        'total_molecules': num_mols,
        'total_atoms': int(num_nodes.sum()),
        'total_bonds': int(num_edges.sum()),
        'avg_atoms_per_molecule': float(num_nodes.mean()),
        'std_atoms_per_molecule': float(num_nodes.std()),
        'min_atoms': int(num_nodes.min()),
        'max_atoms': int(num_nodes.max()),
        'avg_bonds_per_molecule': float(num_edges.mean()),
        'std_bonds_per_molecule': float(num_edges.std()),
        'min_bonds': int(num_edges.min()),
        'max_bonds': int(num_edges.max()),
        'node_feature_dim': graph_data['node_features_list'][0].shape[1] if len(graph_data['node_features_list']) > 0 else 0,
        'edge_feature_dim': graph_data['edge_attr_list'][0].shape[1] if graph_data['edge_attr_list'] is not None and len(graph_data['edge_attr_list']) > 0 else 0
    }


# ============================================================================
# High-Level Pipeline Function
# ============================================================================

def generate_and_save_molecular_graphs(
    data_dir: str = 'data/01_raw',
    output_dir: str = 'data/01_raw',
    smiles_col: str = 'IsomericSMILES',
    cid_col: str = 'CID',
    include_edge_features: bool = True,
    verbose: bool = True
) -> dict:
    """High-level function to generate and save molecular graphs.
    
    This function loads molecules, converts them to graphs, saves to NPZ,
    and returns statistics. Designed for use in scripts.
    
    Args:
        data_dir: Directory containing molecules_raw.npz
        output_dir: Directory to save molecular_graphs.npz
        smiles_col: Column name for SMILES strings
        cid_col: Column name for compound IDs
        include_edge_features: Whether to include edge features
        verbose: Whether to print progress messages
        
    Returns:
        Dictionary containing graph statistics
        
    Example:
        >>> stats = generate_and_save_molecular_graphs(
        ...     data_dir='data/01_raw',
        ...     output_dir='data/01_raw',
        ...     include_edge_features=True
        ... )
        >>> print(f"Generated graphs for {stats['total_molecules']} molecules")
    """
    import os
    from .pyrfume_loader import load_molecules_npz
    
    if verbose:
        print("=" * 70)
        print("Molecular Graph Data Generation")
        print("=" * 70)
        print()
    
    # Load molecules
    if verbose:
        print(f"Loading molecules from {data_dir}/molecules_raw.npz...")
    molecules = load_molecules_npz(data_dir=data_dir)
    if verbose:
        print(f"Loaded {len(molecules)} molecules")
        print()
    
    # Convert to graphs
    if verbose:
        print("Converting molecules to graphs...")
        print(f"  SMILES column: {smiles_col}")
        print(f"  CID column: {cid_col}")
        print(f"  Include edge features: {include_edge_features}")
        print()
    
    graph_data = molecules_to_graphs(
        molecules,
        smiles_col=smiles_col,
        cid_col=cid_col,
        include_edge_features=include_edge_features,
        verbose=verbose
    )
    print()
    
    # Save to NPZ
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'molecular_graphs.npz')
    
    if verbose:
        print(f"Saving graph data to {output_path}...")
    np.savez_compressed(output_path, **graph_data)
    
    file_size_mb = os.path.getsize(output_path) / (1024 ** 2)
    if verbose:
        print(f"Saved! File size: {file_size_mb:.2f} MB")
        print()
    
    # Display statistics
    if verbose:
        print("=" * 70)
        print("Graph Data Statistics")
        print("=" * 70)
    
    stats = graph_statistics(graph_data)
    
    if verbose:
        print("\nMolecules:")
        print(f"  Total molecules: {stats['total_molecules']}")
        print(f"  Total atoms: {stats['total_atoms']:,}")
        print(f"  Total bonds: {stats['total_bonds']:,}")
        
        print("\nAtoms per molecule:")
        print(f"  Mean: {stats['avg_atoms_per_molecule']:.1f} ± {stats['std_atoms_per_molecule']:.1f}")
        print(f"  Range: [{stats['min_atoms']}, {stats['max_atoms']}]")
        
        print("\nBonds per molecule:")
        print(f"  Mean: {stats['avg_bonds_per_molecule']:.1f} ± {stats['std_bonds_per_molecule']:.1f}")
        print(f"  Range: [{stats['min_bonds']}, {stats['max_bonds']}]")
        
        print("\nFeature dimensions:")
        print(f"  Node features: {stats['node_feature_dim']} dims")
        if stats['edge_feature_dim'] > 0:
            print(f"  Edge features: {stats['edge_feature_dim']} dims")
        else:
            print("  Edge features: Not included")
        
        print()
        print("=" * 70)
        print("✓ Graph data generation complete!")
        print("=" * 70)
        print()
    
    return stats


# ============================================================================
# Helper Function for Interactive Exploration
# ============================================================================

def load_graph_by_cid(
    cid: int,
    data_dir: str = "data/01_raw",
    show_image: bool = False,
    save_image: bool = False,
    mode: str = 'detailed',
    projection: str = '2d'
) -> Optional[Dict[str, np.ndarray]]:
    """Load and optionally visualize molecular graph for a specific CID.
    
    This is a helper function similar to activity map loaders - combines
    data loading with optional visualization.
    
    Args:
        cid: Compound ID to load
        data_dir: Directory containing molecular_graphs.npz and molecules_raw.npz
        show_image: If True, display interactive visualization window
        save_image: If True, save visualization to file (only works if show_image is True)
        mode: 'simple' (just molecule) or 'detailed' (molecule + comprehensive info)
        projection: '2d' (2D layout) or '3d' (3D conformer with optimized geometry)
        
    Returns:
        Dictionary with graph data (node_features, edge_index, etc.) or None if not found
        
    Example:
        >>> # Just load the graph data
        >>> graph = load_graph_by_cid(180)
        >>> print(f"Nodes: {graph['num_nodes']}, Edges: {graph['num_edges']}")
        
        >>> # Load and display (simple mode, 2D)
        >>> graph = load_graph_by_cid(180, show_image=True, mode='simple')
        
        >>> # Load and display (simple mode, 3D)
        >>> graph = load_graph_by_cid(180, show_image=True, mode='simple', projection='3d')
        
        >>> # Load, display detailed, and save
        >>> graph = load_graph_by_cid(180, show_image=True, save_image=True, mode='detailed')
    """
    import os
    
    # Load graph data
    graph_data = load_graph_data(data_dir)
    graph = get_graph_by_cid(cid, graph_data)
    
    if graph is None:
        print(f"No graph found for CID {cid}")
        return None
    
    # Print basic info
    print(f"\nCID {cid} Molecular Graph:")
    print(f"  Nodes (atoms): {graph['num_nodes']}")
    print(f"  Edges (bonds): {graph['num_edges']}")
    print(f"  Node features: {graph['node_features'].shape}")
    if 'edge_attr' in graph:
        print(f"  Edge features: {graph['edge_attr'].shape}")
    print()
    
    # Visualize if requested
    if show_image:
        from .graph_viz import visualize_molecular_graph
        from .pyrfume_loader import load_molecules_npz
        
        # Load molecule data for SMILES
        molecules = load_molecules_npz(data_dir)
        
        save_path = None
        if save_image:
            # Save to viz/molecules/ directory
            viz_dir = os.path.join('viz', 'molecules')
            os.makedirs(viz_dir, exist_ok=True)
            save_path = os.path.join(viz_dir, f'CID_{cid}.png')
        
        visualize_molecular_graph(
            cid,
            graph_data,
            molecules,
            save_path=save_path,
            show=True,
            mode=mode,
            projection=projection
        )
        
        if save_path:
            print(f"💡 Visualization saved to: {save_path}")
        else:
            print("💡 Close the visualization window to continue...")
    
    return graph


def visualize_molecule_interactive(
    cid: int,
    data_dir: str = 'data/01_raw',
    output_path: Optional[str] = None,
    width: int = 800,
    height: int = 600,
    style: str = 'stick',
    open_browser: bool = True
):
    """Create an interactive 3D molecular visualization in your web browser.
    
    This function creates an HTML file with an interactive py3Dmol viewer that
    you can rotate, zoom, and explore in 3D.
    
    Args:
        cid: PubChem CID of the molecule
        data_dir: Directory containing molecules.npz
        output_path: Optional path to save HTML file. If None, saves to data_dir
        width: Viewer width in pixels (default: 800)
        height: Viewer height in pixels (default: 600)
        style: Visualization style - 'stick', 'sphere', 'cartoon', 'line', 'cross' (default: 'stick')
        open_browser: Whether to automatically open in browser (default: True)
        
    Returns:
        Path to the generated HTML file, or None if failed
        
    Example:
        >>> # Open vanillin in interactive 3D viewer
        >>> visualize_molecule_interactive(1183)
        
        >>> # Save limonene as sphere style
        >>> visualize_molecule_interactive(440917, style='sphere')
    """
    import os
    from .graph_viz import visualize_molecule_3d_interactive
    from .pyrfume_loader import load_molecules_npz
    
    # Load molecule data for SMILES
    molecules = load_molecules_npz(data_dir)
    
    # Find the molecule
    mol_idx = np.where(molecules['CID'] == cid)[0]
    if not mol_idx.size:  # NumPy array
        print(f"❌ No molecule found for CID {cid}")
        return None
    
    smiles = molecules['IsomericSMILES'][mol_idx[0]]
    mol_name = molecules.get('name', np.array([None] * len(molecules['CID'])))[mol_idx[0]]
    
    # Set output path to viz/molecules/ directory
    if output_path is None:
        viz_dir = os.path.join('viz', 'molecules')
        os.makedirs(viz_dir, exist_ok=True)
        output_path = os.path.join(viz_dir, f'CID_{cid}.html')
    
    # Create interactive visualization
    print(f"\n🌐 Creating interactive 3D viewer for CID {cid}")
    if mol_name:
        print(f"   Molecule: {mol_name}")
    print(f"   Style: {style}")
    
    success = visualize_molecule_3d_interactive(
        smiles,
        output_path,
        width=width,
        height=height,
        style=style
    )
    
    if success:
        print(f"✅ Interactive viewer saved to: {output_path}")
        
        # Open in browser if requested
        if open_browser:
            import webbrowser
            webbrowser.open(f'file://{os.path.abspath(output_path)}')
            print("🌐 Opening in your default web browser...")
        
        return output_path
    else:
        print("❌ Failed to create interactive visualization")
        return None
