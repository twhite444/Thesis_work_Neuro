"""Molecular graph featurization for GNN models.

This module converts SMILES strings to PyTorch Geometric compatible graph data
with node features, edge indices, and optional edge features using RDKit.

Features:
- Atom-level node features (atomic number, degree, hybridization, etc.)
- Bond-level edge features (bond type, conjugation, ring membership)
- Compatible with PyTorch Geometric Data format
- Batch processing for entire molecule datasets
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import warnings


# ============================================================================
# Atom Feature Extraction
# ============================================================================

def get_atom_features(atom: Chem.Atom) -> np.ndarray:
    """Extract feature vector for a single atom.
    
    Features (total: 74 dimensions):
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
        Feature vector as numpy array (74,)
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
        - 'node_features': Node feature matrix (num_atoms, 74)
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
        
        if len(edge_index) == 0:
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
