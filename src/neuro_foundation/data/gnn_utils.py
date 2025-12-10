"""PyTorch Geometric utilities for loading molecular graph data.

This module provides helper functions to load molecular graphs and create
PyTorch Geometric Data objects suitable for GNN training.

Example usage:
    >>> from src.neuro_foundation.data.gnn_utils import create_pyg_dataset
    >>> 
    >>> # Load all graphs as PyG dataset
    >>> dataset = create_pyg_dataset(data_dir='data/01_raw')
    >>> print(f"Dataset size: {len(dataset)}")
    >>> 
    >>> # Access individual graphs
    >>> data = dataset[0]
    >>> print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    >>> 
    >>> # Create a DataLoader for batch training
    >>> from torch_geometric.loader import DataLoader
    >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
"""

import os
import numpy as np
from typing import List, Optional, Union

try:
    import torch
    from torch_geometric.data import Data, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch Geometric not available. Install with: pip install torch torch-geometric")

from .molecular_graphs import load_graph_data, get_graph_by_cid


class MolecularGraphDataset(Dataset):
    """PyTorch Geometric dataset for molecular graphs.
    
    This dataset wraps the molecular graph data stored in NPZ format,
    providing access to individual molecular graphs as PyG Data objects.
    
    Args:
        data_dir: Directory containing molecular_graphs.npz
        transform: Optional transform to apply to each graph
        pre_transform: Optional pre-transform to apply once during initialization
        
    Attributes:
        graph_data: Dictionary containing all graph data from NPZ
        cids: Array of compound IDs
        num_molecules: Total number of molecules in dataset
    """
    
    def __init__(
        self,
        data_dir: str = 'data/01_raw',
        transform=None,
        pre_transform=None
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch Geometric is required for this dataset. "
                "Install with: pip install torch torch-geometric"
            )
        
        self.data_dir = data_dir
        self.graph_data = load_graph_data(data_dir)
        self.cids = self.graph_data['cids']
        self.num_molecules = len(self.cids)
        
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)
    
    def len(self) -> int:
        """Return the number of graphs in the dataset."""
        return self.num_molecules
    
    def get(self, idx: int) -> Data:
        """Get a single molecular graph as PyG Data object.
        
        Args:
            idx: Index of the molecule (0 to len-1)
            
        Returns:
            PyG Data object with:
                - x: Node feature matrix [num_nodes, 137]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, 10] (if available)
                - cid: Compound ID
                - num_nodes: Number of nodes
        """
        # Extract data for this molecule
        node_features = self.graph_data['node_features_list'][idx]
        edge_index = self.graph_data['edge_index_list'][idx]
        cid = self.graph_data['cids'][idx]
        num_nodes = self.graph_data['num_nodes'][idx]
        
        # Convert to PyTorch tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            cid=int(cid),
            num_nodes=int(num_nodes)
        )
        
        # Add edge features if available
        if 'edge_attr_list' in self.graph_data:
            edge_attr = self.graph_data['edge_attr_list'][idx]
            data.edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        return data
    
    def get_by_cid(self, cid: int) -> Optional[Data]:
        """Get a molecular graph by compound ID.
        
        Args:
            cid: Compound ID to retrieve
            
        Returns:
            PyG Data object if CID found, None otherwise
        """
        graph_dict = get_graph_by_cid(cid, self.graph_data)
        if graph_dict is None:
            return None
        
        # Convert to PyG Data
        x = torch.tensor(graph_dict['node_features'], dtype=torch.float)
        edge_index = torch.tensor(graph_dict['edge_index'], dtype=torch.long)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            cid=cid,
            num_nodes=graph_dict['num_nodes']
        )
        
        if 'edge_attr' in graph_dict:
            data.edge_attr = torch.tensor(graph_dict['edge_attr'], dtype=torch.float)
        
        return data


def create_pyg_dataset(
    data_dir: str = 'data/01_raw',
    transform=None,
    pre_transform=None
) -> MolecularGraphDataset:
    """Create a PyTorch Geometric dataset from molecular graphs.
    
    Args:
        data_dir: Directory containing molecular_graphs.npz
        transform: Optional transform to apply to each graph
        pre_transform: Optional pre-transform to apply once
        
    Returns:
        MolecularGraphDataset instance
        
    Example:
        >>> dataset = create_pyg_dataset()
        >>> print(f"Dataset contains {len(dataset)} molecules")
        >>> 
        >>> # Use with DataLoader
        >>> from torch_geometric.loader import DataLoader
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for batch in loader:
        ...     print(f"Batch: {batch.num_graphs} graphs")
        ...     break
    """
    return MolecularGraphDataset(
        data_dir=data_dir,
        transform=transform,
        pre_transform=pre_transform
    )


def create_train_val_test_split(
    dataset: MolecularGraphDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> tuple:
    """Split dataset into train, validation, and test sets.
    
    Args:
        dataset: MolecularGraphDataset to split
        train_ratio: Fraction for training (default: 0.7)
        val_ratio: Fraction for validation (default: 0.15)
        test_ratio: Fraction for testing (default: 0.15)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
        
    Example:
        >>> dataset = create_pyg_dataset()
        >>> train_ds, val_ds, test_ds = create_train_val_test_split(dataset)
        >>> print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for dataset splitting")
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Generate random permutation
    num_samples = len(dataset)
    indices = torch.randperm(num_samples).tolist()
    
    # Calculate split points
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create subset datasets
    from torch.utils.data import Subset
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset


def get_node_feature_names() -> List[str]:
    """Get descriptive names for node features.
    
    Returns:
        List of 137 feature names corresponding to node feature dimensions
    """
    feature_names = []
    
    # Atomic number (100 dims)
    for i in range(1, 101):
        feature_names.append(f'atomic_num_{i}')
    
    # Degree (11 dims)
    for i in range(11):
        feature_names.append(f'degree_{i}')
    
    # Formal charge (5 dims)
    for charge in [-2, -1, 0, 1, 2]:
        feature_names.append(f'charge_{charge}')
    
    # Hybridization (5 dims)
    for hyb in ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']:
        feature_names.append(f'hyb_{hyb}')
    
    # Other features
    feature_names.extend([
        'is_aromatic',
        'total_Hs_0', 'total_Hs_1', 'total_Hs_2', 'total_Hs_3', 'total_Hs_4',
        'radical_e_0', 'radical_e_1', 'radical_e_2', 'radical_e_3', 'radical_e_4',
        'in_ring',
        'chiral_R', 'chiral_S', 'chiral_unspecified', 'chiral_other'
    ])
    
    return feature_names


def get_edge_feature_names() -> List[str]:
    """Get descriptive names for edge features.
    
    Returns:
        List of 10 feature names corresponding to edge feature dimensions
    """
    return [
        'bond_single',
        'bond_double',
        'bond_triple',
        'bond_aromatic',
        'is_conjugated',
        'in_ring',
        'stereo_none',
        'stereo_any',
        'stereo_Z',
        'stereo_E'
    ]


def print_dataset_info(dataset: MolecularGraphDataset) -> None:
    """Print detailed information about the dataset.
    
    Args:
        dataset: MolecularGraphDataset to analyze
    """
    print("=" * 70)
    print("Molecular Graph Dataset Information")
    print("=" * 70)
    print(f"\nDataset size: {len(dataset)} molecules")
    
    # Sample a few graphs to get statistics
    num_nodes_list = []
    num_edges_list = []
    has_edge_attr = None
    
    for i in range(min(len(dataset), 100)):
        data = dataset[i]
        num_nodes_list.append(data.num_nodes)
        num_edges_list.append(data.edge_index.shape[1])
        if has_edge_attr is None:
            has_edge_attr = hasattr(data, 'edge_attr')
    
    print(f"\nGraph statistics (sampled from {min(len(dataset), 100)} graphs):")
    print(f"  Nodes per graph: {np.mean(num_nodes_list):.1f} ± {np.std(num_nodes_list):.1f}")
    print(f"  Edges per graph: {np.mean(num_edges_list):.1f} ± {np.std(num_edges_list):.1f}")
    
    sample_data = dataset[0]
    print(f"\nFeature dimensions:")
    print(f"  Node features: {sample_data.x.shape[1]} dims")
    if has_edge_attr:
        print(f"  Edge features: {sample_data.edge_attr.shape[1]} dims")
    else:
        print(f"  Edge features: Not included")
    
    print("\n" + "=" * 70)
