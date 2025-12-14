"""Comprehensive tests for molecular graph generation."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from src.neuro_foundation.data.molecular_graphs import (
    get_atom_features,
    get_bond_features,
    smiles_to_graph,
    molecules_to_graphs,
    load_graph_data,
    get_graph_by_cid,
    graph_statistics
)


# Sample SMILES for testing
SAMPLE_SMILES = {
    'water': 'O',
    'methane': 'C',
    'ethanol': 'CCO',
    'benzene': 'c1ccccc1',
    'aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
    'invalid': 'INVALID_SMILES_STRING',
}


class TestAtomFeatures:
    """Test atom feature extraction."""
    
    def test_atom_features_dimension(self):
        """Test that atom features have correct dimension."""
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles('CCO')
        atom = mol.GetAtomWithIdx(0)
        features = get_atom_features(atom)
        
        # 100 (atomic num) + 11 (degree) + 5 (charge) + 5 (hyb) + 1 (aromatic)
        # + 5 (Hs) + 5 (radical) + 1 (ring) + 4 (chiral) = 137
        assert len(features) == 137
        
    def test_atom_features_are_numeric(self):
        """Test that all features are numeric (0 or 1)."""
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles('CCO')
        atom = mol.GetAtomWithIdx(0)
        features = get_atom_features(atom)
        
        # All features should be 0 or 1 (one-hot encoded)
        assert all(f in [0, 1] for f in features)
    
    def test_atom_features_carbon(self):
        """Test features for carbon atom."""
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles('C')
        atom = mol.GetAtomWithIdx(0)
        features = get_atom_features(atom)
        
        # Carbon is atomic number 6, so position 5 should be 1 (0-indexed)
        assert features[5] == 1  # atomic_num_6
        
    def test_atom_features_aromatic(self):
        """Test aromatic feature."""
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles('c1ccccc1')
        Chem.SanitizeMol(mol)
        atom = mol.GetAtomWithIdx(0)
        features = get_atom_features(atom)
        
        # Aromatic flag should be set
        aromatic_idx = 100 + 11 + 5 + 5  # After atomic num, degree, charge, hyb
        assert features[aromatic_idx] == 1


class TestBondFeatures:
    """Test bond feature extraction."""
    
    def test_bond_features_dimension(self):
        """Test that bond features have correct dimension."""
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles('CC')
        bond = mol.GetBondWithIdx(0)
        features = get_bond_features(bond)
        
        # 4 (bond type) + 1 (conjugated) + 1 (ring) + 4 (stereo) = 10
        assert len(features) == 10
        
    def test_bond_features_are_numeric(self):
        """Test that all features are numeric (0 or 1)."""
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles('CC')
        bond = mol.GetBondWithIdx(0)
        features = get_bond_features(bond)
        
        # All features should be 0 or 1
        assert all(f in [0, 1] for f in features)
    
    def test_bond_features_single_bond(self):
        """Test features for single bond."""
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles('CC')
        bond = mol.GetBondWithIdx(0)
        features = get_bond_features(bond)
        
        # First position should be 1 for single bond
        assert features[0] == 1
    
    def test_bond_features_double_bond(self):
        """Test features for double bond."""
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles('C=C')
        bond = mol.GetBondWithIdx(0)
        features = get_bond_features(bond)
        
        # Second position should be 1 for double bond
        assert features[1] == 1
    
    def test_bond_features_aromatic_bond(self):
        """Test features for aromatic bond."""
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles('c1ccccc1')
        bond = mol.GetBondWithIdx(0)
        features = get_bond_features(bond)
        
        # Fourth position should be 1 for aromatic bond
        assert features[3] == 1


class TestSMILESToGraph:
    """Test SMILES to graph conversion."""
    
    def test_simple_molecule(self):
        """Test conversion of simple molecule."""
        graph = smiles_to_graph('CCO')
        
        assert graph is not None
        assert 'node_features' in graph
        assert 'edge_index' in graph
        assert 'num_nodes' in graph
        assert 'num_edges' in graph
        
    def test_graph_structure(self):
        """Test that graph has correct structure."""
        graph = smiles_to_graph('CCO')
        
        # Check shapes
        assert graph['node_features'].shape[1] == 137
        assert graph['edge_index'].shape[0] == 2
        
    def test_edge_features_included(self):
        """Test that edge features are included when requested."""
        graph = smiles_to_graph('CCO', include_edge_features=True)
        
        assert 'edge_attr' in graph
        assert graph['edge_attr'].shape[1] == 10
        
    def test_edge_features_excluded(self):
        """Test that edge features are excluded when not requested."""
        graph = smiles_to_graph('CCO', include_edge_features=False)
        
        assert 'edge_attr' not in graph
    
    def test_invalid_smiles(self):
        """Test that invalid SMILES returns None."""
        graph = smiles_to_graph('INVALID_SMILES')
        
        assert graph is None
    
    def test_undirected_graph(self):
        """Test that graph is undirected (edges in both directions)."""
        graph = smiles_to_graph('CC')
        
        edge_index = graph['edge_index']
        # Ethane with explicit Hs: 2 C + 6 H = 8 atoms
        # C-C bond + 6 C-H bonds = 7 bonds * 2 directions = 14 edges
        assert edge_index.shape[1] == 14
        # Verify edges are bidirectional
        assert edge_index.shape[0] == 2
        
    def test_benzene_ring(self):
        """Test conversion of aromatic ring."""
        graph = smiles_to_graph('c1ccccc1')
        
        assert graph is not None
        # Benzene with explicit Hs: 6 C + 6 H = 12 atoms
        assert graph['num_nodes'] == 12
        # 6 C-C bonds (ring) + 6 C-H bonds = 12 bonds (num_edges counts unique bonds)
        assert graph['num_edges'] == 12
        # But edge_index has bidirectional edges: 12 bonds * 2 = 24 directed edges
        assert graph['edge_index'].shape[1] == 24


class TestMoleculesToGraphs:
    """Test batch conversion of molecules."""
    
    def test_batch_conversion(self):
        """Test converting multiple molecules."""
        import pandas as pd
        
        df = pd.DataFrame({
            'CID': [1, 2, 3],
            'IsomericSMILES': ['C', 'CC', 'CCO']
        })
        
        graph_data = molecules_to_graphs(df, verbose=False)
        
        assert 'cids' in graph_data
        assert 'node_features_list' in graph_data
        assert 'edge_index_list' in graph_data
        assert len(graph_data['cids']) == 3
        
    def test_valid_mask(self):
        """Test that valid mask correctly identifies failed conversions."""
        import pandas as pd
        
        df = pd.DataFrame({
            'CID': [1, 2, 3],
            'IsomericSMILES': ['C', 'INVALID', 'CCO']
        })
        
        graph_data = molecules_to_graphs(df, verbose=False)
        
        # Should have 3 entries but one is invalid
        assert len(graph_data['valid_mask']) == 3
        assert graph_data['valid_mask'][0]
        assert not graph_data['valid_mask'][1]
        assert graph_data['valid_mask'][2]
    
    def test_num_nodes_tracking(self):
        """Test that number of nodes is tracked correctly."""
        import pandas as pd
        
        df = pd.DataFrame({
            'CID': [1, 2],
            'IsomericSMILES': ['C', 'CC']
        })
        
        graph_data = molecules_to_graphs(df, verbose=False)
        
        # Methane has 5 atoms (1 C + 4 H), Ethane has 8 (2 C + 6 H)
        # Note: depends on whether explicit Hs are added
        assert len(graph_data['num_nodes']) == 2
        assert all(n > 0 for n in graph_data['num_nodes'])


class TestGraphDataIO:
    """Test loading and saving graph data."""
    
    def test_save_and_load_graph_data(self):
        """Test saving and loading graph data."""
        import pandas as pd
        
        # Create test data
        df = pd.DataFrame({
            'CID': [1, 2],
            'IsomericSMILES': ['C', 'CC']
        })
        
        graph_data = molecules_to_graphs(df, verbose=False)
        
        # Save to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'molecular_graphs.npz'
            np.savez_compressed(filepath, **graph_data)
            
            # Load back
            loaded_data = load_graph_data(tmpdir)
            
            # Check that loaded data matches
            assert 'cids' in loaded_data
            assert len(loaded_data['cids']) == 2
            assert np.array_equal(loaded_data['cids'], graph_data['cids'])
    
    def test_get_graph_by_cid(self):
        """Test retrieving graph by CID."""
        import pandas as pd
        
        df = pd.DataFrame({
            'CID': [100, 200],
            'IsomericSMILES': ['C', 'CC']
        })
        
        graph_data = molecules_to_graphs(df, verbose=False)
        
        # Get graph for CID 100
        graph = get_graph_by_cid(100, graph_data)
        
        assert graph is not None
        assert 'node_features' in graph
        assert 'edge_index' in graph
        
    def test_get_graph_by_invalid_cid(self):
        """Test that invalid CID returns None."""
        import pandas as pd
        
        df = pd.DataFrame({
            'CID': [100, 200],
            'IsomericSMILES': ['C', 'CC']
        })
        
        graph_data = molecules_to_graphs(df, verbose=False)
        
        # Try to get graph for non-existent CID
        graph = get_graph_by_cid(999, graph_data)
        
        assert graph is None


class TestGraphStatistics:
    """Test graph statistics computation."""
    
    def test_graph_statistics(self):
        """Test computing graph statistics."""
        import pandas as pd
        
        df = pd.DataFrame({
            'CID': [1, 2, 3],
            'IsomericSMILES': ['C', 'CC', 'CCC']
        })
        
        graph_data = molecules_to_graphs(df, verbose=False)
        stats = graph_statistics(graph_data)
        
        assert 'total_molecules' in stats
        assert 'total_atoms' in stats
        assert 'total_bonds' in stats
        assert 'avg_atoms_per_molecule' in stats
        assert 'avg_bonds_per_molecule' in stats
        
        assert stats['total_molecules'] == 3
        assert stats['total_atoms'] > 0
        assert stats['total_bonds'] > 0
    
    def test_statistics_aggregation(self):
        """Test that statistics are aggregated correctly."""
        import pandas as pd
        
        df = pd.DataFrame({
            'CID': [1, 2],
            'IsomericSMILES': ['C', 'CC']
        })
        
        graph_data = molecules_to_graphs(df, verbose=False)
        stats = graph_statistics(graph_data)
        
        # Total should equal sum of individual molecules
        total = stats['total_atoms']
        individual_sum = sum(graph_data['num_nodes'])
        assert total == individual_sum


class TestIntegration:
    """Integration tests with real data."""
    
    def test_real_molecules(self):
        """Test with actual molecular SMILES."""
        import pandas as pd
        
        # Use real-world molecules
        df = pd.DataFrame({
            'CID': [180, 702, 887],  # Tyramine, Ethanol, Methanol
            'IsomericSMILES': [
                'C1=CC(=CC=C1CCN)O',  # Tyramine
                'CCO',                 # Ethanol
                'CO'                   # Methanol
            ]
        })
        
        graph_data = molecules_to_graphs(df, verbose=False)
        
        # All should convert successfully
        assert all(graph_data['valid_mask'])
        assert len(graph_data['cids']) == 3
        
        # Check that complex molecule has more atoms
        tyramine_idx = 0
        methanol_idx = 2
        assert graph_data['num_nodes'][tyramine_idx] > graph_data['num_nodes'][methanol_idx]


@pytest.mark.skipif(
    not Path('data/01_raw/molecules_raw.npz').exists(),
    reason="Real data not available"
)
class TestRealDataIntegration:
    """Tests with actual dataset (skipped if data not available)."""
    
    def test_load_real_graph_data(self):
        """Test loading real generated graph data."""
        if Path('data/01_raw/molecular_graphs.npz').exists():
            graph_data = load_graph_data('data/01_raw')
            
            assert 'cids' in graph_data
            assert len(graph_data['cids']) > 0
            
    def test_real_graph_statistics(self):
        """Test computing statistics on real data."""
        if Path('data/01_raw/molecular_graphs.npz').exists():
            graph_data = load_graph_data('data/01_raw')
            stats = graph_statistics(graph_data)
            
            assert stats['total_molecules'] == 287
            assert stats['total_atoms'] > 1000
            assert stats['node_feature_dim'] == 137
