"""Tests for molecular graph visualization functions."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.neuro_foundation.data.molecular_graphs import (
    smiles_to_graph,
    molecules_to_graphs,
    load_graph_by_cid
)


@pytest.fixture
def mock_molecules_df():
    """Create a mock molecules DataFrame."""
    return pd.DataFrame({
        'CID': [180, 240, 702],
        'IsomericSMILES': ['C1=CC(=CC=C1CCN)O', 'CCO', 'CO'],
        'name': ['Tyramine', 'Ethanol', 'Methanol']
    })


@pytest.fixture
def mock_graph_data():
    """Create mock graph data."""
    graph1 = smiles_to_graph('CCO')
    graph2 = smiles_to_graph('CO')
    
    # Create object arrays properly
    node_features_arr = np.empty(2, dtype=object)
    node_features_arr[0] = graph1['node_features']
    node_features_arr[1] = graph2['node_features']
    
    edge_index_arr = np.empty(2, dtype=object)
    edge_index_arr[0] = graph1['edge_index']
    edge_index_arr[1] = graph2['edge_index']
    
    edge_attr_arr = np.empty(2, dtype=object)
    edge_attr_arr[0] = graph1.get('edge_attr')
    edge_attr_arr[1] = graph2.get('edge_attr')
    
    return {
        'cids': np.array([180, 702]),
        'num_nodes': np.array([graph1['num_nodes'], graph2['num_nodes']]),
        'num_edges': np.array([graph1['num_edges'], graph2['num_edges']]),
        'valid_mask': np.array([True, True]),
        'node_features_list': node_features_arr,
        'edge_index_list': edge_index_arr,
        'edge_attr_list': edge_attr_arr
    }


class TestVisualizationPaths:
    """Test that visualizations save to correct directories."""
    
    def test_load_graph_by_cid_saves_to_viz_molecules(self, tmp_path, monkeypatch):
        """Test that load_graph_by_cid saves to viz/molecules/ directory."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)
        
        # Create mock data
        df = pd.DataFrame({
            'CID': [180],
            'IsomericSMILES': ['CCO']
        })
        
        # Create molecules and graphs
        os.makedirs('data/01_raw', exist_ok=True)
        df.to_csv('data/01_raw/molecules_raw.csv', index=False)
        
        # Save molecules as NPZ
        np.savez_compressed(
            'data/01_raw/molecules_raw.npz',
            CID=df['CID'].values,
            IsomericSMILES=df['IsomericSMILES'].values,
            MolecularWeight=np.array([46.07]),  # Ethanol molecular weight
            IUPACName=np.array(['ethanol']),
            name=np.array(['Ethanol'])
        )
        
        # Generate and save graphs
        graph_data = molecules_to_graphs(df, verbose=False)
        np.savez_compressed('data/01_raw/molecular_graphs.npz', **graph_data)
        
        # Mock the visualization function to avoid actually rendering
        with patch('src.neuro_foundation.data.graph_viz.visualize_molecular_graph') as mock_viz:
            # Call with save_image=True
            load_graph_by_cid(180, data_dir='data/01_raw', show_image=True, save_image=True)
            
            # Check that viz directory was created and used in save_path
            mock_viz.assert_called_once()
            call_kwargs = mock_viz.call_args[1]
            save_path = call_kwargs.get('save_path')
            
            assert save_path is not None
            assert 'viz/molecules' in str(save_path)
            assert 'CID_180.png' in str(save_path)
    
    def test_viz_directories_autocreated(self, tmp_path, monkeypatch):
        """Test that viz directories are auto-created when needed."""
        monkeypatch.chdir(tmp_path)
        
        # Create minimal test data
        df = pd.DataFrame({
            'CID': [180],
            'IsomericSMILES': ['CCO']
        })
        
        os.makedirs('data/01_raw', exist_ok=True)
        np.savez_compressed(
            'data/01_raw/molecules_raw.npz',
            CID=df['CID'].values,
            IsomericSMILES=df['IsomericSMILES'].values,
            MolecularWeight=np.array([46.07]),
            IUPACName=np.array(['ethanol']),
            name=np.array(['Ethanol'])
        )
        
        graph_data = molecules_to_graphs(df, verbose=False)
        np.savez_compressed('data/01_raw/molecular_graphs.npz', **graph_data)
        
        # Verify viz/molecules does not exist
        assert not Path('viz/molecules').exists()
        
        # Mock visualization to avoid rendering
        with patch('src.neuro_foundation.data.graph_viz.visualize_molecular_graph'):
            load_graph_by_cid(180, data_dir='data/01_raw', show_image=True, save_image=True)
        
        # Check that directory was created
        assert Path('viz/molecules').exists()
    
    def test_no_visualization_in_data_directory(self, tmp_path, monkeypatch):
        """Test that no visualization files are created in data/ directory."""
        monkeypatch.chdir(tmp_path)
        
        df = pd.DataFrame({
            'CID': [180],
            'IsomericSMILES': ['CCO']
        })
        
        os.makedirs('data/01_raw', exist_ok=True)
        np.savez_compressed(
            'data/01_raw/molecules_raw.npz',
            CID=df['CID'].values,
            IsomericSMILES=df['IsomericSMILES'].values,
            MolecularWeight=np.array([46.07]),
            IUPACName=np.array(['ethanol']),
            name=np.array(['Ethanol'])
        )
        
        graph_data = molecules_to_graphs(df, verbose=False)
        np.savez_compressed('data/01_raw/molecular_graphs.npz', **graph_data)
        
        with patch('src.neuro_foundation.data.graph_viz.visualize_molecular_graph'):
            load_graph_by_cid(180, data_dir='data/01_raw', show_image=True, save_image=True)
        
        # Check that no PNG or HTML files exist in data/ directory
        data_files = list(Path('data').rglob('*.png')) + list(Path('data').rglob('*.html'))
        # Filter out activity_maps_csv files
        viz_files_in_data = [f for f in data_files if 'activity_maps_csv' not in str(f)]
        
        assert len(viz_files_in_data) == 0, f"Found visualization files in data/: {viz_files_in_data}"


class TestLoadGraphByCID:
    """Test the load_graph_by_cid helper function."""
    
    def test_load_only_no_visualization(self, tmp_path, monkeypatch):
        """Test loading graph without visualization."""
        monkeypatch.chdir(tmp_path)
        
        df = pd.DataFrame({
            'CID': [180],
            'IsomericSMILES': ['CCO']
        })
        
        os.makedirs('data/01_raw', exist_ok=True)
        np.savez_compressed(
            'data/01_raw/molecules_raw.npz',
            CID=df['CID'].values,
            IsomericSMILES=df['IsomericSMILES'].values,
            MolecularWeight=np.array([46.07]),
            IUPACName=np.array(['ethanol']),
            name=np.array(['Ethanol'])
        )
        
        graph_data = molecules_to_graphs(df, verbose=False)
        np.savez_compressed('data/01_raw/molecular_graphs.npz', **graph_data)
        
        # Load without visualization
        graph = load_graph_by_cid(180, data_dir='data/01_raw', show_image=False)
        
        assert graph is not None
        assert 'node_features' in graph
        assert 'edge_index' in graph
        assert graph['num_nodes'] > 0
    
    def test_load_with_visualization_modes(self, tmp_path, monkeypatch):
        """Test different visualization modes (simple/detailed, 2d/3d)."""
        monkeypatch.chdir(tmp_path)
        
        df = pd.DataFrame({
            'CID': [180],
            'IsomericSMILES': ['CCO']
        })
        
        os.makedirs('data/01_raw', exist_ok=True)
        np.savez_compressed(
            'data/01_raw/molecules_raw.npz',
            CID=df['CID'].values,
            IsomericSMILES=df['IsomericSMILES'].values,
            MolecularWeight=np.array([46.07]),
            IUPACName=np.array(['ethanol']),
            name=np.array(['Ethanol'])
        )
        
        graph_data = molecules_to_graphs(df, verbose=False)
        np.savez_compressed('data/01_raw/molecular_graphs.npz', **graph_data)
        
        modes = ['simple', 'detailed']
        projections = ['2d', '3d']
        
        with patch('src.neuro_foundation.data.graph_viz.visualize_molecular_graph') as mock_viz:
            for mode in modes:
                for projection in projections:
                    load_graph_by_cid(
                        180,
                        data_dir='data/01_raw',
                        show_image=True,
                        mode=mode,
                        projection=projection
                    )
            
            # Should be called 4 times (2 modes * 2 projections)
            assert mock_viz.call_count == 4
    
    def test_invalid_cid_returns_none(self, tmp_path, monkeypatch):
        """Test that invalid CID returns None gracefully."""
        monkeypatch.chdir(tmp_path)
        
        df = pd.DataFrame({
            'CID': [180],
            'IsomericSMILES': ['CCO']
        })
        
        os.makedirs('data/01_raw', exist_ok=True)
        np.savez_compressed(
            'data/01_raw/molecules_raw.npz',
            CID=df['CID'].values,
            IsomericSMILES=df['IsomericSMILES'].values,
            MolecularWeight=np.array([46.07]),
            IUPACName=np.array(['ethanol']),
            name=np.array(['Ethanol'])
        )
        
        graph_data = molecules_to_graphs(df, verbose=False)
        np.savez_compressed('data/01_raw/molecular_graphs.npz', **graph_data)
        
        # Try to load non-existent CID
        graph = load_graph_by_cid(999999, data_dir='data/01_raw')
        
        assert graph is None


class TestVisualizationFunctions:
    """Test visualization function behavior (mocked rendering)."""
    
    @patch('src.neuro_foundation.data.graph_viz.RDKIT_AVAILABLE', True)
    @patch('src.neuro_foundation.data.graph_viz.draw_molecule_from_smiles')
    def test_visualize_molecular_graph_simple_2d(self, mock_draw, mock_graph_data):
        """Test simple 2D visualization."""
        from src.neuro_foundation.data.graph_viz import visualize_molecular_graph
        
        mock_draw.return_value = MagicMock()  # Mock PIL Image
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test.png'
            
            with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
                visualize_molecular_graph(
                    180,
                    mock_graph_data,
                    save_path=str(save_path),
                    show=False,
                    mode='simple',
                    projection='2d'
                )
    
    @patch('src.neuro_foundation.data.graph_viz.PYMOL_AVAILABLE', True)
    @patch('src.neuro_foundation.data.graph_viz.visualize_molecule_3d_pymol')
    def test_visualize_molecular_graph_simple_3d_pymol(self, mock_pymol, mock_graph_data, mock_molecules_df):
        """Test simple 3D visualization with PyMOL."""
        from src.neuro_foundation.data.graph_viz import visualize_molecular_graph
        
        mock_pymol.return_value = True  # Success
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_3d.png'
            
            visualize_molecular_graph(
                180,
                mock_graph_data,
                mock_molecules_df,
                save_path=str(save_path),
                show=False,
                mode='simple',
                projection='3d'
            )
            
            mock_pymol.assert_called_once()
    
    def test_visualization_creates_output_directory(self, mock_graph_data):
        """Test that visualization creates output directory if needed."""
        from src.neuro_foundation.data.graph_viz import visualize_molecular_graph
        
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / 'deep' / 'nested' / 'path' / 'test.png'
            
            with patch('matplotlib.pyplot.savefig'), \
                 patch('matplotlib.pyplot.close'), \
                 patch('src.neuro_foundation.data.graph_viz.draw_molecule_from_smiles', return_value=MagicMock()):
                
                visualize_molecular_graph(
                    180,
                    mock_graph_data,
                    save_path=str(nested_path),
                    show=False,
                    mode='simple',
                    projection='2d'
                )
            
            # Check that directory was created
            assert nested_path.parent.exists()


class TestInteractiveVisualization:
    """Test interactive py3Dmol visualization."""
    
    def test_visualize_molecule_interactive_creates_html(self, tmp_path, monkeypatch):
        """Test that interactive visualization creates HTML file."""
        monkeypatch.chdir(tmp_path)
        
        # Create test data
        df = pd.DataFrame({
            'CID': [180],
            'IsomericSMILES': ['CCO'],
            'name': ['Ethanol']
        })
        
        os.makedirs('data/01_raw', exist_ok=True)
        np.savez_compressed(
            'data/01_raw/molecules_raw.npz',
            CID=df['CID'].values,
            IsomericSMILES=df['IsomericSMILES'].values,
            MolecularWeight=np.array([46.07]),
            IUPACName=np.array(['ethanol']),
            name=df['name'].values
        )
        
        # Mock the actual visualization to avoid py3Dmol dependency
        from src.neuro_foundation.data.molecular_graphs import visualize_molecule_interactive
        
        with patch('src.neuro_foundation.data.graph_viz.visualize_molecule_3d_interactive', return_value=True):
            output_path = visualize_molecule_interactive(
                180,
                data_dir='data/01_raw',
                open_browser=False
            )
            
            assert output_path is not None
            assert 'viz/molecules' in str(output_path)
            assert 'CID_180.html' in str(output_path)
    
    def test_interactive_viz_saves_to_viz_directory(self, tmp_path, monkeypatch):
        """Test that interactive viz saves to viz/molecules/."""
        monkeypatch.chdir(tmp_path)
        
        df = pd.DataFrame({
            'CID': [180],
            'IsomericSMILES': ['CCO']
        })
        
        os.makedirs('data/01_raw', exist_ok=True)
        np.savez_compressed(
            'data/01_raw/molecules_raw.npz',
            CID=df['CID'].values,
            IsomericSMILES=df['IsomericSMILES'].values,
            MolecularWeight=np.array([46.07]),
            IUPACName=np.array(['ethanol']),
            name=np.array(['Ethanol'])
        )
        
        from src.neuro_foundation.data.molecular_graphs import visualize_molecule_interactive
        
        with patch('src.neuro_foundation.data.graph_viz.visualize_molecule_3d_interactive', return_value=True):
            visualize_molecule_interactive(180, data_dir='data/01_raw', open_browser=False)
        
        # Verify viz/molecules directory was created
        assert Path('viz/molecules').exists()


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_graph_data(self):
        """Test behavior with empty graph data."""
        from src.neuro_foundation.data.molecular_graphs import get_graph_by_cid
        
        empty_data = {
            'cids': np.array([]),
            'num_nodes': np.array([]),
            'num_edges': np.array([]),
            'valid_mask': np.array([]),
            'node_features_list': np.array([]),
            'edge_index_list': np.array([])
        }
        
        graph = get_graph_by_cid(180, empty_data)
        assert graph is None
    
    def test_missing_npz_file(self, tmp_path):
        """Test handling of missing NPZ file."""
        from src.neuro_foundation.data.molecular_graphs import load_graph_by_cid
        
        with pytest.raises(FileNotFoundError):
            load_graph_by_cid(180, data_dir=str(tmp_path))
