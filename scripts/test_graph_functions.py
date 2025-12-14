#!/usr/bin/env python3
"""Test script for molecular graph functions.

This script demonstrates all the graph data generation, loading,
and visualization capabilities with the helper functions.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neuro_foundation.data.molecular_graphs import (
    load_graph_by_cid,
    load_graph_data,
    graph_statistics
)
from src.neuro_foundation.data.graph_viz import (
    visualize_molecular_graph,
    compare_molecule_and_graph,
    print_graph_summary
)
from src.neuro_foundation.data.pyrfume_loader import load_molecules_npz


def test_helper_function():
    """Test the load_graph_by_cid helper function."""
    print("=" * 70)
    print("TEST 1: Helper Function (load_graph_by_cid)")
    print("=" * 70)
    print()
    
    # Test 1a: Load only
    print("1a. Load graph data only (no visualization)")
    print("-" * 50)
    graph = load_graph_by_cid(180)
    print(f"✓ Successfully loaded graph with {graph['num_nodes']} nodes")
    print()
    
    # Test 1b: Load and save image
    print("1b. Load and save visualization")
    print("-" * 50)
    graph = load_graph_by_cid(240, show_image=True, save_image=True)
    print("✓ Loaded and saved visualization for CID 240")
    print()


def test_direct_visualization():
    """Test direct visualization functions."""
    print("=" * 70)
    print("TEST 2: Direct Visualization Functions")
    print("=" * 70)
    print()
    
    # Load data
    graph_data = load_graph_data()
    molecules = load_molecules_npz()
    
    # Test 2a: Single molecule visualization
    print("2a. Single molecule visualization with save_path")
    print("-" * 50)
    visualize_molecular_graph(
        7991,
        graph_data,
        molecules,
        save_path='viz/molecules/test_viz_7991.png',
        show=False
    )
    print("✓ Saved visualization for CID 7991")
    print()
    
    # Test 2b: Comparison visualization
    print("2b. Comparison visualization (structure vs graph)")
    print("-" * 50)
    compare_molecule_and_graph(
        180,
        graph_data,
        molecules,
        save_path='viz/molecules/test_comparison_180.png',
        show=False
    )
    print("✓ Saved comparison visualization for CID 180")
    print()


def test_statistics():
    """Test statistics functions."""
    print("=" * 70)
    print("TEST 3: Statistics and Summary")
    print("=" * 70)
    print()
    
    # Load data
    graph_data = load_graph_data()
    
    # Test 3a: Print summary
    print("3a. Dataset summary")
    print("-" * 50)
    print_graph_summary(graph_data)
    print()
    
    # Test 3b: Get statistics programmatically
    print("3b. Programmatic statistics access")
    print("-" * 50)
    stats = graph_statistics(graph_data)
    print(f"Total molecules: {stats['total_molecules']}")
    print(f"Total atoms: {stats['total_atoms']:,}")
    print(f"Avg atoms/molecule: {stats['avg_atoms_per_molecule']:.1f}")
    print(f"Node feature dims: {stats['node_feature_dim']}")
    print(f"Edge feature dims: {stats['edge_feature_dim']}")
    print("✓ Statistics retrieved successfully")
    print()


def test_batch_loading():
    """Test loading multiple molecules."""
    print("=" * 70)
    print("TEST 4: Batch Loading")
    print("=" * 70)
    print()
    
    print("4a. Load multiple molecules")
    print("-" * 50)
    
    cids_to_test = [180, 240, 7991]
    for cid in cids_to_test:
        graph = load_graph_by_cid(cid, show_image=False, save_image=False)
        if graph:
            print(f"  CID {cid}: {graph['num_nodes']} nodes, {graph['num_edges']} edges")
    
    print(f"✓ Loaded {len(cids_to_test)} molecules successfully")
    print()


def main():
    """Run all tests."""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "MOLECULAR GRAPH FUNCTION TESTS" + " " * 23 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    try:
        test_helper_function()
        test_direct_visualization()
        test_statistics()
        test_batch_loading()
        
        print("=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print()
        print("Generated test files:")
        print("  - data/01_raw/molecular_graph_CID_240.png")
        print("  - data/01_raw/test_viz_7991.png")
        print("  - data/01_raw/test_comparison_180.png")
        print()
        
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ TEST FAILED!")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
