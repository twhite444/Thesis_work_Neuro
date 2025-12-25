#!/usr/bin/env python3
"""Script to visualize molecular graphs.

This script generates visualizations of molecular graphs similar to the
activity map visualizations. Useful for inspecting molecular structures
and understanding the graph featurization.

Usage:
    python scripts/visualize_graphs.py                    # Visualize sample molecules
    python scripts/visualize_graphs.py --cids 180 240     # Specific CIDs
    python scripts/visualize_graphs.py --gallery          # Gallery of molecules
    python scripts/visualize_graphs.py --compare 180      # Structure vs graph
"""

import argparse
import sys
import os

# Add src to path
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # No longer needed with proper __init__.py

from neuro_foundation.data.molecular_graphs import load_graph_data
from neuro_foundation.data.pyrfume_loader import load_molecules_npz
from neuro_foundation.data.graph_viz import (
    visualize_molecular_graph,
    visualize_multiple_graphs,
    compare_molecule_and_graph,
    print_graph_summary
)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize molecular graphs'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/01_raw',
        help='Directory containing molecular_graphs.npz (default: data/01_raw)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/01_raw',
        help='Directory to save visualizations (default: data/01_raw)'
    )
    parser.add_argument(
        '--cids',
        type=int,
        nargs='+',
        help='Specific CIDs to visualize'
    )
    parser.add_argument(
        '--gallery',
        action='store_true',
        help='Create gallery of multiple molecules'
    )
    parser.add_argument(
        '--compare',
        type=int,
        help='Create side-by-side comparison for a specific CID'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print dataset summary only'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=6,
        help='Number of molecules to show in gallery (default: 6)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Molecular Graph Visualization")
    print("=" * 70)
    print()
    
    # Load graph data
    print(f"Loading graph data from {args.data_dir}/molecular_graphs.npz...")
    graph_data = load_graph_data(args.data_dir)
    print(f"✓ Loaded {len(graph_data['cids'])} molecular graphs")
    print()
    
    # Load molecules for SMILES
    print(f"Loading molecule data from {args.data_dir}/molecules_raw.npz...")
    molecules = load_molecules_npz(data_dir=args.data_dir)
    print(f"✓ Loaded {len(molecules)} molecules")
    print()
    
    # Print summary if requested
    if args.summary:
        print_graph_summary(graph_data)
        return
    
    # Compare mode
    if args.compare is not None:
        print(f"Creating comparison visualization for CID {args.compare}...")
        output_path = os.path.join(args.output_dir, f'graph_comparison_CID_{args.compare}.png')
        compare_molecule_and_graph(
            args.compare,
            graph_data,
            molecules,
            output_path=output_path
        )
        print()
        return
    
    # Gallery mode
    if args.gallery:
        # Use provided CIDs or sample from dataset
        if args.cids:
            cids = args.cids[:args.sample_size]
        else:
            # Sample diverse molecules
            import numpy as np
            available_cids = graph_data['cids'][graph_data['valid_mask']]
            
            # Sample based on number of atoms for diversity
            num_nodes = graph_data['num_nodes'][graph_data['valid_mask']]
            # Get indices for small, medium, and large molecules
            sorted_indices = np.argsort(num_nodes)
            step = len(sorted_indices) // args.sample_size
            sampled_indices = sorted_indices[::step][:args.sample_size]
            cids = available_cids[sampled_indices].tolist()
        
        print(f"Creating gallery of {len(cids)} molecules: {cids}")
        print()
        visualize_multiple_graphs(
            cids,
            graph_data,
            molecules,
            output_dir=args.output_dir
        )
        print()
        return
    
    # Individual molecule visualizations
    if args.cids:
        cids = args.cids
    else:
        # Default: visualize a few interesting molecules
        cids = [180, 240, 7991]  # Same as activity maps examples
    
    print(f"Creating visualizations for CIDs: {cids}")
    print()
    
    for cid in cids:
        output_path = os.path.join(args.output_dir, f'molecular_graph_CID_{cid}.png')
        print(f"Visualizing CID {cid}...")
        visualize_molecular_graph(
            cid,
            graph_data,
            molecules,
            output_path=output_path
        )
    
    print()
    print("=" * 70)
    print("✓ Visualization complete!")
    print("=" * 70)
    print()
    print("Generated files:")
    for cid in cids:
        print(f"  {args.output_dir}/molecular_graph_CID_{cid}.png")
    print()
    print("Try these commands:")
    print(f"  python {sys.argv[0]} --gallery                # Gallery view")
    print(f"  python {sys.argv[0]} --compare 180            # Compare view")
    print(f"  python {sys.argv[0]} --cids 58 702 887        # Specific CIDs")
    print(f"  python {sys.argv[0]} --summary                # Dataset summary")
    print()


if __name__ == '__main__':
    main()
