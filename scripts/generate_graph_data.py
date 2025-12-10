#!/usr/bin/env python3
"""Generate molecular graph data for GNN models.

This script converts molecular SMILES strings to graph representations
with node features, edge indices, and edge attributes, storing everything
in a single NPZ file for efficient loading during GNN training.

Usage:
    python scripts/generate_graph_data.py
    python scripts/generate_graph_data.py --no-edge-features
    python scripts/generate_graph_data.py --data-dir data/01_raw --output-dir data/01_raw
"""

import argparse
import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.neuro_foundation.data.pyrfume_loader import load_molecules_npz
from src.neuro_foundation.data.molecular_graphs import molecules_to_graphs, graph_statistics


def main():
    parser = argparse.ArgumentParser(
        description='Generate molecular graph data from SMILES strings'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/01_raw',
        help='Directory containing molecules_raw.npz (default: data/01_raw)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/01_raw',
        help='Directory to save molecular_graphs.npz (default: data/01_raw)'
    )
    parser.add_argument(
        '--no-edge-features',
        action='store_true',
        help='Do not include edge features (saves space)'
    )
    parser.add_argument(
        '--smiles-col',
        type=str,
        default='IsomericSMILES',
        help='Column name for SMILES strings (default: IsomericSMILES)'
    )
    parser.add_argument(
        '--cid-col',
        type=str,
        default='CID',
        help='Column name for compound IDs (default: CID)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Molecular Graph Data Generation for GNN Models")
    print("=" * 70)
    print()
    
    # Load molecules
    print(f"Loading molecules from {args.data_dir}/molecules_raw.npz...")
    molecules = load_molecules_npz(data_dir=args.data_dir)
    print(f"Loaded {len(molecules)} molecules")
    print()
    
    # Convert to graphs
    include_edge_features = not args.no_edge_features
    print(f"Converting molecules to graphs...")
    print(f"  SMILES column: {args.smiles_col}")
    print(f"  CID column: {args.cid_col}")
    print(f"  Include edge features: {include_edge_features}")
    print()
    
    graph_data = molecules_to_graphs(
        molecules,
        smiles_col=args.smiles_col,
        cid_col=args.cid_col,
        include_edge_features=include_edge_features,
        verbose=True
    )
    print()
    
    # Save to NPZ
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'molecular_graphs.npz')
    
    print(f"Saving graph data to {output_path}...")
    np.savez_compressed(output_path, **graph_data)
    
    file_size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"Saved! File size: {file_size_mb:.2f} MB")
    print()
    
    # Display statistics
    print("=" * 70)
    print("Graph Data Statistics")
    print("=" * 70)
    
    stats = graph_statistics(graph_data)
    
    print(f"\nMolecules:")
    print(f"  Total molecules: {stats['total_molecules']}")
    print(f"  Total atoms: {stats['total_atoms']:,}")
    print(f"  Total bonds: {stats['total_bonds']:,}")
    
    print(f"\nAtoms per molecule:")
    print(f"  Mean: {stats['avg_atoms_per_molecule']:.1f} ± {stats['std_atoms_per_molecule']:.1f}")
    print(f"  Range: [{stats['min_atoms']}, {stats['max_atoms']}]")
    
    print(f"\nBonds per molecule:")
    print(f"  Mean: {stats['avg_bonds_per_molecule']:.1f} ± {stats['std_bonds_per_molecule']:.1f}")
    print(f"  Range: [{stats['min_bonds']}, {stats['max_bonds']}]")
    
    print(f"\nFeature dimensions:")
    print(f"  Node features: {stats['node_feature_dim']} dims")
    if stats['edge_feature_dim'] > 0:
        print(f"  Edge features: {stats['edge_feature_dim']} dims")
    else:
        print(f"  Edge features: Not included")
    
    print()
    print("=" * 70)
    print("✓ Graph data generation complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print(f"  1. Load graph data: load_graph_data('{args.output_dir}')")
    print(f"  2. Get graph for specific CID: get_graph_by_cid(180, graph_data)")
    print(f"  3. Use with PyTorch Geometric DataLoader for GNN training")
    print()


if __name__ == '__main__':
    main()
