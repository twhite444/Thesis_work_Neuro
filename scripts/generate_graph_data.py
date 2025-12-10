#!/usr/bin/env python3
"""Generate molecular graph data from SMILES strings.

This script converts molecular SMILES strings to graph representations
with node features, edge indices, and edge attributes, storing everything
in a single NPZ file for efficient loading.

Usage:
    python scripts/generate_graph_data.py
    python scripts/generate_graph_data.py --no-edge-features
    python scripts/generate_graph_data.py --data-dir data/01_raw --output-dir data/01_raw
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neuro_foundation.data.molecular_graphs import generate_and_save_molecular_graphs


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
    
    # Call the src function with parsed arguments
    generate_and_save_molecular_graphs(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        smiles_col=args.smiles_col,
        cid_col=args.cid_col,
        include_edge_features=not args.no_edge_features,
        verbose=True
    )


if __name__ == '__main__':
    main()
