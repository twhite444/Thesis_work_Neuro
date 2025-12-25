"""Script to compute PCA on processed activity maps.

This script:
1. Loads pre-processed activity maps from data/02_processed/processed_maps.npz
2. Fits PCA with specified number of components
3. Saves PCA model and transformed data
4. Generates visualization plots

Usage:
    python scripts/run_pca_on_maps.py --n_components 20
    
    # Or integrate with activity map processing:
    python scripts/run_pca_on_maps.py --process_maps --n_components 20
"""

import argparse
from pathlib import Path

from olfactory_modeling.pipeline.pca_transform import (
    fit_pca_on_maps,
    load_pca_transformed_maps,
    process_activity_maps_with_pca,
    visualize_pca_scatter_2d,
)
from olfactory_modeling.pipeline.activity_maps import load_processed_maps


def main():
    parser = argparse.ArgumentParser(description='Compute PCA on activity maps')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/01_raw',
        help='Directory containing raw activity map data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/02_processed',
        help='Directory to save processed data and PCA model'
    )
    parser.add_argument(
        '--n_components',
        type=int,
        default=20,
        help='Number of PCA components to compute (default: 20)'
    )
    parser.add_argument(
        '--process_maps',
        action='store_true',
        help='Also process raw activity maps before PCA (full pipeline)'
    )
    parser.add_argument(
        '--coverage_threshold',
        type=float,
        default=1.0,
        help='Coverage threshold for global mask (only used with --process_maps)'
    )
    parser.add_argument(
        '--selection_strategy',
        type=str,
        default='quality',
        choices=['quality', 'average', 'median', 'first'],
        help='Map selection strategy (only used with --process_maps)'
    )
    parser.add_argument(
        '--no_visualize',
        action='store_true',
        help='Skip visualization plots'
    )
    
    args = parser.parse_args()
    
    if args.process_maps:
        # Full pipeline: process maps + PCA
        print("\n" + "="*80)
        print("RUNNING FULL PIPELINE: Activity Map Processing + PCA")
        print("="*80)
        
        results = process_activity_maps_with_pca(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_components=args.n_components,
            coverage_threshold=args.coverage_threshold,
            selection_strategy=args.selection_strategy,
            visualize=not args.no_visualize,
        )
        
        pca_maps = results['pca_maps']
        cids = results['cids']
        
    else:
        # PCA only on pre-processed maps
        print("\n" + "="*80)
        print("COMPUTING PCA ON PRE-PROCESSED ACTIVITY MAPS")
        print("="*80)
        
        # Load pre-processed maps
        maps, cids, metadata = load_processed_maps(args.output_dir)
        print(f"Loaded {len(maps)} pre-processed activity maps")
        
        # Fit PCA
        pca_model, pca_maps, pca_metadata = fit_pca_on_maps(
            maps=maps,
            cids=cids,
            n_components=args.n_components,
            output_dir=args.output_dir,
            save_artifacts=True,
            visualize=not args.no_visualize,
        )
    
    # Additional visualization: scatter plot
    if not args.no_visualize:
        print("\nGenerating PCA scatter plot...")
        viz_dir = Path(args.output_dir).parent / 'viz'
        visualize_pca_scatter_2d(
            pca_transformed=pca_maps,
            cids=cids,
            output_dir=str(viz_dir),
        )
    
    print("\n" + "="*80)
    print("✓ PCA COMPUTATION COMPLETE")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  - {args.output_dir}/pca_model.pkl")
    print(f"  - {args.output_dir}/pca_transformed_maps.npz")
    print(f"  - {args.output_dir}/pca_transformed_maps.csv")
    if not args.no_visualize:
        print(f"  - viz/pca_*.png (visualization plots)")
    
    print("\nNext steps:")
    print("  1. Train on PCA components:")
    print("     python scripts/train_baseline_mlp.py --use_pca")
    print("  2. Or compare PCA vs raw maps:")
    print("     python scripts/compare_pca_vs_raw.py")


if __name__ == "__main__":
    main()
