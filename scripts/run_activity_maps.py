#!/usr/bin/env python3
"""Process activity maps: select and mask maps for each CID.

This script runs the complete activity maps preprocessing pipeline:
1. Loads all activity maps from CSV files (preserving NaNs)
2. Computes global mask based on coverage threshold
3. Applies mask to all maps (outside ROI → NaN)
4. Selects one map per CID using specified strategy
5. Applies value policy filtering (filtered values → NaN)
6. Converts ALL NaNs to zeros (for neural network training)
7. Saves processed maps for training

Usage examples:
    # Default: best quality selection, 50% coverage
    python scripts/run_activity_maps.py
    
    # Use averaging instead
    python scripts/run_activity_maps.py --strategy average
    
    # Stricter masking (80% coverage required)
    python scripts/run_activity_maps.py --coverage-threshold 0.8
    
    # Median selection (robust to outliers)
    python scripts/run_activity_maps.py --strategy median
    
    # Keep only positive values inside ROI
    python scripts/run_activity_maps.py --value-policy pos
    
    # Quick test without visualizations
    python scripts/run_activity_maps.py --strategy first --no-visualizations
"""
import argparse
from pathlib import Path

from olfactory_modeling.pipeline.activity_maps import (
    process_activity_maps,
    SelectionStrategy,
)


def main():
    parser = argparse.ArgumentParser(
        description='Process activity maps: select and mask maps for each CID',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: best quality selection, 50%% coverage
  python scripts/run_activity_maps.py
  
  # Use averaging instead
  python scripts/run_activity_maps.py --strategy average
  
  # Stricter masking (80%% coverage required)
  python scripts/run_activity_maps.py --coverage-threshold 0.8
  
  # Median selection (robust to outliers)
  python scripts/run_activity_maps.py --strategy median
  
  # Keep only positive values inside ROI
  python scripts/run_activity_maps.py --value-policy pos
  
  # Quick test without visualizations
  python scripts/run_activity_maps.py --strategy first --no-visualizations
        """
    )
    
    # Input/output paths
    parser.add_argument('--directory-csv', type=str,
                       default='data/01_raw/behavior_data.csv',
                       help='Behavior CSV with activity map paths')
    parser.add_argument('--data-dir', type=str,
                       default='data/01_raw',
                       help='Directory with activity_maps_csv/ folder')
    parser.add_argument('--output-dir', type=str,
                       default='data/02_processed',
                       help='Output directory for processed maps')
    
    # Selection strategy
    parser.add_argument('--strategy', type=str,
                       default='best_quality',
                       choices=['best_quality', 'average', 'median', 'first'],
                       help='Map selection strategy (default: best_quality)')
    
    # Global mask parameters
    parser.add_argument('--coverage-threshold', type=float,
                       default=0.5,
                       help='Fraction of maps required for mask: 0.0-1.0 (default: 0.5)')
    parser.add_argument('--min-region-size', type=int,
                       default=100,
                       help='Minimum connected region size in pixels (default: 100)')
    
    # Value processing parameters
    parser.add_argument('--value-policy', type=str,
                       default='all',
                       choices=['all', 'pos', 'neg'],
                       help='Value filtering inside ROI: "all", "pos", or "neg" (default: all)')
    
    # Options
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip generating visualization plots')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed information')
    
    args = parser.parse_args()
    
    # Validate coverage threshold
    if not 0.0 <= args.coverage_threshold <= 1.0:
        parser.error("Coverage threshold must be between 0.0 and 1.0")
    
    # Run processing pipeline
    results = process_activity_maps(
        directory_csv=args.directory_csv,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        selection_strategy=SelectionStrategy(args.strategy),
        coverage_threshold=args.coverage_threshold,
        min_region_size=args.min_region_size,
        value_policy=args.value_policy,
        save_visualizations=not args.no_visualizations,
        verbose=args.verbose,
    )
    
    # Print summary
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"Molecules processed: {results['n_molecules']}")
    print(f"Selection strategy:  {results['selection_strategy']}")
    print(f"Coverage threshold:  {results['coverage_threshold']}")
    print(f"Value policy:       {results['value_policy']}")
    print(f"Mask coverage:       {results['mask_coverage']:.2%}")
    print("="*80)
    print(f"\nOutputs saved to: {args.output_dir}/")
    print(f"  - processed_maps.npz (maps and CIDs)")
    print(f"  - processed_maps_metadata.csv (metadata)")
    print(f"  - map_statistics.json (QC statistics)")
    print(f"  - global_mask.npy (reusable mask)")
    if not args.no_visualizations:
        print(f"  - visualizations (PNG files)")
    print("="*80)


if __name__ == '__main__':
    main()
