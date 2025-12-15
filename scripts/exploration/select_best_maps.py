#!/usr/bin/env python3
"""Select the best activity map for each CID based on quality metrics.

For CIDs with multiple maps, computes quality metrics and selects the best one
based on a composite score:
    score = z(coverage) + z(range) - 0.5 * z(mean_active)

Where z() is the z-score (standardized value).

Usage:
    python scripts/exploration/select_best_maps.py
    python scripts/exploration/select_best_maps.py --output data/selected_maps.csv
"""
import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.neuro_foundation.data.pyrfume_loader import (
    load_activity_maps_by_cid,
    load_stimuli_npz,
)


def compute_map_metrics(map_data: np.ndarray) -> Dict[str, float]:
    """Compute quality metrics for a single activity map.
    
    Args:
        map_data: Activity map as numpy array
        
    Returns:
        Dictionary with metrics: coverage_frac, range, mean_active, std_active
    """
    total_pixels = map_data.size
    active_mask = map_data > 0
    active_pixels = active_mask.sum()
    
    coverage_frac = active_pixels / total_pixels
    
    min_val = map_data.min()
    max_val = map_data.max()
    value_range = max_val - min_val
    
    active_vals = map_data[active_mask]
    mean_active = active_vals.mean() if len(active_vals) > 0 else 0.0
    std_active = active_vals.std() if len(active_vals) > 0 else 0.0
    
    return {
        'coverage_frac': coverage_frac,
        'range': value_range,
        'mean_active': mean_active,
        'std_active': std_active,
        'min_val': min_val,
        'max_val': max_val,
        'active_pixels': int(active_pixels),
        'total_pixels': int(total_pixels),
    }


def z_score(values: List[float]) -> List[float]:
    """Compute z-scores for a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        List of z-scores (standardized values)
    """
    if len(values) <= 1:
        return [0.0] * len(values)
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample standard deviation
    
    if std == 0:
        return [0.0] * len(values)
    
    return [(v - mean) / std for v in values]


def select_best_maps(data_dir: str = "data/01_raw", close_threshold: float = 0.5) -> pd.DataFrame:
    """Select best activity map for each CID.
    
    Args:
        data_dir: Directory containing the data
        close_threshold: Threshold for flagging "close" scores (in z-score units)
        
    Returns:
        DataFrame with selection results
    """
    # Load metadata
    behavior = pd.read_csv(os.path.join(data_dir, 'behavior_data.csv'))
    stimuli = load_stimuli_npz(data_dir)
    
    # Get unique CIDs
    unique_cids = sorted(behavior['Stimulus'].unique())
    
    results = []
    
    print("="*100)
    print(f"ACTIVITY MAP SELECTION FOR ALL CIDs")
    print("="*100)
    print(f"\nProcessing {len(unique_cids)} CIDs...\n")
    
    for cid in unique_cids:
        # Get molecule name
        cid_info = stimuli[stimuli['CID'] == cid]
        molecule_name = cid_info.iloc[0]['Name'] if len(cid_info) > 0 else f"CID {cid}"
        
        # Load all maps for this CID
        maps = load_activity_maps_by_cid(cid, data_dir)
        n_maps = len(maps)
        
        if n_maps == 0:
            continue
        
        # Compute metrics for each map
        map_metrics = []
        for i, map_data in enumerate(maps):
            metrics = compute_map_metrics(map_data)
            metrics['map_idx'] = i
            map_metrics.append(metrics)
        
        # If only one map, no selection needed
        if n_maps == 1:
            metrics = map_metrics[0]
            results.append({
                'CID': cid,
                'Name': molecule_name,
                'n_maps': 1,
                'selected_idx': 0,
                'removed_indices': '',
                'coverage_frac': metrics['coverage_frac'],
                'range': metrics['range'],
                'mean_active': metrics['mean_active'],
                'std_active': metrics['std_active'],
                'composite_score': 0.0,
                'score_diff': None,
                'close_call': False,
            })
            continue
        
        # Multiple maps: compute composite scores
        coverages = [m['coverage_frac'] for m in map_metrics]
        ranges = [m['range'] for m in map_metrics]
        means_active = [m['mean_active'] for m in map_metrics]
        
        # Compute z-scores for each metric
        z_coverages = z_score(coverages)
        z_ranges = z_score(ranges)
        z_means = z_score(means_active)
        
        # Compute composite score for each map
        scores = []
        for i in range(n_maps):
            score = z_coverages[i] + z_ranges[i] - 0.5 * z_means[i]
            scores.append(score)
            map_metrics[i]['z_coverage'] = z_coverages[i]
            map_metrics[i]['z_range'] = z_ranges[i]
            map_metrics[i]['z_mean'] = z_means[i]
            map_metrics[i]['composite_score'] = score
        
        # Select map with highest score
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        best_metrics = map_metrics[best_idx]
        
        # Get removed indices
        removed_indices = [i for i in range(n_maps) if i != best_idx]
        
        # Check if it's a close call
        sorted_scores = sorted(scores, reverse=True)
        score_diff = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else None
        close_call = score_diff is not None and score_diff < close_threshold
        
        results.append({
            'CID': cid,
            'Name': molecule_name,
            'n_maps': n_maps,
            'selected_idx': best_idx,
            'removed_indices': ','.join(map(str, removed_indices)),
            'coverage_frac': best_metrics['coverage_frac'],
            'range': best_metrics['range'],
            'mean_active': best_metrics['mean_active'],
            'std_active': best_metrics['std_active'],
            'z_coverage': best_metrics['z_coverage'],
            'z_range': best_metrics['z_range'],
            'z_mean': best_metrics['z_mean'],
            'composite_score': best_score,
            'score_diff': score_diff,
            'close_call': close_call,
            'all_scores': scores,
            'all_metrics': map_metrics,
        })
    
    return pd.DataFrame(results)


def print_results(results_df: pd.DataFrame):
    """Print formatted results."""
    
    # Summary statistics
    total_cids = len(results_df)
    single_map = len(results_df[results_df['n_maps'] == 1])
    multi_map = len(results_df[results_df['n_maps'] > 1])
    close_calls = len(results_df[results_df['close_call'] == True])
    
    print("="*100)
    print("SUMMARY")
    print("="*100)
    print(f"Total CIDs: {total_cids}")
    print(f"  - Single map (no selection needed): {single_map}")
    print(f"  - Multiple maps (selection performed): {multi_map}")
    print(f"  - Close calls (score diff < 0.5): {close_calls}")
    print()
    
    # Show only CIDs with multiple maps
    multi_df = results_df[results_df['n_maps'] > 1].copy()
    
    if len(multi_df) == 0:
        print("No CIDs with multiple maps found.")
        return
    
    print("="*100)
    print(f"SELECTION RESULTS FOR {len(multi_df)} CIDs WITH MULTIPLE MAPS")
    print("="*100)
    print()
    
    for idx, row in multi_df.iterrows():
        print(f"CID {row['CID']}: {row['Name']}")
        print(f"  Maps available: {row['n_maps']}")
        print(f"  ✓ SELECTED: Map {row['selected_idx']}")
        print(f"  ✗ REMOVED:  Map(s) {row['removed_indices']}")
        print()
        
        # Show all map scores
        all_scores = row['all_scores']
        all_metrics = row['all_metrics']
        
        for i, (score, metrics) in enumerate(zip(all_scores, all_metrics)):
            marker = "✓✓✓" if i == row['selected_idx'] else "   "
            print(f"    {marker} Map {i}:")
            print(f"          Coverage:    {metrics['coverage_frac']:.4f} (z={metrics['z_coverage']:+.2f})")
            print(f"          Range:       {metrics['range']:.4f} (z={metrics['z_range']:+.2f})")
            print(f"          Mean Active: {metrics['mean_active']:.4f} (z={metrics['z_mean']:+.2f})")
            print(f"          Composite:   {score:.4f}")
        
        if row['close_call']:
            print(f"\n    ⚠️  CLOSE CALL: Score difference = {row['score_diff']:.4f}")
        else:
            print(f"\n    Score difference: {row['score_diff']:.4f}")
        
        print()
    
    # Show close calls separately
    if close_calls > 0:
        print("="*100)
        print(f"⚠️  CLOSE CALLS ({close_calls} CIDs)")
        print("="*100)
        print("\nThese CIDs have similar-quality maps (score diff < 0.5):")
        print()
        
        close_df = results_df[results_df['close_call'] == True].copy()
        for idx, row in close_df.iterrows():
            print(f"  • CID {row['CID']} ({row['Name']}): {row['n_maps']} maps, diff={row['score_diff']:.4f}")
        print()
    
    # Distribution statistics
    print("="*100)
    print("METRIC DISTRIBUTIONS (Selected Maps Only)")
    print("="*100)
    print()
    print(f"Coverage Fraction:")
    print(f"  Min:    {results_df['coverage_frac'].min():.4f}")
    print(f"  Max:    {results_df['coverage_frac'].max():.4f}")
    print(f"  Mean:   {results_df['coverage_frac'].mean():.4f}")
    print(f"  Median: {results_df['coverage_frac'].median():.4f}")
    print()
    print(f"Value Range:")
    print(f"  Min:    {results_df['range'].min():.4f}")
    print(f"  Max:    {results_df['range'].max():.4f}")
    print(f"  Mean:   {results_df['range'].mean():.4f}")
    print(f"  Median: {results_df['range'].median():.4f}")
    print()
    print(f"Mean Active:")
    print(f"  Min:    {results_df['mean_active'].min():.4f}")
    print(f"  Max:    {results_df['mean_active'].max():.4f}")
    print(f"  Mean:   {results_df['mean_active'].mean():.4f}")
    print(f"  Median: {results_df['mean_active'].median():.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Select best activity map for each CID based on quality metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Selection Criteria:
  For each map, compute:
    - coverage_frac = active_pixels / total_pixels
    - range = max - min
    - mean_active = mean of active pixel values
  
  Composite score = z(coverage) + z(range) - 0.5 * z(mean_active)
  
  Select the map with the highest composite score.

Examples:
  # Display selection results
  python scripts/exploration/select_best_maps.py
  
  # Save results to CSV
  python scripts/exploration/select_best_maps.py --output data/selected_maps.csv
  
  # Adjust close-call threshold
  python scripts/exploration/select_best_maps.py --threshold 0.3
        """
    )
    parser.add_argument("--data-dir", default="data/01_raw", 
                        help="Data directory (default: data/01_raw)")
    parser.add_argument("--output", "-o", 
                        help="Output CSV file for results (optional)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for flagging close calls (default: 0.5)")
    
    args = parser.parse_args()
    
    # Perform selection
    results_df = select_best_maps(args.data_dir, args.threshold)
    
    # Print results
    print_results(results_df)
    
    # Save if requested
    if args.output:
        # Create output directory if needed
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save simplified version (without nested lists)
        save_df = results_df.drop(columns=['all_scores', 'all_metrics'], errors='ignore')
        save_df.to_csv(args.output, index=False)
        print(f"\n✓ Results saved to: {args.output}")
        print(f"  ({len(save_df)} CIDs)")


if __name__ == "__main__":
    main()
