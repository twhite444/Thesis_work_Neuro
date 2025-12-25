#!/usr/bin/env python3
"""Batch inspect activity maps for all CIDs and display statistics.

This script loads activity maps for all CIDs and displays statistics
in the terminal without showing or saving visualizations.

Usage:
    python scripts/exploration/batch_inspect_all_cids.py
    python scripts/exploration/batch_inspect_all_cids.py --data-dir data/01_raw
    python scripts/exploration/batch_inspect_all_cids.py --summary-only
"""
import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.olfactory_modeling.data.pyrfume_loader import (
    load_activity_maps_by_cid,
    load_stimuli_npz,
)


def analyze_all_cids(data_dir: str = "data/01_raw", summary_only: bool = False):
    """Analyze activity maps for all CIDs.
    
    Args:
        data_dir: Directory containing the data
        summary_only: If True, only show summary statistics, not per-CID details
    """
    # Load metadata
    behavior = pd.read_csv(os.path.join(data_dir, 'behavior_data.csv'))
    stimuli = load_stimuli_npz(data_dir)
    
    # Get unique CIDs
    unique_cids = sorted(behavior['Stimulus'].unique())
    
    print("="*80)
    print(f"ACTIVITY MAP ANALYSIS FOR ALL CIDs")
    print("="*80)
    print(f"\nData directory: {data_dir}")
    print(f"Total unique CIDs: {len(unique_cids)}")
    print(f"Total activity maps: {len(behavior)}")
    print("="*80)
    
    # Storage for summary statistics
    all_stats = []
    
    # Process each CID
    for i, cid in enumerate(unique_cids, 1):
        # Get molecule name
        cid_info = stimuli[stimuli['CID'] == cid]
        molecule_name = cid_info.iloc[0]['Name'] if len(cid_info) > 0 else f"CID {cid}"
        
        # Load all maps for this CID
        maps = load_activity_maps_by_cid(cid, data_dir)
        n_maps = len(maps)
        
        if n_maps == 0:
            if not summary_only:
                print(f"\n[{i}/{len(unique_cids)}] CID {cid}: {molecule_name}")
                print(f"  ⚠️  No activity maps found")
            continue
        
        # Calculate statistics for each map
        map_stats = []
        for j, map_data in enumerate(maps):
            shape = map_data.shape
            active_pixels = (map_data > 0).sum()
            total_pixels = map_data.size
            coverage = (map_data > 0).mean() * 100
            
            min_val = map_data.min()
            max_val = map_data.max()
            mean_all = map_data.mean()
            
            # Statistics for active pixels only
            active_vals = map_data[map_data > 0]
            mean_active = active_vals.mean() if len(active_vals) > 0 else 0
            std_active = active_vals.std() if len(active_vals) > 0 else 0
            median_active = np.median(active_vals) if len(active_vals) > 0 else 0
            
            map_stats.append({
                'map_idx': j + 1,
                'shape': shape,
                'active_pixels': active_pixels,
                'total_pixels': total_pixels,
                'coverage': coverage,
                'min': min_val,
                'max': max_val,
                'mean_all': mean_all,
                'mean_active': mean_active,
                'median_active': median_active,
                'std_active': std_active,
            })
        
        # Store for summary
        all_stats.append({
            'cid': cid,
            'name': molecule_name,
            'n_maps': n_maps,
            'map_stats': map_stats,
        })
        
        # Display per-CID information if not summary-only
        if not summary_only:
            print(f"\n[{i}/{len(unique_cids)}] CID {cid}: {molecule_name}")
            print(f"  Number of maps: {n_maps}")
            
            for stats in map_stats:
                print(f"\n  Map {stats['map_idx']}:")
                print(f"    Shape: {stats['shape']}")
                print(f"    Coverage: {stats['active_pixels']:,}/{stats['total_pixels']:,} pixels ({stats['coverage']:.2f}%)")
                print(f"    Value range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"    Mean (all pixels): {stats['mean_all']:.4f}")
                print(f"    Mean (active pixels): {stats['mean_active']:.4f}")
                print(f"    Median (active pixels): {stats['median_active']:.4f}")
                print(f"    Std dev (active pixels): {stats['std_active']:.4f}")
    
    # Display summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Overall statistics
    total_maps = sum(s['n_maps'] for s in all_stats)
    cids_with_maps = len([s for s in all_stats if s['n_maps'] > 0])
    cids_multi_maps = len([s for s in all_stats if s['n_maps'] > 1])
    
    print(f"\nCIDs with activity maps: {cids_with_maps}/{len(unique_cids)}")
    print(f"Total activity maps: {total_maps}")
    print(f"CIDs with multiple maps: {cids_multi_maps}")
    
    # Map count distribution
    map_counts = [s['n_maps'] for s in all_stats]
    print(f"\nMaps per CID:")
    print(f"  Min: {min(map_counts)}")
    print(f"  Max: {max(map_counts)}")
    print(f"  Mean: {np.mean(map_counts):.2f}")
    print(f"  Median: {np.median(map_counts):.0f}")
    
    # Coverage statistics across all maps
    all_coverages = [ms['coverage'] for s in all_stats for ms in s['map_stats']]
    print(f"\nCoverage across all maps (% active pixels):")
    print(f"  Min: {min(all_coverages):.2f}%")
    print(f"  Max: {max(all_coverages):.2f}%")
    print(f"  Mean: {np.mean(all_coverages):.2f}%")
    print(f"  Median: {np.median(all_coverages):.2f}%")
    print(f"  Std dev: {np.std(all_coverages):.2f}%")
    
    # Value range statistics (for active pixels)
    all_means = [ms['mean_active'] for s in all_stats for ms in s['map_stats']]
    all_maxs = [ms['max'] for s in all_stats for ms in s['map_stats']]
    all_mins = [ms['min'] for s in all_stats for ms in s['map_stats'] if ms['min'] > 0]
    
    print(f"\nActive pixel values across all maps:")
    print(f"  Global min (active): {min(all_mins) if all_mins else 0:.4f}")
    print(f"  Global max: {max(all_maxs):.4f}")
    print(f"  Mean of means: {np.mean(all_means):.4f}")
    print(f"  Std of means: {np.std(all_means):.4f}")
    
    # Top CIDs by number of maps
    top_n = 10
    sorted_by_maps = sorted(all_stats, key=lambda x: x['n_maps'], reverse=True)[:top_n]
    print(f"\nTop {top_n} CIDs by number of maps:")
    for rank, s in enumerate(sorted_by_maps, 1):
        print(f"  {rank}. CID {s['cid']} ({s['name']}): {s['n_maps']} maps")
    
    # CIDs with highest average coverage
    sorted_by_coverage = sorted(
        all_stats, 
        key=lambda x: np.mean([ms['coverage'] for ms in x['map_stats']]),
        reverse=True
    )[:top_n]
    print(f"\nTop {top_n} CIDs by average coverage:")
    for rank, s in enumerate(sorted_by_coverage, 1):
        avg_cov = np.mean([ms['coverage'] for ms in s['map_stats']])
        print(f"  {rank}. CID {s['cid']} ({s['name']}): {avg_cov:.2f}%")
    
    # CIDs with highest average activity (active pixel mean)
    sorted_by_activity = sorted(
        all_stats,
        key=lambda x: np.mean([ms['mean_active'] for ms in x['map_stats']]),
        reverse=True
    )[:top_n]
    print(f"\nTop {top_n} CIDs by average activity (mean active pixel value):")
    for rank, s in enumerate(sorted_by_activity, 1):
        avg_act = np.mean([ms['mean_active'] for ms in s['map_stats']])
        print(f"  {rank}. CID {s['cid']} ({s['name']}): {avg_act:.4f}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Batch inspect activity maps for all CIDs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show detailed stats for all CIDs
  python scripts/exploration/batch_inspect_all_cids.py
  
  # Show only summary statistics (faster)
  python scripts/exploration/batch_inspect_all_cids.py --summary-only
  
  # Use different data directory
  python scripts/exploration/batch_inspect_all_cids.py --data-dir data/01_raw
        """
    )
    parser.add_argument("--data-dir", default="data/01_raw", 
                        help="Data directory (default: data/01_raw)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Show only summary statistics, skip per-CID details")
    
    args = parser.parse_args()
    
    analyze_all_cids(args.data_dir, args.summary_only)


if __name__ == "__main__":
    main()
