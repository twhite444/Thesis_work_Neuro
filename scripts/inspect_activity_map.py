#!/usr/bin/env python3
"""Inspect individual activity maps by CID or filename.

Usage:
    python scripts/inspect_activity_map.py --cid 180
    python scripts/inspect_activity_map.py --filename 1031_0.csv
    python scripts/inspect_activity_map.py --list-all
"""
import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def list_all_maps(data_dir: str = "data/01_raw"):
    """List all available activity maps with their CIDs."""
    behavior = pd.read_csv(os.path.join(data_dir, 'behavior_data.csv'))
    print(f"\nTotal activity maps: {len(behavior)}")
    print(f"Unique CIDs: {behavior['Stimulus'].nunique()}")
    print("\nFirst 20 maps:")
    print(behavior[['Stimulus', 'Activity Map Path']].head(20).to_string())
    
    # Show CIDs with multiple maps
    cid_counts = behavior['Stimulus'].value_counts()
    multi_maps = cid_counts[cid_counts > 1]
    if len(multi_maps) > 0:
        print(f"\n{len(multi_maps)} CIDs have multiple activity maps:")
        print(multi_maps.head(10))


def load_by_cid(cid: int, data_dir: str = "data/01_raw"):
    """Load all activity maps for a specific CID."""
    behavior = pd.read_csv(os.path.join(data_dir, 'behavior_data.csv'))
    maps_for_cid = behavior[behavior['Stimulus'] == cid]
    
    if len(maps_for_cid) == 0:
        print(f"No activity maps found for CID {cid}")
        return
    
    print(f"\nCID {cid} has {len(maps_for_cid)} activity map(s):\n")
    
    for i, (_, row) in enumerate(maps_for_cid.iterrows(), 1):
        map_filename = os.path.basename(row['Activity Map Path'])
        map_path = os.path.join(data_dir, 'activity_maps_csv', map_filename)
        
        if not os.path.exists(map_path):
            print(f"  {i}. {map_filename} - FILE NOT FOUND")
            continue
        
        activity_map = pd.read_csv(map_path, index_col=0)
        active_pixels = (activity_map.values > 0).sum()
        coverage = (activity_map.values > 0).mean() * 100
        max_value = activity_map.values.max()
        
        print(f"  {i}. {map_filename}")
        print(f"     Shape: {activity_map.shape}")
        print(f"     Active pixels: {active_pixels} ({coverage:.1f}% coverage)")
        print(f"     Value range: [{activity_map.values.min():.4f}, {max_value:.4f}]")
        print(f"     Mean (active): {activity_map.values[activity_map.values > 0].mean():.4f}")
        print()


def load_by_filename(filename: str, data_dir: str = "data/01_raw"):
    """Load a specific activity map by filename."""
    map_path = os.path.join(data_dir, 'activity_maps_csv', filename)
    
    if not os.path.exists(map_path):
        print(f"File not found: {map_path}")
        return
    
    activity_map = pd.read_csv(map_path, index_col=0)
    
    # Find which CID this belongs to
    behavior = pd.read_csv(os.path.join(data_dir, 'behavior_data.csv'))
    matching = behavior[behavior['Activity Map Path'].str.endswith(filename)]
    cid = matching['Stimulus'].iloc[0] if len(matching) > 0 else "Unknown"
    
    print(f"\nActivity Map: {filename}")
    print(f"CID: {cid}")
    print(f"Shape: {activity_map.shape}")
    print(f"Value range: [{activity_map.values.min():.4f}, {activity_map.values.max():.4f}]")
    print(f"Active pixels: {(activity_map.values > 0).sum()} ({(activity_map.values > 0).mean()*100:.1f}%)")
    print(f"Mean (active): {activity_map.values[activity_map.values > 0].mean():.4f}")
    print("\nFirst 5 rows and columns:")
    print(activity_map.iloc[:5, :5])
    
    return activity_map


def main():
    parser = argparse.ArgumentParser(description="Inspect individual activity maps")
    parser.add_argument("--cid", type=int, help="Load maps for specific CID")
    parser.add_argument("--filename", help="Load specific map by filename (e.g., 1031_0.csv)")
    parser.add_argument("--list-all", action="store_true", help="List all available maps")
    parser.add_argument("--data-dir", default="data/01_raw", help="Data directory")
    
    args = parser.parse_args()
    
    if args.list_all:
        list_all_maps(args.data_dir)
    elif args.cid is not None:
        load_by_cid(args.cid, args.data_dir)
    elif args.filename:
        load_by_filename(args.filename, args.data_dir)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/inspect_activity_map.py --list-all")
        print("  python scripts/inspect_activity_map.py --cid 180")
        print("  python scripts/inspect_activity_map.py --filename 1031_0.csv")


if __name__ == "__main__":
    main()
