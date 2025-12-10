#!/usr/bin/env python3
"""Inspect individual activity maps by CID or filename.

Usage:
    # List all available maps
    python scripts/inspect_activity_map.py --list-all
    
    # Show stats for a specific CID
    python scripts/inspect_activity_map.py --cid 180
    
    # Show stats AND interactive visualization (opens popup window)
    python scripts/inspect_activity_map.py --cid 180 --show-images
    
    # Show stats, visualization, AND save to file
    python scripts/inspect_activity_map.py --cid 180 --show-images --save-images
    
    # Inspect a specific map file
    python scripts/inspect_activity_map.py --filename 1031_0.csv
"""
import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neuro_foundation.data.pyrfume_loader import (
    load_activity_maps_by_cid,
    load_stimuli_npz,
)


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


def visualize_maps_for_cid(cid: int, data_dir: str = "data/01_raw", save_path: str = None):
    """Create a visualization showing all activity maps for a specific CID."""
    # Load maps using the efficient NPZ loader
    maps = load_activity_maps_by_cid(cid, data_dir)
    
    if len(maps) == 0:
        print(f"No activity maps found for CID {cid}")
        return None
    
    # Load stimuli metadata to get the name
    try:
        stimuli = load_stimuli_npz(data_dir)
        cid_info = stimuli[stimuli['CID'] == cid]
        molecule_name = cid_info.iloc[0]['Name'] if len(cid_info) > 0 else f"CID {cid}"
    except:
        molecule_name = f"CID {cid}"
    
    # Create figure with subplots for each map
    n_maps = len(maps)
    ncols = min(3, n_maps)  # Max 3 columns
    nrows = (n_maps + ncols - 1) // ncols  # Ceiling division
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    fig.suptitle(f'Activity Maps for {molecule_name} (CID {cid})', 
                 fontsize=14, fontweight='bold')
    
    # Flatten axes array for easier iteration
    if n_maps == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Find global min/max for consistent color scaling (only for non-zero values)
    # We'll mask zeros as NaN for visualization
    non_zero_values = [m[m != 0] for m in maps if (m != 0).any()]
    if non_zero_values:
        vmin = min(v.min() for v in non_zero_values)
        vmax = max(v.max() for v in non_zero_values)
    else:
        vmin, vmax = 0, 1
    
    # Plot each map
    for i, (ax, map_data) in enumerate(zip(axes, maps)):
        # Create a copy with zeros replaced by NaN for visualization only
        display_data = map_data.copy()
        display_data[display_data == 0] = np.nan
        
        im = ax.imshow(display_data, cmap='viridis', aspect='auto', 
                       vmin=vmin, vmax=vmax, interpolation='nearest')
        
        coverage = (map_data > 0).mean() * 100
        mean_active = map_data[map_data > 0].mean() if (map_data > 0).any() else 0
        
        ax.set_title(f'Map {i+1}\nCoverage: {coverage:.1f}%, Mean: {mean_active:.2f}',
                    fontsize=10)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        
        # Add colorbar for each subplot
        plt.colorbar(im, ax=ax, label='Activity')
    
    # Hide extra subplots if any
    for i in range(n_maps, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
        plt.show(block=True)  # Still show even if saved
    else:
        plt.show(block=True)  # Block until user closes window
    
    return fig


def load_by_cid(cid: int, data_dir: str = "data/01_raw", show_images: bool = False, save_images: bool = False):
    """Load all activity maps for a specific CID.
    
    Args:
        cid: The CID to load maps for
        data_dir: Directory containing the data
        show_images: If True, display interactive visualization window
        save_images: If True, save visualization to file (only works if show_images is True)
    """
    behavior = pd.read_csv(os.path.join(data_dir, 'behavior_data.csv'))
    maps_for_cid = behavior[behavior['Stimulus'] == cid]
    
    if len(maps_for_cid) == 0:
        print(f"No activity maps found for CID {cid}")
        return
    
    # Get molecule name from stimuli metadata
    try:
        stimuli = load_stimuli_npz(data_dir)
        cid_info = stimuli[stimuli['CID'] == cid]
        if len(cid_info) > 0:
            molecule_name = cid_info.iloc[0]['Name']
            print(f"\nCID {cid}: {molecule_name}")
            print(f"Has {len(maps_for_cid)} activity map(s):\n")
        else:
            print(f"\nCID {cid} has {len(maps_for_cid)} activity map(s):\n")
    except:
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
    
    # Show visualization if requested
    if show_images:
        save_path = None
        if save_images:
            save_path = os.path.join(data_dir, f'activity_map_CID_{cid}.png')
        
        visualize_maps_for_cid(cid, data_dir, save_path=save_path)
        
        if save_path:
            print(f"\n💡 Visualization saved to: {save_path}")
        else:
            print(f"\n💡 Close the visualization window to continue...")


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
    parser.add_argument("--show-images", action="store_true", 
                        help="Show visualization of activity maps (only works with --cid)")
    parser.add_argument("--save-images", action="store_true",
                        help="Save visualization to file (requires --show-images)")
    parser.add_argument("--data-dir", default="data/01_raw", help="Data directory")
    
    args = parser.parse_args()
    
    if args.list_all:
        list_all_maps(args.data_dir)
    elif args.cid is not None:
        load_by_cid(args.cid, args.data_dir, show_images=args.show_images, save_images=args.save_images)
    elif args.filename:
        load_by_filename(args.filename, args.data_dir)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/inspect_activity_map.py --list-all")
        print("  python scripts/inspect_activity_map.py --cid 180")
        print("  python scripts/inspect_activity_map.py --cid 180 --show-images")
        print("  python scripts/inspect_activity_map.py --cid 180 --show-images --save-images")
        print("  python scripts/inspect_activity_map.py --filename 1031_0.csv")


if __name__ == "__main__":
    main()
