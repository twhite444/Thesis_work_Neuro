#!/usr/bin/env python3
"""Example: Load cached data using helper functions.

Demonstrates fast loading from NPZ files vs CSV files.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

# Add project root to path (go up 2 levels from scripts/examples/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.neuro_foundation.data.pyrfume_loader import (
    load_molecules_csv,
    load_molecules_npz,
    load_behavior_csv,
    load_behavior_npz,
    load_activity_maps_npz,
    load_activity_maps_as_arrays,
)


def benchmark_loading():
    """Compare loading times for CSV vs NPZ formats."""
    print("=" * 70)
    print("Benchmarking data loading speeds")
    print("=" * 70)
    
    # Molecules
    print("\n1. Loading molecules...")
    start = time.time()
    molecules_csv = load_molecules_csv()
    csv_time = time.time() - start
    print(f"   CSV:  {csv_time*1000:.2f} ms → {len(molecules_csv)} molecules")
    
    start = time.time()
    molecules_npz = load_molecules_npz()
    npz_time = time.time() - start
    print(f"   NPZ:  {npz_time*1000:.2f} ms → {len(molecules_npz)} molecules")
    print(f"   Speedup: {csv_time/npz_time:.1f}x faster")
    
    # Behavior
    print("\n2. Loading behavior...")
    start = time.time()
    behavior_csv = load_behavior_csv()
    csv_time = time.time() - start
    print(f"   CSV:  {csv_time*1000:.2f} ms → {len(behavior_csv)} entries")
    
    start = time.time()
    behavior_npz = load_behavior_npz()
    npz_time = time.time() - start
    print(f"   NPZ:  {npz_time*1000:.2f} ms → {len(behavior_npz)} entries")
    print(f"   Speedup: {csv_time/npz_time:.1f}x faster")
    
    # Activity maps (only NPZ format available)
    print("\n3. Loading activity maps...")
    start = time.time()
    activity_records = load_activity_maps_npz()
    load_time = time.time() - start
    print(f"   NPZ (as records): {load_time*1000:.2f} ms → {len(activity_records)} maps")
    
    start = time.time()
    maps_array, cids_array = load_activity_maps_as_arrays()
    array_time = time.time() - start
    print(f"   NPZ (as arrays):  {array_time*1000:.2f} ms → shape {maps_array.shape}")
    
    print("\n" + "=" * 70)
    print("Recommendation: Use NPZ format for production pipelines")
    print("                Use CSV format for inspection/debugging")
    print("=" * 70)


def usage_examples():
    """Show common usage patterns."""
    print("\n\n" + "=" * 70)
    print("Usage Examples")
    print("=" * 70)
    
    # Load molecules and access SMILES
    molecules = load_molecules_npz()
    print("\nExample 1: Access molecule SMILES")
    print(f"  First molecule: CID={molecules['CID'].iloc[0]}, "
          f"SMILES={molecules['IsomericSMILES'].iloc[0]}")
    
    # Load behavior and filter by CID
    behavior = load_behavior_npz()
    print("\nExample 2: Filter behavior by specific CID")
    cid = 180
    filtered = behavior[behavior['Stimulus'] == cid]
    print(f"  CID {cid} has {len(filtered)} activity map(s)")
    
    # Load activity maps efficiently
    maps, cids = load_activity_maps_as_arrays()
    print("\nExample 3: Process all activity maps")
    print(f"  Total maps: {len(cids)}")
    print(f"  Unique CIDs: {len(set(cids))}")
    print(f"  Map dimensions: {maps.shape[1]}x{maps.shape[2]}")
    print(f"  Average coverage: {(maps > 0).mean(axis=(1,2)).mean():.1%} of pixels active")
    
    # Load activity maps by CID
    from src.neuro_foundation.data.pyrfume_loader import (
        load_activity_maps_by_cid,
        load_activity_map_by_cid_averaged
    )
    print("\nExample 4: Load activity maps for a specific CID")
    maps_for_cid = load_activity_maps_by_cid(180)
    print(f"  CID 180 has {len(maps_for_cid)} maps")
    if maps_for_cid:
        print(f"  First map shape: {maps_for_cid[0].shape}")
        print(f"  Coverage: {(maps_for_cid[0] > 0).mean():.1%}")
    
    print("\nExample 5: Get averaged activity map for a CID")
    avg_map = load_activity_map_by_cid_averaged(180)
    if avg_map is not None:
        print(f"  Averaged map shape: {avg_map.shape}")
        print(f"  Coverage: {(avg_map > 0).mean():.1%}")
        print(f"  Value range: [{avg_map.min():.4f}, {avg_map.max():.4f}]")


if __name__ == "__main__":
    try:
        benchmark_loading()
        usage_examples()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run 'python scripts/load_all_data.py' first to generate data files.")
        sys.exit(1)
