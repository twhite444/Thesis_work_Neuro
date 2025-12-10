#!/usr/bin/env python3
"""Quick examples of loading activity maps by CID."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path (go up 2 levels from scripts/examples/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.neuro_foundation.data.pyrfume_loader import (
    load_activity_maps_by_cid,
    load_activity_map_by_cid_averaged,
    load_activity_maps_as_arrays,
)


def main():
    print("=" * 70)
    print("Loading Activity Maps by CID - Quick Examples")
    print("=" * 70)
    
    # Example 1: Load all maps for a specific CID
    print("\n1. Load all maps for CID 180:")
    maps = load_activity_maps_by_cid(180)
    print(f"   Found {len(maps)} maps for CID 180")
    for i, m in enumerate(maps, 1):
        coverage = (m > 0).mean() * 100
        print(f"   Map {i}: shape {m.shape}, coverage {coverage:.1f}%, "
              f"range [{m.min():.2f}, {m.max():.2f}]")
    
    # Example 2: Get averaged map for a CID
    print("\n2. Get averaged map for CID 180:")
    avg_map = load_activity_map_by_cid_averaged(180)
    if avg_map is not None:
        coverage = (avg_map > 0).mean() * 100
        print(f"   Averaged map: shape {avg_map.shape}, coverage {coverage:.1f}%")
        print(f"   Value range: [{avg_map.min():.4f}, {avg_map.max():.4f}]")
        print(f"   Mean (active pixels): {avg_map[avg_map > 0].mean():.4f}")
    
    # Example 3: Check a CID with no maps
    print("\n3. Try loading a non-existent CID:")
    maps = load_activity_maps_by_cid(99999)
    print(f"   Found {len(maps)} maps for CID 99999 (expected 0)")
    
    # Example 4: Find CIDs with most maps
    print("\n4. Find CIDs with multiple maps:")
    all_maps, all_cids = load_activity_maps_as_arrays()
    unique_cids, counts = np.unique(all_cids, return_counts=True)
    multi_map_cids = unique_cids[counts > 1]
    print(f"   {len(multi_map_cids)} CIDs have multiple activity maps")
    
    # Show top 5 CIDs with most maps
    top_indices = np.argsort(counts)[-5:][::-1]
    print("\n   Top 5 CIDs with most maps:")
    for idx in top_indices:
        cid = unique_cids[idx]
        count = counts[idx]
        maps_for_cid = load_activity_maps_by_cid(cid)
        avg_coverage = np.mean([(m > 0).mean() for m in maps_for_cid]) * 100
        print(f"   - CID {cid}: {count} maps (avg coverage: {avg_coverage:.1f}%)")
    
    # Example 5: Batch load multiple CIDs
    print("\n5. Batch load multiple CIDs:")
    test_cids = [180, 240, 58, 106]
    for cid in test_cids:
        maps = load_activity_maps_by_cid(cid)
        if maps:
            avg_map = load_activity_map_by_cid_averaged(cid)
            print(f"   CID {cid}: {len(maps)} map(s), "
                  f"avg coverage: {(avg_map > 0).mean()*100:.1f}%")
        else:
            print(f"   CID {cid}: No maps found")
    
    print("\n" + "=" * 70)
    print("Quick Reference:")
    print("=" * 70)
    print("""
    # Load all maps for a CID (returns list of numpy arrays)
    maps = load_activity_maps_by_cid(180)
    
    # Load averaged map for a CID (returns single numpy array)
    avg_map = load_activity_map_by_cid_averaged(180)
    
    # Load all maps and CIDs (for bulk processing)
    all_maps, all_cids = load_activity_maps_as_arrays()
    """)


if __name__ == "__main__":
    main()
