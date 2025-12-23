#!/usr/bin/env python3
"""Load all data from Pyrfume: molecules, behavior, and activity maps.

This script downloads data from Pyrfume and saves it in both formats:
- CSV: Human-readable, for inspection and compatibility
- NPZ: Compressed binary, for fast loading in pipelines

Usage:
    python scripts/load_all_data.py --output-dir data/01_raw
    python scripts/load_all_data.py --output-dir data/01_raw --skip-activity-maps
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neuro_foundation.data.pyrfume_loader import PyrfumeLoader


def main(output_dir: str, skip_activity_maps: bool = False):
    """Load all data from Pyrfume and save to disk."""
    loader = PyrfumeLoader(output_dir=output_dir)
    
    print("=" * 70)
    print("Loading molecules from Pyrfume...")
    print("=" * 70)
    molecules = loader.load_molecules()
    print(f"✓ Loaded {len(molecules)} unique molecules\n")
    
    print("=" * 70)
    print("Computing Mordred molecular descriptors...")
    print("=" * 70)
    mordred_features = loader.compute_mordred_features(molecules)
    print(f"✓ Computed {mordred_features.shape[1]} descriptors for {len(mordred_features)} molecules\n")
    
    print("=" * 70)
    print("Loading behavior data from Pyrfume...")
    print("=" * 70)
    behavior = loader.load_behavior()
    print(f"✓ Loaded {len(behavior)} behavior entries\n")
    
    print("=" * 70)
    print("Loading stimuli metadata from Pyrfume...")
    print("=" * 70)
    stimuli = loader.load_stimuli()
    print(f"✓ Loaded {len(stimuli)} stimuli entries\n")
    
    if not skip_activity_maps:
        print("=" * 70)
        print("Loading activity maps from Pyrfume (this may take a minute)...")
        print("=" * 70)
        # Always save individual CSVs by default
        activity_maps = loader.load_activity_maps(save_individual_csvs=True)
        print(f"✓ Loaded {len(activity_maps)} activity maps\n")
    else:
        print("Skipping activity maps (--skip-activity-maps flag set)\n")
    
    print("=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"  - molecules_raw.csv / .npz: {len(molecules)} molecules")
    print(f"  - mordred_features_raw.csv / .npz: {mordred_features.shape[1]} descriptors")
    print(f"  - behavior_data.csv / .npz: {len(behavior)} entries")
    print(f"  - stimuli_metadata.csv / .npz: {len(stimuli)} entries")
    if not skip_activity_maps:
        print(f"  - activity_maps.npz: {len(activity_maps)} maps")
        print(f"  - activity_maps_csv/: {len(activity_maps)} individual CSV files")
    print("\nFiles can be loaded using:")
    print("  from src.neuro_foundation.data.pyrfume_loader import (")
    print("      load_molecules_csv, load_mordred_features_npz,")
    print("      load_behavior_csv, load_stimuli_csv, load_activity_maps_npz")
    print("  )")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load all data from Pyrfume")
    parser.add_argument("--output-dir", default="data/01_raw", 
                        help="Directory to save raw data")
    parser.add_argument("--skip-activity-maps", action="store_true",
                        help="Skip loading activity maps (saves time)")
    args = parser.parse_args()
    main(args.output_dir, args.skip_activity_maps)
