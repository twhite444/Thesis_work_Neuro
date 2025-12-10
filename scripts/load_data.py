from __future__ import annotations
import argparse
import os
from pathlib import Path
import pyrfume


def main(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load manifests as in legacy
    _ = pyrfume.load_manifest('arshamian_2022')
    _ = pyrfume.load_manifest('leon')
    
    # Load molecules and behavior_1 from leon
    molecules = pyrfume.load_data('leon/molecules.csv')
    molecules.reset_index(inplace=True)
    molecules.rename(columns={'index': 'CID'}, inplace=True)

     # Check for duplicate CIDs
    duplicate_cids = molecules[molecules.duplicated(subset='CID', keep=False)]
    print(f"Duplicate CIDs in molecules before removal:\n{duplicate_cids}")

    # Remove duplicate CIDs by keeping the first occurrence
    molecules = molecules.drop_duplicates(subset='CID', keep='first')

    # Debug: Check for duplicates after removal
    duplicate_cids_after = molecules[molecules.duplicated(subset='CID', keep=False)]
    print(f"Duplicate CIDs in molecules after removal:\n{duplicate_cids_after}")


    # Load behavior data
    behavior = pyrfume.load_data('leon/behavior_1.csv')
    
    
    # Save raw data to output_dir
    molecules.to_csv(os.path.join(output_dir, 'molecules_raw.csv'), index=False)
    behavior.to_csv(os.path.join(output_dir, 'behavior_data.csv'), index=True)

    print(f"Saved molecules to {output_dir}/molecules_raw.csv ({molecules.shape})")
    print(f"Saved behavior to {output_dir}/behavior_data.csv ({behavior.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load raw data via Pyrfume")
    parser.add_argument("--output-dir", default="data/01_raw", help="Directory to save raw artifacts")
    args = parser.parse_args()
    main(args.output_dir)
