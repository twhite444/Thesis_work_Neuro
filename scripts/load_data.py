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
    behavior = pyrfume.load_data('leon/behavior_1.csv')
    
    # Save raw data to output_dir
    molecules.to_csv(os.path.join(output_dir, 'molecules_raw.csv'), index=False)
    behavior.to_csv(os.path.join(output_dir, 'behavior_data.csv'), index=False)
    
    print(f"Saved molecules to {output_dir}/molecules_raw.csv ({molecules.shape})")
    print(f"Saved behavior to {output_dir}/behavior_data.csv ({behavior.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load raw data via Pyrfume")
    parser.add_argument("--output-dir", default="data/01_raw", help="Directory to save raw artifacts")
    args = parser.parse_args()
    main(args.output_dir)
