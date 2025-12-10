from __future__ import annotations
import argparse
from pathlib import Path
from src.neuro_foundation.data.pyrfume_loader import PyrfumeLoader
import os
import pyrfume


def main(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    loader = PyrfumeLoader(output_dir=output_dir)
    molecules = loader.load_molecules()
    images = loader.load_images()  # optional

    # Strictly load the leon behavior listing (behavior_1.csv) and save it
    behavior_df = pyrfume.load_data('leon/behavior_1.csv')
    behavior_path = Path(output_dir) / 'behavior_data.csv'
    behavior_df.to_csv(behavior_path, index=False)

    print(f"Saved raw molecules to {output_dir}/molecules_raw.csv ({molecules.shape})")
    if images is not None:
        print(f"Saved image data to {output_dir}/image_data.csv ({images.shape})")
    else:
        print("Image data not available; skipped saving.")
    print(f"Saved behavior listing to {behavior_path} ({behavior_df.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load raw data via Pyrfume")
    parser.add_argument("--output-dir", default="data/01_raw", help="Directory to save raw artifacts")
    args = parser.parse_args()
    main(args.output_dir)
