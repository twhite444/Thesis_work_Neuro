from __future__ import annotations
import argparse
from pathlib import Path
from src.neuro_foundation.data.pyrfume_loader import PyrfumeLoader


def main(output_dir: str):
    loader = PyrfumeLoader(output_dir=output_dir)
    molecules = loader.load_molecules()
    _ = loader.load_images()  # optional
    print(f"Saved raw molecules to {output_dir}/molecules_raw.csv ({molecules.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load raw data via Pyrfume")
    parser.add_argument("--output-dir", default="data/01_raw", help="Directory to save raw artifacts")
    args = parser.parse_args()
    main(args.output_dir)
