from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neuro_foundation.pipeline.preprocess import featurize_and_standardize
from src.neuro_foundation.data.pyrfume_loader import (
    PyrfumeLoader,
    load_molecules_npz,
    load_molecules_csv,
)


def main(output_dir: str, use_cached: bool, data_dir: str):
    """Featurize molecules and standardize features.
    
    Args:
        output_dir: Directory to save processed data
        use_cached: If True, load from local cache; if False, download fresh data
        data_dir: Directory containing raw data files
    """
    if use_cached:
        # Use fast NPZ loading for cached data
        print(f"Loading molecules from {data_dir}/molecules_raw.npz...")
        molecules = load_molecules_npz(data_dir)
    else:
        # Download fresh data from Pyrfume
        print("Downloading fresh data from Pyrfume...")
        loader = PyrfumeLoader(output_dir=data_dir)
        molecules = loader.load_molecules()
    
    print(f"Loaded {len(molecules)} molecules")
    df = featurize_and_standardize(molecules, output_dir=output_dir)
    print(f"Saved cleaned data to {output_dir}/cleaned_data.csv ({df.shape})")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Featurize SMILES and standardize")
    parser.add_argument("--output-dir", default="data/02_processed", 
                       help="Directory to save cleaned data")
    parser.add_argument("--data-dir", default="data/01_raw",
                       help="Directory containing raw data files")
    parser.add_argument("--use-cached", action="store_true", default=True,
                       help="Use cached raw molecules (default: True, fast NPZ loading)")
    parser.add_argument("--force-download", action="store_true",
                       help="Force fresh download from Pyrfume (overrides --use-cached)")
    args = parser.parse_args()
    
    # If force-download is specified, set use_cached to False
    use_cached = args.use_cached and not args.force_download
    
    main(args.output_dir, use_cached, args.data_dir)
