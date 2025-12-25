from __future__ import annotations
import argparse
from pathlib import Path

from neuro_foundation.pipeline.preprocess import featurize_and_standardize
from neuro_foundation.data.pyrfume_loader import (
    PyrfumeLoader,
    load_molecules_npz,
)


def main(
    output_dir: str, 
    use_cached: bool, 
    data_dir: str,
    variance_threshold: float,
    drop_nan: bool,
    drop_zero: bool,
    standardize: bool,
    save_intermediate: bool,
):
    """Featurize molecules and apply preprocessing pipeline.
    
    Args:
        output_dir: Directory to save processed data
        use_cached: If True, load from local cache; if False, download fresh data
        data_dir: Directory containing raw data files
        variance_threshold: Minimum variance for feature selection
        drop_nan: Whether to drop columns with NaN
        drop_zero: Whether to drop zero-only columns
        standardize: Whether to standardize features
        save_intermediate: Whether to save intermediate outputs
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
    
    print(f"Loaded {len(molecules)} molecules\n")
    print("="*60)
    print("PREPROCESSING PIPELINE")
    print("="*60)
    
    df = featurize_and_standardize(
        molecules=molecules,
        variance_threshold=variance_threshold,
        drop_nan_columns=drop_nan,
        drop_zero_columns=drop_zero,
        standardize=standardize,
        output_dir=output_dir,
        save_intermediate=save_intermediate,
    )
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Output: {output_dir}/cleaned_data.csv")
    print(f"Shape: {df.shape[0]} samples × {df.shape[1]} features")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Featurize SMILES and apply configurable preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: drop NaN/zeros, remove constants, standardize
  python scripts/preprocess.py
  
  # Remove low-variance features (threshold=0.01)
  python scripts/preprocess.py --variance-threshold 0.01
  
  # Keep raw features (no standardization)
  python scripts/preprocess.py --no-standardize
  
  # Save both unscaled and scaled features
  python scripts/preprocess.py --save-intermediate
  
  # Minimal processing (only drop NaN)
  python scripts/preprocess.py --variance-threshold 0 --no-drop-zero --no-standardize
        """
    )
    
    # Data loading arguments
    parser.add_argument("--output-dir", default="data/02_processed", 
                       help="Directory to save processed data (default: data/02_processed)")
    parser.add_argument("--data-dir", default="data/01_raw",
                       help="Directory containing raw data files (default: data/01_raw)")
    parser.add_argument("--force-download", action="store_true",
                       help="Force fresh download from Pyrfume (slower, requires internet)")
    parser.add_argument("--no-cache", action="store_true",
                       help="Don't use cached molecules, download fresh data (same as --force-download)")
    
    # Feature selection and filtering arguments
    parser.add_argument("--variance-threshold", type=float, default=0.0,
                       help="Variance threshold for feature selection (default: 0.0 = remove only constants). "
                            "Applied BEFORE standardization. Try 0.01 or 0.1 to remove low-variance features.")
    parser.add_argument("--no-drop-nan", action="store_true",
                       help="Don't drop columns with NaN values (default: drop NaN)")
    parser.add_argument("--no-drop-zero", action="store_true",
                       help="Don't drop zero-only columns (default: drop zeros)")
    
    # Standardization arguments
    parser.add_argument("--no-standardize", action="store_true",
                       help="Don't standardize features (default: standardize with StandardScaler)")
    
    # Output arguments
    parser.add_argument("--save-intermediate", action="store_true",
                       help="Save intermediate unscaled features to unscaled_features.csv")
    
    args = parser.parse_args()
    
    # Use cached by default, unless --force-download or --no-cache is specified
    use_cached = not (args.force_download or args.no_cache)
    
    main(
        output_dir=args.output_dir,
        use_cached=use_cached,
        data_dir=args.data_dir,
        variance_threshold=args.variance_threshold,
        drop_nan=not args.no_drop_nan,
        drop_zero=not args.no_drop_zero,
        standardize=not args.no_standardize,
        save_intermediate=args.save_intermediate,
    )
