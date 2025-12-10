import argparse
import sys
from pathlib import Path
# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neuro_foundation.pipeline.activity_maps import pipeline_load_and_mask


def main():
    parser = argparse.ArgumentParser(description='Run activity maps load+mask pipeline using local data.')
    parser.add_argument('--directory-csv', type=str, default='data/01_raw/behavior_data.csv', 
                       help='Path to behavior/activity CSV')
    parser.add_argument('--data-dir', type=str, default='data/01_raw', 
                       help='Directory containing activity_maps_csv/ folder')
    parser.add_argument('--coverage-threshold', type=float, default=0.5, 
                       help='Fraction of maps required to consider a pixel covered')
    parser.add_argument('--output-dir', type=str, default='data/02_processed', 
                       help='Directory to save outputs')
    parser.add_argument('--verbose', action='store_true', help='Print directory info')
    args = parser.parse_args()

    maps, cids, mask = pipeline_load_and_mask(
        directory_csv=args.directory_csv,
        data_dir=args.data_dir,
        coverage_threshold=args.coverage_threshold,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )
    print(f"Averaged maps: {len(maps)} | Unique CIDs: {len(cids)} | Mask shape: {mask.shape}")


if __name__ == '__main__':
    main()
