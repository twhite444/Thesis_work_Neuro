import argparse
from src.neuro_foundation.pipeline.activity_maps import pipeline_load_and_mask


def main():
    parser = argparse.ArgumentParser(description='Run activity maps load+mask pipeline (no PCA).')
    parser.add_argument('--directory-csv', type=str, default='output_data/behavior_data.csv', help='Path to behavior/activity CSV')
    parser.add_argument('--base-path', type=str, default='leon', help='Base path for activity maps relative to pyrfume datasets')
    parser.add_argument('--coverage-threshold', type=float, default=0.5, help='Fraction of maps required to consider a pixel covered')
    parser.add_argument('--output-dir', type=str, default='output_data', help='Directory to save outputs')
    parser.add_argument('--verbose', action='store_true', help='Print directory info')
    args = parser.parse_args()

    maps, cids, mask = pipeline_load_and_mask(
        directory_csv=args.directory_csv,
        base_path=args.base_path,
        coverage_threshold=args.coverage_threshold,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )
    print(f"Averaged maps: {len(maps)} | Unique CIDs: {len(cids)} | Mask shape: {mask.shape}")


if __name__ == '__main__':
    main()
