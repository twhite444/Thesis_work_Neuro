from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from scipy.ndimage import label, binary_dilation, binary_erosion
import matplotlib.pyplot as plt


@dataclass
class ActivityMapRecord:
    cid: int
    map: np.ndarray


def load_directory_csv(path: str) -> pd.DataFrame:
    """Load behavior/activity directory CSV and derive CID column from Stimulus.
    Expects columns: Stimulus, Activity Map Path
    """
    df = pd.read_csv(path)
    if 'Stimulus' not in df.columns or 'Activity Map Path' not in df.columns:
        raise ValueError("Expected columns 'Stimulus' and 'Activity Map Path' in directory CSV")
    df['CID'] = df['Stimulus'].map(lambda x: int(str(x).split('_')[0]))
    df = df[df['CID'] > 0]
    return df


def load_activity_maps(directory_df: pd.DataFrame, data_dir: str = 'data/01_raw') -> List[ActivityMapRecord]:
    """Load activity maps from local CSV files in data_dir/activity_maps_csv/.
    
    Args:
        directory_df: DataFrame with 'CID' and 'Activity Map Path' columns
        data_dir: Base directory containing activity_maps_csv/ folder
    
    Returns:
        List of ActivityMapRecord objects
    """
    records: List[ActivityMapRecord] = []
    total = len(directory_df)
    csv_dir = os.path.join(data_dir, 'activity_maps_csv')
    
    if not os.path.exists(csv_dir):
        raise FileNotFoundError(f"Activity maps CSV directory not found: {csv_dir}. "
                               f"Run 'python scripts/load_all_data.py' first to download the data.")
    
    for i, (_, row) in enumerate(directory_df.iterrows(), start=1):
        print(f'\rLoading maps: {i}/{total}', end='', flush=True)
        
        # Get the filename from the path
        map_filename = os.path.basename(row['Activity Map Path'])
        map_path = os.path.join(csv_dir, map_filename)
        
        if not os.path.exists(map_path):
            print(f"\nWarning: Map file not found: {map_path}, skipping...")
            continue
            
        # Load CSV and convert to numpy array
        map_df = pd.read_csv(map_path, index_col=0)
        arr = np.nan_to_num(map_df.to_numpy(), nan=0)
        records.append(ActivityMapRecord(cid=int(row['CID']), map=arr))
    
    print()
    return records


def compute_global_mask(records: List[ActivityMapRecord], coverage_threshold: float) -> np.ndarray:
    """Compute a global mask based on coverage across maps.
    coverage_threshold in [0,1] indicates fraction of maps that must have non-zero values.
    Note: Since NaNs are converted to 0 during loading, we count non-zero pixels.
    """
    if not records:
        raise ValueError("No activity maps provided")
    shape = records[0].map.shape
    valid_counts = np.zeros(shape, dtype=int)
    for r in records:
        valid_counts += (r.map != 0)  # Count non-zero pixels, not non-NaN
    required = int(coverage_threshold * len(records))
    global_mask = valid_counts >= max(required, 1)
    refined = binary_erosion(binary_dilation(global_mask))
    # Keep sufficiently large regions (>=100 pixels) similar to legacy
    labeled_mask, _ = label(refined)
    counts = np.bincount(labeled_mask.ravel())
    valid_regions = np.isin(labeled_mask, np.where(counts >= 100)[0])
    refined &= valid_regions
    return refined


def apply_mask(records: List[ActivityMapRecord], mask: np.ndarray) -> List[ActivityMapRecord]:
    """Apply global mask to each map and return masked records."""
    masked_records: List[ActivityMapRecord] = []
    for r in records:
        arr = np.nan_to_num(r.map, nan=0.0)
        masked = arr * mask
        masked_records.append(ActivityMapRecord(cid=r.cid, map=masked))
    return masked_records


def average_by_cid(records: List[ActivityMapRecord]) -> Tuple[List[np.ndarray], List[int]]:
    """Average masked maps per CID (to handle multiple maps per molecule)."""
    by_cid: Dict[int, List[np.ndarray]] = {}
    for r in records:
        by_cid.setdefault(r.cid, []).append(r.map)
    averaged_maps: List[np.ndarray] = []
    cids: List[int] = []
    for cid, maps in by_cid.items():
        stacked = np.stack(maps, axis=0)
        mean_map = np.nanmean(stacked, axis=0)
        averaged_maps.append(mean_map)
        cids.append(cid)
    return averaged_maps, cids


def visualize_map(arr: np.ndarray, title: str, output_path: str, cmap: str = 'viridis') -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(arr, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def visualize_global_mask(mask: np.ndarray, output_path: str) -> None:
    visualize_map(mask.astype(float), title='Refined Global Mask', output_path=output_path, cmap='gray')


def visualize_coverage(valid_counts: np.ndarray, output_path: str) -> None:
    """Visualize coverage counts with zeros masked as NaN for clarity."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a copy with zeros replaced by NaN for visualization
    display_data = valid_counts.astype(float).copy()
    display_data[display_data == 0] = np.nan
    
    plt.figure(figsize=(8, 6))
    im = plt.imshow(display_data, cmap='magma', interpolation='nearest')
    plt.colorbar(im, label='Number of maps covering pixel')
    plt.title('Coverage Counts (zeros = no coverage)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_coverage_histogram(valid_counts: np.ndarray, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    flat = valid_counts.ravel()
    plt.hist(flat, bins=20, color='steelblue', alpha=0.8)
    plt.title('Pixel Coverage Histogram')
    plt.xlabel('Number of maps covering pixel')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def pipeline_load_and_mask(directory_csv: str, data_dir: str = 'data/01_raw', coverage_threshold: float = 1.0,
                            output_dir: str = 'output_data', verbose: bool = False) -> Tuple[List[np.ndarray], List[int], np.ndarray]:
    """High-level function: load directory, load maps, compute+apply mask, average by CID, and visualize.
    
    Args:
        directory_csv: Path to behavior CSV with activity map paths
        data_dir: Directory containing activity_maps_csv/ folder (default: 'data/01_raw')
        coverage_threshold: Fraction of maps required to consider a pixel covered
        output_dir: Directory to save visualizations
        verbose: Print debug information
    
    Returns:
        Tuple of (averaged_maps, cids, mask)
    """
    os.makedirs(output_dir, exist_ok=True)
    df = load_directory_csv(directory_csv)
    if verbose:
        print(f"Directory rows: {len(df)}")
        print(df.head())
    records = load_activity_maps(df, data_dir=data_dir)
    # coverage visualization (before mask)
    # Count how many maps have non-zero values at each pixel (since we convert NaN to 0 in loading)
    shape = records[0].map.shape if records else (0, 0)
    valid_counts = np.zeros(shape, dtype=int)
    for r in records:
        valid_counts += (r.map != 0)  # Count non-zero pixels, not non-NaN
    visualize_coverage(valid_counts, os.path.join(output_dir, 'coverage_counts.png'))
    visualize_coverage_histogram(valid_counts, os.path.join(output_dir, 'coverage_histogram.png'))

    mask = compute_global_mask(records, coverage_threshold=coverage_threshold)
    masked_records = apply_mask(records, mask)
    averaged_maps, cids = average_by_cid(masked_records)
    # Visualizations
    visualize_global_mask(mask, os.path.join(output_dir, 'global_mask.png'))
    # Save example masked maps and a small gallery if available
    if averaged_maps:
        visualize_map(averaged_maps[0], title=f'Masked Averaged Map CID={cids[0]}',
                      output_path=os.path.join(output_dir, 'masked_averaged_example.png'))
        # Gallery of up to 6 averaged maps
        n = min(len(averaged_maps), 6)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = np.array(axes).reshape(-1)
        for i in range(n):
            ax = axes[i]
            ax.imshow(averaged_maps[i], cmap='viridis')
            ax.set_title(f'CID={cids[i]}')
            ax.axis('off')
        # Hide unused axes
        for j in range(n, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'masked_averaged_gallery.png'), dpi=300)
        plt.close()
    return averaged_maps, cids, mask
