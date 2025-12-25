from __future__ import annotations
import os
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from scipy.ndimage import label, binary_dilation, binary_erosion
import matplotlib.pyplot as plt


class SelectionStrategy(str, Enum):
    """Available activity map selection strategies."""
    BEST_QUALITY = "best_quality"  # Use composite score based on coverage, range, mean
    AVERAGE = "average"             # Average all maps per CID
    MEDIAN = "median"               # Median of maps per CID (robust to outliers)
    FIRST = "first"                 # Just use first map (simple baseline)


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
        arr = map_df.to_numpy()
        records.append(ActivityMapRecord(cid=int(row['CID']), map=arr))
    
    print()
    return records


def compute_global_mask(records: List[ActivityMapRecord], coverage_threshold: float, min_region_size: int = 100) -> np.ndarray:
    """Compute a global mask based on coverage across maps.
    coverage_threshold in [0,1] indicates fraction of maps that must have finite values.
    Coverage is defined as pixel existence (np.isfinite).
    """
    if not records:
        raise ValueError("No activity maps provided")
    shape = records[0].map.shape
    valid_counts = np.zeros(shape, dtype=int)
    for r in records:
        valid_counts += np.isfinite(r.map).astype(int)  # Count finite pixels
    required = math.ceil(coverage_threshold * len(records))
    global_mask = valid_counts >= max(required, 1)
    refined = binary_erosion(binary_dilation(global_mask))
    # Keep sufficiently large regions
    labeled_mask, _ = label(refined)
    counts = np.bincount(labeled_mask.ravel())
    valid_regions = np.isin(labeled_mask, np.where(counts >= min_region_size)[0])
    refined &= valid_regions
    return refined


def apply_mask(records: List[ActivityMapRecord], mask: np.ndarray) -> List[ActivityMapRecord]:
    """Apply global mask to each map and return masked records.
    Pixels outside the mask become NaN, pixels inside keep original values.
    """
    masked_records: List[ActivityMapRecord] = []
    for r in records:
        masked = np.where(mask, r.map, np.nan)
        masked_records.append(ActivityMapRecord(cid=r.cid, map=masked))
    return masked_records


def apply_value_policy(map_arr: np.ndarray, mask: np.ndarray, value_policy: str) -> np.ndarray:
    """Apply value filtering policy inside the ROI only.
    
    Args:
        map_arr: Activity map array
        mask: Boolean ROI mask
        value_policy: "all", "pos", or "neg"
        
    Returns:
        Array with policy applied inside ROI, dropped values set to 0
    """
    result = map_arr.copy()
    if value_policy == "all":
        pass  # Keep all values
    elif value_policy == "pos":
        # Keep only >0 inside ROI, set others to 0
        inside_roi = mask
        result[inside_roi & (result <= 0)] = 0.0
    elif value_policy == "neg":
        # Keep only <0 inside ROI, set others to 0
        inside_roi = mask
        result[inside_roi & (result >= 0)] = 0.0
    else:
        raise ValueError(f"Unknown value_policy: {value_policy}")
    return result


def mask_for_display(arr: np.ndarray, mask: np.ndarray, hide_zeros_inside_roi: bool = False) -> np.ndarray:
    """Prepare array for visualization with proper masking.
    
    Args:
        arr: Activity map array
        mask: Boolean ROI mask
        hide_zeros_inside_roi: If True, set zeros inside ROI to NaN for readability
        
    Returns:
        Array suitable for visualization (outside ROI = NaN)
    """
    display_arr = arr.copy()
    # Outside ROI always NaN
    display_arr[~mask] = np.nan
    # Optionally hide zeros inside ROI
    if hide_zeros_inside_roi:
        display_arr[mask & (display_arr == 0)] = np.nan
    return display_arr


def compute_map_stats(map_arr: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Compute robust statistics for a single activity map inside ROI.
    
    Args:
        map_arr: Activity map array
        mask: Boolean ROI mask
        
    Returns:
        Dictionary with statistics
    """
    roi_pixels = map_arr[mask]
    roi_finite_mask = np.isfinite(roi_pixels)
    roi_finite = roi_pixels[roi_finite_mask]
    
    stats = {
        'roi_total_pixels': mask.sum(),
        'roi_finite_pixels': len(roi_finite),
        'roi_finite_fraction': len(roi_finite) / mask.sum() if mask.sum() > 0 else 0.0,
    }
    
    if len(roi_finite) > 0:
        stats.update({
            'roi_min': float(roi_finite.min()),
            'roi_max': float(roi_finite.max()),
            'roi_mean': float(roi_finite.mean()),
            'roi_std': float(roi_finite.std()),
        })
        
        # Positive values
        pos_mask = roi_finite > 0
        pos_vals = roi_finite[pos_mask]
        stats['n_pos_pixels'] = len(pos_vals)
        stats['pos_frac_in_roi'] = len(pos_vals) / len(roi_finite) if len(roi_finite) > 0 else 0.0
        if len(pos_vals) > 0:
            stats.update({
                'pos_min': float(pos_vals.min()),
                'pos_max': float(pos_vals.max()),
                'pos_mean': float(pos_vals.mean()),
                'pos_std': float(pos_vals.std()),
            })
        
        # Negative values
        neg_mask = roi_finite < 0
        neg_vals = roi_finite[neg_mask]
        stats['n_neg_pixels'] = len(neg_vals)
        stats['neg_frac_in_roi'] = len(neg_vals) / len(roi_finite) if len(roi_finite) > 0 else 0.0
        if len(neg_vals) > 0:
            stats.update({
                'neg_min': float(neg_vals.min()),
                'neg_max': float(neg_vals.max()),
                'neg_mean': float(neg_vals.mean()),
                'neg_std': float(neg_vals.std()),
            })
        
        # Nonzero values
        nonzero_mask = roi_finite != 0
        nonzero_vals = roi_finite[nonzero_mask]
        stats['n_nonzero_pixels'] = len(nonzero_vals)
        stats['nonzero_frac_in_roi'] = len(nonzero_vals) / len(roi_finite) if len(roi_finite) > 0 else 0.0
        if len(nonzero_vals) > 0:
            stats.update({
                'nonzero_min': float(nonzero_vals.min()),
                'nonzero_max': float(nonzero_vals.max()),
                'nonzero_mean': float(nonzero_vals.mean()),
                'nonzero_std': float(nonzero_vals.std()),
            })
    
    return stats


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


# ============================================================================
# Selection Strategy Functions
# ============================================================================

def _compute_map_quality_metrics(map_data: np.ndarray) -> Dict[str, float]:
    """Compute quality metrics for a single activity map.
    
    Args:
        map_data: Activity map as numpy array
        
    Returns:
        Dictionary with metrics: coverage_frac, range, mean_active, std_active
    """
    total_pixels = map_data.size
    active_mask = map_data > 0
    active_pixels = active_mask.sum()
    
    coverage_frac = active_pixels / total_pixels
    
    min_val = map_data.min()
    max_val = map_data.max()
    value_range = max_val - min_val
    
    active_vals = map_data[active_mask]
    mean_active = active_vals.mean() if active_vals.size > 0 else 0.0  # NumPy: use .size
    std_active = active_vals.std() if active_vals.size > 0 else 0.0
    
    return {
        'coverage_frac': coverage_frac,
        'range': value_range,
        'mean_active': mean_active,
        'std_active': std_active,
    }


def _z_score(values: List[float]) -> List[float]:
    """Compute z-scores for a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        List of z-scores (standardized values)
    """
    if len(values) <= 1:
        return [0.0] * len(values)
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample standard deviation
    
    if std == 0:
        return [0.0] * len(values)
    
    return [(v - mean) / std for v in values]


def select_best_by_quality(records: List[ActivityMapRecord]) -> Tuple[List[np.ndarray], List[int], Dict]:
    """Select best map per CID using composite quality score.
    
    Composite score = z(coverage) + z(range) - 0.5 * z(mean_active)
    
    Higher coverage and range are better (more informative maps).
    Lower mean_active can be better (less baseline activity, more dynamic range).
    
    Args:
        records: All activity map records
        
    Returns:
        Tuple of (selected_maps, cids, metadata)
    """
    by_cid: Dict[int, List[ActivityMapRecord]] = {}
    for r in records:
        by_cid.setdefault(r.cid, []).append(r)
    
    selected_maps: List[np.ndarray] = []
    cids: List[int] = []
    metadata = {
        'n_single_map': 0,
        'n_multi_map': 0,
        'selection_details': []
    }
    
    for cid, cid_records in sorted(by_cid.items()):
        n_maps = len(cid_records)
        
        if n_maps == 1:
            # Only one map, no selection needed
            selected_maps.append(cid_records[0].map)
            cids.append(cid)
            metadata['n_single_map'] += 1
            continue
        
        # Multiple maps: compute quality metrics
        map_metrics = []
        for rec in cid_records:
            metrics = _compute_map_quality_metrics(rec.map)
            map_metrics.append(metrics)
        
        # Extract metric lists
        coverages = [m['coverage_frac'] for m in map_metrics]
        ranges = [m['range'] for m in map_metrics]
        means_active = [m['mean_active'] for m in map_metrics]
        
        # Compute z-scores
        z_coverages = _z_score(coverages)
        z_ranges = _z_score(ranges)
        z_means = _z_score(means_active)
        
        # Compute composite score
        scores = []
        for i in range(n_maps):
            score = z_coverages[i] + z_ranges[i] - 0.5 * z_means[i]
            scores.append(score)
        
        # Select best
        best_idx = int(np.argmax(scores))
        selected_maps.append(cid_records[best_idx].map)
        cids.append(cid)
        metadata['n_multi_map'] += 1
        
        metadata['selection_details'].append({
            'cid': cid,
            'n_maps': n_maps,
            'selected_idx': best_idx,
            'scores': scores,
        })
    
    return selected_maps, cids, metadata


def select_by_averaging(records: List[ActivityMapRecord]) -> Tuple[List[np.ndarray], List[int], Dict]:
    """Select maps by averaging all maps per CID.
    
    Most conservative approach - combines all maps for each molecule.
    
    Args:
        records: All activity map records
        
    Returns:
        Tuple of (selected_maps, cids, metadata)
    """
    averaged_maps, cids = average_by_cid(records)
    
    metadata = {
        'strategy': 'average',
        'n_molecules': len(cids),
    }
    
    return averaged_maps, cids, metadata


def select_by_median(records: List[ActivityMapRecord]) -> Tuple[List[np.ndarray], List[int], Dict]:
    """Select maps by taking median across all maps per CID.
    
    More robust to outliers than averaging.
    
    Args:
        records: All activity map records
        
    Returns:
        Tuple of (selected_maps, cids, metadata)
    """
    by_cid: Dict[int, List[np.ndarray]] = {}
    for r in records:
        by_cid.setdefault(r.cid, []).append(r.map)
    
    median_maps: List[np.ndarray] = []
    cids: List[int] = []
    
    for cid, maps in sorted(by_cid.items()):
        stacked = np.stack(maps, axis=0)
        median_map = np.nanmedian(stacked, axis=0)
        median_maps.append(median_map)
        cids.append(cid)
    
    metadata = {
        'strategy': 'median',
        'n_molecules': len(cids),
    }
    
    return median_maps, cids, metadata


def select_first_map(records: List[ActivityMapRecord]) -> Tuple[List[np.ndarray], List[int], Dict]:
    """Select first map for each CID (simple baseline).
    
    Args:
        records: All activity map records
        
    Returns:
        Tuple of (selected_maps, cids, metadata)
    """
    by_cid: Dict[int, ActivityMapRecord] = {}
    for r in records:
        if r.cid not in by_cid:
            by_cid[r.cid] = r
    
    selected_maps = [r.map for r in sorted(by_cid.values(), key=lambda x: x.cid)]
    cids = sorted(by_cid.keys())
    
    metadata = {
        'strategy': 'first',
        'n_molecules': len(cids),
    }
    
    return selected_maps, cids, metadata


def select_maps_by_strategy(
    records: List[ActivityMapRecord],
    strategy: SelectionStrategy,
    **kwargs
) -> Tuple[List[np.ndarray], List[int], Dict]:
    """Select one map per CID using specified strategy.
    
    Args:
        records: All loaded activity map records
        strategy: Which selection method to use
        **kwargs: Strategy-specific parameters
        
    Returns:
        Tuple of (selected_maps, cids, metadata)
    """
    if strategy == SelectionStrategy.BEST_QUALITY:
        return select_best_by_quality(records)
    elif strategy == SelectionStrategy.AVERAGE:
        return select_by_averaging(records)
    elif strategy == SelectionStrategy.MEDIAN:
        return select_by_median(records)
    elif strategy == SelectionStrategy.FIRST:
        return select_first_map(records)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def visualize_map(arr: np.ndarray, title: str, output_path: str, cmap: str = 'viridis', mask: Optional[np.ndarray] = None, hide_zeros_inside_roi: bool = False) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    if mask is not None:
        display_arr = mask_for_display(arr, mask, hide_zeros_inside_roi)
    else:
        display_arr = arr
    plt.imshow(display_arr, cmap=cmap)
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


# ============================================================================
# Global Mask Save/Load Functions
# ============================================================================

def compute_and_save_global_mask(
    records: List[ActivityMapRecord],
    coverage_threshold: float = 0.5,
    min_region_size: int = 100,
    output_path: str = "data/02_processed/global_mask.npy"
) -> np.ndarray:
    """Compute global mask and save for reuse.
    
    Args:
        records: All activity map records
        coverage_threshold: Fraction of maps required (0.0 to 1.0)
            - 1.0 = pixel must be active in ALL maps (strictest)
            - 0.5 = pixel must be active in 50% of maps (balanced)
            - 0.1 = pixel must be active in 10% of maps (lenient)
        min_region_size: Minimum connected region size in pixels
        output_path: Where to save mask for reuse
        
    Returns:
        Binary mask array (79, 43)
    """
    # Compute mask using existing function
    mask = compute_global_mask(records, coverage_threshold, min_region_size)
    
    # Save mask
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, mask)
    
    # Save metadata
    metadata = {
        'coverage_threshold': coverage_threshold,
        'min_region_size': min_region_size,
        'n_maps': len(records),
        'active_pixels': int(mask.sum()),
        'total_pixels': int(mask.size),
        'coverage_fraction': float(mask.sum() / mask.size)
    }
    
    meta_path = output_path.replace('.npy', '_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return mask


def load_global_mask(mask_path: str = "data/02_processed/global_mask.npy") -> np.ndarray:
    """Load pre-computed global mask.
    
    Args:
        mask_path: Path to saved mask file
        
    Returns:
        Binary mask array
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Global mask not found at {mask_path}")
    return np.load(mask_path)


# ============================================================================
# Processed Maps Save/Load Functions
# ============================================================================

def save_processed_maps(
    maps: List[np.ndarray],
    cids: List[int],
    output_dir: str,
    metadata: Dict
) -> None:
    """Save processed maps in multiple formats.
    
    Args:
        maps: List of processed activity maps
        cids: Corresponding CID for each map
        output_dir: Directory to save outputs
        metadata: Processing metadata to save
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as NPZ (efficient for loading)
    maps_array = np.stack(maps, axis=0)  # (n_molecules, 79, 43)
    cids_array = np.array(cids)
    
    np.savez(
        os.path.join(output_dir, 'processed_maps.npz'),
        maps=maps_array,
        cids=cids_array,
        **metadata
    )
    
    # Save metadata CSV
    metadata_df = pd.DataFrame({
        'CID': cids,
        'selection_strategy': metadata.get('selection_strategy', 'unknown'),
        'coverage_threshold': metadata.get('coverage_threshold', 0.0),
        'map_shape_h': [m.shape[0] for m in maps],
        'map_shape_w': [m.shape[1] for m in maps],
    })
    metadata_df.to_csv(
        os.path.join(output_dir, 'processed_maps_metadata.csv'),
        index=False
    )
    
    print(f"Saved {len(maps)} processed maps to {output_dir}/processed_maps.npz")
    print(f"Saved metadata to {output_dir}/processed_maps_metadata.csv")


def load_processed_maps(data_dir: str = "data/02_processed") -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load pre-processed activity maps.
    
    Args:
        data_dir: Directory containing processed_maps.npz
        
    Returns:
        Tuple of (maps, cids, metadata)
    """
    maps_path = os.path.join(data_dir, 'processed_maps.npz')
    if not os.path.exists(maps_path):
        raise FileNotFoundError(f"Processed maps not found at {maps_path}")
    
    data = np.load(maps_path)
    maps = data['maps']
    cids = data['cids']
    
    # Extract metadata
    metadata = {k: data[k].item() if hasattr(data[k], 'item') else data[k] 
                for k in data.files if k not in ['maps', 'cids']}
    
    return maps, cids, metadata


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_processing_results(
    maps: List[np.ndarray],
    cids: List[int],
    mask: np.ndarray,
    output_dir: str
) -> None:
    """Generate visualizations of processed maps.
    
    Args:
        maps: Processed activity maps
        cids: Corresponding CIDs
        mask: Global mask used
        output_dir: Where to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize global mask
    visualize_global_mask(mask, os.path.join(output_dir, 'global_mask.png'))
    
    # Example map
    if maps:
        visualize_map(
            maps[0],
            title=f'Processed Map CID={cids[0]}',
            output_path=os.path.join(output_dir, 'processed_map_example.png'),
            mask=mask
        )
        
        # Gallery of up to 6 maps
        n = min(len(maps), 6)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = np.array(axes).reshape(-1)
        
        for i in range(n):
            ax = axes[i]
            display_arr = mask_for_display(maps[i], mask)
            ax.imshow(display_arr, cmap='viridis')
            ax.set_title(f'CID={cids[i]}')
            ax.axis('off')
        
        # Hide unused axes
        for j in range(n, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'processed_maps_gallery.png'), dpi=300)
        plt.close()


# ============================================================================
# Main Processing Pipeline
# ============================================================================

def process_activity_maps(
    directory_csv: str = "data/01_raw/behavior_data.csv",
    data_dir: str = "data/01_raw",
    output_dir: str = "data/02_processed",
    selection_strategy: SelectionStrategy = SelectionStrategy.BEST_QUALITY,
    coverage_threshold: float = 0.5,
    min_region_size: int = 100,
    nan_policy: str = "to_zero",
    value_policy: str = "all",
    save_visualizations: bool = True,
    verbose: bool = False,
) -> Dict[str, any]:
    """Complete activity maps processing pipeline.
    
    Pipeline steps:
    1. Load all activity maps from CSV files
    2. Compute global mask based on coverage threshold
    3. Apply global mask to all maps
    4. Select one map per CID using specified strategy
    5. Apply value policy and NaN policy
    6. Compute QC statistics
    7. Save processed maps and metadata
    8. Generate visualizations (optional)
    
    Args:
        directory_csv: Path to behavior CSV
        data_dir: Directory with activity_maps_csv/
        output_dir: Where to save processed outputs
        selection_strategy: Which selection method to use
        coverage_threshold: Fraction for global mask (0.0-1.0)
        min_region_size: Min pixels in connected regions
        nan_policy: "to_zero" or "keep" - how to handle NaNs at end
        value_policy: "all", "pos", or "neg" - value filtering inside ROI
        save_visualizations: Whether to generate plots
        verbose: Print detailed info
        
    Returns:
        Dictionary with processing results and metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("ACTIVITY MAPS PROCESSING PIPELINE")
    print("="*80)
    
    # Step 1: Load all maps
    print(f"\n[1/6] Loading activity maps from {data_dir}...")
    df = load_directory_csv(directory_csv)
    if verbose:
        print(f"      Directory rows: {len(df)}")
    records = load_activity_maps(df, data_dir=data_dir)
    print(f"      ✓ Loaded {len(records)} activity maps")
    
    # Step 2: Compute global mask
    print(f"\n[2/6] Computing global mask (threshold={coverage_threshold})...")
    mask = compute_and_save_global_mask(
        records,
        coverage_threshold=coverage_threshold,
        min_region_size=min_region_size,
        output_path=os.path.join(output_dir, 'global_mask.npy')
    )
    mask_coverage = mask.sum() / mask.size
    print(f"      ✓ Mask coverage: {mask_coverage:.2%} ({mask.sum()}/{mask.size} pixels)")
    
    # Step 3: Apply mask
    print(f"\n[3/6] Applying global mask to all maps...")
    masked_records = apply_mask(records, mask)
    print(f"      ✓ Applied mask to {len(masked_records)} maps")
    
    # Step 4: Select maps by strategy
    print(f"\n[4/8] Selecting maps using strategy: {selection_strategy.value}...")
    selected_maps, cids, selection_metadata = select_maps_by_strategy(
        masked_records,
        strategy=selection_strategy
    )
    print(f"      ✓ Selected {len(selected_maps)} maps (one per CID)")
    if 'n_single_map' in selection_metadata:
        print(f"        - Single map CIDs: {selection_metadata['n_single_map']}")
        print(f"        - Multi map CIDs: {selection_metadata['n_multi_map']}")
    
    # Step 5: Apply value policy
    print(f"\n[5/8] Applying value policy: {value_policy}...")
    processed_maps = []
    for map_arr in selected_maps:
        processed_maps.append(apply_value_policy(map_arr, mask, value_policy))
    print(f"      ✓ Applied value policy to {len(processed_maps)} maps")
    
    # Step 6: Apply NaN policy
    print(f"\n[6/8] Applying NaN policy: {nan_policy}...")
    if nan_policy == "to_zero":
        final_maps = []
        for map_arr in processed_maps:
            final_maps.append(np.nan_to_num(map_arr, nan=0.0))
        print(f"      ✓ Converted NaNs to zeros in {len(final_maps)} maps")
    elif nan_policy == "keep":
        final_maps = processed_maps
        print(f"      ✓ Kept NaNs intact in {len(final_maps)} maps")
    else:
        raise ValueError(f"Unknown nan_policy: {nan_policy}")
    
    # Step 7: Compute QC statistics
    print(f"\n[7/8] Computing QC statistics...")
    all_stats = []
    for map_arr in processed_maps:  # Use processed_maps before nan_policy for stats
        stats = compute_map_stats(map_arr, mask)
        all_stats.append(stats)
    
    # Save stats to JSON
    stats_path = os.path.join(output_dir, 'map_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump({
            'cids': cids,
            'statistics': all_stats,
            'summary': {
                'n_maps': len(all_stats),
                'value_policy': value_policy,
                'nan_policy': nan_policy,
            }
        }, f, indent=2)
    print(f"      ✓ Computed and saved statistics for {len(all_stats)} maps")
    
    # Step 8: Save processed maps
    print(f"\n[8/8] Saving processed maps to {output_dir}...")
    save_processed_maps(
        maps=final_maps,
        cids=cids,
        output_dir=output_dir,
        metadata={
            'selection_strategy': selection_strategy.value,
            'coverage_threshold': coverage_threshold,
            'min_region_size': min_region_size,
            'nan_policy': nan_policy,
            'value_policy': value_policy,
            'n_maps': len(final_maps),
            'mask_active_pixels': int(mask.sum()),
            'mask_coverage_fraction': float(mask_coverage),
        }
    )
    if save_visualizations:
        print(f"\n[9/9] Generating visualizations...")
        visualize_processing_results(
            maps=final_maps,
            cids=cids,
            mask=mask,
            output_dir=output_dir
        )
        print(f"      ✓ Saved visualizations to {output_dir}")
    else:
        print(f"\n[9/9] Skipping visualizations")
    
    print("\n" + "="*80)
    print("✓ PROCESSING COMPLETE")
    print("="*80)
    
    return {
        'n_molecules': len(final_maps),
        'selection_strategy': selection_strategy.value,
        'coverage_threshold': coverage_threshold,
        'nan_policy': nan_policy,
        'value_policy': value_policy,
        'mask_coverage': float(mask_coverage),
    }


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
    # Create output directory for visualizations
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_directory_csv(directory_csv)
    if verbose:
        print(f"Directory rows: {len(df)}")
        print(df.head())
    records = load_activity_maps(df, data_dir=data_dir)
    # coverage visualization (before mask)
    # Count how many maps have finite values at each pixel
    shape = records[0].map.shape if records else (0, 0)
    valid_counts = np.zeros(shape, dtype=int)
    for r in records:
        valid_counts += np.isfinite(r.map).astype(int)  # Count finite pixels
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
                      output_path=os.path.join(output_dir, 'masked_averaged_example.png'), mask=mask)
        # Gallery of up to 6 averaged maps
        n = min(len(averaged_maps), 6)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = np.array(axes).reshape(-1)
        for i in range(n):
            ax = axes[i]
            display_arr = mask_for_display(averaged_maps[i], mask)
            ax.imshow(display_arr, cmap='viridis')
            ax.set_title(f'CID={cids[i]}')
            ax.axis('off')
        # Hide unused axes
        for j in range(n, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'masked_averaged_gallery.png'), dpi=300)
        plt.close()
    return averaged_maps, cids, mask
