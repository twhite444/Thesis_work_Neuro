from __future__ import annotations
import os
from typing import List, NamedTuple
import numpy as np
import pandas as pd
import pyrfume
from .interfaces import DatasetLoader


class ActivityMapRecord(NamedTuple):
    """Single activity map with associated CID."""
    cid: int
    map: np.ndarray  # 2D array (79, 43)


class PyrfumeLoader(DatasetLoader):
    """Pyrfume-backed loader replicating legacy data sources (read-only).

    Loads:
    - arshamian_2022 manifest (not used directly here but validated)
    - leon manifest
    - leon/molecules.csv
    - leon/behavior_1.csv
    - leon/csvs/*.csv (activity maps)
    
    Saves both CSV (human-readable) and NPZ (efficient binary) formats.
    """

    def __init__(self, output_dir: str = "data/01_raw"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_molecules(self) -> pd.DataFrame:
        """Load molecules from Pyrfume, deduplicate by CID, and save to CSV + NPZ."""
        _ = pyrfume.load_manifest('leon')
        molecules = pyrfume.load_data('leon/molecules.csv')

        molecules.reset_index(inplace=True)
        molecules.rename(columns={'index': 'CID'}, inplace=True)

        # Check for duplicate CIDs
        duplicate_cids = molecules[molecules.duplicated(subset='CID', keep=False)]
        if not duplicate_cids.empty:
            print(f"Found {len(duplicate_cids)} duplicate CID entries, keeping first occurrence")

        # Remove duplicate CIDs by keeping the first occurrence
        molecules = molecules.drop_duplicates(subset='CID', keep='first')

        # Save CSV for human readability
        csv_path = os.path.join(self.output_dir, 'molecules_raw.csv')
        molecules.to_csv(csv_path, index=False)
        
        # Save NPZ for fast loading
        npz_path = os.path.join(self.output_dir, 'molecules_raw.npz')
        np.savez_compressed(
            npz_path,
            CID=molecules['CID'].values,
            MolecularWeight=molecules['MolecularWeight'].values,
            IsomericSMILES=molecules['IsomericSMILES'].values,
            IUPACName=molecules['IUPACName'].values,
            name=molecules['name'].values
        )
        
        print(f"Saved {len(molecules)} molecules to {csv_path} and {npz_path}")
        return molecules
    
    def load_behavior(self) -> pd.DataFrame:
        """Load behavior data from Pyrfume, clean, and save to CSV + NPZ."""
        behavior = pyrfume.load_data('leon/behavior_1.csv')
        behavior.reset_index(inplace=True)
        
        # Normalize stimulus by removing any suffix after '_' (e.g. "153_2" -> "153")
        behavior['Stimulus'] = behavior['Stimulus'].astype(str).str.split('_', n=1).str[0]
        # Convert Stimulus to numeric (coerce invalid values to NaN) and filter valid non-negative entries
        behavior['Stimulus'] = pd.to_numeric(behavior['Stimulus'], errors='coerce')
        behavior = behavior[behavior['Stimulus'].notna() & (behavior['Stimulus'] >= 0)]
        
        # Save CSV for human readability
        csv_path = os.path.join(self.output_dir, 'behavior_data.csv')
        behavior.to_csv(csv_path, index=False)
        
        # Save NPZ for fast loading
        npz_path = os.path.join(self.output_dir, 'behavior_data.npz')
        np.savez_compressed(
            npz_path,
            Stimulus=behavior['Stimulus'].values.astype(int),
            ActivityMapPath=behavior['Activity Map Path'].values
        )
        
        print(f"Saved {len(behavior)} behavior entries to {csv_path} and {npz_path}")
        return behavior

    def load_activity_maps(self, save_individual_csvs: bool = True) -> List[ActivityMapRecord]:
        """Load all activity maps from Pyrfume and save to NPZ and individual CSVs.
        
        Args:
            save_individual_csvs: If True, save each activity map as a separate CSV file
                                 in data/01_raw/activity_maps_csv/ directory.
        
        Returns:
            List of ActivityMapRecord with CID and 2D numpy array per map.
        """
        # Ensure behavior data exists to get map paths
        behavior_csv = os.path.join(self.output_dir, 'behavior_data.csv')
        if os.path.exists(behavior_csv):
            behavior = pd.read_csv(behavior_csv)
            print(f"Loaded existing behavior CSV: {behavior_csv}")
        else:
            print(f"Behavior CSV not found. Loading from Pyrfume...")
            behavior = self.load_behavior()
        
        # Load manifest
        _ = pyrfume.load_manifest('leon')
        
        # Create directory for individual CSV files if requested
        if save_individual_csvs:
            csv_dir = os.path.join(self.output_dir, 'activity_maps_csv')
            os.makedirs(csv_dir, exist_ok=True)
            print(f"Individual CSVs will be saved to: {csv_dir}")
        
        # Load all activity maps
        records: List[ActivityMapRecord] = []
        total = len(behavior)
        
        print(f"Loading {total} activity maps from Pyrfume...")
        for i, (_, row) in enumerate(behavior.iterrows(), start=1):
            print(f'\rLoading maps: {i}/{total}', end='', flush=True)
            
            map_path = row['Activity Map Path']
            # Load from pyrfume (format: csvs/1031_0.csv)
            try:
                map_df = pyrfume.load_data(f'leon/{map_path}')
                arr = np.nan_to_num(map_df.to_numpy(), nan=0.0)
                cid = int(row['Stimulus'])
                records.append(ActivityMapRecord(cid=cid, map=arr))
                
                # Save individual CSV if requested
                if save_individual_csvs:
                    # Extract filename from path (e.g., "csvs/1031_0.csv" -> "1031_0.csv")
                    csv_filename = os.path.basename(map_path)
                    csv_path = os.path.join(csv_dir, csv_filename)
                    # Save with CID in index for reference
                    df_to_save = pd.DataFrame(arr)
                    df_to_save.to_csv(csv_path, index=True)
                    
            except Exception as e:
                print(f"\nWarning: Could not load {map_path}: {e}")
                continue
        
        print(f"\nSuccessfully loaded {len(records)} activity maps")
        
        # Save to NPZ for efficient storage and loading
        npz_path = os.path.join(self.output_dir, 'activity_maps.npz')
        maps_array = np.stack([r.map for r in records])  # shape: (n_maps, 79, 43)
        cids_array = np.array([r.cid for r in records])
        
        np.savez_compressed(
            npz_path,
            maps=maps_array,
            cids=cids_array
        )
        
        size_mb = maps_array.nbytes / (1024 ** 2)
        print(f"Saved activity maps to {npz_path} ({size_mb:.2f} MB uncompressed)")
        
        if save_individual_csvs:
            print(f"Saved {len(records)} individual CSV files to {csv_dir}")
        
        return records

    def load_images(self) -> pd.DataFrame | None:
        """Legacy method for compatibility with DatasetLoader interface.
        
        Note: Use load_activity_maps() instead for the refactored pipeline.
        """
        return None


# ============================================================================
# Helper functions for loading cached data (CSV and NPZ)
# ============================================================================

def load_molecules_csv(data_dir: str = "data/01_raw") -> pd.DataFrame:
    """Load molecules from cached CSV file."""
    csv_path = os.path.join(data_dir, 'molecules_raw.csv')
    return pd.read_csv(csv_path)


def load_molecules_npz(data_dir: str = "data/01_raw") -> pd.DataFrame:
    """Load molecules from cached NPZ file (faster than CSV)."""
    npz_path = os.path.join(data_dir, 'molecules_raw.npz')
    data = np.load(npz_path, allow_pickle=True)
    return pd.DataFrame({
        'CID': data['CID'],
        'MolecularWeight': data['MolecularWeight'],
        'IsomericSMILES': data['IsomericSMILES'],
        'IUPACName': data['IUPACName'],
        'name': data['name']
    })


def load_behavior_csv(data_dir: str = "data/01_raw") -> pd.DataFrame:
    """Load behavior data from cached CSV file."""
    csv_path = os.path.join(data_dir, 'behavior_data.csv')
    return pd.read_csv(csv_path)


def load_behavior_npz(data_dir: str = "data/01_raw") -> pd.DataFrame:
    """Load behavior data from cached NPZ file (faster than CSV)."""
    npz_path = os.path.join(data_dir, 'behavior_data.npz')
    data = np.load(npz_path, allow_pickle=True)
    return pd.DataFrame({
        'Stimulus': data['Stimulus'],
        'Activity Map Path': data['ActivityMapPath']
    })


def load_activity_maps_npz(data_dir: str = "data/01_raw") -> List[ActivityMapRecord]:
    """Load activity maps from cached NPZ file.
    
    Returns:
        List of ActivityMapRecord with CID and 2D numpy array per map.
    """
    npz_path = os.path.join(data_dir, 'activity_maps.npz')
    data = np.load(npz_path)
    maps = data['maps']  # shape: (n_maps, 79, 43)
    cids = data['cids']  # shape: (n_maps,)
    
    return [ActivityMapRecord(cid=int(cid), map=map_arr) 
            for cid, map_arr in zip(cids, maps)]


def load_activity_maps_as_arrays(data_dir: str = "data/01_raw") -> tuple[np.ndarray, np.ndarray]:
    """Load activity maps as raw numpy arrays for efficient processing.
    
    Returns:
        Tuple of (maps, cids) where:
        - maps: numpy array of shape (n_maps, 79, 43)
        - cids: numpy array of shape (n_maps,) with CID for each map
    """
    npz_path = os.path.join(data_dir, 'activity_maps.npz')
    data = np.load(npz_path)
    return data['maps'], data['cids']


def load_activity_maps_by_cid(cid: int, data_dir: str = "data/01_raw") -> List[np.ndarray]:
    """Load all activity maps for a specific CID from NPZ file.
    
    Args:
        cid: The CID (molecule identifier) to filter by
        data_dir: Directory containing activity_maps.npz
    
    Returns:
        List of 2D numpy arrays (79, 43) for all maps matching the CID.
        Returns empty list if no maps found for this CID.
    
    Example:
        >>> maps = load_activity_maps_by_cid(180)
        >>> print(f"CID 180 has {len(maps)} maps")
        >>> print(f"First map shape: {maps[0].shape}")
    """
    maps, cids = load_activity_maps_as_arrays(data_dir)
    
    # Find indices where CID matches
    indices = np.where(cids == cid)[0]
    
    if len(indices) == 0:
        return []
    
    # Return list of maps for this CID
    return [maps[i] for i in indices]


def load_activity_map_by_cid_averaged(cid: int, data_dir: str = "data/01_raw") -> np.ndarray | None:
    """Load and average all activity maps for a specific CID.
    
    Args:
        cid: The CID (molecule identifier) to filter by
        data_dir: Directory containing activity_maps.npz
    
    Returns:
        2D numpy array (79, 43) averaged across all maps for this CID.
        Returns None if no maps found for this CID.
    
    Example:
        >>> avg_map = load_activity_map_by_cid_averaged(180)
        >>> print(f"Averaged map shape: {avg_map.shape}")
    """
    maps_for_cid = load_activity_maps_by_cid(cid, data_dir)
    
    if len(maps_for_cid) == 0:
        return None
    
    # Average across all maps using nanmean to handle any NaN values
    return np.nanmean(np.stack(maps_for_cid), axis=0)
