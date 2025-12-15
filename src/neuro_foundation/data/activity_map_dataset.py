"""PyTorch datasets for molecular structure to activity map prediction.

Uses pre-computed and selected features from data/02_processed/selected_features.csv
instead of computing features on the fly.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MoleculeActivityMapDataset(Dataset):
    """Dataset for molecular structure → activity map prediction.
    
    Loads:
    - Pre-computed molecular features from data/02_processed/selected_features.csv
    - Activity maps (79×43 spatial patterns) from data/selected_maps.csv
    
    Features are already standardized and variance-selected (268 features).
    CID is used as the index to align features with activity maps.
    
    Args:
        processed_dir: Directory containing processed data (default: data/02_processed)
        raw_data_dir: Directory containing raw data like selected_maps.csv (default: data)
        transform: Optional transform to apply to activity maps
        split: Which split to use ('train', 'val', 'test', or None for all data)
        random_seed: Random seed for reproducible splits (default: 42)
    """
    
    def __init__(
        self,
        processed_dir: str = "data/02_processed",
        raw_data_dir: str = "data",
        transform: Optional[callable] = None,
        split: Optional[str] = None,
        random_seed: int = 42,
    ):
        super().__init__()
        
        self.processed_dir = Path(processed_dir)
        self.raw_data_dir = Path(raw_data_dir)
        self.transform = transform
        self.split = split
        self.random_seed = random_seed
        
        # Load data
        self._load_data()
        
        # Apply split if specified
        if split is not None:
            self._apply_split()
    
    @property
    def feature_dim(self):
        """Return the number of input features."""
        return self.features.shape[1]
    
    
    def _load_data(self):
        """Load pre-computed molecular features and activity map metadata."""
        # Load pre-computed selected features (CID as index)
        features_path = self.processed_dir / "selected_features.csv"
        if not features_path.exists():
            raise FileNotFoundError(
                f"Selected features not found at {features_path}. "
                "Run 'python scripts/preprocess.py' and 'python scripts/select_features.py' first."
            )
        
        self.features = pd.read_csv(features_path, index_col='CID')
        
        # Load selected maps metadata
        maps_path = self.raw_data_dir / "selected_maps.csv"
        if not maps_path.exists():
            raise FileNotFoundError(f"Selected maps not found at {maps_path}")
        
        selected_maps = pd.read_csv(maps_path)
        
        # Load behavior data to get activity map paths
        behavior_path = self.raw_data_dir / "01_raw" / "behavior_data.csv"
        if not behavior_path.exists():
            behavior_path = Path("data/01_raw/behavior_data.csv")
        
        behavior = pd.read_csv(behavior_path)
        
        # Merge features with maps metadata using CID
        self.data = selected_maps.merge(
            self.features,
            left_on='CID',
            right_index=True,
            how='inner'
        )
        
        # Get activity map paths for each molecule
        map_paths = []
        for _, row in self.data.iterrows():
            cid = row['CID']
            selected_idx = int(row['selected_idx'])
            
            # Get all maps for this CID
            cid_maps = behavior[behavior['Stimulus'] == cid]
            
            if len(cid_maps) > selected_idx:
                map_path = cid_maps.iloc[selected_idx]['Activity Map Path']
                map_paths.append(map_path)
            else:
                map_paths.append(None)
        
        self.data['activity_map_path'] = map_paths
        
        # Remove entries with missing activity maps
        self.data = self.data.dropna(subset=['activity_map_path']).reset_index(drop=True)
        
        print(f"Loaded {len(self.data)} molecules with {self.feature_dim} features each")
    
    def _apply_split(self):
        """Apply train/val/test split."""
        assert self.split in ['train', 'val', 'test'], \
            f"Split must be 'train', 'val', or 'test', got {self.split}"
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Create split indices
        n_samples = len(self.data)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        # 70% train, 15% val, 15% test
        train_end = int(0.70 * n_samples)
        val_end = int(0.85 * n_samples)
        
        if self.split == 'train':
            split_indices = indices[:train_end]
        elif self.split == 'val':
            split_indices = indices[train_end:val_end]
        else:  # test
            split_indices = indices[val_end:]
        
        self.data = self.data.iloc[split_indices].reset_index(drop=True)
        print(f"{self.split.capitalize()} split: {len(self.data)} samples")
    
    def _load_activity_map(self, map_filename: str) -> np.ndarray:
        """Load activity map from CSV file.
        
        Args:
            map_filename: Filename of activity map CSV
            
        Returns:
            Activity map as numpy array (79, 43)
        """
        # Try multiple possible locations for activity maps
        possible_paths = [
            Path("data/01_raw/activity_maps_csv") / map_filename,
            self.raw_data_dir / "01_raw" / "activity_maps_csv" / map_filename,
            self.raw_data_dir / "activity_maps_csv" / map_filename,
        ]
        
        for map_path in possible_paths:
            if map_path.exists():
                activity_map = pd.read_csv(map_path, index_col=0)
                return activity_map.values.astype(np.float32)
        
        # If not found, return zeros
        print(f"Warning: Activity map not found: {map_filename}")
        return np.zeros((79, 43), dtype=np.float32)
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, activity_map, metadata)
            - features: Pre-computed molecular features (268-dim tensor)
            - activity_map: Activity map (79, 43 tensor)
            - metadata: Dict with CID, Name, coverage, etc.
        """
        row = self.data.iloc[idx]
        
        # Get pre-computed features for this molecule
        # Features are in columns after the metadata columns
        feature_cols = self.features.columns
        features = row[feature_cols].values.astype(np.float32)
        
        # Load activity map
        map_filename = os.path.basename(row['activity_map_path'])
        activity_map = self._load_activity_map(map_filename)
        
        # Apply transform if specified
        if self.transform is not None:
            activity_map = self.transform(activity_map)
        
        # Convert to tensors
        features = torch.tensor(features, dtype=torch.float32)
        activity_map = torch.tensor(activity_map, dtype=torch.float32)
        
        # Metadata
        metadata = {
            'CID': int(row['CID']),
            'Name': row['Name'],
            'coverage_frac': float(row['coverage_frac']),
            'mean_active': float(row['mean_active']),
        }
        
        return features, activity_map, metadata


def get_dataloaders(
    processed_dir: str = "data/02_processed",
    raw_data_dir: str = "data",
    batch_size: int = 32,
    num_workers: int = 4,
    random_seed: int = 42,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train, validation, and test dataloaders using pre-computed features.
    
    Args:
        processed_dir: Directory containing processed features
        raw_data_dir: Directory containing raw data (selected_maps.csv, etc.)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        random_seed: Random seed for reproducible splits
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        
    Example:
        >>> train_loader, val_loader, test_loader = get_dataloaders(
        ...     processed_dir="data/02_processed",
        ...     raw_data_dir="data",
        ...     batch_size=32
        ... )
    """
    from torch.utils.data import DataLoader
    
    # Create datasets for each split
    train_dataset = MoleculeActivityMapDataset(
        processed_dir=processed_dir,
        raw_data_dir=raw_data_dir,
        split='train',
        random_seed=random_seed,
    )
    
    val_dataset = MoleculeActivityMapDataset(
        processed_dir=processed_dir,
        raw_data_dir=raw_data_dir,
        split='val',
        random_seed=random_seed,
    )
    
    test_dataset = MoleculeActivityMapDataset(
        processed_dir=processed_dir,
        raw_data_dir=raw_data_dir,
        split='test',
        random_seed=random_seed,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
