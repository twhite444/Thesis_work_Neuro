"""PyTorch datasets for molecular structure to activity map prediction.

Uses:
- Pre-computed molecular features from data/02_processed/selected_features.csv
- Pre-processed activity maps from data/02_processed/processed_maps.npz

Both are already selected (one per CID), masked (global mask applied), and aligned.
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
    - Pre-processed activity maps from data/02_processed/processed_maps.npz
    
    Features are already standardized and variance-selected (268 features).
    Maps are already selected (one per CID) and masked (global mask applied).
    CID is used to align features with activity maps.
    
    Args:
        processed_dir: Directory containing processed data (default: data/02_processed)
        transform: Optional transform to apply to activity maps
        split: Which split to use ('train', 'val', 'test', or None for all data)
        random_seed: Random seed for reproducible splits (default: 42)
    """
    
    def __init__(
        self,
        processed_dir: str = "data/02_processed",
        transform: Optional[callable] = None,
        split: Optional[str] = None,
        random_seed: int = 42,
    ):
        super().__init__()
        
        self.processed_dir = Path(processed_dir)
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
        """Load pre-computed molecular features and pre-processed activity maps."""
        # Load pre-computed selected features (CID as index)
        features_path = self.processed_dir / "selected_features.csv"
        if not features_path.exists():
            raise FileNotFoundError(
                f"Selected features not found at {features_path}. "
                "Run 'python scripts/preprocess.py' and 'python scripts/select_features.py' first."
            )
        
        self.features = pd.read_csv(features_path, index_col='CID')
        
        # Load pre-processed activity maps
        maps_path = self.processed_dir / "processed_maps.npz"
        if not maps_path.exists():
            raise FileNotFoundError(
                f"Processed maps not found at {maps_path}. "
                "Run 'python scripts/run_activity_maps.py' first to generate processed maps."
            )
        
        maps_data = np.load(maps_path)
        self.maps = maps_data['maps']  # (n_molecules, 79, 43)
        self.map_cids = maps_data['cids']
        
        # Align features with maps using CID
        common_cids = np.intersect1d(self.features.index, self.map_cids)
        
        if not common_cids.size:  # NumPy array - use .size
            raise ValueError(
                "No common CIDs found between features and maps. "
                "Regenerate both processed_maps.npz and selected_features.csv"
            )
        
        # Filter and align
        self.features = self.features.loc[common_cids]
        map_indices = [np.where(self.map_cids == cid)[0][0] for cid in common_cids]
        self.maps = self.maps[map_indices]
        self.cids = common_cids
        
        print(f"Loaded {len(self.cids)} molecules with {self.feature_dim} features and aligned maps")
    
    def _apply_split(self):
        """Apply train/val/test split."""
        assert self.split in ['train', 'val', 'test'], \
            f"Split must be 'train', 'val', or 'test', got {self.split}"
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Create split indices
        n_samples = len(self.features)
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
        
        # Apply split
        self.features = self.features.iloc[split_indices].reset_index(drop=True)
        self.maps = self.maps[split_indices]
        self.cids = self.cids[split_indices]
        
        print(f"{self.split.capitalize()} split: {len(self.cids)} samples")
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, activity_map, metadata)
            - features: Pre-computed molecular features (268-dim tensor)
            - activity_map: Pre-processed activity map (79, 43 tensor)
            - metadata: Dict with CID and index
        """
        # Get pre-computed features
        features = self.features.iloc[idx].values.astype(np.float32)
        
        # Get pre-processed activity map (already selected and masked)
        activity_map = self.maps[idx].astype(np.float32)
        
        # Apply transform if specified
        if self.transform is not None:
            activity_map = self.transform(activity_map)
        
        # Convert to tensors
        features = torch.tensor(features, dtype=torch.float32)
        activity_map = torch.tensor(activity_map, dtype=torch.float32)
        
        # Metadata
        metadata = {
            'cid': int(self.cids[idx]),
            'index': idx,
        }
        
        return features, activity_map, metadata


def get_dataloaders(
    processed_dir: str = "data/02_processed",
    batch_size: int = 32,
    num_workers: int = 0,
    random_seed: int = 42,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train, validation, and test dataloaders using pre-processed data.
    
    Args:
        processed_dir: Directory containing processed features and maps
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading (0 recommended for macOS/MPS)
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        
    Note:
        For optimal performance on macOS with MPS backend:
        - Use num_workers=0 to avoid multiprocessing overhead
        - Use pin_memory=False to avoid MPS compatibility warnings
        - Batch size 32 provides good balance of speed and memory
        
    Example:
        >>> train_loader, val_loader, test_loader = get_dataloaders(
        ...     processed_dir="data/02_processed",
        ...     batch_size=32,
        ...     num_workers=0
        ... )
    """
    from torch.utils.data import DataLoader
    
    # Create datasets for each split
    train_dataset = MoleculeActivityMapDataset(
        processed_dir=processed_dir,
        split='train',
        random_seed=random_seed,
    )
    
    val_dataset = MoleculeActivityMapDataset(
        processed_dir=processed_dir,
        split='val',
        random_seed=random_seed,
    )
    
    test_dataset = MoleculeActivityMapDataset(
        processed_dir=processed_dir,
        split='test',
        random_seed=random_seed,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Avoid MPS compatibility warnings
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    return train_loader, val_loader, test_loader
