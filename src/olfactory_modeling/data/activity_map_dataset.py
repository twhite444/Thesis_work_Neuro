"""PyTorch datasets for molecular structure to activity map prediction.


Uses:
- Pre-computed molecular features from data/02_processed/cleaned_data.csv
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
        - Pre-computed molecular features from data/02_processed/cleaned_data.csv
        - Pre-processed activity maps from data/02_processed/processed_maps.npz
            OR PCA-transformed maps from data/02_processed/pca_transformed_maps.npz
    
    Features are already standardized and variance-selected (see preprocess pipeline).
    Maps are already selected (one per CID) and masked (global mask applied).
    CID is used to align features with activity maps.
    
    Args:
        processed_dir: Directory containing processed data (default: data/02_processed)
        transform: Optional transform to apply to activity maps
        split: Which split to use ('train', 'val', 'test', or None for all data)
        random_seed: Random seed for reproducible splits (default: 42)
        use_pca: If True, load PCA-transformed maps instead of raw maps (default: False)
    """
    
    def __init__(
        self,
        processed_dir: str = "data/02_processed",
        transform: Optional[callable] = None,
        split: Optional[str] = None,
        random_seed: int = 42,
        use_pca: bool = False,
    ):
        super().__init__()
        
        self.processed_dir = Path(processed_dir)
        self.transform = transform
        self.split = split
        self.random_seed = random_seed
        self.use_pca = use_pca
        
        # Load data
        self._load_data()
        
        # Apply split if specified
        if split is not None:
            self._apply_split()
    
    @property
    def feature_dim(self):
        """Return the number of input features."""
        return self.features.shape[1]
    
    @property
    def output_dim(self):
        """Return the number of output dimensions (map size or n_components for PCA)."""
        if self.use_pca:
            return self.maps.shape[1]  # n_components
        else:
            return self.maps.shape[1] * self.maps.shape[2]  # height * width
    
    @property
    def output_shape(self):
        """Return the output shape (for raw maps) or None (for PCA)."""
        if self.use_pca:
            return None  # PCA outputs are 1D
        else:
            return (self.maps.shape[1], self.maps.shape[2])  # (height, width)
    
    def _load_data(self):
        """Load pre-computed molecular features and pre-processed activity maps."""
        # Load pre-computed features (CID as index)
        features_path = self.processed_dir / "cleaned_data.csv"
        if not features_path.exists():
            raise FileNotFoundError(
                f"Processed features not found at {features_path}. "
                "Run 'python scripts/preprocess.py' first."
            )
        self.features = pd.read_csv(features_path, index_col='CID')
        
        # Load activity maps (raw or PCA-transformed)
        if self.use_pca:
            maps_path = self.processed_dir / "pca_transformed_maps.npz"
            if not maps_path.exists():
                raise FileNotFoundError(
                    f"PCA-transformed maps not found at {maps_path}. "
                    "Run fit_pca_on_maps() or process_activity_maps_with_pca() first."
                )
            maps_data = np.load(maps_path)
            self.maps = maps_data['pca_maps']  # (n_molecules, n_components)
            self.map_cids = maps_data['cids']
            self.n_components = maps_data['n_components'].item()
            print(f"Loading PCA-transformed maps with {self.n_components} components")
        else:
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
                "Regenerate both processed_maps.npz and cleaned_data.csv"
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
            Tuple of (features, target, metadata)
            - features: Pre-computed molecular features (268-dim tensor)
            - target: Activity map (79, 43 tensor) OR PCA components (n_components tensor)
            - metadata: Dict with CID, index, and target type
        """
        # Get pre-computed features
        features = self.features.iloc[idx].values.astype(np.float32)
        
        # Get target (activity map or PCA components)
        target = self.maps[idx].astype(np.float32)
        
        # Apply transform if specified
        if self.transform is not None:
            target = self.transform(target)
        
        # Convert to tensors
        features = torch.tensor(features, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        
        # Metadata
        metadata = {
            'cid': int(self.cids[idx]),
            'index': idx,
            'target_type': 'pca' if self.use_pca else 'raw_map',
        }
        
        return features, target, metadata


def get_dataloaders(
    processed_dir: str = "data/02_processed",
    batch_size: int = 32,
    num_workers: int = 0,
    random_seed: int = 42,
    use_pca: bool = False,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train, validation, and test dataloaders using pre-processed data.
    
    Args:
        processed_dir: Directory containing processed features and maps
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading (0 recommended for macOS/MPS)
        random_seed: Random seed for reproducible splits
        use_pca: If True, use PCA-transformed maps as targets instead of raw maps
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        
    Note:
        For optimal performance on macOS with MPS backend:
        - Use num_workers=0 to avoid multiprocessing overhead
        - Use pin_memory=False to avoid MPS compatibility warnings
        - Batch size 32 provides good balance of speed and memory
        
    Example:
        >>> # Load raw activity maps
        >>> train_loader, val_loader, test_loader = get_dataloaders(
        ...     processed_dir="data/02_processed",
        ...     batch_size=32,
        ...     use_pca=False
        ... )
        >>> 
        >>> # Load PCA-transformed maps (faster training, lower memory)
        >>> train_loader, val_loader, test_loader = get_dataloaders(
        ...     processed_dir="data/02_processed",
        ...     batch_size=32,
        ...     use_pca=True
        ... )
    """
    from torch.utils.data import DataLoader
    
    # Create datasets for each split
    train_dataset = MoleculeActivityMapDataset(
        processed_dir=processed_dir,
        split='train',
        random_seed=random_seed,
        use_pca=use_pca,
    )
    
    val_dataset = MoleculeActivityMapDataset(
        processed_dir=processed_dir,
        split='val',
        random_seed=random_seed,
        use_pca=use_pca,
    )
    
    test_dataset = MoleculeActivityMapDataset(
        processed_dir=processed_dir,
        split='test',
        random_seed=random_seed,
        use_pca=use_pca,
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
