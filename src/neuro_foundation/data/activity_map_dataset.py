"""PyTorch datasets for molecular structure to activity map prediction."""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


class MoleculeActivityMapDataset(Dataset):
    """Dataset for molecular structure → activity map prediction.
    
    Loads:
    - Molecular structures (SMILES) and computes features (ECFP fingerprints or descriptors)
    - Activity maps (79×43 spatial patterns) from selected_maps.csv
    
    Args:
        data_dir: Root data directory containing molecules.csv, selected_maps.csv, etc.
        feature_type: Type of molecular features ('ecfp', 'rdkit', 'morgan')
        ecfp_radius: Radius for ECFP fingerprints (default: 2)
        ecfp_bits: Number of bits for ECFP fingerprints (default: 2048)
        transform: Optional transform to apply to activity maps
        split: Which split to use ('train', 'val', 'test', or None for all data)
        random_seed: Random seed for reproducible splits (default: 42)
    """
    
    def __init__(
        self,
        data_dir: str = "data/01_raw",
        feature_type: str = "ecfp",
        ecfp_radius: int = 2,
        ecfp_bits: int = 2048,
        transform: Optional[callable] = None,
        split: Optional[str] = None,
        random_seed: int = 42,
    ):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.feature_type = feature_type
        self.ecfp_radius = ecfp_radius
        self.ecfp_bits = ecfp_bits
        self.transform = transform
        self.split = split
        self.random_seed = random_seed
        
        # Load data
        self._load_data()
        
        # Apply split if specified
        if split is not None:
            self._apply_split()
    
    def _load_data(self):
        """Load molecular structures and activity maps."""
        # Load selected maps
        selected_maps = pd.read_csv(self.data_dir / "selected_maps.csv")
        
        # Load molecules
        molecules = pd.read_csv(self.data_dir / "molecules.csv")
        
        # Load behavior data to get activity map paths
        behavior = pd.read_csv(self.data_dir / "behavior_data.csv")
        
        # Merge to get molecule info with selected maps
        self.data = selected_maps.merge(
            molecules[['CID', 'SMILES', 'InChIKey']], 
            on='CID', 
            how='left'
        )
        
        # Get activity map paths for selected maps
        # For each CID, get the map at index selected_idx
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
        
        # Remove entries with missing SMILES or activity maps
        self.data = self.data.dropna(subset=['SMILES', 'activity_map_path']).reset_index(drop=True)
        
        print(f"Loaded {len(self.data)} molecules with activity maps")
    
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
    
    def _compute_ecfp_features(self, smiles: str) -> np.ndarray:
        """Compute ECFP (Extended Connectivity Fingerprints) features.
        
        Args:
            smiles: SMILES string
            
        Returns:
            ECFP fingerprint as numpy array
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.ecfp_bits)
        
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, 
            radius=self.ecfp_radius, 
            nBits=self.ecfp_bits
        )
        return np.array(fp)
    
    def _compute_rdkit_features(self, smiles: str) -> np.ndarray:
        """Compute RDKit molecular descriptors.
        
        Args:
            smiles: SMILES string
            
        Returns:
            RDKit descriptors as numpy array
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(200)  # Default descriptor count
        
        # Compute all RDKit descriptors
        descriptor_names = [desc[0] for desc in Descriptors._descList]
        descriptors = []
        
        for name in descriptor_names:
            try:
                desc_fn = getattr(Descriptors, name)
                value = desc_fn(mol)
                # Handle NaN values
                if np.isnan(value) or np.isinf(value):
                    value = 0.0
                descriptors.append(value)
            except:
                descriptors.append(0.0)
        
        return np.array(descriptors, dtype=np.float32)
    
    def _load_activity_map(self, map_filename: str) -> np.ndarray:
        """Load activity map from CSV file.
        
        Args:
            map_filename: Filename of activity map CSV
            
        Returns:
            Activity map as numpy array (79, 43)
        """
        map_path = self.data_dir / "activity_maps_csv" / map_filename
        
        if not map_path.exists():
            return np.zeros((79, 43), dtype=np.float32)
        
        # Load CSV
        activity_map = pd.read_csv(map_path, index_col=0)
        
        return activity_map.values.astype(np.float32)
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, activity_map, metadata)
            - features: Molecular features (tensor)
            - activity_map: Activity map (79, 43 tensor)
            - metadata: Dict with CID, SMILES, Name, etc.
        """
        row = self.data.iloc[idx]
        
        # Compute molecular features
        smiles = row['SMILES']
        if self.feature_type == 'ecfp':
            features = self._compute_ecfp_features(smiles)
        elif self.feature_type == 'rdkit':
            features = self._compute_rdkit_features(smiles)
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
        
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
            'SMILES': smiles,
            'coverage_frac': float(row['coverage_frac']),
            'mean_active': float(row['mean_active']),
        }
        
        return features, activity_map, metadata


def get_dataloaders(
    data_dir: str = "data/01_raw",
    feature_type: str = "ecfp",
    batch_size: int = 32,
    num_workers: int = 4,
    random_seed: int = 42,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Root data directory
        feature_type: Type of molecular features ('ecfp' or 'rdkit')
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        
    Example:
        >>> train_loader, val_loader, test_loader = get_dataloaders(
        ...     data_dir="data/01_raw",
        ...     feature_type="ecfp",
        ...     batch_size=32
        ... )
    """
    from torch.utils.data import DataLoader
    
    # Create datasets for each split
    train_dataset = MoleculeActivityMapDataset(
        data_dir=data_dir,
        feature_type=feature_type,
        split='train',
        random_seed=random_seed,
    )
    
    val_dataset = MoleculeActivityMapDataset(
        data_dir=data_dir,
        feature_type=feature_type,
        split='val',
        random_seed=random_seed,
    )
    
    test_dataset = MoleculeActivityMapDataset(
        data_dir=data_dir,
        feature_type=feature_type,
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
