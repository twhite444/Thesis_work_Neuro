"""
Test Data Alignment Script

This script tests the prepare_training_data function to ensure molecular
features (X) are correctly aligned with brain PCA scores (y).

Usage:
    python scripts/test_alignment.py
    
Output:
    Validates alignment and prints statistics
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.neuro_smell.stages.preprocessing import prepare_training_data


def main():
    """Test the data alignment pipeline."""
    
    print("="*80)
    print("TESTING DATA ALIGNMENT")
    print("="*80)
    
    # Define paths
    molecular_features_path = project_root / "data" / "02_processed" / "selected_features.csv"
    brain_pca_path = project_root / "data" / "02_processed" / "brain_pca_scores.csv"
    
    # Check if files exist
    print("\n1. Checking input files...")
    if not molecular_features_path.exists():
        print(f"❌ Molecular features not found: {molecular_features_path}")
        print("\nPlease run the molecular feature extraction pipeline first:")
        print("  python scripts/run_legacy_pipeline.py")
        return 1
    print(f"✅ Found: {molecular_features_path}")
    
    if not brain_pca_path.exists():
        print(f"❌ Brain PCA scores not found: {brain_pca_path}")
        print("\nPlease run the brain processing pipeline first:")
        print("  python scripts/process_brain_maps.py")
        return 1
    print(f"✅ Found: {brain_pca_path}")
    
    # Load molecular features
    print("\n2. Loading molecular features...")
    molecular_df = pd.read_csv(molecular_features_path, index_col=0)
    print(f"   Shape: {molecular_df.shape}")
    print(f"   Columns (first 10): {molecular_df.columns[:10].tolist()}")
    print(f"   Index name: {molecular_df.index.name}")
    
    # Test alignment
    print("\n3. Testing alignment function...")
    try:
        X, y, common_cids, metadata = prepare_training_data(
            molecular_features_df=molecular_df,
            brain_pca_scores_path=str(brain_pca_path),
            cid_column='CID'
        )
        
        print("\n4. Validation checks...")
        
        # Check shapes match
        assert X.shape[0] == y.shape[0], "X and y must have same number of samples!"
        print(f"✅ X and y have matching sample counts: {X.shape[0]}")
        
        # Check no NaNs in training data
        assert not np.isnan(X).any(), "X contains NaN values!"
        assert not np.isnan(y).any(), "y contains NaN values!"
        print(f"✅ No NaN values in X or y")
        
        # Check no infinites
        assert not np.isinf(X).any(), "X contains infinite values!"
        assert not np.isinf(y).any(), "y contains infinite values!"
        print(f"✅ No infinite values in X or y")
        
        # Check y has 5 targets (5 PCA components)
        assert y.shape[1] == 5, f"y should have 5 targets, got {y.shape[1]}"
        print(f"✅ y has correct number of targets: {y.shape[1]}")
        
        # Check reasonable value ranges
        print(f"\n5. Value range checks...")
        print(f"   X range: [{X.min():.4f}, {X.max():.4f}]")
        print(f"   y range: [{y.min():.4f}, {y.max():.4f}]")
        
        # Check metadata
        print(f"\n6. Metadata:")
        print(f"   Common CIDs: {metadata['n_samples']}")
        print(f"   Features: {metadata['n_features']}")
        print(f"   Targets: {metadata['n_targets']}")
        print(f"   Missing in brain: {len(metadata['missing_in_brain'])}")
        print(f"   Missing in molecular: {len(metadata['missing_in_molecular'])}")
        
        # Sample some training examples
        print(f"\n7. Sample training examples:")
        for i in range(min(3, len(common_cids))):
            cid = common_cids[i]
            print(f"   CID {cid}:")
            print(f"      X: {X[i, :5]}... ({X.shape[1]} features)")
            print(f"      y: {y[i]} (5 PCA scores)")
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✅")
        print("="*80)
        print("\nData is ready for model training!")
        print(f"X shape: {X.shape} (molecules × molecular features)")
        print(f"y shape: {y.shape} (molecules × brain PCA scores)")
        print("\nNext step:")
        print("  Train neural network with this aligned data")
        print("  Expected result: R² ≈ 0.506 (thesis benchmark)")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during alignment: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
