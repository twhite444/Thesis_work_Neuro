#!/usr/bin/env python3
"""
Verify PCA Masking Implementation

This script tests that our new PCA masking implementation
produces results consistent with the legacy/pca_copy.py approach.

Usage:
    python scripts/verify_pca_masking.py
    
    # Or with custom parameters:
    python scripts/verify_pca_masking.py --n-components 50 --threshold 0.1
    
Prerequisites:
    - Run scripts/run_legacy_pipeline.py first to generate preprocessed data
    - Or have data/02_processed/selected_features.csv available

Copyright (c) 2025 Tom White
Licensed under the MIT License
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from neuro_smell.stages.pca_masking import PCAMasking


def verify_pca_masking(
    n_components: int = 50,
    threshold: float = 0.1,
    data_path: str = None
):
    """
    Test PCA masking with real data.
    
    Args:
        n_components: Number of PCA components
        threshold: Masking threshold
        data_path: Path to preprocessed data (optional)
    """
    
    print("="*70)
    print("🧪 Testing PCA Masking Implementation")
    print("="*70)
    print(f"\nParameters:")
    print(f"  n_components: {n_components}")
    print(f"  threshold: {threshold}")
    
    # Determine data path
    if data_path is None:
        data_path = project_root / "data" / "02_processed" / "selected_features.csv"
    else:
        data_path = Path(data_path)
    
    if not data_path.exists():
        print(f"\n❌ Data not found: {data_path}")
        print("\n💡 Run this command first:")
        print("   python scripts/run_legacy_pipeline.py")
        return False
    
    # Load preprocessed data
    print(f"\n📊 Loading data from: {data_path}")
    df = pd.read_csv(data_path, index_col=0)
    X = df.values
    
    print(f"   Shape: {X.shape}")
    print(f"   Mean: {X.mean():.4f} (should be ~0 after StandardScaler)")
    print(f"   Std: {X.std():.4f} (should be ~1 after StandardScaler)")
    
    if abs(X.mean()) > 0.1 or abs(X.std() - 1.0) > 0.1:
        print(f"\n⚠️  Warning: Data doesn't appear to be standardized!")
        print(f"   Expected: mean ≈ 0, std ≈ 1")
        print(f"   Got: mean = {X.mean():.4f}, std = {X.std():.4f}")
    
    # Apply PCA masking
    print(f"\n🎭 Applying PCA Masking...")
    
    masker = PCAMasking(n_components=n_components, threshold=threshold)
    X_masked, mask = masker.fit_transform(X)
    
    print(f"\n✅ Masking complete!")
    print(f"   Input features: {X.shape[1]}")
    print(f"   Output features: {X_masked.shape[1]}")
    print(f"   Reduction: {(1 - X_masked.shape[1]/X.shape[1])*100:.1f}%")
    
    # Show PCA info
    info = masker.get_info()
    print(f"\n📊 PCA Information:")
    print(f"   Variance explained: {info['variance_explained']*100:.2f}%")
    print(f"   Features selected: {info['features_selected']} / {info['features_total']}")
    print(f"   Reduction: {info['reduction_percent']:.1f}%")
    
    # Generate visualizations
    output_dir = project_root / "test_output" / "pca_analysis"
    print(f"\n📊 Generating visualizations...")
    masker.visualize(str(output_dir))
    
    print(f"\n✅ Visualizations saved to: {output_dir}/")
    print(f"   📈 global_mask.png - Feature importance with threshold")
    print(f"   📈 top_3_components.png - First 3 PC loadings")
    print(f"   📈 pca_scree.png - Explained variance per component")
    print(f"   📈 pca_cumulative.png - Cumulative variance")
    
    # Save mask for reproducibility
    mask_path = output_dir / "feature_mask.csv"
    masker.save_mask(str(mask_path))
    
    # Compare with legacy if available
    legacy_pca_data = project_root / "legacy" / "pca_transformed_data.csv"
    if legacy_pca_data.exists():
        print(f"\n🔍 Comparing with legacy output...")
        df_legacy = pd.read_csv(legacy_pca_data, index_col=0)
        
        print(f"   Legacy shape: {df_legacy.shape}")
        print(f"   New shape: {X_masked.shape}")
        
        if df_legacy.shape[1] == X_masked.shape[1]:
            print(f"   ✅ Feature count matches!")
        else:
            print(f"   ⚠️  Feature count differs:")
            print(f"      Legacy: {df_legacy.shape[1]} features")
            print(f"      New: {X_masked.shape[1]} features")
            print(f"      Difference: {abs(df_legacy.shape[1] - X_masked.shape[1])} features")
            print(f"\n   💡 Try adjusting threshold:")
            if X_masked.shape[1] > df_legacy.shape[1]:
                print(f"      Increase threshold (e.g., {threshold + 0.05:.2f}) to select fewer features")
            else:
                print(f"      Decrease threshold (e.g., {threshold - 0.05:.2f}) to select more features")
    else:
        print(f"\n💡 Legacy data not found at: {legacy_pca_data}")
        print(f"   Run full legacy pipeline to compare results")
    
    # Statistical validation
    print(f"\n📊 Masked Feature Statistics:")
    print(f"   Mean: {X_masked.mean():.6f}")
    print(f"   Std: {X_masked.std():.6f}")
    print(f"   Min: {X_masked.min():.6f}")
    print(f"   Max: {X_masked.max():.6f}")
    
    # Show top features by importance
    print(f"\n🏆 Top 10 Most Important Features:")
    top_indices = np.argsort(masker.feature_importance)[-10:][::-1]
    for i, idx in enumerate(top_indices, 1):
        importance = masker.feature_importance[idx]
        selected = "✅" if masker.global_mask[idx] else "❌"
        print(f"   {i:2d}. Feature {idx:3d}: {importance:.4f} {selected}")
    
    print("\n" + "="*70)
    print("✅ PCA Masking Test Complete!")
    print("="*70)
    print(f"\n📁 Output files:")
    print(f"   {output_dir}/global_mask.png")
    print(f"   {output_dir}/top_3_components.png")
    print(f"   {output_dir}/pca_scree.png")
    print(f"   {output_dir}/pca_cumulative.png")
    print(f"   {output_dir}/feature_mask.csv")
    print(f"\n💡 Next steps:")
    print(f"   1. Review visualizations to verify PCA behavior")
    print(f"   2. Adjust threshold if needed to match legacy feature count")
    print(f"   3. Integrate into full pipeline with configs/preprocessing/")
    
    return True


def main():
    """Parse arguments and run verification."""
    parser = argparse.ArgumentParser(
        description="Verify PCA masking implementation against legacy"
    )
    parser.add_argument(
        '--n-components', 
        type=int, 
        default=50,
        help='Number of PCA components (default: 50)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.1,
        help='Masking threshold (default: 0.1)'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to preprocessed data CSV (default: data/02_processed/selected_features.csv)'
    )
    
    args = parser.parse_args()
    
    success = verify_pca_masking(
        n_components=args.n_components,
        threshold=args.threshold,
        data_path=args.data_path
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
