#!/usr/bin/env python3
"""
Legacy Build Pipeline - Exact replica of UPDATED legacy/build.py with smart caching

This script replicates the UPDATED legacy build.py exactly:
1. Load Pyrfume Leon dataset (with duplicate CID handling)
2. Extract Mordred features via smiles_to_mordred() on ALL SMILES at once
3. Maintain CID as index throughout pipeline
4. Preprocess: dropna(), remove zeros, StandardScaler, VarianceThreshold(1.0)
5. Save processed features with CID index

NEW: Adds smart caching so reruns are instant!

Usage:
    python scripts/run_legacy_pipeline.py
    
    # Force rerun (ignore cache)
    python scripts/run_legacy_pipeline.py --force
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Legacy imports (matching build.py)
try:
    import pyrfume
    from pyrfume.features import smiles_to_mordred
    from rdkit import Chem
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\nInstall requirements:")
    print("  pip install pyrfume mordred rdkit")
    sys.exit(1)

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# New: smart cache
from neuro_smell.utils.smart_cache import get_cache_manager


def is_valid_smiles(smiles):
    """Validate SMILES (from legacy build.py)"""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def load_data():
    """Load Pyrfume data (from UPDATED legacy build.py)"""
    print("\n" + "="*70)
    print("STAGE 1: Load Pyrfume Data")
    print("="*70)
    
    print("Loading manifests...")
    arshamian_manifest = pyrfume.load_manifest('arshamian_2022')
    leon_manifest = pyrfume.load_manifest('leon')
    
    print("Loading molecules...")
    molecules = pyrfume.load_data('leon/molecules.csv')
    molecules.reset_index(inplace=True)
    molecules.rename(columns={'index': 'CID'}, inplace=True)
    
    # Check for duplicate CIDs (NEW IN UPDATED VERSION)
    duplicate_cids = molecules[molecules.duplicated(subset='CID', keep=False)]
    print(f"Duplicate CIDs in molecules before removal:\n{duplicate_cids}")
    
    # Remove duplicate CIDs by keeping the first occurrence (NEW)
    molecules = molecules.drop_duplicates(subset='CID', keep='first')
    
    # Debug: Check for duplicates after removal (NEW)
    duplicate_cids_after = molecules[molecules.duplicated(subset='CID', keep=False)]
    print(f"Duplicate CIDs in molecules after removal:\n{duplicate_cids_after}")
    
    print("Loading behavior data...")
    behavior_data = pyrfume.load_data('leon/behavior_1.csv')
    
    print("Loading image data...")
    image_data = pyrfume.load_data('leon/csvs/1031_0.csv')
    
    # Save raw data
    output_dir = project_root / "data" / "00_raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    molecules.to_csv(output_dir / "molecules_raw.csv", index=True)
    behavior_data.to_csv(output_dir / "behavior_data.csv", index=True)
    image_data.to_csv(output_dir / "image_data.csv", index=True)
    
    print(f"\n✅ Loaded {len(molecules)} molecules (after deduplication)")
    print(f"   Molecules index: {molecules.index.name}")
    print(f"   Saved to: {output_dir}")
    
    return molecules


def preprocess_data(molecules):
    """Preprocess features (from UPDATED legacy build.py)"""
    print("\n" + "="*70)
    print("STAGE 2: Feature Extraction & Preprocessing")
    print("="*70)
    
    # Featurize molecules - UPDATED: use unique SMILES and batch process
    smiles = molecules["IsomericSMILES"].unique().tolist()
    print(f"Number of SMILES strings: {len(smiles)}")
    
    # Check validity of SMILES strings
    valid_smiles = [s for s in smiles if is_valid_smiles(s)]
    print(f"Number of valid SMILES strings: {len(valid_smiles)}")
    
    # Extract Mordred features for all SMILES at once (UPDATED METHOD)
    print("Extracting Mordred descriptors (batch processing)...")
    mordred_features = smiles_to_mordred(smiles)
    print(f"Mordred features shape: {mordred_features.shape}")
    
    # Add CID to mordred features for alignment (UPDATED)
    print("Molecules CID values:", molecules["CID"].head())
    mordred_features["CID"] = molecules["CID"].values[:mordred_features.shape[0]]  # Ensure alignment
    mordred_features.set_index("CID", inplace=True)
    print("Mordred features index after setting CID:")
    print(mordred_features.index)
    
    # Remove rows with NaN values and columns with zero variance
    initial_columns = mordred_features.shape[1]
    print(f"Initial columns: {initial_columns}")
    
    # Step 1: Drop NaN columns
    filtered_data = mordred_features.dropna(axis=1, how='any')
    after_nan_removal_columns = filtered_data.shape[1]
    print(f"After NaN removal: {after_nan_removal_columns} (removed {initial_columns - after_nan_removal_columns})")
    
    # Step 2: Remove zero-variance columns
    filtered_data = filtered_data.loc[:, ~(filtered_data.eq(0).any(axis=0))]
    after_zero_variance_removal_columns = filtered_data.shape[1]
    print(f"After zero removal: {after_zero_variance_removal_columns} (removed {after_nan_removal_columns - after_zero_variance_removal_columns})")
    
    # Step 3: Standardize the data
    print("Applying StandardScaler...")
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(filtered_data)
    
    standardized_df = pd.DataFrame(
        standardized_data, 
        columns=filtered_data.columns, 
        index=filtered_data.index  # MAINTAIN CID INDEX
    )
    
    # Debug: Print index information (UPDATED)
    print("Cleaned data index:", standardized_df.index.name)
    print("First few rows of cleaned data:")
    print(standardized_df.head())
    
    print(f"\n✅ Preprocessing complete!")
    print(f"   Output shape: {standardized_df.shape}")
    
    return standardized_df


def select_features(data, variance_threshold=1.0):
    """Feature selection (from UPDATED legacy build.py)"""
    print("\n" + "="*70)
    print("STAGE 3: Feature Selection")
    print("="*70)
    
    print(f"Variance threshold: {variance_threshold}")
    
    # Select features with variance above threshold
    selector = VarianceThreshold(threshold=variance_threshold)
    selected_data = selector.fit_transform(data)
    selected_features = data.columns[selector.get_support()]
    
    selected_df = pd.DataFrame(
        selected_data, 
        columns=selected_features, 
        index=data.index  # MAINTAIN CID INDEX
    )
    
    # Debug: Print index information (UPDATED)
    print("Selected features index:", selected_df.index.name)
    print("First few rows of selected features:")
    print(selected_df.head())
    
    print(f"\n✅ Feature selection complete!")
    print(f"   Features selected: {len(selected_features)}/{len(data.columns)}")
    print(f"   Output shape: {selected_df.shape}")
    
    return selected_df


def process_all(variance_threshold=1.0, use_cache=True, force_rerun=False):
    """
    Main pipeline (from UPDATED legacy build.py + smart caching)
    
    Args:
        variance_threshold: VarianceThreshold parameter (legacy: 1.0)
        use_cache: Enable smart caching
        force_rerun: Force rerun all stages
    """
    print("\n" + "="*70)
    print("🧬 LEGACY PIPELINE - UPDATED build.py replica with smart caching")
    print("="*70)
    print(f"Variance threshold: {variance_threshold}")
    print(f"Smart caching: {'enabled' if use_cache else 'disabled'}")
    print(f"Force rerun: {force_rerun}")
    
    # Initialize cache manager
    cache_manager = get_cache_manager("legacy_pipeline") if use_cache else None
    
    # Check cache status
    if cache_manager and not force_rerun:
        print("\n📊 Cache Status:")
        cache_manager.print_cache_status()
    
    # Stage 1: Load data
    stage_config = {"variance_threshold": variance_threshold}
    if cache_manager and not force_rerun:
        should_run = cache_manager.should_rerun_stage("load_data", stage_config, force_rerun)
    else:
        should_run = True
    
    if should_run:
        molecules = load_data()
        print("Molecules index after loading:", molecules.index.name)
        
        if cache_manager:
            output_file = project_root / "data" / "00_raw" / "molecules_raw.csv"
            cache_manager.mark_stage_complete("load_data", stage_config, str(output_file))
    else:
        print("\n✅ Using cached molecules data")
        molecules = pd.read_csv(project_root / "data" / "00_raw" / "molecules_raw.csv", index_col=0)
    
    # Stage 2: Preprocess (includes featurization)
    if cache_manager and not force_rerun:
        should_run = cache_manager.should_rerun_stage("preprocess", stage_config, force_rerun)
    else:
        should_run = True
    
    if should_run:
        cleaned_data = preprocess_data(molecules)
        print("Cleaned data index after preprocessing:", cleaned_data.index.name)
        
        # Save cleaned data
        output_dir = project_root / "data" / "02_processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "cleaned_data.csv"
        cleaned_data.to_csv(output_file, index=True)  # Save with CID index
        
        if cache_manager:
            cache_manager.mark_stage_complete("preprocess", stage_config, str(output_file))
    else:
        print("\n✅ Using cached cleaned data")
        output_file = project_root / "data" / "02_processed" / "cleaned_data.csv"
        cleaned_data = pd.read_csv(output_file, index_col=0)  # Load with CID index
    
    # Stage 3: Feature selection
    if cache_manager and not force_rerun:
        should_run = cache_manager.should_rerun_stage("select_features", stage_config, force_rerun)
    else:
        should_run = True
    
    if should_run:
        selected_features = select_features(cleaned_data, variance_threshold)
        print("Selected features index after feature selection:", selected_features.index.name)
        
        # Save selected features
        output_dir = project_root / "data" / "02_processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "selected_features.csv"
        selected_features.to_csv(output_file, index=True)  # Save with CID index
        
        if cache_manager:
            cache_manager.mark_stage_complete("select_features", stage_config, str(output_file))
    else:
        print("\n✅ Using cached selected features")
        output_file = project_root / "data" / "02_processed" / "selected_features.csv"
        selected_features = pd.read_csv(output_file, index_col=0)  # Load with CID index
    
    # Final summary
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE")
    print("="*70)
    print(f"Final features: {selected_features.shape}")
    print(f"CID index maintained: {selected_features.index.name}")
    print(f"\nOutput saved to:")
    print(f"  - data/00_raw/molecules_raw.csv")
    print(f"  - data/02_processed/cleaned_data.csv (with CID index)")
    print(f"  - data/02_processed/selected_features.csv (with CID index)")
    
    if cache_manager:
        print(f"\n📊 Cache saved to: data/.cache/legacy_pipeline/")
        print(f"   Next run will be instant! ⚡")
    
    return selected_features


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="UPDATED legacy build.py replica with caching")
    parser.add_argument("--variance-threshold", type=float, default=1.0,
                       help="Variance threshold for feature selection (default: 1.0)")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable caching")
    parser.add_argument("--force", action="store_true",
                       help="Force rerun all stages (ignore cache)")
    
    args = parser.parse_args()
    
    try:
        selected_features = process_all(
            variance_threshold=args.variance_threshold,
            use_cache=not args.no_cache,
            force_rerun=args.force
        )
        print("\n✅ Success!")
        print(f"\nFinal output shape: {selected_features.shape}")
        print(f"Index: {selected_features.index.name}")
        print(f"\nFirst few rows:")
        print(selected_features.head())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
