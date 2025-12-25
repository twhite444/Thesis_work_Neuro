from __future__ import annotations
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


def featurize_and_standardize(
    molecules: pd.DataFrame, 
    variance_threshold: float = 0.0,
    drop_nan_columns: bool = True,
    drop_zero_columns: bool = True,
    standardize: bool = True,
    output_dir: str = "data/02_processed",
    save_intermediate: bool = False,
) -> pd.DataFrame:
    """Featurize SMILES to Mordred descriptors and apply preprocessing pipeline.
    
    Pipeline steps (all optional and configurable):
    1. SMILES → Mordred molecular descriptors
    2. Drop columns with NaN values (optional)
    3. Drop zero-only columns (optional)
    4. Apply variance threshold filtering (BEFORE standardization)
    5. Standardize features to mean=0, std=1 (optional)
    
    Args:
        molecules: DataFrame with 'IsomericSMILES' and 'CID' columns
        variance_threshold: Minimum variance for feature selection (default: 0.0).
                          0.0 = remove only constants
                          Higher values remove low-variance features
                          Applied BEFORE standardization to be meaningful
        drop_nan_columns: Whether to drop columns containing any NaN (default: True)
        drop_zero_columns: Whether to drop columns that are entirely zero (default: True)
        standardize: Whether to standardize features with StandardScaler (default: True)
        output_dir: Directory to save processed data
        save_intermediate: Whether to save intermediate outputs (pre-standardization) (default: False)
        
    Returns:
        DataFrame with processed features and CID as index
        
    Note:
        Variance threshold is applied BEFORE standardization because:
        - StandardScaler forces all features to have variance ≈ 1.0
        - Applying variance threshold after standardization is meaningless
        - We filter low-variance features on the raw (but cleaned) data
    """
    os.makedirs(output_dir, exist_ok=True)

    if 'IsomericSMILES' not in molecules.columns:
        raise ValueError("Expected 'IsomericSMILES' column in molecules")
    if 'CID' not in molecules.columns:
        raise ValueError("Expected 'CID' column in molecules")

    smiles = molecules['IsomericSMILES'].tolist()
    cids = molecules['CID'].values
    
    # Try to load cached Mordred features
    print(f"Loading Mordred descriptors for {len(smiles)} molecules...")
    try:
        from src.olfactory_modeling.data.pyrfume_loader import load_mordred_features_npz
        raw_data_dir = output_dir.replace('02_processed', '01_raw')
        mordred_features = load_mordred_features_npz(raw_data_dir)
        print(f"  ✓ Loaded {mordred_features.shape[1]} descriptors from cache ({raw_data_dir}/mordred_features_raw.npz)")
        
        # Align features with the provided molecules (in case of CID mismatch)
        mordred_features = mordred_features.loc[cids]
        
    except (FileNotFoundError, ImportError) as e:
        # Fallback to computing if cache not available
        print(f"  Cache not found, computing Mordred descriptors from SMILES...")
        print(f"  💡 Tip: Run 'python scripts/load_all_data.py' first to cache features for faster preprocessing")
        from pyrfume.features import smiles_to_mordred
        mordred_features = smiles_to_mordred(smiles)
        mordred_features.index = cids
        print(f"  ✓ Computed {mordred_features.shape[1]} descriptors")
    
    filtered = mordred_features.copy()
    
    # Step 1: Drop NaN columns
    if drop_nan_columns:
        n_before = filtered.shape[1]
        filtered = filtered.dropna(axis=1, how='any')
        n_dropped = n_before - filtered.shape[1]
        if n_dropped > 0:
            print(f"  Dropped {n_dropped} columns with NaN values")
    
    # Step 2: Drop zero-only columns
    if drop_zero_columns:
        n_before = filtered.shape[1]
        zero_only = filtered.eq(0).all(axis=0)
        filtered = filtered.loc[:, ~zero_only]
        n_dropped = n_before - filtered.shape[1]
        if n_dropped > 0:
            print(f"  Dropped {n_dropped} zero-only columns")
    
    # Step 3: Variance threshold (BEFORE standardization!)
    if variance_threshold > 0:
        print(f"\nApplying variance threshold: {variance_threshold}")
        n_before = filtered.shape[1]
        
        # Calculate variances for reporting
        variances = filtered.var()
        print(f"  Variance range: [{variances.min():.4f}, {variances.max():.4f}]")
        
        selector = VarianceThreshold(threshold=variance_threshold)
        selected_data = selector.fit_transform(filtered)
        selected_cols = filtered.columns[selector.get_support()]
        filtered = pd.DataFrame(selected_data, columns=selected_cols, index=filtered.index)
        
        n_dropped = n_before - filtered.shape[1]
        if n_dropped > 0:
            print(f"  Removed {n_dropped} low-variance features")
            print(f"  Kept {filtered.shape[1]} features")
        else:
            print(f"  No features removed (all have variance > {variance_threshold})")
    
    # Add CID index
    filtered.index = cids
    filtered.index.name = 'CID'
    
    # Save intermediate (pre-standardization) data if requested
    if save_intermediate:
        intermediate_path = os.path.join(output_dir, 'unscaled_features.csv')
        filtered.to_csv(intermediate_path, index=True)
        print(f"\n✓ Saved unscaled features to {intermediate_path}")
    
    # Step 4: Standardization
    if standardize:
        print(f"\nStandardizing features...")
        scaler = StandardScaler()
        standardized = scaler.fit_transform(filtered)
        result_df = pd.DataFrame(standardized, columns=filtered.columns, index=cids)
        result_df.index.name = 'CID'
        
        # Persist scaler stats for reproducibility
        stats = {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist(),
            'features': filtered.columns.tolist(),
        }
        pd.Series(stats).to_json(os.path.join(output_dir, 'scaler_stats.json'))
        
        cleaned_path = os.path.join(output_dir, 'cleaned_data.csv')
        result_df.to_csv(cleaned_path, index=True)
        print(f"✓ Saved standardized features to {cleaned_path}")
    else:
        result_df = filtered
        cleaned_path = os.path.join(output_dir, 'cleaned_data.csv')
        result_df.to_csv(cleaned_path, index=True)
        print(f"✓ Saved features to {cleaned_path} (not standardized)")
    
    # Save metadata
    metadata = {
        'n_samples': len(result_df),
        'n_features': len(result_df.columns),
        'variance_threshold': variance_threshold,
        'drop_nan_columns': drop_nan_columns,
        'drop_zero_columns': drop_zero_columns,
        'standardized': standardize,
    }
    pd.Series(metadata).to_json(os.path.join(output_dir, 'preprocess_metadata.json'))
    
    print(f"\nFinal feature set: {result_df.shape[0]} samples × {result_df.shape[1]} features")
    
    return result_df
