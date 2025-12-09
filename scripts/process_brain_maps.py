"""
Process Brain Activity Maps

This script demonstrates the complete brain activity processing pipeline:
1. Load individual brain maps (405 stimulus presentations)
2. Average by CID (handle multiple concentrations/repetitions)
3. Apply PCA to extract principal spatial patterns
4. Extract first 5 PC scores as model targets

Usage:
    python scripts/process_brain_maps.py
    
Output:
    - data/02_processed/brain_pca_scores.csv: 287 × 5 target values
    - data/02_processed/brain_maps_averaged.npz: Averaged brain maps
    - test_output/brain_pca/: Visualizations
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.neuro_smell.stages.brain_activity import BrainActivityProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run brain activity processing pipeline."""
    
    logger.info("=" * 80)
    logger.info("BRAIN ACTIVITY PROCESSING PIPELINE")
    logger.info("=" * 80)
    
    # Define paths
    data_dir = project_root / "data" / "00_raw"
    molecules_csv = data_dir / "molecules_raw.csv"
    behavior_csv = data_dir / "behavior_data.csv"
    csvs_dir = data_dir / "csvs"
    
    output_dir = project_root / "data" / "02_processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    viz_dir = project_root / "test_output" / "brain_pca"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify files exist
    logger.info("\n1. Verifying input files...")
    for path in [molecules_csv, behavior_csv, csvs_dir]:
        if not path.exists():
            logger.error(f"❌ Missing: {path}")
            logger.error("Please ensure Pyrfume data is downloaded.")
            return 1
        logger.info(f"✅ Found: {path}")
    
    # Load molecules
    logger.info("\n2. Loading molecules...")
    molecules_df = pd.read_csv(molecules_csv)
    logger.info(f"Loaded {len(molecules_df)} molecules")
    logger.info(f"Columns: {molecules_df.columns.tolist()}")
    
    # Check if CID column exists
    if 'CID' not in molecules_df.columns:
        logger.error("❌ 'CID' column not found in molecules_raw.csv")
        logger.info(f"Available columns: {molecules_df.columns.tolist()}")
        return 1
    
    logger.info(f"Unique CIDs: {molecules_df['CID'].nunique()}")
    
    # Initialize processor
    logger.info("\n3. Initializing BrainActivityProcessor...")
    processor = BrainActivityProcessor(
        n_components=50,
        n_targets=5,
        standardize=True
    )
    
    # Load and average brain maps
    logger.info("\n4. Loading and averaging brain maps by CID...")
    try:
        averaged_maps_df = processor.load_and_average_maps(
            behavior_csv=str(behavior_csv),
            csvs_dir=str(csvs_dir),
            molecules_df=molecules_df,
            cid_column='CID'
        )
        
        logger.info(f"✅ Averaged brain maps: {len(averaged_maps_df)} molecules")
        logger.info(f"Brain map shape: {averaged_maps_df['brain_map'].iloc[0].shape}")
        logger.info(
            f"Repetitions per molecule: "
            f"mean={averaged_maps_df['n_reps'].mean():.2f}, "
            f"median={averaged_maps_df['n_reps'].median():.0f}, "
            f"max={averaged_maps_df['n_reps'].max()}"
        )
        
        # Show examples of high-repetition molecules
        high_rep = averaged_maps_df.nlargest(5, 'n_reps')
        logger.info("\nTop 5 molecules by repetitions (likely concentration series):")
        for _, row in high_rep.iterrows():
            logger.info(f"  CID {row['CID']}: {row['n_reps']} repetitions")
        
    except Exception as e:
        logger.error(f"❌ Error loading brain maps: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Apply PCA
    logger.info("\n5. Applying PCA to brain activity data...")
    try:
        pca_scores = processor.apply_pca()
        
        logger.info(f"✅ PCA scores shape: {pca_scores.shape}")
        logger.info(
            f"Variance explained by first {processor.n_targets} components: "
            f"{np.sum(processor.pca.explained_variance_ratio_[:processor.n_targets])*100:.2f}%"
        )
        
    except Exception as e:
        logger.error(f"❌ Error applying PCA: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Extract targets
    logger.info("\n6. Extracting target values (first 5 PC scores)...")
    try:
        targets = processor.extract_targets()
        
        logger.info(f"✅ Targets shape: {targets.shape}")
        logger.info(f"Targets statistics:")
        logger.info(f"  Mean: {targets.mean(axis=0)}")
        logger.info(f"  Std: {targets.std(axis=0)}")
        logger.info(f"  Min: {targets.min(axis=0)}")
        logger.info(f"  Max: {targets.max(axis=0)}")
        
    except Exception as e:
        logger.error(f"❌ Error extracting targets: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save outputs
    logger.info("\n7. Saving outputs...")
    
    # Save PCA scores as targets
    targets_df = pd.DataFrame(
        targets,
        columns=[f'PC{i+1}' for i in range(targets.shape[1])]
    )
    targets_df['CID'] = averaged_maps_df['CID'].values
    targets_df = targets_df[['CID'] + [f'PC{i+1}' for i in range(targets.shape[1])]]
    
    targets_path = output_dir / "brain_pca_scores.csv"
    targets_df.to_csv(targets_path, index=False)
    logger.info(f"✅ Saved targets: {targets_path}")
    
    # Save averaged brain maps (for future analysis)
    brain_maps_path = output_dir / "brain_maps_averaged.npz"
    np.savez_compressed(
        brain_maps_path,
        brain_matrix=processor.brain_matrix,
        cids=averaged_maps_df['CID'].values,
        n_reps=averaged_maps_df['n_reps'].values
    )
    logger.info(f"✅ Saved averaged brain maps: {brain_maps_path}")
    
    # Save PCA model
    pca_model_path = output_dir / "brain_pca_model.npz"
    np.savez(
        pca_model_path,
        components=processor.pca.components_,
        explained_variance=processor.pca.explained_variance_,
        explained_variance_ratio=processor.pca.explained_variance_ratio_,
        mean=processor.pca.mean_,
        n_components=processor.pca.n_components_
    )
    logger.info(f"✅ Saved PCA model: {pca_model_path}")
    
    # Create visualizations
    logger.info("\n8. Creating visualizations...")
    try:
        processor.visualize_pca(
            output_dir=str(viz_dir),
            n_components_to_plot=5
        )
        logger.info(f"✅ Saved visualizations: {viz_dir}")
        
    except Exception as e:
        logger.warning(f"⚠️ Error creating visualizations: {e}")
        # Non-critical, continue
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE ✅")
    logger.info("=" * 80)
    logger.info(f"Input: {len(molecules_df)} molecules, 405 brain map presentations")
    logger.info(f"Output: {len(averaged_maps_df)} averaged brain maps")
    logger.info(f"Targets: {targets.shape[0]} × {targets.shape[1]} PCA scores")
    logger.info(f"Variance explained: {np.sum(processor.pca.explained_variance_ratio_[:5])*100:.2f}%")
    logger.info("\nOutput files:")
    logger.info(f"  1. {targets_path.relative_to(project_root)}")
    logger.info(f"  2. {brain_maps_path.relative_to(project_root)}")
    logger.info(f"  3. {pca_model_path.relative_to(project_root)}")
    logger.info(f"  4. {viz_dir.relative_to(project_root)}/pca_*.png")
    logger.info("\nNext steps:")
    logger.info("  - Align molecular features (X) with brain PCA scores (y)")
    logger.info("  - Train neural network: X → y")
    logger.info("  - Validate R² ≈ 0.506 (thesis result)")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
