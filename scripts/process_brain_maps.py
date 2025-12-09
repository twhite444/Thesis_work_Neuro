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
        
        logger.info(f"✅ Processed {len(averaged_maps_df)} molecules")
        logger.info(f"✅ Brain maps averaged by CID")
        
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
        logger.info(f"PC1 variance: {processor.pca.explained_variance_ratio_[0]*100:.2f}%")
        logger.info(f"PC2 variance: {processor.pca.explained_variance_ratio_[1]*100:.2f}%")
        logger.info(
            f"Total (first 5): "
            f"{np.sum(processor.pca.explained_variance_ratio_[:5])*100:.2f}%"
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
    # Add CID column
    targets_df['CID'] = averaged_maps_df['CID'].values
    # Reorder columns to have CID first
    targets_df = targets_df[['CID'] + [f'PC{i+1}' for i in range(targets.shape[1])]]
    
    targets_path = output_dir / "brain_pca_scores.csv"
    targets_df.to_csv(targets_path, index=False)
    logger.info(f"✅ Saved targets: {targets_path}")
    
    # Save averaged brain maps
    brain_maps_path = output_dir / "brain_maps_averaged.npz"
    brain_maps_array = np.stack([arr for arr in averaged_maps_df['brain_map'].values])
    np.savez_compressed(
        brain_maps_path,
        brain_matrix=brain_maps_array,
        cids=averaged_maps_df['CID'].values
    )
    logger.info(f"✅ Saved averaged brain maps: {brain_maps_path}")
    
    # Save PCA model
    pca_model_path = output_dir / "brain_pca_model.npz"
    np.savez(
        pca_model_path,
        components=processor.pca.components_,
        explained_variance=processor.pca.explained_variance_,
        explained_variance_ratio=processor.pca.explained_variance_ratio_,
        mean=processor.pca.mean_
    )
    logger.info(f"✅ Saved PCA model: {pca_model_path}")
    
    # Create visualizations
    logger.info("\n8. Creating visualizations...")
    try:
        processor.visualize_pca(output_dir=str(viz_dir))
        logger.info(f"✅ Saved visualizations: {viz_dir}")
        
    except Exception as e:
        logger.warning(f"⚠️ Error creating visualizations: {e}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE ✅")
    logger.info("=" * 80)
    logger.info(f"Molecules: {len(averaged_maps_df)}")
    logger.info(f"Targets: {targets.shape}")
    logger.info(f"Variance: {np.sum(processor.pca.explained_variance_ratio_[:5])*100:.2f}%")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
