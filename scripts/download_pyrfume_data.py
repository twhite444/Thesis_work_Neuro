"""
Download Pyrfume Leon Dataset

This script downloads the complete Johnson & Leon (2007) dataset from Pyrfume,
including:
1. molecules.csv - Molecular identifiers and SMILES strings
2. behavior_1.csv - Stimulus-to-brain-map lookup table
3. csvs/*.csv - Individual brain activity maps (405 files)

The brain activity maps are 2D spatial patterns of glomerular activation
measured using 2-deoxyglucose (2-DG) imaging.

Usage:
    python scripts/download_pyrfume_data.py
    
Output:
    data/00_raw/molecules_raw.csv
    data/00_raw/behavior_data.csv
    data/00_raw/csvs/*.csv (405 brain map files)
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd

try:
    import pyrfume
except ImportError:
    print("ERROR: pyrfume package not installed.")
    print("Install with: pip install pyrfume")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Download complete Pyrfume Leon dataset."""
    
    logger.info("=" * 80)
    logger.info("DOWNLOADING PYRFUME LEON DATASET")
    logger.info("=" * 80)
    
    # Setup output directory
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "00_raw"
    csvs_dir = output_dir / "csvs"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    csvs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Download molecules
    logger.info("\n1. Downloading molecules.csv...")
    try:
        molecules = pyrfume.load_data('leon/molecules.csv')
        
        # CID is the index, check for duplicate indices
        n_before = len(molecules)
        molecules = molecules[~molecules.index.duplicated(keep='first')]
        n_after = len(molecules)
        
        molecules_path = output_dir / "molecules_raw.csv"
        molecules.to_csv(molecules_path, index=True)
        
        logger.info(f"✅ Downloaded molecules: {n_before} total, {n_after} unique CIDs")
        logger.info(f"   Columns: {molecules.columns.tolist()}")
        logger.info(f"   Saved to: {molecules_path}")
        
    except Exception as e:
        logger.error(f"❌ Error downloading molecules: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 2. Download behavior data (stimulus mapping)
    logger.info("\n2. Downloading behavior_1.csv (stimulus mapping)...")
    try:
        behavior = pyrfume.load_data('leon/behavior_1.csv')
        
        behavior_path = output_dir / "behavior_data.csv"
        behavior.to_csv(behavior_path, index=True)
        
        logger.info(f"✅ Downloaded behavior data: {len(behavior)} stimulus presentations")
        logger.info(f"   Columns: {behavior.columns.tolist()}")
        logger.info(f"   Saved to: {behavior_path}")
        
        # Stimulus is the index - extract for downloading brain maps
        stimuli = behavior.index.values
        logger.info(f"   Found {len(stimuli)} brain maps to download")
        
    except Exception as e:
        logger.error(f"❌ Error downloading behavior data: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 3. Download all brain activity maps
    logger.info("\n3. Downloading brain activity maps (this may take a while)...")
    
    downloaded = 0
    failed = []
    
    for i, stimulus in enumerate(stimuli, 1):
        # Construct Pyrfume path
        pyrfume_path = f'leon/csvs/{stimulus}.csv'
        output_path = csvs_dir / f"{stimulus}.csv"
        
        # Skip if already downloaded
        if output_path.exists():
            logger.debug(f"   [{i}/{len(stimuli)}] Skipping {stimulus}.csv (already exists)")
            downloaded += 1
            continue
        
        try:
            brain_map = pyrfume.load_data(pyrfume_path)
            brain_map.to_csv(output_path, index=False, header=False)
            
            downloaded += 1
            
            if i % 50 == 0:
                logger.info(f"   [{i}/{len(stimuli)}] Downloaded {downloaded} brain maps...")
                
        except Exception as e:
            logger.warning(f"   Failed to download {stimulus}.csv: {e}")
            failed.append(stimulus)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 80)
    logger.info(f"✅ Molecules: {n_after} unique molecules")
    logger.info(f"✅ Behavior data: {len(behavior)} stimulus presentations")
    logger.info(f"✅ Brain maps: {downloaded} / {len(stimuli)} files downloaded")
    
    if failed:
        logger.warning(f"⚠️  Failed downloads: {len(failed)} files")
        logger.warning(f"   First few failures: {failed[:10]}")
    
    logger.info("\nFiles saved to:")
    logger.info(f"  - {output_dir}/molecules_raw.csv")
    logger.info(f"  - {output_dir}/behavior_data.csv")
    logger.info(f"  - {csvs_dir}/*.csv ({downloaded} files)")
    
    logger.info("\nNext step:")
    logger.info("  python scripts/process_brain_maps.py")
    logger.info("=" * 80)
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
