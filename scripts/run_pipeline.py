#!/usr/bin/env python3
"""
Full Pipeline Script - Run complete pipeline with stage caching

Usage:
    # Run full pipeline with default config
    python scripts/run_pipeline.py
    
    # Use specific experiment config
    python scripts/run_pipeline.py experiment=example_no_pca
    
    # Force recompute all stages (ignore cache)
    python scripts/run_pipeline.py --force-recompute
    
    # Skip training (just extract features and preprocess)
    python scripts/run_pipeline.py --skip-training

This script runs all three stages:
1. Feature Extraction (from SMILES or use existing features)
2. Preprocessing (optional PCA, scaling, feature selection)
3. Training (PyTorch Lightning with GPU support)

Each stage is cached based on config, so changing model architecture
doesn't re-run feature extraction!
"""

import sys
from pathlib import Path
import argparse

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import hydra
from omegaconf import DictConfig, OmegaConf

from neuro_smell.utils.cache_manager import CacheManager
from neuro_smell.stages.feature_extraction import extract_features
from neuro_smell.stages.preprocessing import preprocess_data
from neuro_smell.stages.training import train_model


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    """
    Run full pipeline with intelligent caching.
    
    Args:
        config: Hydra configuration
    """
    print("\n" + "="*60)
    print("🔬 Full Olfactory Prediction Pipeline")
    print("="*60)
    
    # Parse additional arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-recompute', action='store_true',
                        help='Force recompute all stages (ignore cache)')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training stage')
    args, _ = parser.parse_known_args()
    
    # Print configuration
    print("\n📋 Configuration:")
    print(OmegaConf.to_yaml(config))
    
    # Initialize cache manager
    cache_root = Path(config.paths.cache)
    cache_manager = CacheManager(cache_root, config)
    
    print(f"\n💾 Cache directory: {cache_root}")
    
    if args.force_recompute:
        print("⚠️  Force recompute enabled - ignoring all caches")
    
    # Stage 1: Feature Extraction
    print("\n" + "="*60)
    print("📊 Stage 1: Feature Extraction")
    print("="*60)
    
    df_features = extract_features(
        config=config,
        cache_manager=cache_manager,
        force_recompute=args.force_recompute
    )
    
    print(f"✅ Features extracted: {df_features.shape}")
    
    # Stage 2: Preprocessing
    print("\n" + "="*60)
    print("🔧 Stage 2: Preprocessing")
    print("="*60)
    
    df_preprocessed, preprocessor = preprocess_data(
        df=df_features,
        config=config,
        target_column=config.data.target_column,
        cache_manager=cache_manager,
        force_recompute=args.force_recompute
    )
    
    print(f"✅ Preprocessing complete: {df_preprocessed.shape}")
    
    # Show preprocessing info
    prep_info = preprocessor.get_preprocessing_info()
    print(f"\n📊 Preprocessing Summary:")
    print(f"   Input features: {prep_info.get('input_dim', 'N/A')}")
    print(f"   Output features: {prep_info.get('output_dim', 'N/A')}")
    print(f"   Scaling: {prep_info.get('scaling', 'none')}")
    print(f"   PCA: {prep_info.get('pca_enabled', False)}")
    if prep_info.get('variance_explained'):
        print(f"   Variance explained: {prep_info['variance_explained']*100:.2f}%")
    
    # Save preprocessed data for training
    processed_data_path = Path(config.paths.processed) / f"{config.experiment_name}_processed.csv"
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    df_preprocessed.to_csv(processed_data_path, index=False)
    print(f"\n💾 Saved preprocessed data: {processed_data_path}")
    
    # Update config to use preprocessed data
    config.data.data_path = str(processed_data_path)
    
    # Stage 3: Training
    if not args.skip_training:
        print("\n" + "="*60)
        print("🏋️  Stage 3: Training")
        print("="*60)
        
        results = train_model(config)
        
        # Final summary
        print("\n" + "="*60)
        print("✅ Pipeline Complete!")
        print("="*60)
        print(f"\nExperiment: {results['experiment_name']}")
        print(f"Output directory: {results['output_dir']}")
        
        if results['test_results']:
            print(f"\n🎯 Final Test Results:")
            for key, value in results['test_results'].items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
    else:
        print("\n⏭️  Skipping training stage")
        print("\n" + "="*60)
        print("✅ Preprocessing Complete!")
        print("="*60)
        print(f"\nPreprocessed data saved: {processed_data_path}")
    
    # Show cache info
    print("\n💾 Cache Summary:")
    cache_size = cache_manager.get_cache_size()
    print(f"   Total cache size: {cache_size:.2f} MB")
    
    cache_list = cache_manager.list_caches()
    for stage, caches in cache_list.items():
        if caches:
            print(f"   {stage}: {len(caches)} cache(s)")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
