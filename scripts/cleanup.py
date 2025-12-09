#!/usr/bin/env python3
"""
Cleanup Utility - Manage cached data and experiment outputs

Copyright (c) 2025 Tom White
Licensed under the MIT License

Usage:
    # Show what would be deleted (dry run)
    python scripts/cleanup.py --dry-run
    
    # Clear all caches
    python scripts/cleanup.py --cache all
    
    # Clear specific stage cache
    python scripts/cleanup.py --cache features
    python scripts/cleanup.py --cache preprocessing
    
    # Clear old caches (older than N days)
    python scripts/cleanup.py --cache all --older-than 30
    
    # Clear experiment outputs
    python scripts/cleanup.py --outputs
    
    # Clear old experiments (older than N days)
    python scripts/cleanup.py --outputs --older-than 30
    
    # Clear processed data
    python scripts/cleanup.py --processed
    
    # Clear everything (be careful!)
    python scripts/cleanup.py --all

This utility helps manage disk space by cleaning up:
- Cached feature extractions
- Cached preprocessing results
- Old experiment outputs
- Processed data files
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import shutil

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import hydra
from omegaconf import DictConfig

from neuro_smell.utils.cache_manager import CacheManager


def get_dir_size(directory: Path) -> float:
    """Get directory size in MB"""
    total = 0
    try:
        for item in directory.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
    except Exception:
        pass
    return total / (1024 * 1024)


def delete_old_files(directory: Path, days: int, dry_run: bool = False) -> tuple:
    """Delete files older than N days"""
    cutoff = datetime.now() - timedelta(days=days)
    deleted_count = 0
    deleted_size = 0
    
    if not directory.exists():
        return deleted_count, deleted_size
    
    for item in directory.iterdir():
        try:
            mtime = datetime.fromtimestamp(item.stat().st_mtime)
            if mtime < cutoff:
                size = get_dir_size(item) if item.is_dir() else item.stat().st_size / (1024 * 1024)
                
                if not dry_run:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                
                deleted_count += 1
                deleted_size += size
        except Exception as e:
            print(f"⚠️  Error processing {item}: {e}")
    
    return deleted_count, deleted_size


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    """
    Cleanup utility main function.
    
    Args:
        config: Hydra configuration
    """
    parser = argparse.ArgumentParser(description='Cleanup cached data and outputs')
    parser.add_argument('--cache', choices=['all', 'features', 'preprocessing', 'training'],
                        help='Clear cache for specified stage(s)')
    parser.add_argument('--outputs', action='store_true',
                        help='Clear experiment outputs')
    parser.add_argument('--processed', action='store_true',
                        help='Clear processed data files')
    parser.add_argument('--all', action='store_true',
                        help='Clear everything (cache, outputs, processed)')
    parser.add_argument('--older-than', type=int, metavar='DAYS',
                        help='Only delete items older than N days')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be deleted without deleting')
    
    args, _ = parser.parse_known_args()
    
    print("\n" + "="*60)
    print("🗑️  Cleanup Utility")
    print("="*60)
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - Nothing will be deleted\n")
    
    # Initialize cache manager
    cache_root = Path(config.paths.cache)
    cache_manager = CacheManager(cache_root, config)
    
    total_size_freed = 0
    
    # Clear cache
    if args.cache or args.all:
        print("\n🗄️  Clearing cache...")
        
        # Get current cache size
        cache_size_before = cache_manager.get_cache_size()
        print(f"   Current cache size: {cache_size_before:.2f} MB")
        
        if not args.dry_run:
            stage = args.cache if args.cache != 'all' else None
            cache_manager.clear_cache(
                stage=stage,
                older_than_days=args.older_than
            )
            
            cache_size_after = cache_manager.get_cache_size()
            freed = cache_size_before - cache_size_after
            total_size_freed += freed
            print(f"   ✅ Freed {freed:.2f} MB")
        else:
            # Count what would be deleted
            cache_list = cache_manager.list_caches()
            stage_filter = args.cache if args.cache != 'all' else None
            
            for stage, caches in cache_list.items():
                if stage_filter and stage != stage_filter:
                    continue
                
                for cache_info in caches:
                    if args.older_than:
                        created = datetime.fromisoformat(cache_info['created'])
                        age_days = (datetime.now() - created).days
                        if age_days <= args.older_than:
                            continue
                    
                    print(f"   Would delete: {stage}/{cache_info['cache_key']} ({cache_info['size_mb']:.2f} MB)")
                    total_size_freed += cache_info['size_mb']
    
    # Clear outputs
    if args.outputs or args.all:
        print("\n📁 Clearing experiment outputs...")
        
        outputs_dir = Path(config.paths.outputs)
        
        if outputs_dir.exists():
            size_before = get_dir_size(outputs_dir)
            print(f"   Current outputs size: {size_before:.2f} MB")
            
            if args.older_than:
                count, size = delete_old_files(outputs_dir, args.older_than, args.dry_run)
                print(f"   {'Would delete' if args.dry_run else 'Deleted'} {count} experiment(s) ({size:.2f} MB)")
                total_size_freed += size
            else:
                if not args.dry_run:
                    shutil.rmtree(outputs_dir)
                    outputs_dir.mkdir(parents=True)
                    print(f"   ✅ Cleared all outputs ({size_before:.2f} MB)")
                    total_size_freed += size_before
                else:
                    print(f"   Would delete all outputs ({size_before:.2f} MB)")
                    total_size_freed += size_before
        else:
            print("   No outputs directory found")
    
    # Clear processed data
    if args.processed or args.all:
        print("\n📊 Clearing processed data...")
        
        processed_dir = Path(config.paths.processed)
        
        if processed_dir.exists():
            size_before = get_dir_size(processed_dir)
            print(f"   Current processed data size: {size_before:.2f} MB")
            
            if args.older_than:
                count, size = delete_old_files(processed_dir, args.older_than, args.dry_run)
                print(f"   {'Would delete' if args.dry_run else 'Deleted'} {count} file(s) ({size:.2f} MB)")
                total_size_freed += size
            else:
                if not args.dry_run:
                    shutil.rmtree(processed_dir)
                    processed_dir.mkdir(parents=True)
                    print(f"   ✅ Cleared all processed data ({size_before:.2f} MB)")
                    total_size_freed += size_before
                else:
                    print(f"   Would delete all processed data ({size_before:.2f} MB)")
                    total_size_freed += size_before
        else:
            print("   No processed data directory found")
    
    # Print summary
    print("\n" + "="*60)
    if args.dry_run:
        print(f"📊 Would free: {total_size_freed:.2f} MB")
        print("\nRun without --dry-run to actually delete files")
    else:
        print(f"✅ Cleanup complete! Freed {total_size_freed:.2f} MB")
    print("="*60 + "\n")
    
    # Show remaining disk usage
    print("💾 Current Disk Usage:")
    
    cache_size = cache_manager.get_cache_size()
    print(f"   Cache: {cache_size:.2f} MB")
    
    if Path(config.paths.outputs).exists():
        outputs_size = get_dir_size(Path(config.paths.outputs))
        print(f"   Outputs: {outputs_size:.2f} MB")
    
    if Path(config.paths.processed).exists():
        processed_size = get_dir_size(Path(config.paths.processed))
        print(f"   Processed: {processed_size:.2f} MB")
    
    print()


if __name__ == "__main__":
    main()
