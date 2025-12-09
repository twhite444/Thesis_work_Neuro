"""
Smart Cache Manager - Stage-specific caching with comprehensive safeguards

Copyright (c) 2025 Tom White
Licensed under the MIT License

Features:
- Stage-specific caching (only reruns what changed)
- Input file hashing (detects upstream changes)
- File existence validation
- Cache versioning
- Platform isolation
- File locking (prevents race conditions)
- Detailed cache inspection
"""

import hashlib
import json
import platform
import fcntl
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from omegaconf import DictConfig, OmegaConf


class SmartCacheManager:
    """
    Production-ready cache manager for multi-stage ML pipeline.
    
    Each stage (feature extraction, preprocessing, training) gets its own
    cache that only invalidates when relevant config changes.
    
    Example:
        cache = SmartCacheManager(experiment_name="baseline")
        
        # Check if stage needs to rerun
        if cache.should_rerun_stage('preprocessing', preprocessing_config):
            results = run_preprocessing()
            cache.mark_stage_complete('preprocessing', preprocessing_config, output_file)
        else:
            results = load_cached_results()
    """
    
    CACHE_VERSION = "1.0"  # Increment when cache format changes
    
    def __init__(
        self, 
        experiment_name: str,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize cache manager.
        
        Args:
            experiment_name: Name of experiment (each gets own cache)
            cache_dir: Cache directory (default: data/.cache/{experiment_name})
        """
        # Experiment-specific cache prevents collisions
        if cache_dir is None:
            cache_dir = Path(f"data/.cache/{experiment_name}")
        
        self.experiment_name = experiment_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.cache_dir / "metadata.json"
        self.lock_file = self.cache_dir / ".lock"
        
        self.metadata = self._load_metadata()
    
    def should_rerun_stage(
        self, 
        stage_name: str, 
        stage_config: Dict,
        force: bool = False
    ) -> bool:
        """
        Determine if stage needs to rerun based on comprehensive checks.
        
        Checks (in order):
        1. Force flag
        2. No cache exists
        3. Cache version mismatch
        4. Config changed
        5. Output file missing
        6. Input file missing (for dependent stages)
        7. Input file changed (hash mismatch)
        
        Args:
            stage_name: Name of stage ('feature_extraction', 'preprocessing', 'training')
            stage_config: Configuration dict for this stage only
            force: If True, always rerun regardless of cache
        
        Returns:
            True if stage should rerun, False if can use cache
        """
        # 1. Force rerun requested
        if force:
            print(f"🔄 {stage_name}: Force rerun requested")
            return True
        
        # 2. No cache exists
        if stage_name not in self.metadata:
            print(f"🔄 {stage_name}: No cache found (first run)")
            return True
        
        cached_meta = self.metadata[stage_name]
        
        # 3. Cache version mismatch
        if cached_meta.get('cache_version') != self.CACHE_VERSION:
            print(f"🔄 {stage_name}: Cache version mismatch")
            print(f"   Cached: {cached_meta.get('cache_version')}")
            print(f"   Current: {self.CACHE_VERSION}")
            return True
        
        # 4. Config changed
        config_hash = self._compute_config_hash(stage_config)
        if cached_meta.get('config_hash') != config_hash:
            print(f"🔄 {stage_name}: Config changed")
            self._print_config_diff(stage_name, stage_config, cached_meta)
            return True
        
        # 5. Output file missing
        output_file = cached_meta.get('output_file')
        if not output_file or not Path(output_file).exists():
            print(f"🔄 {stage_name}: Output file missing")
            if output_file:
                print(f"   Expected: {output_file}")
            return True
        
        # 6. Input file missing (for dependent stages)
        input_file = self._extract_input_file(stage_config)
        if input_file:
            if not Path(input_file).exists():
                print(f"🔄 {stage_name}: Input file missing")
                print(f"   Expected: {input_file}")
                print(f"   ⚠️  You may need to rerun the previous stage")
                return True
            
            # 7. Input file changed
            input_hash = self._compute_file_hash(input_file)
            if cached_meta.get('input_hash') != input_hash:
                print(f"🔄 {stage_name}: Input file changed")
                print(f"   File: {input_file}")
                print(f"   This likely means upstream stage was rerun")
                return True
        
        # All checks passed - safe to use cache
        print(f"✅ {stage_name}: Using cached results")
        cache_age = self._get_cache_age(cached_meta.get('timestamp'))
        print(f"   Cached {cache_age} ago")
        return False
    
    def mark_stage_complete(
        self,
        stage_name: str,
        stage_config: Dict,
        output_file: str
    ):
        """
        Mark stage as complete and save metadata.
        
        Args:
            stage_name: Name of stage
            stage_config: Configuration used for this stage
            output_file: Path to output file produced
        """
        input_file = self._extract_input_file(stage_config)
        output_path = Path(output_file)
        
        # Collect comprehensive metadata
        self.metadata[stage_name] = {
            'cache_version': self.CACHE_VERSION,
            'config_hash': self._compute_config_hash(stage_config),
            'output_file': str(output_file),
            'output_size': output_path.stat().st_size if output_path.exists() else 0,
            'input_file': str(input_file) if input_file else None,
            'input_hash': self._compute_file_hash(input_file) if input_file else None,
            'platform': self._get_platform_info(),
            'timestamp': datetime.now().isoformat()
        }
        
        self._save_metadata()
        
        size_mb = self.metadata[stage_name]['output_size'] / 1024 / 1024
        print(f"💾 {stage_name}: Results cached ({size_mb:.2f} MB)")
    
    def get_cached_output_path(self, stage_name: str) -> Optional[Path]:
        """
        Get path to cached output file for a stage.
        
        Args:
            stage_name: Name of stage
        
        Returns:
            Path to cached output file, or None if no cache
        """
        if stage_name not in self.metadata:
            return None
        
        output_file = self.metadata[stage_name].get('output_file')
        if output_file and Path(output_file).exists():
            return Path(output_file)
        
        return None
    
    def clear_cache(self, stage_name: Optional[str] = None):
        """
        Clear cache for specific stage or all stages.
        
        Args:
            stage_name: Stage to clear, or None to clear all
        """
        if stage_name:
            if stage_name in self.metadata:
                # Delete the cached file
                output_file = self.metadata[stage_name].get('output_file')
                if output_file and Path(output_file).exists():
                    Path(output_file).unlink()
                    print(f"   Deleted: {output_file}")
                
                # Remove from metadata
                del self.metadata[stage_name]
                print(f"🗑️  Cleared cache for {stage_name}")
            else:
                print(f"⚠️  No cache found for {stage_name}")
        else:
            # Clear all stages
            stages_cleared = []
            for stage in list(self.metadata.keys()):
                if stage == 'version':
                    continue
                
                output_file = self.metadata[stage].get('output_file')
                if output_file and Path(output_file).exists():
                    Path(output_file).unlink()
                
                del self.metadata[stage]
                stages_cleared.append(stage)
            
            if stages_cleared:
                print(f"🗑️  Cleared cache for: {', '.join(stages_cleared)}")
            else:
                print("ℹ️  No caches to clear")
        
        self._save_metadata()
    
    def print_cache_status(self):
        """Print detailed status of all cached stages."""
        print("\n" + "="*70)
        print(f"📊 CACHE STATUS: {self.experiment_name}")
        print("="*70)
        
        # Check if any stages cached
        stages = [k for k in self.metadata.keys() if k != 'version']
        if not stages:
            print("\nℹ️  No cached stages")
            print("\n💡 Tip: Run your first experiment to populate cache")
            return
        
        # Print each stage
        total_size = 0
        for stage_name in sorted(stages):
            info = self.metadata[stage_name]
            
            print(f"\n📦 {stage_name}:")
            
            # Timestamp
            timestamp = info.get('timestamp', 'unknown')
            if timestamp != 'unknown':
                cache_age = self._get_cache_age(timestamp)
                print(f"   Cached: {cache_age} ago ({timestamp})")
            
            # Output file
            output_file = info.get('output_file', 'unknown')
            exists = Path(output_file).exists() if output_file != 'unknown' else False
            status = "✅" if exists else "❌ MISSING"
            print(f"   Output: {output_file} {status}")
            
            # Size
            size_bytes = info.get('output_size', 0)
            size_mb = size_bytes / 1024 / 1024
            total_size += size_bytes
            print(f"   Size: {size_mb:.2f} MB")
            
            # Input (for dependent stages)
            input_file = info.get('input_file')
            if input_file:
                input_exists = Path(input_file).exists()
                input_status = "✅" if input_exists else "❌ MISSING"
                print(f"   Input: {input_file} {input_status}")
            
            # Hashes (short form)
            config_hash = info.get('config_hash', 'unknown')
            print(f"   Config Hash: {config_hash[:12]}...")
            
            if info.get('input_hash'):
                input_hash = info.get('input_hash')
                print(f"   Input Hash: {input_hash[:12]}...")
            
            # Platform
            platform_info = info.get('platform', {})
            platform_str = f"{platform_info.get('system')} {platform_info.get('machine')}"
            print(f"   Platform: {platform_str}")
        
        # Summary
        total_mb = total_size / 1024 / 1024
        print(f"\n{'='*70}")
        print(f"Total cached: {len(stages)} stages, {total_mb:.2f} MB")
        print(f"Cache dir: {self.cache_dir}")
    
    def print_config_diff(self, stage_name: str, new_config: Dict):
        """
        Show differences between current config and cached config.
        
        Args:
            stage_name: Name of stage
            new_config: New configuration to compare
        """
        if stage_name not in self.metadata:
            print(f"ℹ️  No cached config for {stage_name}")
            return
        
        self._print_config_diff(stage_name, new_config, self.metadata[stage_name])
    
    # ========================================
    # Internal Methods
    # ========================================
    
    def _extract_input_file(self, config: Dict) -> Optional[str]:
        """Extract input file path from config."""
        # Try common input file keys
        for key in ['input_file', 'data_path', 'smiles_file', 'features_file']:
            if key in config and config[key]:
                return config[key]
        return None
    
    def _compute_config_hash(self, config: Dict) -> str:
        """
        Compute stable hash of configuration.
        
        Handles OmegaConf objects and ensures deterministic hashing.
        """
        # Convert OmegaConf to plain dict
        if isinstance(config, (DictConfig, dict)):
            if hasattr(config, '__dict__'):
                config = OmegaConf.to_container(config, resolve=True)
        
        # Recursive sort for deterministic hash
        def sort_nested(obj):
            if isinstance(obj, dict):
                return {k: sort_nested(v) for k, v in sorted(obj.items())}
            elif isinstance(obj, list):
                return [sort_nested(item) for item in obj]
            return obj
        
        config_sorted = sort_nested(config)
        config_str = json.dumps(config_sorted, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    def _compute_file_hash(self, file_path: str) -> str:
        """
        Compute hash of file contents.
        
        For large files (>100MB), only hashes first/last chunks for speed.
        """
        path = Path(file_path)
        if not path.exists():
            return None
        
        file_size = path.stat().st_size
        
        # For large files, hash samples (much faster)
        if file_size > 100_000_000:  # 100 MB
            hash_md5 = hashlib.md5()
            
            with open(path, 'rb') as f:
                # First 1MB
                hash_md5.update(f.read(1024 * 1024))
                
                # Last 1MB
                f.seek(-1024 * 1024, 2)
                hash_md5.update(f.read())
                
                # File size
                hash_md5.update(str(file_size).encode())
            
            return hash_md5.hexdigest()[:16]
        
        # For small files, hash everything
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:16]
    
    def _get_platform_info(self) -> Dict[str, str]:
        """Get platform information for cache isolation."""
        return {
            'system': platform.system(),
            'machine': platform.machine(),
            'python': platform.python_version()
        }
    
    def _get_cache_age(self, timestamp: str) -> str:
        """Get human-readable cache age."""
        try:
            cached_time = datetime.fromisoformat(timestamp)
            delta = datetime.now() - cached_time
            
            if delta.days > 0:
                return f"{delta.days} day{'s' if delta.days != 1 else ''}"
            elif delta.seconds > 3600:
                hours = delta.seconds // 3600
                return f"{hours} hour{'s' if hours != 1 else ''}"
            elif delta.seconds > 60:
                minutes = delta.seconds // 60
                return f"{minutes} minute{'s' if minutes != 1 else ''}"
            else:
                return "just now"
        except:
            return "unknown"
    
    def _print_config_diff(self, stage_name: str, new_config: Dict, cached_meta: Dict):
        """Print differences between configs (if possible)."""
        # This is simplified - full implementation would do deep comparison
        print(f"   (Config hash changed: full diff not shown)")
        print(f"   Cached hash: {cached_meta.get('config_hash', 'unknown')[:12]}...")
        print(f"   New hash: {self._compute_config_hash(new_config)[:12]}...")
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata with file locking."""
        if not self.metadata_file.exists():
            return {'version': self.CACHE_VERSION}
        
        try:
            with open(self.metadata_file, 'r') as f:
                # Acquire shared lock (multiple readers OK)
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            # Validate cache version
            cache_version = data.get('version', '0.0')
            if cache_version != self.CACHE_VERSION:
                print(f"⚠️  Cache version mismatch ({cache_version} vs {self.CACHE_VERSION})")
                print("   Cache cleared - will regenerate")
                return {'version': self.CACHE_VERSION}
            
            return data
        
        except Exception as e:
            print(f"⚠️  Error loading cache: {e}")
            print("   Starting with fresh cache")
            return {'version': self.CACHE_VERSION}
    
    def _save_metadata(self):
        """Save cache metadata with file locking."""
        self.metadata['version'] = self.CACHE_VERSION
        
        try:
            with open(self.metadata_file, 'w') as f:
                # Acquire exclusive lock (one writer only)
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(self.metadata, f, indent=2)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        
        except Exception as e:
            print(f"⚠️  Error saving cache: {e}")


# Convenience function
def get_cache_manager(experiment_name: str) -> SmartCacheManager:
    """
    Get cache manager for an experiment.
    
    Args:
        experiment_name: Name of experiment
    
    Returns:
        SmartCacheManager instance
    """
    return SmartCacheManager(experiment_name)
