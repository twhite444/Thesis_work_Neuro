"""Pipeline metadata collection utilities."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
from ..utils.logging_config import get_logger
logger = get_logger(__name__)

def load_json_safe(filepath: str) -> Dict[str, Any]:
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Metadata file not found: {filepath}")
            return {}
    except Exception as e:
        logger.warning(f"Could not load metadata from {filepath}: {e}")
        return {}

def load_npz_metadata(filepath: str, metadata_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    metadata = {}
    try:
        if os.path.exists(filepath):
            data = np.load(filepath)
            if metadata_keys is None:
                for key in data.keys():
                    arr = data[key]
                    if arr.size < 100:
                        if arr.size == 1:
                            metadata[key] = arr.item()
                        else:
                            metadata[key] = arr.tolist()
            else:
                for key in metadata_keys:
                    if key in data:
                        arr = data[key]
                        if arr.size == 1:
                            metadata[key] = arr.item()
                        else:
                            metadata[key] = arr.tolist()
        else:
            logger.warning(f"NPZ file not found: {filepath}")
    except Exception as e:
        logger.warning(f"Could not load metadata from {filepath}: {e}")
    return metadata

def collect_pipeline_metadata(processed_dir: str = "data/02_processed", use_pca: bool = False) -> Dict[str, Any]:
    processed_path = Path(processed_dir)
    metadata = {}
    activity_maps_meta = {}
    global_mask_meta = load_json_safe(str(processed_path / "global_mask_metadata.json"))
    if global_mask_meta:
        activity_maps_meta['global_mask'] = global_mask_meta
    try:
        import pandas as pd
        maps_meta_path = processed_path / "processed_maps_metadata.csv"
        if maps_meta_path.exists():
            maps_df = pd.read_csv(maps_meta_path)
            activity_maps_meta['n_molecules'] = len(maps_df)
            activity_maps_meta['selection_strategy'] = maps_df['selection_strategy'].iloc[0] if len(maps_df) > 0 else None
            activity_maps_meta['coverage_threshold'] = maps_df['coverage_threshold'].iloc[0] if len(maps_df) > 0 else None
            activity_maps_meta['value_policy'] = maps_df['value_policy'].iloc[0] if len(maps_df) > 0 else None
            activity_maps_meta['map_shape'] = [
                int(maps_df['map_shape_h'].iloc[0]),
                int(maps_df['map_shape_w'].iloc[0])
            ] if len(maps_df) > 0 else None
    except Exception as e:
        logger.warning(f"Could not load processed maps metadata CSV: {e}")
    map_stats_path = processed_path / "map_statistics.json"
    if map_stats_path.exists():
        try:
            with open(map_stats_path, 'r') as f:
                map_stats = json.load(f)
                if map_stats:
                    activity_maps_meta['n_maps_with_stats'] = len(map_stats)
                    first_cid = list(map_stats.keys())[0]
                    activity_maps_meta['stats_keys'] = list(map_stats[first_cid].keys())
        except Exception as e:
            logger.warning(f"Could not load map statistics: {e}")
    metadata['activity_maps'] = activity_maps_meta
    if use_pca:
        pca_meta = {}
        pca_npz_path = processed_path / "pca_transformed_maps.npz"
        pca_npz_meta = load_npz_metadata(
            str(pca_npz_path),
            metadata_keys=['n_components', 'n_samples', 'original_shape', 
                          'explained_variance_ratio', 'cumulative_variance', 
                          'total_variance_explained']
        )
        pca_meta.update(pca_npz_meta)
        metadata['pca'] = pca_meta
    preprocess_meta = load_json_safe(str(processed_path / "preprocess_metadata.json"))
    if preprocess_meta:
        metadata['preprocessing'] = preprocess_meta
    scaler_stats_path = processed_path / "scaler_stats.json"
    if scaler_stats_path.exists():
        try:
            with open(scaler_stats_path, 'r') as f:
                scaler_stats = json.load(f)
                metadata['scaler'] = {
                    'n_features': len(scaler_stats.get('means', [])),
                    'method': 'StandardScaler',
                }
        except Exception as e:
            logger.warning(f"Could not load scaler stats: {e}")
    return metadata
