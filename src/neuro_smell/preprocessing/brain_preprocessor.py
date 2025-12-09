"""
Brain Preprocessor - Derive and apply NaN mask for voxel-wise reliability.
"""
import numpy as np
from typing import Tuple


def derive_mask(brain_matrix: np.ndarray, nan_threshold: float = 0.05) -> np.ndarray:
    """Return boolean mask of voxels to keep: fraction of NaNs <= threshold."""
    nan_frac = np.mean(np.isnan(brain_matrix), axis=0)
    return nan_frac <= nan_threshold


def apply_mask(brain_matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return brain_matrix[:, mask]
