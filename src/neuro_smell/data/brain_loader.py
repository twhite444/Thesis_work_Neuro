"""
Brain Loader - Load brain maps or averaged vectors per CID.

Contract:
- Input: path to npz containing `brain_matrix` and `cids` (averaged voxels per CID) OR raw map paths.
- Output: tuple (brain_matrix: np.ndarray [n_samples, n_voxels], cids: list[int])
"""
from pathlib import Path
import numpy as np
from typing import Tuple, List


def load_brain_averaged_npz(path: str) -> Tuple[np.ndarray, List[int]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Brain npz not found: {path}")
    data = np.load(p, allow_pickle=True)
    brain_matrix = data.get('brain_matrix')
    cids = data.get('cids')
    if brain_matrix is None or cids is None:
        raise ValueError("npz must contain 'brain_matrix' and 'cids'")
    return brain_matrix, list(cids)
