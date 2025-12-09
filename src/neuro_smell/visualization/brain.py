"""
Brain Visualization Utilities
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def _save_meta(path: Path, meta: dict):
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def plot_brain_vector(vector: np.ndarray, shape: Tuple[int, ...], title: str, out_path: str):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if np.prod(shape) != vector.size:
        raise ValueError(f"Vector size {vector.size} does not match shape {shape}")
    arr = vector.reshape(shape)
    # Show middle slice if 3D
    if len(shape) == 3:
        mid = shape[2] // 2
        img = arr[:, :, mid]
    else:
        img = arr
    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    _save_meta(out, {"shape": shape, "title": title})


def plot_brain_grid(vectors: np.ndarray, shape: Tuple[int, ...], n_cols: int, out_path: str):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = vectors.shape[0]
    n_rows = int(np.ceil(n / n_cols))
    plt.figure(figsize=(n_cols * 3, n_rows * 3))
    for i in range(n):
        arr = vectors[i].reshape(shape)
        img = arr[:, :, shape[2] // 2] if len(shape) == 3 else arr
        ax = plt.subplot(n_rows, n_cols, i + 1)
        ax.imshow(img, cmap="viridis")
        ax.axis("off")
        ax.set_title(f"idx {i}", fontsize=8)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    _save_meta(out, {"shape": shape, "n_items": int(n)})


def plot_mask(mask: np.ndarray, shape: Tuple[int, ...], title: str, out_path: str):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if np.prod(shape) != mask.size:
        raise ValueError("Mask size does not match shape")
    arr = mask.reshape(shape)
    img = arr[:, :, shape[2] // 2] if len(shape) == 3 else arr
    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    kept = int(mask.sum())
    _save_meta(out, {"shape": shape, "kept_voxels": kept, "total_voxels": int(mask.size), "keep_pct": kept / mask.size})


def overlay_mask(vector: np.ndarray, mask: np.ndarray, shape: Tuple[int, ...], out_path: str):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = vector.reshape(shape)
    m = mask.reshape(shape)
    img = arr[:, :, shape[2] // 2] if len(shape) == 3 else arr
    mimg = m[:, :, shape[2] // 2] if len(shape) == 3 else m
    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap="viridis")
    plt.imshow(np.ma.masked_where(~mimg, mimg), cmap="Reds", alpha=0.3)
    plt.title("Overlay mask on brain slice")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    _save_meta(out, {"shape": shape})
