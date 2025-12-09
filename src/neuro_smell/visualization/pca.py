"""
PCA Visualization Utilities
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _save_meta(path: Path, meta: dict):
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def plot_scree(explained_variance: np.ndarray, out_path: str):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(explained_variance) + 1), explained_variance, marker='o')
    plt.xlabel("Component")
    plt.ylabel("Explained Variance")
    plt.title("PCA Scree Plot")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    _save_meta(out, {"n_components": int(len(explained_variance))})


def plot_cumulative_variance(explained_variance_ratio: np.ndarray, out_path: str):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cum = np.cumsum(explained_variance_ratio)
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(cum) + 1), cum, marker='o')
    plt.xlabel("Component")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("PCA Cumulative Variance")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    _save_meta(out, {"n_components": int(len(explained_variance_ratio)), "final_cum": float(cum[-1])})


def plot_pca_component(component_vector: np.ndarray, shape: tuple, pc_index: int, out_path: str):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if np.prod(shape) != component_vector.size:
        raise ValueError("Component size does not match shape")
    arr = component_vector.reshape(shape)
    img = arr[:, :, shape[2] // 2] if len(shape) == 3 else arr
    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap="coolwarm")
    plt.colorbar()
    plt.title(f"PCA Component PC{pc_index}")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    _save_meta(out, {"pc_index": pc_index, "shape": shape})


def plot_scores_scatter(scores_df: pd.DataFrame, pc_x: str, pc_y: str, out_path: str):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.scatter(scores_df[pc_x], scores_df[pc_y], s=20, alpha=0.7)
    plt.xlabel(pc_x)
    plt.ylabel(pc_y)
    plt.title(f"Scores Scatter: {pc_x} vs {pc_y}")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    _save_meta(out, {"pc_x": pc_x, "pc_y": pc_y, "n_points": int(len(scores_df))})
