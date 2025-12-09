"""
Brain Targets (PCA) Stage
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import Dict, Optional, Tuple
from ..preprocessing.brain_preprocessor import derive_mask, apply_mask
from ..visualization.pca import plot_scree, plot_cumulative_variance, plot_pca_component, plot_scores_scatter


def fit_pca_and_save(brain_matrix: np.ndarray, cids: np.ndarray, shape: Optional[Tuple[int, int]] = None, n_components: int = 5, out_dir: str = "test_output/brain_pca") -> Dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # derive mask
    mask = derive_mask(brain_matrix, nan_threshold=0.05)
    masked = apply_mask(brain_matrix, mask)
    # fit PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(np.nan_to_num(masked))
    explained = pca.explained_variance_
    ratio = pca.explained_variance_ratio_
    # save model and artifacts
    np.savez(out / "brain_pca_model.npz", components=pca.components_, mean=pca.mean_, explained_variance=explained, explained_ratio=ratio, mask=mask)
    # scores CSV
    cols = [f"PC{i+1}" for i in range(scores.shape[1])]
    df = pd.DataFrame(scores, columns=cols)
    df.insert(0, "CID", cids)
    df.to_csv(out / "brain_pca_scores.csv", index=False)
    # visuals
    plot_scree(explained, str(out / "pca_scree.png"))
    plot_cumulative_variance(ratio, str(out / "pca_cumulative.png"))
    # component maps (first min(10, n_components))
    comp = pca.components_
    if shape is not None:
        k = min(10, comp.shape[0])
        for i in range(k):
            # expand back to full space using mask
            full = np.zeros(mask.shape, dtype=np.float32)
            full[mask] = comp[i]
            plot_pca_component(full, shape, i+1, str(out / f"pca_component_pc{i+1}.png"))
    # score scatter PC1 vs PC2 (if available)
    if scores.shape[1] >= 2:
        plot_scores_scatter(df, "PC1", "PC2", str(out / "scores_pc1_pc2.png"))
    return {"scores_path": str(out / "brain_pca_scores.csv"), "model_path": str(out / "brain_pca_model.npz")}
