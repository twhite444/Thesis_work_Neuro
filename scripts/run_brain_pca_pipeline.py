#!/usr/bin/env python3
"""
Brain PCA + Canonical Dataset Pipeline
- Derives brain mask
- Fits PCA on averaged brain maps
- Saves PCA model + scores + visuals
- Assembles canonical features_and_targets.csv

Usage:
  python scripts/run_brain_pca_pipeline.py
"""
import os
import sys
import json
import pandas as pd
from pathlib import Path

# Ensure `src` is on PYTHONPATH when running as a script
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from neuro_smell.data.brain_loader import load_brain_averaged_npz  # type: ignore
from neuro_smell.preprocessing.brain_preprocessor import derive_mask, apply_mask  # type: ignore
from neuro_smell.stages.brain_targets import fit_pca_and_save  # type: ignore
from neuro_smell.datasets.assembler import merge_features_and_targets  # type: ignore

DATA_DIR = "data/02_processed"
TEST_OUT = "test_output/brain_pca"


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(TEST_OUT, exist_ok=True)


def main():
    ensure_dirs()

    features_csv = os.path.join(DATA_DIR, "selected_features.csv")
    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"Expected descriptors at {features_csv}")
    features_df = pd.read_csv(features_csv)
    if "CID" not in features_df.columns:
        raise ValueError("selected_features.csv must include 'CID' column")

    brain_npz = os.path.join(DATA_DIR, "brain_maps_averaged.npz")
    if not os.path.exists(brain_npz):
        raise FileNotFoundError(f"Expected averaged brain maps at {brain_npz}")
    brain_matrix, brain_cids = load_brain_averaged_npz(brain_npz)

    mask = derive_mask(brain_matrix, nan_threshold=0.5)
    masked_matrix = apply_mask(brain_matrix, mask)

    meta = {
        "shape": list(brain_matrix.shape),
        "masked_shape": list(masked_matrix.shape),
        "mask_voxels_kept": int(mask.sum()),
        "mask_voxels_dropped": int((~mask).sum()),
    }
    with open(os.path.join(TEST_OUT, "mask_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    pca_model_path = os.path.join(DATA_DIR, "brain_pca_model.npz")
    scores_csv_path = os.path.join(DATA_DIR, "brain_pca_scores.csv")
    fit_pca_and_save(
        brain_matrix=brain_matrix,
        cids=brain_cids,
        shape=None,
        n_components=5,
        out_dir=TEST_OUT,
    )

    out_csv = os.path.join(DATA_DIR, "features_and_targets.csv")
    merge_features_and_targets(
        features_csv=features_csv,
        scores_csv=scores_csv_path,
        out_csv=out_csv,
    )
    print(f"✅ Pipeline complete. Canonical dataset: {out_csv}")


if __name__ == "__main__":
    main()
