"""
Brain Activity Data Processing Module

This module handles the processing of fMRI brain activation maps from the
Johnson & Leon (2007) dataset. The key steps are:

1. Load individual brain activity maps (405 stimulus presentations)
2. Extract CID from stimulus identifier (format: "{CID}_{repetition}")
3. Average multiple presentations of the same molecule by CID
4. Apply PCA to averaged brain maps to extract principal components
5. Extract first 5 PC scores as model targets

Critical Understanding:
- Input: 405 2D brain activity maps (some molecules have multiple presentations)
- Processing: Average by CID → 287 unique molecules × ~1,000 voxels
- PCA: Applied to BRAIN DATA (not molecular features!)
- Output: 287 × 5 PCA component scores (targets for neural network)
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class BrainActivityProcessor:
    """
    Process fMRI brain activation maps and extract PCA-based targets.
    
    This class handles the critical step of converting raw 2D brain activity maps
    into PCA-reduced representations that serve as targets for predicting
    neural responses from molecular features.
    
    Attributes:
        n_components (int): Number of PCA components to compute (default: 50)
        n_targets (int): Number of PC scores to use as targets (default: 5)
        pca (PCA): Fitted PCA model
        scaler (StandardScaler): Scaler for brain activity data
        averaged_maps (pd.DataFrame): Averaged brain maps by CID
        pca_scores (np.ndarray): PCA-transformed brain data
    """
    
    def __init__(
        self,
        n_components: int = 50,
        n_targets: int = 5,
        standardize: bool = True
    ):
        """
        Initialize BrainActivityProcessor.
        
        Args:
            n_components: Number of PCA components to compute
            n_targets: Number of PC scores to use as model targets
            standardize: Whether to standardize brain data before PCA
        """
        self.n_components = n_components
        self.n_targets = n_targets
        self.standardize = standardize
        
        self.pca: Optional[PCA] = None
        self.scaler: Optional[StandardScaler] = None
        self.averaged_maps: Optional[pd.DataFrame] = None
        self.pca_scores: Optional[np.ndarray] = None
        self.brain_matrix: Optional[np.ndarray] = None
        
        logger.info(
            f"Initialized BrainActivityProcessor: "
            f"n_components={n_components}, n_targets={n_targets}, "
            f"standardize={standardize}"
        )
    
    def load_and_average_maps(
        self,
        behavior_csv: str,
        csvs_dir: str,
        molecules_df: pd.DataFrame,
        cid_column: str = 'CID'
    ) -> pd.DataFrame:
        """
        Load brain activity maps and average by CID.
        
        This handles the critical data alignment step:
        1. Load stimulus mapping (405 presentations)
        2. Extract CID from stimulus identifier (e.g., "7991_3" → "7991")
        3. Load each 2D brain map and flatten to 1D
        4. Group by CID and average across repetitions/concentrations
        5. Align with molecular features by CID
        
        Args:
            behavior_csv: Path to behavior_data.csv (stimulus → map lookup)
            csvs_dir: Directory containing individual brain map CSV files
            molecules_df: DataFrame with CID column for alignment
            cid_column: Name of CID column in molecules_df
        
        Returns:
            DataFrame with columns:
            - CID: Molecule identifier
            - brain_map: Flattened, averaged activation pattern (1D array)
            - n_reps: Number of presentations averaged
        
        Raises:
            FileNotFoundError: If behavior_csv or brain map files not found
            ValueError: If CID alignment fails
        """
        logger.info(f"Loading brain activity maps from {behavior_csv}")
        
        # Load stimulus mapping
        if not os.path.exists(behavior_csv):
            raise FileNotFoundError(f"Behavior CSV not found: {behavior_csv}")
        
        behavior = pd.read_csv(behavior_csv, index_col=0)
        logger.info(f"Loaded {len(behavior)} stimulus presentations")
        
        # Stimulus is the index - extract CID from it (format: "{CID}_{rep}")
        # Handle both string and numeric CIDs
        behavior['CID'] = behavior.index.astype(str).str.split('_').str[0]
        behavior['Stimulus'] = behavior.index
        
        unique_cids = behavior['CID'].nunique()
        logger.info(f"Found {unique_cids} unique CIDs in behavior data")
        
        # Load all brain maps grouped by CID
        brain_maps_by_cid: Dict[str, List[np.ndarray]] = {}
        missing_files = []
        
        for idx, row in behavior.iterrows():
            cid = row['CID']
            stimulus = row['Stimulus']
            
            # Construct full path to brain map CSV
            # behavior_data.csv contains paths like "csvs/1031_0.csv"
            map_filename = f"{stimulus}.csv"
            map_path = os.path.join(csvs_dir, map_filename)
            
            if not os.path.exists(map_path):
                missing_files.append(map_path)
                continue
            
            try:
                # Load 2D brain map (no header, pure numeric data)
                brain_map_2d = pd.read_csv(map_path, header=None).values
                
                # Flatten to 1D vector
                brain_map_flat = brain_map_2d.flatten()
                
                # Store by CID (accumulate if multiple)
                if cid not in brain_maps_by_cid:
                    brain_maps_by_cid[cid] = []
                brain_maps_by_cid[cid].append(brain_map_flat)
                
            except Exception as e:
                logger.warning(f"Error loading {map_path}: {e}")
                missing_files.append(map_path)
        
        if missing_files:
            logger.warning(
                f"Could not load {len(missing_files)} brain map files. "
                f"First few: {missing_files[:5]}"
            )
        
        logger.info(f"Successfully loaded brain maps for {len(brain_maps_by_cid)} CIDs")
        
        # Average maps for each CID
        averaged_maps = {}
        n_reps = {}
        
        for cid, maps in brain_maps_by_cid.items():
            # Stack and average across first axis (repetitions)
            stacked = np.stack(maps, axis=0)  # (n_reps, n_voxels)
            averaged = stacked.mean(axis=0)    # (n_voxels,)
            
            averaged_maps[cid] = averaged
            n_reps[cid] = len(maps)
            
            if len(maps) > 1:
                logger.debug(f"CID {cid}: Averaged {len(maps)} repetitions")
        
        # Check for molecules with many repetitions (likely concentration series)
        high_rep_cids = {cid: n for cid, n in n_reps.items() if n > 5}
        if high_rep_cids:
            logger.info(
                f"CIDs with >5 repetitions (likely concentration series): "
                f"{dict(sorted(high_rep_cids.items(), key=lambda x: x[1], reverse=True)[:10])}"
            )
        
        # Align with molecules DataFrame
        molecules_cids = molecules_df[cid_column].astype(str).values
        
        # Create result DataFrame
        result_data = []
        missing_cids = []
        
        for cid in molecules_cids:
            if str(cid) in averaged_maps:
                result_data.append({
                    'CID': cid,
                    'brain_map': averaged_maps[str(cid)],
                    'n_reps': n_reps[str(cid)]
                })
            else:
                missing_cids.append(cid)
        
        if missing_cids:
            logger.warning(
                f"Missing brain maps for {len(missing_cids)} molecules: "
                f"{missing_cids[:10]}..."
            )
        
        result_df = pd.DataFrame(result_data)
        
        logger.info(
            f"Created averaged brain map dataset: "
            f"{len(result_df)} molecules × {len(averaged_maps[list(averaged_maps.keys())[0]])} voxels"
        )
        logger.info(
            f"Average repetitions per molecule: {np.mean(result_df['n_reps']):.2f} "
            f"(range: {result_df['n_reps'].min()}-{result_df['n_reps'].max()})"
        )
        
        self.averaged_maps = result_df
        
        # Convert to matrix for PCA
        self.brain_matrix = np.vstack(result_df['brain_map'].values)
        
        # Check for NaN values and log statistics
        n_nans = np.isnan(self.brain_matrix).sum()
        if n_nans > 0:
            logger.warning(
                f"Brain matrix contains {n_nans} NaN values "
                f"({n_nans / self.brain_matrix.size * 100:.2f}% of total)"
            )
            logger.info("NaNs will be replaced with 0 (background regions)")
            self.brain_matrix = np.nan_to_num(self.brain_matrix, nan=0.0)
        
        return result_df
    
    def apply_pca(
        self,
        brain_matrix: Optional[np.ndarray] = None,
        save_model: bool = True
    ) -> np.ndarray:
        """
        Apply PCA to brain activation maps.
        
        This is the CRITICAL step that differs from typical molecular pipelines.
        PCA is applied to BRAIN DATA (not molecular features!) to extract
        principal spatial patterns of glomerular activation.
        
        Args:
            brain_matrix: Brain activity matrix (n_molecules × n_voxels).
                         If None, uses self.brain_matrix from load_and_average_maps.
            save_model: Whether to save fitted PCA model to self.pca
        
        Returns:
            PCA scores (n_molecules × n_components)
        
        Raises:
            ValueError: If brain_matrix is None and not previously loaded
        """
        if brain_matrix is None:
            if self.brain_matrix is None:
                raise ValueError(
                    "No brain matrix available. Call load_and_average_maps first "
                    "or provide brain_matrix argument."
                )
            brain_matrix = self.brain_matrix
        
        logger.info(
            f"Applying PCA to brain activity data: "
            f"shape={brain_matrix.shape}, n_components={self.n_components}"
        )
        
        # Optionally standardize before PCA
        if self.standardize:
            logger.info("Standardizing brain activity data before PCA")
            self.scaler = StandardScaler()
            brain_matrix_scaled = self.scaler.fit_transform(brain_matrix)
        else:
            brain_matrix_scaled = brain_matrix
        
        # Fit PCA
        self.pca = PCA(n_components=self.n_components)
        pca_scores = self.pca.fit_transform(brain_matrix_scaled)
        
        # Log explained variance
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        
        logger.info("PCA completed:")
        for i in range(min(5, self.n_components)):
            logger.info(
                f"  PC{i+1}: {self.pca.explained_variance_ratio_[i]*100:.2f}% "
                f"(cumulative: {cumulative_variance[i]*100:.2f}%)"
            )
        
        logger.info(
            f"First {self.n_targets} components explain "
            f"{cumulative_variance[self.n_targets-1]*100:.2f}% of variance"
        )
        
        self.pca_scores = pca_scores
        
        return pca_scores
    
    def extract_targets(
        self,
        pca_scores: Optional[np.ndarray] = None,
        n_targets: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract first N PCA component scores as model targets.
        
        These PCA scores represent the "neural code" - the spatial patterns
        of brain activation that we're trying to predict from molecular structure.
        
        Args:
            pca_scores: PCA scores array (n_molecules × n_components).
                       If None, uses self.pca_scores from apply_pca.
            n_targets: Number of PC scores to extract. If None, uses self.n_targets.
        
        Returns:
            Target array (n_molecules × n_targets)
        
        Raises:
            ValueError: If pca_scores is None and not previously computed
        """
        if pca_scores is None:
            if self.pca_scores is None:
                raise ValueError(
                    "No PCA scores available. Call apply_pca first "
                    "or provide pca_scores argument."
                )
            pca_scores = self.pca_scores
        
        if n_targets is None:
            n_targets = self.n_targets
        
        if n_targets > pca_scores.shape[1]:
            raise ValueError(
                f"n_targets ({n_targets}) exceeds available components "
                f"({pca_scores.shape[1]})"
            )
        
        targets = pca_scores[:, :n_targets]
        
        logger.info(
            f"Extracted {n_targets} PCA component scores as targets: "
            f"shape={targets.shape}"
        )
        
        return targets
    
    def visualize_pca(
        self,
        output_dir: str,
        n_components_to_plot: int = 5
    ) -> None:
        """
        Create visualizations of PCA results.
        
        Generates:
        1. Scree plot (variance explained by each component)
        2. Cumulative variance plot
        3. Component loadings heatmap (spatial patterns)
        
        Args:
            output_dir: Directory to save plots
            n_components_to_plot: Number of components to visualize
        
        Raises:
            ValueError: If PCA has not been fitted
        """
        if self.pca is None:
            raise ValueError("PCA has not been fitted. Call apply_pca first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Scree plot
        fig, ax = plt.subplots(figsize=(10, 6))
        components = range(1, len(self.pca.explained_variance_ratio_) + 1)
        variance_pct = self.pca.explained_variance_ratio_ * 100
        
        ax.bar(components, variance_pct, alpha=0.7, color='steelblue')
        ax.plot(components, variance_pct, 'o-', color='darkblue', linewidth=2)
        ax.axvline(x=self.n_targets, color='red', linestyle='--', 
                   label=f'First {self.n_targets} components (used as targets)')
        ax.set_xlabel('Principal Component', fontsize=12)
        ax.set_ylabel('Variance Explained (%)', fontsize=12)
        ax.set_title('PCA Scree Plot: Brain Activity Patterns', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        scree_path = os.path.join(output_dir, 'pca_scree.png')
        plt.tight_layout()
        plt.savefig(scree_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved scree plot: {scree_path}")
        
        # 2. Cumulative variance plot
        fig, ax = plt.subplots(figsize=(10, 6))
        cumulative = np.cumsum(variance_pct)
        
        ax.plot(components, cumulative, 'o-', linewidth=2, markersize=6)
        ax.axhline(y=cumulative[self.n_targets-1], color='red', linestyle='--',
                   label=f'First {self.n_targets} components: {cumulative[self.n_targets-1]:.1f}%')
        ax.axvline(x=self.n_targets, color='red', linestyle='--')
        ax.fill_between(components[:self.n_targets], 0, cumulative[:self.n_targets], 
                        alpha=0.2, color='red')
        ax.set_xlabel('Number of Components', fontsize=12)
        ax.set_ylabel('Cumulative Variance Explained (%)', fontsize=12)
        ax.set_title('Cumulative Variance Explained by PCA Components', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        cumulative_path = os.path.join(output_dir, 'pca_cumulative.png')
        plt.tight_layout()
        plt.savefig(cumulative_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved cumulative variance plot: {cumulative_path}")
        
        # 3. Top components loadings (spatial patterns)
        fig, axes = plt.subplots(1, min(3, n_components_to_plot), figsize=(15, 5))
        if n_components_to_plot == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes[:min(3, n_components_to_plot)]):
            loadings = self.pca.components_[i]
            
            # Try to reshape to approximate spatial map (assume square-ish)
            n_voxels = len(loadings)
            side_length = int(np.sqrt(n_voxels))
            
            if side_length ** 2 == n_voxels:
                # Perfect square
                loadings_2d = loadings.reshape(side_length, side_length)
            else:
                # Approximate rectangle
                n_cols = side_length
                n_rows = int(np.ceil(n_voxels / n_cols))
                padded = np.pad(loadings, (0, n_rows * n_cols - n_voxels), 
                               mode='constant', constant_values=0)
                loadings_2d = padded.reshape(n_rows, n_cols)
            
            im = ax.imshow(loadings_2d, cmap='RdBu_r', aspect='auto')
            ax.set_title(
                f"PC{i+1} ({self.pca.explained_variance_ratio_[i]*100:.1f}%)",
                fontsize=12, fontweight='bold'
            )
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle('Spatial Patterns of Top Principal Components', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        components_path = os.path.join(output_dir, 'top_3_components.png')
        plt.tight_layout()
        plt.savefig(components_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved component patterns: {components_path}")
        
        # 4. Create summary statistics
        summary = {
            'n_molecules': self.brain_matrix.shape[0],
            'n_voxels': self.brain_matrix.shape[1],
            'n_components_computed': self.n_components,
            'n_targets_used': self.n_targets,
            'variance_explained_by_targets': cumulative[self.n_targets-1],
            'pc1_variance': variance_pct[0],
            'pc2_variance': variance_pct[1] if len(variance_pct) > 1 else None,
            'pc3_variance': variance_pct[2] if len(variance_pct) > 2 else None,
            'pc4_variance': variance_pct[3] if len(variance_pct) > 3 else None,
            'pc5_variance': variance_pct[4] if len(variance_pct) > 4 else None,
        }
        
        summary_path = os.path.join(output_dir, 'pca_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("=== Brain Activity PCA Summary ===\n\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"Saved PCA summary: {summary_path}")
