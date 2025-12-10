from __future__ import annotations
import os
import pandas as pd
import pyrfume
from .interfaces import DatasetLoader

class PyrfumeLoader(DatasetLoader):
    """Pyrfume-backed loader replicating legacy data sources (read-only).

    Loads:
    - arshamian_2022 manifest (not used directly here but validated)
    - leon manifest
    - leon/molecules.csv
    - leon/csvs/1031_0.csv (image/brain data) if available
    """

    def __init__(self, output_dir: str = "data/01_raw"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_molecules(self) -> pd.DataFrame:
        # Ensure manifests can be referenced (read-only check)
        _ = pyrfume.load_manifest('leon')
        molecules = pyrfume.load_data('leon/molecules.csv')

        molecules.reset_index(inplace=True)
        molecules.rename(columns={'index': 'CID'}, inplace=True)

        # Check for duplicate CIDs
        duplicate_cids = molecules[molecules.duplicated(subset='CID', keep=False)]
        print(f"Duplicate CIDs in molecules before removal:\n{duplicate_cids}")

        # Remove duplicate CIDs by keeping the first occurrence
        molecules = molecules.drop_duplicates(subset='CID', keep='first')

        # Debug: Check for duplicates after removal
        duplicate_cids_after = molecules[molecules.duplicated(subset='CID', keep=False)]
        print(f"Duplicate CIDs in molecules after removal:\n{duplicate_cids_after}")


        # Persist raw for provenance
        molecules.to_csv(os.path.join(self.output_dir, 'molecules_raw.csv'), index=False)
        return molecules
    
    def load_behavior(self) -> pd.DataFrame:
        behavior = pyrfume.load_data('leon/behavior_1.csv')
        behavior.reset_index(inplace=True)
        # Persist raw for provenance
        # Normalize stimulus by removing any suffix after '_' (e.g. "153_2" -> "153")
        behavior['Stimulus'] = behavior['Stimulus'].astype(str).str.split('_', n=1).str[0]
        # Convert Stimulus to numeric (coerce invalid values to NaN) and filter valid non-negative entries
        behavior['Stimulus'] = pd.to_numeric(behavior['Stimulus'], errors='coerce')
        behavior = behavior[behavior['Stimulus'].notna() & (behavior['Stimulus'] >= 0)]
        behavior.to_csv(os.path.join(self.output_dir, 'behavior_data.csv'), index=False)
        return behavior

    def load_images(self) -> pd.DataFrame | None:
        try:
            image_df = pyrfume.load_data('leon/csvs/1031_0.csv')
            image_df.to_csv(os.path.join(self.output_dir, 'image_data.csv'), index=False)
            return image_df
        except Exception:
            return None
