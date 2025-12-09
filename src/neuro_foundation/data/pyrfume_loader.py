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
        _ = pyrfume.load_manifest('arshamian_2022')
        _ = pyrfume.load_manifest('leon')
        molecules = pyrfume.load_data('leon/molecules.csv')
        # Persist raw for provenance
        molecules.to_csv(os.path.join(self.output_dir, 'molecules_raw.csv'), index=False)
        return molecules

    def load_images(self) -> pd.DataFrame | None:
        try:
            image_df = pyrfume.load_data('leon/csvs/1031_0.csv')
            image_df.to_csv(os.path.join(self.output_dir, 'image_data.csv'), index=False)
            return image_df
        except Exception:
            return None
