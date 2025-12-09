from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import pandas as pd

class DatasetLoader(ABC):
    """Abstract dataset loader to support multiple sources.

    Contract:
    - load_molecules() -> pd.DataFrame with at least column 'IsomericSMILES' and optionally 'CID'.
    - load_images() -> Optional[pd.DataFrame] for any image/brain-related data (optional).
    """

    @abstractmethod
    def load_molecules(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def load_images(self) -> Optional[pd.DataFrame]:
        return None
