"""
CID Alignment Utilities
"""
import pandas as pd
from typing import Tuple, List


def align_by_cid(features_df: pd.DataFrame, target_df: pd.DataFrame, cid_col: str = "CID", strict: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    f_cids = set(features_df[cid_col])
    t_cids = set(target_df[cid_col])
    common = sorted(list(f_cids & t_cids))
    if strict:
        missing_f = sorted(list(t_cids - f_cids))
        missing_t = sorted(list(f_cids - t_cids))
        if missing_f or missing_t:
            raise ValueError(f"CID mismatch: missing in features={missing_t[:5]}..., missing in targets={missing_f[:5]}... (counts: features_only={len(missing_t)}, targets_only={len(missing_f)})")
    f_aligned = features_df.set_index(cid_col).loc[common].reset_index()
    t_aligned = target_df.set_index(cid_col).loc[common].reset_index()
    return f_aligned, t_aligned, common
