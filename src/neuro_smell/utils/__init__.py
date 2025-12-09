"""
Utility functions and helpers.

Available utilities:
- CacheManager: Legacy caching for pipeline stages
- SmartCacheManager: New stage-specific smart caching
- metrics: Custom evaluation metrics (Pearson correlation, R², MAE, RMSE)
- data_utils: Data validation, summarization, and split helpers
"""

from .cache_manager import CacheManager
from .smart_cache import SmartCacheManager, get_cache_manager
from .metrics import (
    pearson_correlation,
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
    compute_all_metrics,
    numpy_pearson_correlation,
)
from .data_utils import (
    validate_csv_file,
    summarize_dataframe,
    print_data_summary,
    check_data_distribution,
    resolve_data_path,
    split_by_stratification,
    save_data_splits,
)

__all__ = [
    'CacheManager',
    'SmartCacheManager',
    'get_cache_manager',
    'pearson_correlation',
    'r2_score',
    'mean_absolute_error',
    'root_mean_squared_error',
    'compute_all_metrics',
    'numpy_pearson_correlation',
    'validate_csv_file',
    'summarize_dataframe',
    'print_data_summary',
    'check_data_distribution',
    'resolve_data_path',
    'split_by_stratification',
    'save_data_splits',
]
