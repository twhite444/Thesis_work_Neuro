"""Utility modules for the olfactory_modeling package."""

from .profiling import Timer, EpochTimer, profile_dataloader, compare_device_performance
from .metrics import (
    to_numpy,
    flatten_arrays,
    compute_correlation,
    compute_mse,
    compute_mae,
    compute_statistics,
    sample_for_plotting,
    DEFAULT_MAX_SCATTER_POINTS,
    DEFAULT_MAX_REPORT_POINTS,
    DEFAULT_RANDOM_SEED,
)
from .logging_config import (
    setup_logging,
    get_logger,
    quick_setup,
    log_function_call,
)

__all__ = [
    # Profiling
    'Timer',
    'EpochTimer', 
    'profile_dataloader',
    'compare_device_performance',
    # Metrics
    "to_numpy",
    "flatten_arrays",
    "compute_correlation",
    "compute_mse",
    "compute_mae",
    "compute_statistics",
    "sample_for_plotting",
    "DEFAULT_MAX_SCATTER_POINTS",
    "DEFAULT_MAX_REPORT_POINTS",
    "DEFAULT_RANDOM_SEED",
    # Logging
    "setup_logging",
    "get_logger",
    "quick_setup",
    "log_function_call",
]
