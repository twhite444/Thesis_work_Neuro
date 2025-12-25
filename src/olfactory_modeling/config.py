"""Centralized configuration management for neuro_foundation.

Provides typed configuration classes using dataclasses with sensible defaults,
environment variable overrides, and validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    
    # Directories
    raw_data_dir: Path = Path("data/01_raw")
    processed_data_dir: Path = Path("data/02_processed")
    output_dir: Path = Path("data/03_output")
    
    # Activity map processing
    activity_map_coverage_threshold: float = 0.5
    activity_map_shape: tuple[int, int] = (79, 43)
    
    # Feature selection
    min_variance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    
    # Sampling
    random_seed: int = 42
    max_scatter_points: int = 10000
    max_report_points: int = 5000
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure paths are Path objects
        self.raw_data_dir = Path(self.raw_data_dir)
        self.processed_data_dir = Path(self.processed_data_dir)
        self.output_dir = Path(self.output_dir)
        
        # Validate thresholds
        if not 0 <= self.activity_map_coverage_threshold <= 1:
            raise ValueError("coverage_threshold must be between 0 and 1")
        if not 0 <= self.min_variance_threshold:
            raise ValueError("min_variance_threshold must be non-negative")
        if not 0 <= self.correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be between 0 and 1")


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Model architecture
    hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    dropout_rate: float = 0.3
    activation: str = "relu"
    
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    early_stopping_patience: Optional[int] = 10
    
    # Optimization
    optimizer: str = "adam"
    weight_decay: float = 0.0
    lr_scheduler: Optional[str] = None
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    
    # Cross-validation
    n_splits: int = 5
    shuffle_splits: bool = True
    
    # Device
    device: str = "cuda:0"  # or "cpu", "cuda:1", etc.
    
    # Checkpointing
    save_best_only: bool = True
    checkpoint_dir: Path = Path("checkpoints")
    
    def __post_init__(self):
        """Validate configuration."""
        self.checkpoint_dir = Path(self.checkpoint_dir)
        
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be positive")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout_rate must be in [0, 1)")


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    
    log_level: str = "INFO"
    log_dir: Path = Path("logs")
    console_output: bool = True
    file_output: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        self.log_dir = Path(self.log_dir)
        
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")


@dataclass
class Config:
    """Master configuration combining all sub-configurations."""
    
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_env(cls) -> Config:
        """Create configuration from environment variables.
        
        Environment variables:
            NEURO_DATA_DIR: Override raw_data_dir
            NEURO_LOG_LEVEL: Override log_level
            NEURO_DEVICE: Override device (cpu/cuda)
            NEURO_BATCH_SIZE: Override batch_size
            NEURO_LEARNING_RATE: Override learning_rate
        
        Returns:
            Configured Config instance
        """
        config = cls()
        
        # Data config overrides
        if data_dir := os.getenv("NEURO_DATA_DIR"):
            config.data.raw_data_dir = Path(data_dir)
        
        # Training config overrides
        if device := os.getenv("NEURO_DEVICE"):
            config.training.device = device
        if batch_size := os.getenv("NEURO_BATCH_SIZE"):
            config.training.batch_size = int(batch_size)
        if lr := os.getenv("NEURO_LEARNING_RATE"):
            config.training.learning_rate = float(lr)
        
        # Logging config overrides
        if log_level := os.getenv("NEURO_LOG_LEVEL"):
            config.logging.log_level = log_level.upper()
        
        return config


# Default global config instance
default_config = Config()
