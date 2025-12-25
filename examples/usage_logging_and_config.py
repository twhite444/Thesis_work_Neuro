"""Example: Using the new logging and configuration infrastructure.

This demonstrates the professional infrastructure now available for
production neural network training and experiment tracking.
"""

# ============================================================================
# Example 1: Basic Logging Setup
# ============================================================================

from olfactory_modeling.utils import setup_logging, get_logger
from pathlib import Path

# Setup logging for a training script
setup_logging(
    log_level="INFO",
    log_dir=Path("logs"),
    console=True  # Also print to console
)

logger = get_logger(__name__)

logger.info("Starting experiment")
logger.debug("This won't show with INFO level")
logger.warning("Learning rate might be too high")
logger.error("GPU out of memory", exc_info=True)  # Full stack trace


# ============================================================================
# Example 2: Quick Setup for Notebooks
# ============================================================================

from olfactory_modeling.utils import quick_setup

# Quick setup for interactive work
logger = quick_setup(
    verbose=True,  # DEBUG level
    log_file=Path("experiment_2024-12-16.log")
)

logger.debug("Detailed debug info now visible")


# ============================================================================
# Example 3: Environment-Aware Configuration
# ============================================================================

from olfactory_modeling.config import Config
import os

# Set environment variables (usually in shell or cluster job script)
os.environ['NEURO_DEVICE'] = 'cuda:0'
os.environ['NEURO_BATCH_SIZE'] = '128'
os.environ['NEURO_LEARNING_RATE'] = '0.001'
os.environ['NEURO_LOG_LEVEL'] = 'INFO'

# Load config with environment overrides
config = Config.from_env()

print(f"Device: {config.training.device}")  # cuda:0
print(f"Batch size: {config.training.batch_size}")  # 128
print(f"Learning rate: {config.training.learning_rate}")  # 0.001
print(f"Log level: {config.logging.log_level}")  # INFO


# ============================================================================
# Example 4: Using Configuration in Training
# ============================================================================

from olfactory_modeling.config import default_config

# Access configuration
data_dir = default_config.data.raw_data_dir
batch_size = default_config.training.batch_size
learning_rate = default_config.training.learning_rate

logger.info(f"Loading data from {data_dir}")
logger.info(f"Training with batch_size={batch_size}, lr={learning_rate}")


# ============================================================================
# Example 5: Training with Logging (train_nn.py now does this)
# ============================================================================

import torch
import torch.nn as nn
from olfactory_modeling.utils import get_logger

logger = get_logger(__name__)

def train_epoch(model, dataloader, optimizer, device):
    """Example training function with logging."""
    model.train()
    total_loss = 0
    
    logger.info(f"Starting epoch on {device}")
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            logger.debug(f"Batch {batch_idx}: loss={loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    logger.info(f"Epoch complete. Avg loss: {avg_loss:.4f}")
    
    return avg_loss


# ============================================================================
# Example 6: Creating Models with BaseNeuralModel
# ============================================================================

from olfactory_modeling.models.base import BaseNeuralModel
import torch.nn as nn

class MoleculeEncoder(BaseNeuralModel):
    """Example model using the new base class."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Store metadata
        self.set_metadata('input_dim', input_dim)
        self.set_metadata('hidden_dim', hidden_dim)
        self.set_metadata('output_dim', output_dim)
    
    def forward(self, x):
        return self.network(x)
    
    def get_input_dim(self) -> int:
        return self.get_metadata('input_dim')
    
    def get_output_dim(self) -> int:
        return self.get_metadata('output_dim')


# Use the model
model = MoleculeEncoder(input_dim=268, hidden_dim=512, output_dim=1680)

# Get feature importance (provided by base class)
importance = model.get_feature_importance(top_n=20)
logger.info(f"Top 20 most important features: {list(importance.keys())}")

# Count parameters
n_params = model.count_parameters()
logger.info(f"Model has {n_params:,} trainable parameters")

# Save checkpoint with metadata
model.save_checkpoint(
    Path("checkpoints/molecule_encoder.pt"),
    optimizer_state=None,
    epoch=10,
    metrics={'val_loss': 0.042, 'correlation': 0.87}
)


# ============================================================================
# Example 7: Production Logging for Long Jobs
# ============================================================================

from olfactory_modeling.utils import setup_logging, get_logger
from pathlib import Path
from datetime import datetime

# Setup for cluster job
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = Path(f"logs/experiment_{timestamp}")
log_dir.mkdir(parents=True, exist_ok=True)

setup_logging(
    log_level="INFO",
    log_file=log_dir / "training.log",
    console=True
)

logger = get_logger(__name__)

# Log experiment configuration
logger.info("="*70)
logger.info("EXPERIMENT START")
logger.info("="*70)
logger.info(f"Timestamp: {timestamp}")
logger.info(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"Log directory: {log_dir}")
logger.info("="*70)

# Your training code here...
logger.info("Training started")

# On completion
logger.info("="*70)
logger.info("EXPERIMENT COMPLETE")
logger.info(f"Total training time: 3h 45m")
logger.info(f"Best validation loss: 0.0234")
logger.info(f"Final correlation: 0.92")
logger.info("="*70)


# ============================================================================
# Example 8: Debugging with Function Call Logging
# ============================================================================

from olfactory_modeling.utils import get_logger, log_function_call

logger = get_logger(__name__, level="DEBUG")

@log_function_call(logger, level="DEBUG")
def preprocess_data(molecules, normalize=True, remove_outliers=False):
    """Example function with automatic call logging."""
    # ... processing ...
    return processed_molecules

# Automatically logs:
# "Calling preprocess_data(molecules=<DataFrame>, normalize=True, remove_outliers=False)"
# "preprocess_data returned <DataFrame>"

result = preprocess_data(molecules, normalize=True, remove_outliers=True)


# ============================================================================
# Example 9: Searching Logs After Training
# ============================================================================

"""
After training completes, you can search logs:

# Find all errors
$ grep "ERROR" logs/experiment_*/training.log

# Track validation loss over time
$ grep "val_loss" logs/experiment_*/training.log

# Find when early stopping triggered
$ grep "Early stopping" logs/experiment_*/training.log

# Get training summary
$ grep "EXPERIMENT" logs/experiment_*/training.log

# Find specific epoch
$ grep "Epoch 47" logs/experiment_*/training.log

# Monitor specific hyperparameter
$ grep "learning_rate" logs/experiment_*/training.log
"""


# ============================================================================
# Example 10: Cluster Job Script
# ============================================================================

"""
Example SLURM job script using the new infrastructure:

#!/bin/bash
#SBATCH --job-name=neuro_training
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

# Set environment variables
export NEURO_DEVICE=cuda:0
export NEURO_BATCH_SIZE=128
export NEURO_LEARNING_RATE=0.001
export NEURO_LOG_LEVEL=INFO
export NEURO_DATA_DIR=/scratch/data

# Run training
python -m olfactory_modeling.pipeline.train_nn \\
    --config-env \\
    --output-dir results/run_$SLURM_JOB_ID

# Logs will be in:
# - results/run_$SLURM_JOB_ID/logs/
# - Searchable even after job completes
# - Full stack traces on errors
# - Timestamped experiment records
"""

print("✅ All examples demonstrate the new production-ready infrastructure!")
