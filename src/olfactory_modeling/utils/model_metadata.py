"""Model metadata collection utilities."""
from __future__ import annotations
import torch
from typing import Dict, Any
from ..utils.logging_config import get_logger
logger = get_logger(__name__)

def collect_model_metadata(model: torch.nn.Module) -> Dict[str, Any]:
    metadata = {}
    metadata['model_class'] = model.__class__.__name__
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metadata['total_parameters'] = total_params
    metadata['trainable_parameters'] = trainable_params
    metadata['n_layers'] = len(list(model.modules()))
    metadata['architecture'] = str(model)
    return metadata
