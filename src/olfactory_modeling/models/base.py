"""Abstract base classes for neural network models.

Enforces consistent interface across all model implementations and
provides common functionality for feature importance, checkpointing,
and evaluation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

import torch
import torch.nn as nn


class BaseNeuralModel(nn.Module, ABC):
    """Abstract base class for all neural network models.
    
    Enforces:
    - Consistent forward pass interface
    - Feature importance extraction
    - Model metadata and checkpointing
    - Evaluation mode handling
    """
    
    def __init__(self):
        super().__init__()
        self._metadata: Dict[str, Any] = {}
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output predictions
        """
        pass
    
    @abstractmethod
    def get_input_dim(self) -> int:
        """Return the expected input dimension."""
        pass
    
    @abstractmethod
    def get_output_dim(self) -> int:
        """Return the model output dimension."""
        pass
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> Dict[int, float]:
        """Extract feature importance scores from first layer weights.
        
        Args:
            top_n: Number of top features to return. If None, returns all
            
        Returns:
            Dictionary mapping feature index to importance score
            
        Raises:
            NotImplementedError: If model doesn't support feature importance
        """
        first_layer = self._get_first_linear_layer()
        if first_layer is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have a Linear first layer"
            )
        
        weights = first_layer.weight.data.abs().mean(dim=0).cpu().numpy()
        importance = {i: float(w) for i, w in enumerate(weights)}
        
        if top_n is not None:
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        return importance
    
    def _get_first_linear_layer(self) -> Optional[nn.Linear]:
        """Find first Linear layer in the model."""
        # Check direct attributes
        for module in self.modules():
            if isinstance(module, nn.Linear):
                return module
        
        # Check common attribute names
        for attr_name in ['network', 'encoder', 'layers', 'fc']:
            if hasattr(self, attr_name):
                network = getattr(self, attr_name)
                for module in network.modules():
                    if isinstance(module, nn.Linear):
                        return module
        
        return None
    
    def save_checkpoint(
        self,
        filepath: Path,
        optimizer_state: Optional[Dict] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict] = None
    ) -> None:
        """Save model checkpoint with metadata.
        
        Args:
            filepath: Path to save checkpoint
            optimizer_state: Optional optimizer state dict
            epoch: Optional epoch number
            metrics: Optional metrics dictionary
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_class': self.__class__.__name__,
            'metadata': self._metadata,
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, filepath)
    
    @classmethod
    def load_checkpoint(cls, filepath: Path, map_location: Optional[str] = None) -> BaseNeuralModel:
        """Load model from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            map_location: Device to map tensors to (e.g., 'cpu', 'cuda:0')
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(filepath, map_location=map_location)
        
        # This is a basic implementation - subclasses should override
        # to properly instantiate with correct architecture parameters
        raise NotImplementedError(
            "Subclasses must implement load_checkpoint with architecture params"
        )
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Store metadata about the model (hyperparameters, training config, etc)."""
        self._metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Retrieve metadata value."""
        return self._metadata.get(key, default)
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters.
        
        Args:
            trainable_only: If True, count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
