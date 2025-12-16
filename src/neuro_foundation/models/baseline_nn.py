"""Baseline neural network models for molecular structure to activity map prediction.

Models:
- MoleculeToActivityMapMLP: Simple feedforward network (descriptors → flat output)
- MoleculeToActivityMapCNN: CNN decoder network (descriptors → spatial map)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MoleculeToActivityMapMLP(nn.Module):
    """Simple MLP baseline for predicting activity maps from molecular descriptors.
    
    Architecture:
        Input: Molecular descriptors (e.g., ECFP fingerprints)
        Hidden layers: 512 → 256 → 128 (matching reference architecture)
        Dropout: 0.35 (following reference paper)
        Output: Flattened activity map (reshaped to spatial dimensions)
    
    Reference architecture from paper:
        - 3 hidden layers: 512, 256, 128 neurons
        - ReLU activation
        - Dropout 0.35 for regularization
    
    Args:
        input_dim: Dimension of input molecular descriptors
        hidden_dims: List of hidden layer dimensions (default: [512, 256, 128])
        output_shape: Tuple of (height, width) for activity map
        dropout: Dropout probability (default: 0.35, from reference paper)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [512, 256, 128],
        output_shape: Tuple[int, int] = (79, 43),
        dropout: float = 0.35,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_shape = output_shape
        self.output_dim = output_shape[0] * output_shape[1]  # 79 * 43 = 3397
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input molecular descriptors (batch_size, input_dim)
            
        Returns:
            Predicted activity map (batch_size, height, width)
        """
        # Pass through MLP
        flat_output = self.network(x)  # (batch_size, 3397)
        
        # Reshape to spatial dimensions
        batch_size = x.shape[0]
        output = flat_output.view(batch_size, self.output_shape[0], self.output_shape[1])
        
        return output


class MoleculeToActivityMapCNN(nn.Module):
    """CNN decoder for predicting activity maps from molecular descriptors.
    
    Uses transposed convolutions to generate spatial activity patterns.
    This architecture better respects the 2D spatial structure of brain activity maps.
    
    Architecture:
        Input: Molecular descriptors
        Encoder: Dense layers (512 → 256 → 128) matching reference
        Decoder: Transposed convolutions to upsample to (79, 43)
        Dropout: 0.35 for regularization
    
    Args:
        input_dim: Dimension of input molecular descriptors
        latent_dim: Dimension of latent representation
        output_shape: Tuple of (height, width) for activity map
        dropout: Dropout probability (default: 0.35, from reference)
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 512,
        output_shape: Tuple[int, int] = (79, 43),
        dropout: float = 0.35,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        
        # Encoder: Compress molecular descriptors to latent representation
        # Following reference architecture: 512 → 256 → 128
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Initial spatial dimensions after reshaping latent vector
        # We'll reshape 128 to (64, 4, 4) then upsample
        self.initial_channels = 64
        self.initial_h = 5
        self.initial_w = 3
        
        # Project latent to initial spatial representation
        self.to_spatial = nn.Sequential(
            nn.Linear(128, self.initial_channels * self.initial_h * self.initial_w),
            nn.ReLU(),
        )
        
        # Decoder: Transposed convolutions to upsample
        # From (5, 3) → (10, 6) → (20, 11) → (40, 22) → (79, 43)
        self.decoder = nn.Sequential(
            # (64, 5, 3) → (32, 10, 6)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            
            # (32, 10, 6) → (16, 20, 11)
            nn.ConvTranspose2d(32, 16, kernel_size=(4, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            
            # (16, 20, 11) → (8, 40, 22)
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            
            # (8, 40, 22) → (1, 79, 43)
            nn.ConvTranspose2d(8, 1, kernel_size=(4, 3), stride=2, padding=1, output_padding=(1, 1)),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input molecular descriptors (batch_size, input_dim)
            
        Returns:
            Predicted activity map (batch_size, height, width)
        """
        batch_size = x.shape[0]
        
        # Encode to latent representation
        latent = self.encoder(x)  # (batch_size, latent_dim)
        
        # Project to spatial representation
        spatial = self.to_spatial(latent)  # (batch_size, channels * h * w)
        
        # Reshape to spatial dimensions
        spatial = spatial.view(
            batch_size, 
            self.initial_channels, 
            self.initial_h, 
            self.initial_w
        )
        
        # Decode through transposed convolutions
        output = self.decoder(spatial)  # (batch_size, 1, height, width)
        
        # Remove channel dimension
        output = output.squeeze(1)  # (batch_size, height, width)
        
        return output


def get_model(model_type: str, input_dim: int, **kwargs) -> nn.Module:
    """Factory function to create models.
    
    Args:
        model_type: Type of model ('mlp' or 'cnn')
        input_dim: Input dimension for molecular descriptors
        **kwargs: Additional arguments passed to model constructor
        
    Returns:
        Initialized model
        
    Example:
        >>> model = get_model('mlp', input_dim=2048, hidden_dims=[512, 1024])
        >>> model = get_model('cnn', input_dim=2048, latent_dim=512)
    """
    models = {
        'mlp': MoleculeToActivityMapMLP,
        'cnn': MoleculeToActivityMapCNN,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    return models[model_type](input_dim=input_dim, **kwargs)
