"""Relation Embedding Parameterization.

This module implements x_obs parameterization for edge embeddings using
a combination of random projection and learnable linear transform.

Formula:
    x_obs = normalize(W_rp @ φ) + W_lin @ φ
    
    - W_rp: Fixed random projection (provides base structure)
    - W_lin: ES-learnable linear term (allows adaptation)

Reference:
    - spec/20_impl_plan.ir.yml: IMPL_019
    - User roadmap: "random projection + linear term"

Author: Claude (via IMPL_019)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import numpy as np

logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class RelationParamConfig:
    """Relation encoder configuration.
    
    Reference: spec/11_interfaces.ir.yml (implied by EdgeEmbeddingState)
    """
    input_dim: int = 64         # φ_ij dimension (depends on pair features)
    output_dim: int = 32        # x_obs dimension (edge embedding size)
    random_proj_seed: int = 42  # Fixed seed for reproducibility
    use_linear_term: bool = True  # Whether to use learnable W_lin
    normalize_rp: bool = True   # Normalize random projection output


# ==============================================================================
# RELATION ENCODER
# ==============================================================================

class RelationEncoder:
    """Relation embedding encoder.
    
    Transforms pair features φ_ij into initial edge embeddings x_obs.
    
    Uses random projection for base structure, with optional learnable
    linear term for ES-based adaptation.
    """
    
    def __init__(self, config: RelationParamConfig):
        """Initialize the encoder.
        
        Args:
            config: Encoder configuration
        """
        self.config = config
        
        # Fixed random projection matrix
        rng = np.random.RandomState(config.random_proj_seed)
        W_rp_np = rng.randn(config.output_dim, config.input_dim)
        W_rp_np = W_rp_np / np.sqrt(config.input_dim)  # Normalize
        self.W_rp = torch.tensor(W_rp_np, dtype=torch.float32)
        
        # Learnable linear term (initialized to zero)
        if config.use_linear_term:
            self.W_lin = torch.zeros(config.output_dim, config.input_dim)
        else:
            self.W_lin = None
        
        logger.debug(f"RelationEncoder: {config.input_dim} -> {config.output_dim}")
    
    def encode(self, phi: torch.Tensor) -> torch.Tensor:
        """Encode pair features to edge embeddings.
        
        Args:
            phi: Pair features, shape (m, input_dim) or (input_dim,)
            
        Returns:
            Edge embeddings x_obs, shape (m, output_dim) or (output_dim,)
        """
        single = phi.dim() == 1
        if single:
            phi = phi.unsqueeze(0)
        
        # Random projection: W_rp @ φ^T -> (output_dim, m) -> transpose
        x_rp = torch.mm(self.W_rp.to(phi.device), phi.t()).t()
        
        # Normalize if configured
        if self.config.normalize_rp:
            x_rp = torch.nn.functional.normalize(x_rp, dim=1)
        
        # Add learnable linear term
        if self.W_lin is not None:
            x_lin = torch.mm(self.W_lin.to(phi.device), phi.t()).t()
            x_obs = x_rp + x_lin
        else:
            x_obs = x_rp
        
        if single:
            x_obs = x_obs.squeeze(0)
        
        return x_obs
    
    def set_linear_weights(self, W_lin: torch.Tensor) -> None:
        """Set the learnable linear weights (from ES).
        
        Args:
            W_lin: Linear weight matrix, shape (output_dim, input_dim)
        """
        if self.W_lin is None:
            raise ValueError("Linear term not enabled in config")
        
        if W_lin.shape != self.W_lin.shape:
            raise ValueError(f"Shape mismatch: {W_lin.shape} vs {self.W_lin.shape}")
        
        self.W_lin = W_lin.clone()
    
    def get_linear_weights(self) -> Optional[torch.Tensor]:
        """Get the current linear weights."""
        return self.W_lin.clone() if self.W_lin is not None else None


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

def encode_relation(
    phi: torch.Tensor,
    config: Optional[RelationParamConfig] = None,
    W_lin: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Encode pair features to edge embeddings (stateless).
    
    x_obs = normalize(W_rp @ φ) + W_lin @ φ
    
    Args:
        phi: Pair features, shape (m, input_dim)
        config: Encoder configuration
        W_lin: Learnable linear weights, shape (output_dim, input_dim)
        
    Returns:
        Edge embeddings x_obs
    """
    if config is None:
        # Infer dimensions from input
        input_dim = phi.shape[-1]
        config = RelationParamConfig(input_dim=input_dim)
    
    encoder = RelationEncoder(config)
    
    if W_lin is not None:
        encoder.set_linear_weights(W_lin)
    
    return encoder.encode(phi)


def random_projection(
    phi: torch.Tensor,
    output_dim: int = 32,
    seed: int = 42,
) -> torch.Tensor:
    """Apply random projection to pair features.
    
    Args:
        phi: Pair features, shape (m, d)
        output_dim: Output dimension
        seed: Random seed
        
    Returns:
        Projected features, shape (m, output_dim)
    """
    input_dim = phi.shape[-1]
    
    rng = np.random.RandomState(seed)
    W = rng.randn(output_dim, input_dim) / np.sqrt(input_dim)
    W = torch.tensor(W, dtype=phi.dtype, device=phi.device)
    
    if phi.dim() == 1:
        return torch.mv(W, phi)
    else:
        return torch.mm(W, phi.t()).t()


def create_default_encoder(input_dim: int = 64) -> RelationEncoder:
    """Create a relation encoder with default configuration.
    
    Args:
        input_dim: Pair feature dimension
        
    Returns:
        Configured RelationEncoder
    """
    config = RelationParamConfig(input_dim=input_dim)
    return RelationEncoder(config)
