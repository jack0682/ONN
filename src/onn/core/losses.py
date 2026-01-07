"""Loss Functions for Edge-Embedding ONN Solver.

This module implements the loss terms for the Projection-Consensus solver:
    L_total = L_data + λ_ctx * L_context + λ_ricci * L_ricci 
              + λ_smooth * L_smooth + λ_var * L_variance

Loss Terms:
    - L_data: Stay close to observations (||x - x_obs||²)
    - L_context: Soft cycle constraint (||Cx - τ||²)
    - L_ricci: Curvature regularization (Σ Ric(e)²)
    - L_smooth: Temporal stability (||x_t - x_{t-1}||²)
    - L_variance: Anti-collapse (penalize low variance)

Reference:
    - spec/20_impl_plan.ir.yml: IMPL_017
    - User's roadmap: "Anti-collapse is critical for 32D embeddings"

Author: Claude (via IMPL_017)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from onn.core.graph import EdgeGraph
from onn.core.cycles import CycleBasis

logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class LossConfig:
    """Loss function weights and parameters.
    
    Reference: spec/11_interfaces.ir.yml -> ONNSolverConfig
    """
    lambda_context: float = 1.0     # Context constraint weight
    lambda_ricci: float = 0.1       # Ricci curvature weight
    lambda_smooth: float = 0.1      # Temporal smoothness weight
    lambda_var: float = 1.0         # Anti-collapse variance weight
    
    # Anti-collapse parameters
    min_variance: float = 0.01      # Minimum variance threshold per dimension
    
    # Numerical stability
    eps: float = 1e-8


# ==============================================================================
# INDIVIDUAL LOSS TERMS
# ==============================================================================

def loss_data(
    x: torch.Tensor,
    x_obs: torch.Tensor,
) -> torch.Tensor:
    """Data fidelity loss: ||x - x_obs||².
    
    Keeps the learned embeddings close to observations.
    
    Args:
        x: Current embeddings of shape (m, p)
        x_obs: Observed embeddings of shape (m, p)
        
    Returns:
        Scalar loss tensor
    """
    diff = x - x_obs
    return torch.sum(diff ** 2)


def loss_context(
    x: torch.Tensor,
    C: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Context constraint loss: ||Cx - τ||² (soft cycle constraint).
    
    Encourages cycle consistency without hard projection.
    Used as a monitoring metric and soft regularizer.
    
    Args:
        x: Edge embeddings of shape (m, p)
        C: Cycle matrix of shape (q, m)
        tau: Target values of shape (q, p)
        
    Returns:
        Scalar loss tensor
    """
    if C.shape[0] == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    
    if x.dim() == 1:
        x = x.unsqueeze(1)
    if tau.dim() == 1:
        tau = tau.unsqueeze(1)
    
    # Cx - τ: (q, m) @ (m, p) - (q, p) = (q, p)
    residual = torch.mm(C, x) - tau
    return torch.sum(residual ** 2)


def loss_ricci(
    x: torch.Tensor,
    edge_graph: EdgeGraph,
    weight_from_embedding: bool = False,
) -> torch.Tensor:
    """Ricci curvature regularization: Σ_e Ric(e)².
    
    Forman-Ricci curvature for each edge:
        Ric(e) = 4 - d_i - d_j + 3 * Δ_ij
    
    where d_i, d_j are node degrees and Δ_ij is the number of triangles
    containing edge e = (i, j).
    
    This regularizes graph structure, preventing curvature explosion.
    
    Args:
        x: Edge embeddings of shape (m, p) (used for weighted version)
        edge_graph: EdgeGraph instance
        weight_from_embedding: If True, weight curvature by embedding norm
        
    Returns:
        Scalar loss tensor
        
    Reference:
        Forman-Ricci curvature: spec/02_onn_math_spec.md Section 2.3
    """
    m = edge_graph.num_edges
    
    if m == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    
    # Compute node degrees
    degrees = {}
    for ek in edge_graph.edge_keys:
        degrees[ek.source_id] = degrees.get(ek.source_id, 0) + 1
        degrees[ek.target_id] = degrees.get(ek.target_id, 0) + 1
    
    # Count triangles per edge
    adj = edge_graph._get_adjacency()
    
    ricci_values = []
    for ek in edge_graph.edge_keys:
        d_i = degrees.get(ek.source_id, 0)
        d_j = degrees.get(ek.target_id, 0)
        
        # Count triangles: common neighbors
        neighbors_i = set(adj.get(ek.source_id, []))
        neighbors_j = set(adj.get(ek.target_id, []))
        triangles = len(neighbors_i & neighbors_j)
        
        # Forman-Ricci curvature
        ric = 4 - d_i - d_j + 3 * triangles
        ricci_values.append(ric)
    
    ricci_tensor = torch.tensor(ricci_values, device=x.device, dtype=x.dtype)
    
    if weight_from_embedding:
        # Weight by embedding norm (higher norm edges contribute more)
        weights = torch.norm(x, dim=1)
        weights = weights / (weights.sum() + 1e-8)
        return torch.sum(weights * ricci_tensor ** 2)
    else:
        return torch.sum(ricci_tensor ** 2)


def loss_smooth(
    x: torch.Tensor,
    x_prev: torch.Tensor,
) -> torch.Tensor:
    """Temporal smoothness loss: ||x_t - x_{t-1}||².
    
    Prevents sudden jumps in embeddings between time steps.
    
    Args:
        x: Current embeddings of shape (m, p)
        x_prev: Previous embeddings of shape (m, p)
        
    Returns:
        Scalar loss tensor
    """
    diff = x - x_prev
    return torch.sum(diff ** 2)


def loss_variance(
    x: torch.Tensor,
    min_variance: float = 0.01,
) -> torch.Tensor:
    """Anti-collapse variance loss.
    
    Penalizes when embedding dimensions have low variance.
    This prevents all embeddings from collapsing to the same vector.
    
    Loss = Σ_k max(0, σ²_min - Var(x_{:,k}))²
    
    Args:
        x: Edge embeddings of shape (m, p)
        min_variance: Minimum required variance per dimension
        
    Returns:
        Scalar loss tensor
        
    Reference:
        User roadmap: "Without loss_variance, embeddings collapse to constant vectors"
    """
    if x.dim() == 1:
        x = x.unsqueeze(1)
    
    m, p = x.shape
    
    if m < 2:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    
    # Compute variance per dimension: Var(x_{:,k})
    variance_per_dim = torch.var(x, dim=0)  # (p,)
    
    # Penalty for low variance: max(0, σ²_min - Var)²
    deficit = F.relu(min_variance - variance_per_dim)
    
    return torch.sum(deficit ** 2)


# ==============================================================================
# TOTAL LOSS
# ==============================================================================

def total_loss(
    x: torch.Tensor,
    x_obs: torch.Tensor,
    cycle_basis: CycleBasis,
    edge_graph: EdgeGraph,
    x_prev: Optional[torch.Tensor] = None,
    config: Optional[LossConfig] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute weighted total loss with breakdown.
    
    L_total = L_data + λ_ctx * L_context + λ_ricci * L_ricci 
              + λ_smooth * L_smooth + λ_var * L_variance
    
    Args:
        x: Current embeddings of shape (m, p)
        x_obs: Observed embeddings of shape (m, p)
        cycle_basis: CycleBasis with C and τ
        edge_graph: EdgeGraph for Ricci computation
        x_prev: Previous embeddings (optional, for smoothness)
        config: Loss configuration with weights
        
    Returns:
        Tuple of (total_loss_tensor, breakdown_dict)
        
    Reference:
        spec/20_impl_plan.ir.yml: IMPL_017
    """
    if config is None:
        config = LossConfig()
    
    # Data fidelity (always weight = 1.0)
    l_data = loss_data(x, x_obs)
    
    # Context (soft cycle constraint)
    l_context = loss_context(x, cycle_basis.cycle_matrix, cycle_basis.tau)
    
    # Ricci curvature
    l_ricci = loss_ricci(x, edge_graph)
    
    # Smoothness (only if x_prev provided)
    if x_prev is not None:
        l_smooth = loss_smooth(x, x_prev)
    else:
        l_smooth = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    
    # Anti-collapse variance
    l_var = loss_variance(x, config.min_variance)
    
    # Weighted total
    total = (
        l_data 
        + config.lambda_context * l_context
        + config.lambda_ricci * l_ricci
        + config.lambda_smooth * l_smooth
        + config.lambda_var * l_var
    )
    
    # Breakdown for logging/debugging
    breakdown = {
        "data": l_data.item(),
        "context": l_context.item(),
        "ricci": l_ricci.item(),
        "smooth": l_smooth.item(),
        "variance": l_var.item(),
        "total": total.item(),
    }
    
    return total, breakdown


# ==============================================================================
# GRADIENT UTILITIES
# ==============================================================================

def compute_gradient(
    x: torch.Tensor,
    x_obs: torch.Tensor,
    cycle_basis: CycleBasis,
    edge_graph: EdgeGraph,
    x_prev: Optional[torch.Tensor] = None,
    config: Optional[LossConfig] = None,
) -> torch.Tensor:
    """Compute gradient of total loss w.r.t. x using autograd.
    
    Args:
        x: Current embeddings (must have requires_grad=True)
        x_obs: Observed embeddings
        cycle_basis: Cycle constraint basis
        edge_graph: Graph structure
        x_prev: Previous embeddings (optional)
        config: Loss configuration
        
    Returns:
        Gradient tensor of same shape as x
    """
    x_grad = x.clone().requires_grad_(True)
    
    loss, _ = total_loss(x_grad, x_obs, cycle_basis, edge_graph, x_prev, config)
    
    grad = torch.autograd.grad(loss, x_grad, create_graph=False)[0]
    
    return grad


def gradient_step(
    x: torch.Tensor,
    grad: torch.Tensor,
    step_size: float,
    max_grad_norm: Optional[float] = None,
    use_delta: bool = False,
    delta_beta: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply gradient descent step with optional gradient clipping.

    Supports two modes:
    1. Standard Euler: x_new = x - η * grad
    2. Delta ODE (DDL): x_new = x - β * η * k * (k^T grad)
       where k = normalize(grad)

    Args:
        x: Current embeddings (m, p)
        grad: Gradient tensor (m, p)
        step_size: Learning rate (η)
        max_grad_norm: Maximum gradient norm (for clipping)
        use_delta: If True, use Delta ODE update instead of Euler
        delta_beta: β value for delta update (0=identity, 1=projection, 2=reflection)
        eps: Numerical stability epsilon

    Returns:
        Updated embeddings
    """
    # Gradient clipping
    if max_grad_norm is not None:
        grad_norm = torch.norm(grad)
        if grad_norm > max_grad_norm:
            grad = grad * (max_grad_norm / grad_norm)

    if use_delta:
        # Delta ODE update (DDL paper Eq 4.1)
        # x_new = x - β * η * k * (k^T grad)
        # where k = grad / ||grad||
        grad_norm = torch.norm(grad, dim=-1, keepdim=True) + eps
        k = grad / grad_norm  # Normalized direction (m, p)

        # k^T @ grad for each row: sum over embedding dim
        k_dot_grad = (k * grad).sum(dim=-1, keepdim=True)  # (m, 1)

        # Delta update: rank-1 projected gradient
        return x - delta_beta * step_size * k * k_dot_grad
    else:
        # Standard Euler step
        return x - step_size * grad
