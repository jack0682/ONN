"""Projection Operators for Cycle Constraint Enforcement.

This module implements the projection operators P_C that enforce the cycle
constraint Cx = τ on edge embeddings.

Mathematical Foundation:
    Hard Projection:
        P_C(x) = x - C^T (CC^T)^{-1} (Cx - τ)
    
    Relaxed Projection (α ∈ (0, 1]):
        P_C^{α}(x) = x - α * C^T (CC^T)^{-1} (Cx - τ)

The projection ensures that after applying P_C, we have Cx = τ (for α = 1).

Reference:
    - spec/20_impl_plan.ir.yml: IMPL_016
    - Convex optimization: affine projection onto constraint set

Author: Claude (via IMPL_016)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from onn.core.cycles import CycleBasis

logger = logging.getLogger(__name__)


# ==============================================================================
# PROJECTION OPERATORS
# ==============================================================================

def project_constraints(
    x: torch.Tensor,
    C: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Hard projection: x - C^T (CC^T)^{-1} (Cx - τ).
    
    Projects x onto the constraint set {x : Cx = τ}.
    After projection, Cx = τ exactly (within numerical precision).
    
    Args:
        x: Edge embeddings of shape (m, p) or (m,)
        C: Cycle matrix of shape (q, m)
        tau: Target values of shape (q, p) or (q,)
        
    Returns:
        Projected embeddings of same shape as x
        
    Raises:
        ValueError: If dimensions don't match
        
    Reference:
        Affine projection: P = I - C^T (CC^T)^{-1} C, then x + C^T (CC^T)^{-1} τ
    """
    return relaxed_projection(x, C, tau, alpha=1.0)


def relaxed_projection(
    x: torch.Tensor,
    C: torch.Tensor,
    tau: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Relaxed projection: x - α * C^T (CC^T)^{-1} (Cx - τ).
    
    When α = 1.0, this is the hard projection.
    When α < 1.0, this is a partial step towards the constraint set.
    
    Using α < 1.0 helps when:
    - Training with ES (allows gradual constraint satisfaction)
    - Avoiding oscillation in alternating projection schemes
    
    Args:
        x: Edge embeddings of shape (m, p) or (m,)
        C: Cycle matrix of shape (q, m)
        tau: Target values of shape (q, p) or (q,)
        alpha: Relaxation strength in (0, 1]. Default 1.0 (hard projection).
        
    Returns:
        Projected embeddings of same shape as x
        
    Raises:
        ValueError: If alpha not in (0, 1] or dimensions don't match
    """
    # Validate alpha
    if not (0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    
    # Handle empty constraint case
    q, m = C.shape
    if q == 0:
        logger.debug("No constraints, returning x unchanged")
        return x
    
    # Ensure x is 2D: (m, p)
    x_was_1d = x.dim() == 1
    if x_was_1d:
        x = x.unsqueeze(1)
    
    # Ensure tau is 2D: (q, p)
    if tau.dim() == 1:
        tau = tau.unsqueeze(1)
    
    # Validate dimensions
    m_x, p = x.shape
    q_tau, p_tau = tau.shape
    
    if m_x != m:
        raise ValueError(f"x has {m_x} edges but C expects {m}")
    if q_tau != q:
        raise ValueError(f"tau has {q_tau} rows but C has {q}")
    if p != p_tau:
        raise ValueError(f"x has {p} dims but tau has {p_tau}")
    
    # Compute residual: Cx - τ, shape (q, p)
    residual = torch.mm(C, x) - tau
    
    # Compute (CC^T)^{-1} @ residual
    # CC^T is (q, q), symmetric positive semi-definite
    CCT = torch.mm(C, C.t())  # (q, q)
    
    # Add small regularization for numerical stability
    CCT = CCT + 1e-6 * torch.eye(q, device=CCT.device, dtype=CCT.dtype)
    
    try:
        # Solve (CC^T) @ y = residual for y
        # Using torch.linalg.solve for numerical stability
        correction_coeff = torch.linalg.solve(CCT, residual)  # (q, p)
    except RuntimeError as e:
        logger.warning(f"Singular matrix in projection, using pseudoinverse: {e}")
        CCT_pinv = torch.linalg.pinv(CCT)
        correction_coeff = torch.mm(CCT_pinv, residual)
    
    # Correction: C^T @ (CC^T)^{-1} @ residual, shape (m, p)
    correction = torch.mm(C.t(), correction_coeff)
    
    # Apply relaxed projection
    x_proj = x - alpha * correction
    
    if x_was_1d:
        x_proj = x_proj.squeeze(1)
    
    return x_proj


def project_with_basis(
    x: torch.Tensor,
    cycle_basis: CycleBasis,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Project using a CycleBasis object.
    
    Convenience wrapper around relaxed_projection.
    
    Args:
        x: Edge embeddings of shape (m, p)
        cycle_basis: CycleBasis with C and τ
        alpha: Relaxation strength
        
    Returns:
        Projected embeddings
    """
    return relaxed_projection(x, cycle_basis.cycle_matrix, cycle_basis.tau, alpha)


# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================

def compute_projection_error(
    x: torch.Tensor,
    C: torch.Tensor,
    tau: torch.Tensor,
) -> float:
    """Compute ||Cx - τ||_2 (constraint violation).
    
    Args:
        x: Edge embeddings of shape (m, p) or (m,)
        C: Cycle matrix of shape (q, m)
        tau: Target values of shape (q, p) or (q,)
        
    Returns:
        Scalar violation (L2 norm)
    """
    if x.dim() == 1:
        x = x.unsqueeze(1)
    if tau.dim() == 1:
        tau = tau.unsqueeze(1)
    
    residual = torch.mm(C, x) - tau
    return torch.norm(residual).item()


def is_in_constraint_set(
    x: torch.Tensor,
    C: torch.Tensor,
    tau: torch.Tensor,
    tolerance: float = 1e-6,
) -> bool:
    """Check if x satisfies Cx = τ within tolerance.
    
    Args:
        x: Edge embeddings
        C: Cycle matrix
        tau: Target values
        tolerance: Maximum allowed violation
        
    Returns:
        True if ||Cx - τ|| < tolerance
    """
    error = compute_projection_error(x, C, tau)
    return error < tolerance


def verify_projection_idempotent(
    C: torch.Tensor,
    tau: torch.Tensor,
    m: int,
    p: int,
    tolerance: float = 1e-5,
) -> bool:
    """Verify that P_C(P_C(x)) = P_C(x) (idempotency).
    
    A valid projection operator should be idempotent.
    
    Args:
        C: Cycle matrix of shape (q, m)
        tau: Target values
        m: Number of edges
        p: Embedding dimension
        tolerance: Maximum allowed deviation
        
    Returns:
        True if projection is idempotent
    """
    # Random test point
    x = torch.randn(m, p)
    
    # Project once
    x_proj = project_constraints(x, C, tau)
    
    # Project twice
    x_proj2 = project_constraints(x_proj, C, tau)
    
    # Check idempotency
    diff = torch.norm(x_proj2 - x_proj).item()
    
    if diff > tolerance:
        logger.error(f"Projection not idempotent: ||P(P(x)) - P(x)|| = {diff}")
        return False
    
    return True


def compute_null_space_projection(
    C: torch.Tensor,
) -> torch.Tensor:
    """Compute the projection matrix onto null space of C.
    
    P_null = I - C^T (CC^T)^{-1} C
    
    This projects onto the space where Cx = 0.
    
    Args:
        C: Cycle matrix of shape (q, m)
        
    Returns:
        Projection matrix of shape (m, m)
    """
    q, m = C.shape
    
    if q == 0:
        return torch.eye(m)
    
    CCT = torch.mm(C, C.t()) + 1e-6 * torch.eye(q, device=C.device, dtype=C.dtype)
    CCT_inv = torch.linalg.inv(CCT)
    
    # P = I - C^T (CC^T)^{-1} C
    P = torch.eye(m, device=C.device, dtype=C.dtype) - torch.mm(C.t(), torch.mm(CCT_inv, C))
    
    return P
