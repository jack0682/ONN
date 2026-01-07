"""Differentiable K-step Unroll Solver.

Phase 2 Implementation:
- K-step gradient descent with create_graph=True
- Allows end-to-end gradient flow from loss back to W_lin
- Compatible with existing ProjectionConsensusSolver interface

Key difference from standard solver:
    Standard:  grad = autograd.grad(loss, x, create_graph=False)
    Diff:      grad = autograd.grad(loss, x, create_graph=True)

This enables:
    x_obs = encoder(phi)  # phi depends on W_lin
    x_final = diff_solve(x_obs, K_steps)
    fitness = compute_fitness(x_final)
    fitness.backward()  # Gradient flows to W_lin!

Reference:
    - "Differentiable Optimization" (Amos & Kolter, 2017)
    - "Meta-Learning with Differentiable Convex Optimization"

Author: Claude (Phase 2 - Differentiable Solver)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn

from onn.core.graph import EdgeGraph
from onn.core.cycles import CycleBasis, build_cycle_basis
from onn.core.projection import relaxed_projection
from onn.core.losses import LossConfig, total_loss

logger = logging.getLogger(__name__)


# ==============================================================================
# DELTA ODE UTILITIES (DDL Paper - Section 9, Eq 4.1)
# ==============================================================================

def delta_ode_step(
    x: torch.Tensor,
    grad: torch.Tensor,
    step_size: float,
    beta: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Perform Delta ODE step based on DDL paper.

    Implements the rank-1 gated gradient update from the Delta Rule (Eq 4.1):

        X_new = X + beta * k * (v^T - k^T X)

    For gradient descent, we interpret:
        - k = normalize(grad): Direction from gradient
        - v = k^T X - step_size * (k^T grad): Correction signal

    This gives:
        (v^T - k^T X) = -step_size * (k^T grad)
        X_new = X - beta * step_size * k * (k^T grad)

    Which is a rank-1 projected gradient step along direction k.

    Args:
        x: Current state (m, p)
        grad: Gradient of loss w.r.t. x (m, p)
        step_size: Learning rate
        beta: Gate value in [0, 2] (default 1.0 for projection)
        eps: Numerical stability

    Returns:
        x_new: Updated state (m, p)
    """
    # k = normalize(grad) along feature dimension
    grad_norm = torch.norm(grad, dim=-1, keepdim=True) + eps
    k = grad / grad_norm  # (m, p) normalized rows

    # Projection: k^T grad (dot product per edge)
    k_dot_grad = (k * grad).sum(dim=-1, keepdim=True)  # (m, 1)

    # Delta update: X_new = X - beta * step_size * k * (k^T grad)
    # This is rank-1 projected gradient descent
    x_new = x - beta * step_size * k * k_dot_grad

    return x_new


def compute_adaptive_beta(
    grad: torch.Tensor,
    beta_learnable: Optional[torch.Tensor] = None,
    beta_mode: str = 'fixed',
    beta_value: float = 1.0,
) -> torch.Tensor:
    """Compute adaptive beta for Delta ODE.

    Args:
        grad: Gradient tensor (m, p)
        beta_learnable: Optional learnable beta parameter
        beta_mode: 'fixed', 'adaptive', or 'learnable'
        beta_value: Fixed beta value (used if mode='fixed')

    Returns:
        beta: Scalar or tensor beta value
    """
    if beta_mode == 'fixed':
        return torch.tensor(beta_value, device=grad.device, dtype=grad.dtype)
    elif beta_mode == 'adaptive':
        # Adaptive: beta based on gradient magnitude
        # Higher gradient -> lower beta (more conservative)
        grad_mag = torch.norm(grad).item()
        beta = 2.0 * torch.sigmoid(torch.tensor(-grad_mag))
        return beta.to(grad.device)
    elif beta_mode == 'learnable' and beta_learnable is not None:
        # Learnable beta with sigmoid to bound [0, 2]
        return 2.0 * torch.sigmoid(beta_learnable)
    else:
        return torch.tensor(beta_value, device=grad.device, dtype=grad.dtype)


@dataclass
class DiffSolverConfig:
    """Configuration for differentiable solver.

    Key parameters:
        K: Number of unroll steps (tradeoff: more = better solution, slower backprop)
        create_graph: If True, enables gradient flow through iterations

    Delta ODE parameters (DDL Paper - Section 9, Eq 4.1):
        use_delta_update: If True, use rank-1 Delta update instead of Euler step
        delta_beta_mode: 'fixed', 'adaptive', or 'learnable'
        delta_beta_value: Fixed beta value (used if mode='fixed')
    """
    # Solver parameters
    K: int = 10                     # Number of unroll steps
    step_size: float = 0.01         # Gradient step size (eta)
    projection_alpha: float = 1.0   # Relaxed projection strength

    # Loss weights
    lambda_context: float = 1.0
    lambda_ricci: float = 0.1
    lambda_smooth: float = 0.1
    lambda_var: float = 1.0
    min_variance: float = 0.01

    # Gradient control
    max_grad_norm: float = 1.0
    create_graph: bool = True       # Enable end-to-end gradient flow

    # Checkpointing (memory optimization)
    checkpoint_every: int = 0       # 0 = no checkpointing

    # Delta ODE parameters (DDL Paper)
    use_delta_update: bool = False  # Use Delta ODE instead of Euler
    delta_beta_mode: str = 'fixed'  # 'fixed', 'adaptive', or 'learnable'
    delta_beta_value: float = 1.0   # Beta for projection (1.0 = orthogonal proj)

    def to_loss_config(self) -> LossConfig:
        return LossConfig(
            lambda_context=self.lambda_context,
            lambda_ricci=self.lambda_ricci,
            lambda_smooth=self.lambda_smooth,
            lambda_var=self.lambda_var,
            min_variance=self.min_variance,
        )


class DifferentiableSolver(nn.Module):
    """Differentiable K-step solver for end-to-end learning.

    This solver uses `create_graph=True` in autograd, allowing gradients
    to flow from the final output back through all K iterations to the
    input x_obs (and thus to W_lin that produced x_obs).

    Usage:
        >>> solver = DifferentiableSolver(config)
        >>> x_obs = encoder(phi)  # x_obs depends on W_lin
        >>> x_final, info = solver(x_obs, edge_graph)
        >>> loss = fitness_loss(x_final)
        >>> loss.backward()  # Gradients flow to encoder.W_lin!

    Note:
        Memory usage scales with K due to computation graph storage.
        Use checkpoint_every > 0 for large K to reduce memory.
    """

    def __init__(self, config: DiffSolverConfig):
        super().__init__()
        self.config = config
        self.loss_config = config.to_loss_config()

        logger.info(
            f"DifferentiableSolver: K={config.K}, η={config.step_size}, "
            f"create_graph={config.create_graph}"
        )

    def _gradient_step(
        self,
        x: torch.Tensor,
        x_obs: torch.Tensor,
        cycle_basis: CycleBasis,
        edge_graph: EdgeGraph,
        x_prev: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Single gradient step with optional graph creation.

        Supports two modes:
        1. Standard Euler: x_new = x - step_size * grad
        2. Delta ODE (DDL Paper Eq 4.1):
           x_new = x - beta * step_size * k * (k^T grad)
           where k = normalize(grad)

        Args:
            x: Current embeddings (m, p) - requires_grad if create_graph
            x_obs: Observed embeddings
            cycle_basis: Cycle constraints
            edge_graph: Graph structure
            x_prev: Previous timestep embeddings

        Returns:
            Updated x and loss breakdown
        """
        # Compute loss
        loss, breakdown = total_loss(
            x, x_obs, cycle_basis, edge_graph, x_prev, self.loss_config
        )

        # Compute gradient WITH graph creation for backprop
        grad = torch.autograd.grad(
            loss, x,
            create_graph=self.config.create_graph,
            retain_graph=self.config.create_graph,
        )[0]

        # Gradient clipping (differentiable)
        if self.config.max_grad_norm > 0:
            grad_norm = torch.norm(grad)
            scale = torch.clamp(
                self.config.max_grad_norm / (grad_norm + 1e-8),
                max=1.0
            )
            grad = grad * scale

        # Gradient step: Euler or Delta ODE
        if self.config.use_delta_update:
            # Delta ODE step (DDL Paper - Section 9, Eq 4.1)
            # x_new = x - beta * step_size * k * (k^T grad)
            beta = compute_adaptive_beta(
                grad,
                beta_learnable=getattr(self, 'beta_learnable', None),
                beta_mode=self.config.delta_beta_mode,
                beta_value=self.config.delta_beta_value,
            )
            x_new = delta_ode_step(
                x, grad, self.config.step_size, beta=beta.item()
            )
            breakdown['delta_beta'] = beta.item()
        else:
            # Standard Euler step
            x_new = x - self.config.step_size * grad

        return x_new, breakdown

    def _projection_step(
        self,
        x: torch.Tensor,
        cycle_basis: CycleBasis,
    ) -> torch.Tensor:
        """Relaxed projection step (differentiable).

        The relaxed projection is already differentiable:
        x_proj = x + α * C^T @ (C @ C^T)^{-1} @ (τ - C @ x)
        """
        return relaxed_projection(
            x,
            cycle_basis.cycle_matrix,
            cycle_basis.tau,
            alpha=self.config.projection_alpha,
        )

    def forward(
        self,
        x_obs: torch.Tensor,
        edge_graph: EdgeGraph,
        x_prev: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """K-step differentiable solve.

        Args:
            x_obs: Observed embeddings (m, p) - can have grad from encoder
            edge_graph: Graph structure
            x_prev: Previous timestep embeddings
            return_trajectory: If True, return all intermediate states

        Returns:
            x_final: Final embeddings after K steps
            info: Dictionary with loss history and optionally trajectory
        """
        m, p = x_obs.shape

        # Build cycle basis
        cycle_basis = build_cycle_basis(edge_graph, p)

        # Initialize x = x_obs (maintains gradient connection!)
        x = x_obs.clone()
        if self.config.create_graph:
            x = x.requires_grad_(True)

        # Tracking
        loss_history = []
        trajectory = [x.detach().clone()] if return_trajectory else None

        # K-step unroll
        for k in range(self.config.K):
            # Gradient step
            x, breakdown = self._gradient_step(
                x, x_obs, cycle_basis, edge_graph, x_prev
            )

            # Projection step
            x = self._projection_step(x, cycle_basis)

            # Track
            loss_history.append(breakdown['total'])

            if return_trajectory:
                trajectory.append(x.detach().clone())

            # Gradient checkpointing (memory optimization)
            if (self.config.checkpoint_every > 0 and
                k > 0 and k % self.config.checkpoint_every == 0):
                x = x.detach().requires_grad_(True)

        info = {
            'loss_history': loss_history,
            'final_loss': loss_history[-1] if loss_history else 0.0,
            'K': self.config.K,
        }

        if return_trajectory:
            info['trajectory'] = trajectory

        return x, info


class DifferentiableEncoder(nn.Module):
    """Differentiable encoder with W_lin for end-to-end learning.

    Combines:
    - Fixed random projection W_rp
    - Learnable W_lin (gradient flows here!)
    - Optional geometric features W_geo

    Formula: x_obs = α * norm(W_rp @ φ) + β * W_lin @ φ + γ * W_geo @ ξ
    """

    def __init__(
        self,
        input_dim: int = 64,
        output_dim: int = 32,
        geometric_dim: int = 6,
        alpha: float = 0.3,
        beta: float = 0.5,
        gamma: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Fixed random projection
        rng = torch.Generator().manual_seed(seed)
        W_rp = torch.randn(output_dim, input_dim, generator=rng) / (input_dim ** 0.5)
        self.register_buffer('W_rp', W_rp)

        # Learnable W_lin - THIS IS WHAT WE'RE OPTIMIZING END-TO-END!
        self.W_lin = nn.Parameter(torch.zeros(output_dim, input_dim))
        nn.init.xavier_normal_(self.W_lin)

        # Fixed geometric projection
        W_geo = torch.randn(output_dim, geometric_dim, generator=rng) / (geometric_dim ** 0.5)
        self.register_buffer('W_geo', W_geo)

        logger.info(
            f"DifferentiableEncoder: in={input_dim}, out={output_dim}, "
            f"W_lin params={input_dim * output_dim}"
        )

    def forward(
        self,
        phi: torch.Tensor,
        xi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode pair features to edge embeddings.

        All operations are differentiable, so gradients flow to W_lin.

        Args:
            phi: Pair features (batch, input_dim)
            xi: Geometric features (batch, geometric_dim) - optional

        Returns:
            x_obs: Edge embeddings (batch, output_dim)
        """
        # Random projection (fixed, but still in graph)
        x_rp = torch.mm(phi, self.W_rp.t())
        x_rp = torch.nn.functional.normalize(x_rp, dim=1, eps=1e-8)

        # Learnable linear (GRADIENT FLOWS HERE!)
        x_lin = torch.mm(phi, self.W_lin.t())

        # Geometric (fixed)
        if xi is not None:
            x_geo = torch.mm(xi, self.W_geo.t())
        else:
            x_geo = torch.zeros_like(x_lin)

        # Combine
        x_obs = self.alpha * x_rp + self.beta * x_lin + self.gamma * x_geo

        return x_obs


class EndToEndTrainer:
    """End-to-end trainer using differentiable solver.

    Combines:
    - DifferentiableEncoder (W_lin - learnable)
    - DifferentiableSolver (K-step unroll)
    - Fitness-based loss

    Training loop:
        phi -> Encoder(W_lin) -> x_obs -> DiffSolver(K) -> x_final -> Loss
                   ↑                                                    |
                   └────────────────── backward ────────────────────────┘

    Usage:
        >>> trainer = EndToEndTrainer(encoder, solver)
        >>> loss = trainer.train_step(phi, edge_graph)
        >>> # W_lin is updated!
    """

    def __init__(
        self,
        encoder: DifferentiableEncoder,
        solver: DifferentiableSolver,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.encoder = encoder
        self.solver = solver

        # Optimizer for W_lin
        self.optimizer = torch.optim.AdamW(
            encoder.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.step_count = 0
        self.loss_history: List[float] = []

    def compute_fitness_loss(
        self,
        x_final: torch.Tensor,
        x_obs: torch.Tensor,
        edge_graph: EdgeGraph,
    ) -> torch.Tensor:
        """Compute differentiable fitness loss.

        Lower is better (we minimize this).

        Components:
        - Violation: How well constraints are satisfied
        - Drift: How far x_final is from x_obs
        - Collapse: Variance penalty
        """
        # Violation (from solver - already minimized)
        # We want x_final to be a good solution

        # Drift from observation
        drift = torch.mean((x_final - x_obs) ** 2)

        # Anti-collapse
        variance = torch.var(x_final, dim=0).mean()
        collapse = torch.relu(0.01 - variance)

        # Smoothness (within batch)
        if x_final.shape[0] > 1:
            smoothness = torch.mean((x_final[1:] - x_final[:-1]) ** 2)
        else:
            smoothness = torch.tensor(0.0, device=x_final.device)

        # Total loss (lower = better fitness)
        loss = drift + 10.0 * collapse + 0.1 * smoothness

        return loss

    def train_step(
        self,
        phi: torch.Tensor,
        edge_graph: EdgeGraph,
        xi: Optional[torch.Tensor] = None,
        x_prev: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Single training step with end-to-end gradient.

        Args:
            phi: Pair features (m, input_dim)
            edge_graph: Graph structure
            xi: Geometric features (optional)
            x_prev: Previous embeddings (optional)

        Returns:
            Loss breakdown dictionary
        """
        self.encoder.train()
        self.optimizer.zero_grad()

        # Forward: phi -> x_obs -> x_final
        x_obs = self.encoder(phi, xi)
        x_final, solver_info = self.solver(x_obs, edge_graph, x_prev)

        # Compute fitness loss
        loss = self.compute_fitness_loss(x_final, x_obs, edge_graph)

        # Backward: gradient flows to W_lin!
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)

        # Update W_lin
        self.optimizer.step()

        self.step_count += 1
        self.loss_history.append(loss.item())

        return {
            'loss': loss.item(),
            'solver_loss': solver_info['final_loss'],
            'step': self.step_count,
        }

    def evaluate(
        self,
        phi: torch.Tensor,
        edge_graph: EdgeGraph,
        xi: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Evaluate without gradient update."""
        self.encoder.eval()
        with torch.no_grad():
            x_obs = self.encoder(phi, xi)
            x_final, info = self.solver(x_obs, edge_graph)
            loss = self.compute_fitness_loss(x_final, x_obs, edge_graph)

        return x_final, {
            'loss': loss.item(),
            **info,
        }


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

def create_diff_solver(
    K: int = 10,
    step_size: float = 0.01,
    create_graph: bool = True,
) -> DifferentiableSolver:
    """Create differentiable solver.

    Args:
        K: Number of unroll steps
        step_size: Gradient step size
        create_graph: Enable end-to-end gradients

    Returns:
        Configured DifferentiableSolver
    """
    config = DiffSolverConfig(
        K=K,
        step_size=step_size,
        create_graph=create_graph,
    )
    return DifferentiableSolver(config)


def create_diff_encoder(
    input_dim: int = 64,
    output_dim: int = 32,
    alpha: float = 0.3,
    beta: float = 0.5,
    gamma: float = 0.2,
) -> DifferentiableEncoder:
    """Create differentiable encoder."""
    return DifferentiableEncoder(
        input_dim=input_dim,
        output_dim=output_dim,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )


def create_e2e_trainer(
    input_dim: int = 64,
    output_dim: int = 32,
    K: int = 10,
    lr: float = 1e-3,
) -> EndToEndTrainer:
    """Create end-to-end trainer.

    Args:
        input_dim: Pair feature dimension
        output_dim: Embedding dimension
        K: Solver unroll steps
        lr: Learning rate for W_lin

    Returns:
        Configured EndToEndTrainer
    """
    encoder = create_diff_encoder(input_dim, output_dim)
    solver = create_diff_solver(K=K)
    return EndToEndTrainer(encoder, solver, lr=lr)
