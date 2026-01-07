"""Projection-Consensus Solver for Edge Embeddings.

This module implements the main ONN solver that stabilizes edge embeddings
using alternating gradient descent and constraint projection.

Algorithm (K-step Projection-Consensus):
    for k = 1 to K:
        x ← x - η ∇_x L_total(x)    # Gradient step
        x ← P_C^(α)(x)               # Relaxed projection

The solver ensures:
1. Data fidelity: x stays close to observations
2. Cycle consistency: Cx ≈ τ (enforced by projection)
3. Structural regularity: Ricci energy bounded
4. Temporal stability: x_t close to x_{t-1}
5. Non-collapse: embedding variance maintained

Reference:
    - spec/11_interfaces.ir.yml: ONNSolverConfig
    - spec/20_impl_plan.ir.yml: IMPL_018
    - User roadmap: "Projection-Consensus loop"

Author: Claude (via IMPL_018)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from onn.core.graph import EdgeGraph
from onn.core.cycles import CycleBasis, build_cycle_basis
from onn.core.projection import relaxed_projection
from onn.core.losses import LossConfig, total_loss, gradient_step

logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class PCSolverConfig:
    """Projection-Consensus solver hyperparameters.

    Reference: spec/11_interfaces.ir.yml -> ONNSolverConfig
    """
    # Solver parameters
    step_size: float = 0.01         # η: gradient step size
    steps: int = 10                 # K: inner loop iterations
    projection_alpha: float = 1.0   # α: relaxed projection strength

    # Loss weights (forwarded to LossConfig)
    lambda_context: float = 1.0     # Context constraint weight
    lambda_ricci: float = 0.1       # Ricci curvature weight
    lambda_smooth: float = 0.1      # Temporal smoothness weight
    lambda_var: float = 1.0         # Anti-collapse variance weight

    # Anti-collapse
    min_variance: float = 0.01      # Minimum variance threshold

    # Gradient control
    max_grad_norm: Optional[float] = 1.0  # Gradient clipping (None = no clipping)

    # Delta Update (DDL paper integration)
    use_delta_update: bool = False  # Use delta ODE instead of Euler
    delta_beta: float = 1.0         # β ∈ [0, 2]: 0=identity, 1=projection, 2=reflection

    # Convergence
    tolerance: float = 1e-6         # Stop if loss change < tolerance

    # Device
    device: str = "cpu"
    
    def to_loss_config(self) -> LossConfig:
        """Convert to LossConfig for loss computation."""
        return LossConfig(
            lambda_context=self.lambda_context,
            lambda_ricci=self.lambda_ricci,
            lambda_smooth=self.lambda_smooth,
            lambda_var=self.lambda_var,
            min_variance=self.min_variance,
        )


@dataclass
class SolverResult:
    """Result of a solve() call.
    
    Contains the final embeddings and diagnostic information.
    """
    x: torch.Tensor                 # Final embeddings (m, p)
    converged: bool                 # Whether solver converged
    iterations_used: int            # Number of iterations performed
    final_loss: float               # Final total loss
    loss_history: List[float] = field(default_factory=list)
    violation_history: List[float] = field(default_factory=list)
    breakdown: Dict[str, float] = field(default_factory=dict)


# ==============================================================================
# SOLVER CLASS
# ==============================================================================

class ProjectionConsensusSolver:
    """K-step solver with gradient descent + relaxed projection.
    
    This is the core ONN solver that stabilizes edge embeddings by:
    1. Gradient descent on the total loss
    2. Relaxed projection onto the constraint set
    
    Example:
        >>> solver = ProjectionConsensusSolver(config)
        >>> result = solver.solve(x_obs, edge_graph)
        >>> print(f"Converged: {result.converged}, Loss: {result.final_loss}")
    
    Reference:
        spec/20_impl_plan.ir.yml: IMPL_018
    """
    
    def __init__(self, config: Optional[PCSolverConfig] = None):
        """Initialize the solver.
        
        Args:
            config: Solver configuration. Uses defaults if None.
        """
        self.config = config or PCSolverConfig()
        self.device = torch.device(self.config.device)
        
        # Cache for warm starting
        self._last_result: Optional[SolverResult] = None
        self._last_cycle_basis: Optional[CycleBasis] = None
        
        logger.debug(f"Initialized PC solver with K={self.config.steps}, η={self.config.step_size}")
    
    def step(
        self,
        x: torch.Tensor,
        x_obs: torch.Tensor,
        cycle_basis: CycleBasis,
        edge_graph: EdgeGraph,
        x_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Single PC step: gradient descent followed by projection.
        
        Args:
            x: Current embeddings of shape (m, p)
            x_obs: Observed embeddings of shape (m, p)
            cycle_basis: Cycle constraint basis
            edge_graph: Graph structure for Ricci
            x_prev: Previous embeddings for smoothness (optional)
            
        Returns:
            Tuple of (updated_x, loss_breakdown)
        """
        # Ensure x has gradients
        x_var = x.clone().requires_grad_(True)
        
        # Compute loss
        loss_config = self.config.to_loss_config()
        loss, breakdown = total_loss(
            x_var, x_obs, cycle_basis, edge_graph, x_prev, loss_config
        )
        
        # Compute gradient
        grad = torch.autograd.grad(loss, x_var, create_graph=False)[0]
        
        # Gradient step (Euler or Delta ODE)
        x_new = gradient_step(
            x, grad,
            step_size=self.config.step_size,
            max_grad_norm=self.config.max_grad_norm,
            use_delta=self.config.use_delta_update,
            delta_beta=self.config.delta_beta,
        )
        
        # Relaxed projection
        x_proj = relaxed_projection(
            x_new,
            cycle_basis.cycle_matrix,
            cycle_basis.tau,
            alpha=self.config.projection_alpha
        )
        
        return x_proj, breakdown
    
    def solve(
        self,
        x_obs: torch.Tensor,
        edge_graph: EdgeGraph,
        x_prev: Optional[torch.Tensor] = None,
        warm_start: bool = False,
    ) -> SolverResult:
        """Full solve: K steps of PC iteration.
        
        Args:
            x_obs: Observed embeddings of shape (m, p)
            edge_graph: Graph structure
            x_prev: Previous embeddings for smoothness (optional)
            warm_start: If True, start from last solution
            
        Returns:
            SolverResult with final embeddings and diagnostics
            
        Raises:
            ValueError: If x_obs is empty or malformed
        """
        m, p = x_obs.shape
        
        # Validation
        if m == 0:
            raise ValueError("Cannot solve: empty embedding tensor")
        if m != edge_graph.num_edges:
            raise ValueError(f"x_obs has {m} edges but graph has {edge_graph.num_edges}")
        
        # Move to device
        x_obs = x_obs.to(self.device)
        if x_prev is not None:
            x_prev = x_prev.to(self.device)
        
        # Build or reuse cycle basis
        cycle_basis = build_cycle_basis(edge_graph, embedding_dim=p)
        cycle_basis.cycle_matrix = cycle_basis.cycle_matrix.to(self.device)
        cycle_basis.tau = cycle_basis.tau.to(self.device)
        self._last_cycle_basis = cycle_basis
        
        logger.debug(f"Solving: {m} edges, {cycle_basis.num_cycles} cycles, K={self.config.steps}")
        
        # Initialize x
        if warm_start and self._last_result is not None:
            x = self._last_result.x.clone()
            if x.shape != x_obs.shape:
                logger.warning("Warm start shape mismatch, using x_obs")
                x = x_obs.clone()
        else:
            x = x_obs.clone()
        
        # Tracking
        loss_history = []
        violation_history = []
        converged = False
        prev_loss = float('inf')
        
        # Main loop
        for k in range(self.config.steps):
            x, breakdown = self.step(x, x_obs, cycle_basis, edge_graph, x_prev)
            
            current_loss = breakdown["total"]
            loss_history.append(current_loss)
            
            # Track constraint violation
            violation = breakdown["context"]
            violation_history.append(violation)
            
            # Convergence check
            loss_change = abs(prev_loss - current_loss)
            if loss_change < self.config.tolerance:
                logger.debug(f"Converged at step {k+1}: Δloss = {loss_change:.2e}")
                converged = True
                break
            
            prev_loss = current_loss
        
        # Final result
        result = SolverResult(
            x=x.detach(),
            converged=converged,
            iterations_used=len(loss_history),
            final_loss=loss_history[-1] if loss_history else 0.0,
            loss_history=loss_history,
            violation_history=violation_history,
            breakdown=breakdown,
        )
        
        self._last_result = result
        
        logger.info(
            f"Solve complete: {result.iterations_used} iters, "
            f"loss={result.final_loss:.4f}, converged={converged}"
        )
        
        return result
    
    def reset(self) -> None:
        """Reset solver state (clears warm start cache)."""
        self._last_result = None
        self._last_cycle_basis = None
    
    def get_last_result(self) -> Optional[SolverResult]:
        """Get the last solve result."""
        return self._last_result
    
    def get_last_cycle_basis(self) -> Optional[CycleBasis]:
        """Get the last computed cycle basis."""
        return self._last_cycle_basis


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

def create_default_pc_solver() -> ProjectionConsensusSolver:
    """Create a PC solver with default configuration.
    
    Returns:
        Configured ProjectionConsensusSolver
    """
    return ProjectionConsensusSolver(PCSolverConfig())


def create_solver_from_dict(params: Dict[str, float]) -> ProjectionConsensusSolver:
    """Create a solver from a parameter dictionary (for ES).

    Args:
        params: Dictionary with keys like 'step_size', 'steps', etc.
                Supports delta update params: 'use_delta_update', 'delta_beta'

    Returns:
        Configured ProjectionConsensusSolver
    """
    config = PCSolverConfig(
        step_size=params.get('step_size', 0.01),
        steps=int(params.get('steps', 10)),
        projection_alpha=params.get('projection_alpha', 1.0),
        lambda_context=params.get('lambda_context', 1.0),
        lambda_ricci=params.get('lambda_ricci', 0.1),
        lambda_smooth=params.get('lambda_smooth', 0.1),
        lambda_var=params.get('lambda_var', 1.0),
        # Delta update options
        use_delta_update=bool(params.get('use_delta_update', False)),
        delta_beta=params.get('delta_beta', 1.0),
    )
    return ProjectionConsensusSolver(config)


# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================

def analyze_convergence(result: SolverResult) -> Dict[str, float]:
    """Analyze convergence behavior from a solver result.
    
    Args:
        result: SolverResult from solve()
        
    Returns:
        Dictionary with convergence metrics
    """
    if not result.loss_history:
        return {"error": "No loss history"}
    
    losses = result.loss_history
    violations = result.violation_history
    
    metrics = {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_reduction": losses[0] - losses[-1] if len(losses) > 1 else 0,
        "loss_reduction_ratio": (losses[0] - losses[-1]) / (losses[0] + 1e-8) if len(losses) > 1 else 0,
        "initial_violation": violations[0] if violations else 0,
        "final_violation": violations[-1] if violations else 0,
        "converged": float(result.converged),
        "iterations": result.iterations_used,
    }
    
    # Loss decrease monotonicity
    if len(losses) > 1:
        decreases = sum(1 for i in range(1, len(losses)) if losses[i] < losses[i-1])
        metrics["monotonic_ratio"] = decreases / (len(losses) - 1)
    
    return metrics
