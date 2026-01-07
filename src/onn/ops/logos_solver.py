"""
LOGOS (Logical Ontological Generator for Self-Adjustment) Operator

The Correction Operator in the ONN Trinity.
Iteratively projects the graph onto the Constraint Manifold C using PyTorch autograd.

Core Math (from spec/02_onn_math_spec.md Section 2.2):
    S^{k+1} = Π_C ( S^k - η ∇_S L_total(S^k, E^k) )

Loss Function (from spec/02_onn_math_spec.md Section 4.1):
    L_total = λ_data * L_data + λ_phys * L_phys + λ_logic * L_logic

Where:
    L_data  = Σ_i ||S_i - S_i^raw||²           (Data Fidelity)
    L_phys  = Σ_{i≠j} ReLU(Sim(B_i,B_j) - θ)   (Physical Validity)
    L_logic = Σ_{(i,j)∈E} w_ij ||S_i + r_ij - S_j||²  (Logical Consistency / TransE)

Hyperparameters (from spec/02_onn_math_spec.md Section 4.2):
    λ_data = 1.0, λ_phys = 10.0, λ_logic = 2.0

Framework-agnostic: No ROS2, DDS, or external framework dependencies.
Uses PyTorch for automatic differentiation.

Reference:
    - spec/02_onn_math_spec.md
    - spec/10_architecture.ir.yml -> modules[logos_consensus_solver]
    - spec/11_interfaces.ir.yml -> data_schemas[StabilizedGraph]
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import logging
import copy


from onn.core.tensors import (
    SemanticNode,
    SemanticEdge,
    RawSemanticGraph,
    StabilizedGraph,
    BOUND_TENSOR_DIM,
    FORM_TENSOR_DIM,
    INTENT_TENSOR_DIM,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants from spec/02_onn_math_spec.md
# -----------------------------------------------------------------------------

STATE_DIM = BOUND_TENSOR_DIM + FORM_TENSOR_DIM + INTENT_TENSOR_DIM  # 64


# -----------------------------------------------------------------------------
# Configuration (from spec/02_onn_math_spec.md Section 4.2)
# -----------------------------------------------------------------------------


@dataclass
class LOGOSConfig:
    """
    Configuration for the LOGOS solver.

    Hyperparameters from spec/02_onn_math_spec.md Section 4.2:
        λ_data = 1.0   : Trust sensors significantly
        λ_phys = 10.0  : Physics violations are expensive
        λ_logic = 2.0  : Logic guides the structure
    """

    # Loss weights (spec Section 4.2)
    lambda_data: float = 1.0
    lambda_phys: float = 10.0
    lambda_logic: float = 2.0

    # Physical constraint threshold
    overlap_threshold: float = 0.5

    # Solver parameters
    max_iterations: int = 10
    learning_rate: float = 0.01
    tolerance: float = 1.0

    # Backtracking line search (Optional, but solver expects them)
    max_backtracking_steps: int = 6
    backtracking_shrink: float = 0.5
    backtracking_tolerance: float = 1e-4

    # Edge pruning
    edge_prune_threshold: float = 0.1

    # Device
    device: str = "cpu"

    # Autonomous gating parameters (INVARIANT G)
    gate_lr: float = 0.1
    gate_threshold: float = 0.5
    gate_min: float = 0.0
    gate_max: float = 1.0

    # Autonomous uncertainty parameters (INVARIANT U)
    uncertainty_lr: float = 0.05
    uncertainty_min: float = 0.01
    uncertainty_max: float = 10.0
    uncertainty_target_residual: float = 0.1


# -----------------------------------------------------------------------------
# Loss Functions (from spec/02_onn_math_spec.md Section 4.1)
# -----------------------------------------------------------------------------


def compute_obs_residual(state: torch.Tensor, state_raw: torch.Tensor) -> torch.Tensor:
    """
    Observation residual: difference between current state and raw observation.

    r_obs = 2*(S - S_raw)
    This matches ∇(Σ||S - S_raw||²) so that a GD step reduces the data term.
    """
    if state.shape != state_raw.shape:
        raise ValueError(
            f"state shape {state.shape} must match state_raw shape {state_raw.shape} for r_obs"
        )
    return 2.0 * (state - state_raw)


def compute_cons_residual(
    state: torch.Tensor,
    edge_indices: torch.Tensor,
    relation_embeddings: torch.Tensor,
    edge_weights: torch.Tensor,
    config: LOGOSConfig,
) -> torch.Tensor:
    """
    Consistency residual: gradient of physical and logical consistency losses.

    r_cons = ∇(L_phys + L_logic)

    Args:
        state: Current state tensor, shape (N, STATE_DIM)
        ... (and other loss arguments)

    Returns:
        Residual tensor (gradient) of shape (N, STATE_DIM)
    """
    l_phys = compute_phys_loss(state, config.overlap_threshold)
    l_logic = compute_logic_loss(state, edge_indices, relation_embeddings, edge_weights)

    # Autograd computes the gradient of the sum
    consistency_loss = config.lambda_phys * l_phys + config.lambda_logic * l_logic

    if state.grad is not None:
        state.grad.zero_()

    consistency_loss.backward(retain_graph=True)

    if state.grad is None:
        return torch.zeros_like(state)

    return state.grad.clone()


def compute_data_loss(state: torch.Tensor, state_raw: torch.Tensor) -> torch.Tensor:
    """
    Data Fidelity Loss: Keep state close to observation.

    L_data = Σ_i ||S_i - S_i^raw||²

    Reference: spec/02_onn_math_spec.md Section 4.1 Equation (1)

    Args:
        state: Current state tensor, shape (N, 64)
        state_raw: Raw observation state, shape (N, 64)

    Returns:
        Scalar loss tensor
    """
    diff = state - state_raw
    return torch.sum(diff**2)


def compute_phys_loss(
    state: torch.Tensor, overlap_threshold: float = 0.5
) -> torch.Tensor:
    """
    Physical Validity Loss: Penalize invalid overlaps.

    L_phys = Σ_{i≠j} ReLU(Sim(B_i, B_j) - θ_overlap)

    reference: spec/02_onn_math_spec.md Section 4.1 Equation (2)

    Args:
        state: Current state tensor, shape (N, 64)
        overlap_threshold: Unused in CPL_009 (retained for API compatibility)

    Returns:
        Scalar loss tensor
    """
    # CPL_009: Sphere Collision Penalty
    # L_phys = Σ_{i≠j} ReLU(R_i + R_j - ||p_i - p_j||)

    # Extract position (b_0:3) and radius (b_11)
    # Assumes SEGO populates these correctly (verified in CPL_009 SEGO update)
    pos = state[:, 0:3]  # (N, 3)
    # Enforce non-negative, non-zero radius to avoid degenerate collisions
    radius = torch.clamp(torch.abs(state[:, 11]), min=1e-3)  # (N,)

    # Compute pairwise Euclidean distance
    # shape: (N, N)
    dist_matrix = torch.cdist(pos, pos, p=2)

    # Compute sum of radii for all pairs
    # shape: (N, N) via broadcasting
    r_sum_matrix = radius.unsqueeze(1) + radius.unsqueeze(0)

    # Calculate penetration depth: (Ri + Rj) - dist
    # Positive value means collision (overlap)
    penetration = r_sum_matrix - dist_matrix

    # Apply ReLU to get penalty (0 if no collision)
    violation = F.relu(penetration)

    # Zero out diagonal (self-collision is always R+R - 0 > 0, but invalid physically)
    # Self-collision should be ignored as an object always overlaps itself
    n = state.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=state.device)
    violation = violation * mask.float()

    # Sum violations (divide by 2 because (i,j) and (j,i) are symmetric)
    loss = torch.sum(violation) / 2.0

    return loss


def compute_logic_loss(
    state: torch.Tensor,
    edge_indices: torch.Tensor,
    relation_embeddings: torch.Tensor,
    edge_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Logical Consistency Loss: TransE-style embedding loss.

    L_logic = Σ_{(i,j)∈E} w_ij ||S_i + r_ij - S_j||²

    Reference: spec/02_onn_math_spec.md Section 4.1 Equation (3)

    Args:
        state: Current state tensor, shape (N, 64)
        edge_indices: Edge source/target indices, shape (E, 2)
        relation_embeddings: r_ij vectors, shape (E, 64) or (E, 16) padded
        edge_weights: w_ij connection stiffness, shape (E,)

    Returns:
        Scalar loss tensor
    """
    if edge_indices.shape[0] == 0:
        return torch.tensor(0.0, device=state.device)

    source_idx = edge_indices[:, 0]  # (E,)
    target_idx = edge_indices[:, 1]  # (E,)

    # Get source and target states
    s_i = state[source_idx]  # (E, 64)
    s_j = state[target_idx]  # (E, 64)

    # Pad relation embeddings to state dim if needed
    if relation_embeddings.shape[1] < STATE_DIM:
        padding = torch.zeros(
            relation_embeddings.shape[0],
            STATE_DIM - relation_embeddings.shape[1],
            device=relation_embeddings.device,
        )
        relation_embeddings = torch.cat([relation_embeddings, padding], dim=1)

    # TransE: S_i + r_ij ≈ S_j
    translation_error = s_i + relation_embeddings - s_j  # (E, 64)

    # Weighted squared error
    error_norm_sq = torch.sum(translation_error**2, dim=1)  # (E,)
    weighted_error = edge_weights * error_norm_sq  # (E,)

    return torch.sum(weighted_error)


def compute_total_loss(
    state: torch.Tensor,
    state_raw: torch.Tensor,
    edge_indices: torch.Tensor,
    relation_embeddings: torch.Tensor,
    edge_weights: torch.Tensor,
    config: LOGOSConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Total Loss Function for LOGOS.

    L_total = λ_data * L_data + λ_phys * L_phys + λ_logic * L_logic

    Reference: spec/02_onn_math_spec.md Section 4.1

    Args:
        state: Current state tensor, shape (N, 64)
        state_raw: Raw observation state, shape (N, 64)
        edge_indices: Edge source/target indices, shape (E, 2)
        relation_embeddings: r_ij vectors, shape (E, d)
        edge_weights: w_ij connection stiffness, shape (E,)
        config: LOGOS configuration with lambda weights

    Returns:
        Tuple of (total_loss, loss_breakdown_dict)
    """
    l_data = compute_data_loss(state, state_raw)
    l_phys = compute_phys_loss(state, config.overlap_threshold)
    l_logic = compute_logic_loss(state, edge_indices, relation_embeddings, edge_weights)

    total = (
        config.lambda_data * l_data
        + config.lambda_phys * l_phys
        + config.lambda_logic * l_logic
    )

    breakdown = {
        "data": float(l_data.detach()),
        "phys": float(l_phys.detach()),
        "logic": float(l_logic.detach()),
        "total": float(total.detach()),
    }

    return total, breakdown


# -----------------------------------------------------------------------------
# Projection Operator (from spec/02_onn_math_spec.md Section 5.1)
# -----------------------------------------------------------------------------


def project_to_manifold(state: torch.Tensor) -> torch.Tensor:
    """
    Hard Projection Π_C: Project state onto constraint manifold.
    - Clamps radius to be positive.
    - Normalizes bound tensor.
    - Clamps intent tensor to [0, 1].
    """
    bound = state[:, :BOUND_TENSOR_DIM].clone()
    form = state[:, BOUND_TENSOR_DIM : BOUND_TENSOR_DIM + FORM_TENSOR_DIM]
    intent = state[:, BOUND_TENSOR_DIM + FORM_TENSOR_DIM :]

    bound[:, 11] = torch.clamp(torch.abs(bound[:, 11]), min=1e-3)

    bound_norm = torch.norm(bound, p=2, dim=1, keepdim=True).clamp(min=1e-8)
    bound_projected = bound / bound_norm

    intent_projected = torch.clamp(intent, 0.0, 1.0)

    return torch.cat([bound_projected, form, intent_projected], dim=1)


# -----------------------------------------------------------------------------
# LOGOS Solver
# -----------------------------------------------------------------------------


@dataclass
class SolverResult:
    """Result of a LOGOS solve operation."""

    final_state: torch.Tensor
    iterations: int
    converged: bool
    energy_history: List[float]
    residual_norm_history: List[float]
    final_breakdown: Dict[str, float]
    gate_history: Optional[List[torch.Tensor]] = None
    uncertainty_history: Optional[List[torch.Tensor]] = None
    final_gates: Optional[torch.Tensor] = None
    final_uncertainties: Optional[torch.Tensor] = None


class LOGOSSolver:
    """
    The LOGOS Operator implementation using PyTorch autograd.

    Solves for topological validity by iteratively projecting
    the semantic graph onto the constraint manifold.

    Algorithm (from spec/02_onn_math_spec.md Section 5.1):
        1. Initialize S from raw observation
        2. For k in range(max_iterations):
            a. loss = compute_total_energy(S, edges, config)
            b. grads = torch.autograd.grad(loss, S)
            c. S = S - learning_rate * grads
            d. S = project_to_manifold(S)  # Hard constraints
            e. if loss < tolerance: break
        3. Return G_valid(S, edges)

    Hypothesis H-01: Energy monotonically decreases during the loop.

    Reference:
        - spec/02_onn_math_spec.md Section 5.1
        - spec/10_architecture.ir.yml -> modules[logos_consensus_solver]
    """

    def __init__(self, config: Optional[LOGOSConfig] = None):
        """
        Initialize the LOGOS solver.

        Args:
            config: Solver configuration. Uses defaults from spec if None.
        """
        self.config = config or LOGOSConfig()
        self.device = torch.device(self.config.device)
        self._last_result: Optional[SolverResult] = None

        # === CPL_003: Hold last valid graph for safety ===
        self._last_valid_graph: Optional[StabilizedGraph] = None

    def solve(
        self, raw_graph: RawSemanticGraph, warm_start: bool = False
    ) -> StabilizedGraph:
        """
        Solve for a topologically valid graph.

        This is the main entry point for the LOGOS operator.

        Args:
            raw_graph: Unstabilized graph from SEGO
            warm_start: Whether to use previous solution as starting point

        Returns:
            StabilizedGraph with constraint-satisfied nodes and pruned edges

        Raises:
            ValueError: If raw_graph has no nodes

        Reference: spec/10_architecture.ir.yml -> modules[logos_consensus_solver]
        """
        if not raw_graph.nodes:
            raise ValueError("Cannot solve empty graph: no nodes provided")

        state_raw, node_id_map = self._nodes_to_tensor(raw_graph.nodes)
        edge_indices, relation_embeddings, edge_weights = self._edges_to_tensors(
            raw_graph.edge_candidates, node_id_map
        )

        if warm_start and self._last_result is not None:
            state = self._last_result.final_state.clone().detach()
            if state.shape[0] != state_raw.shape[0]:
                state = state_raw.clone()
        else:
            state = state_raw.clone()

        num_nodes = state.shape[0]
        num_edges = edge_indices.shape[0]

        gates = torch.ones(num_edges, device=self.device)
        for i, edge in enumerate(raw_graph.edge_candidates):
            if i < num_edges:
                gates[i] = edge.gate

        uncertainties = torch.full(
            (num_nodes,), self.config.uncertainty_min, device=self.device
        )
        for node in raw_graph.nodes:
            idx = node_id_map.get(node.node_id)
            if idx is not None and idx < num_nodes:
                uncertainties[idx] = max(node.uncertainty, self.config.uncertainty_min)

        with torch.no_grad():
            state = project_to_manifold(state)

        energy_history: List[float] = []
        residual_norm_history: List[float] = []
        gate_history: List[torch.Tensor] = []
        uncertainty_history: List[torch.Tensor] = []
        converged = False
        breakdown: Dict[str, float] = {}
        iteration = 0
        prev_edge_residuals: Optional[torch.Tensor] = None

        for iteration in range(self.config.max_iterations):
            state.requires_grad_(True)

            r_obs = compute_obs_residual(state, state_raw)
            r_cons = compute_cons_residual(
                state, edge_indices, relation_embeddings, edge_weights, self.config
            )

            beta_obs = getattr(self.config, "beta_obs", 1.0)
            beta_cons = getattr(self.config, "beta_cons", 1.0)

            edge_residuals = self._compute_per_edge_residual(
                state, edge_indices, relation_embeddings
            )

            with torch.no_grad():
                if prev_edge_residuals is not None and num_edges > 0:
                    conflict = edge_residuals - prev_edge_residuals
                    gate_delta = -self.config.gate_lr * (
                        conflict + self.config.gate_threshold
                    )
                    gates = torch.clamp(
                        gates + gate_delta, self.config.gate_min, self.config.gate_max
                    )

                prev_edge_residuals = edge_residuals.clone()

                residual_norm = torch.norm(r_obs + r_cons).item()
                uncertainty_delta = self.config.uncertainty_lr * (
                    residual_norm - self.config.uncertainty_target_residual
                )
                uncertainties = torch.clamp(
                    uncertainties + uncertainty_delta,
                    self.config.uncertainty_min,
                    self.config.uncertainty_max,
                )

            gate_history.append(gates.clone())
            uncertainty_history.append(uncertainties.clone())

            if num_edges > 0:
                gate_weights = gates[None, :].expand(num_nodes, -1).mean(dim=1)
            else:
                gate_weights = torch.ones(num_nodes, device=self.device)

            grads = beta_obs * r_obs + beta_cons * r_cons * gate_weights[:, None]

            total_loss, breakdown = compute_total_loss(
                state,
                state_raw,
                edge_indices,
                relation_embeddings,
                edge_weights * gates if num_edges > 0 else edge_weights,
                self.config,
            )
            energy_history.append(breakdown["total"])
            residual_norm_history.append(torch.norm(grads).item())

            if total_loss.item() < self.config.tolerance:
                converged = True
                with torch.no_grad():
                    state = project_to_manifold(state)
                break

            with torch.no_grad():
                eps = 1e-6
                step_scale = self.config.learning_rate / (uncertainties[:, None] + eps)
                updated_state = state - step_scale * grads
                state = project_to_manifold(updated_state)

        if not converged:
            with torch.no_grad():
                state = project_to_manifold(state)

        final_breakdown = breakdown if breakdown else {}
        self._last_result = SolverResult(
            final_state=state.detach(),
            iterations=iteration + 1,
            converged=converged,
            energy_history=energy_history,
            residual_norm_history=residual_norm_history,
            final_breakdown=final_breakdown,
            gate_history=gate_history,
            uncertainty_history=uncertainty_history,
            final_gates=gates.clone() if num_edges > 0 else None,
            final_uncertainties=uncertainties.clone(),
        )

        stabilized_nodes = self._tensor_to_nodes(state.detach(), node_id_map)
        pruned_edges = self._prune_edges(raw_graph.edge_candidates, node_id_map)

        # === CPL_003: Mark validity and iteration count ===
        result_graph = StabilizedGraph(
            timestamp_ns=raw_graph.timestamp_ns,
            nodes=stabilized_nodes,
            edges=pruned_edges,
            global_energy=final_breakdown.get("total", 0.0),
            is_valid=converged,
            iterations_used=iteration + 1,
        )

        # === CPL_003: Cache valid graph for fallback ===
        if converged:
            self._last_valid_graph = result_graph
        else:
            logger.warning(
                f"LOGOS did not converge after {iteration + 1} iterations. "
                f"is_valid=False, final_energy={final_breakdown.get('total', 0.0):.6f}"
            )

        return result_graph

    def _nodes_to_tensor(
        self, nodes: List[SemanticNode]
    ) -> Tuple[torch.Tensor, Dict[int, int]]:
        """
        Convert list of SemanticNodes to a state tensor.

        Returns:
            state: Tensor of shape (N, 64)
            node_id_map: Dict mapping node_id -> tensor row index
        """
        node_id_map = {node.node_id: i for i, node in enumerate(nodes)}

        states = []
        for node in nodes:
            combined = np.concatenate(
                [node.bound_tensor, node.form_tensor, node.intent_tensor]
            )
            states.append(combined)

        state = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        return state, node_id_map

    def _edges_to_tensors(
        self, edges: List[SemanticEdge], node_id_map: Dict[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert list of SemanticEdges to tensors.

        Returns:
            edge_indices: Tensor of shape (E, 2)
            relation_embeddings: Tensor of shape (E, d)
            edge_weights: Tensor of shape (E,)
        """
        if not edges:
            return (
                torch.zeros((0, 2), dtype=torch.long, device=self.device),
                torch.zeros((0, STATE_DIM), dtype=torch.float32, device=self.device),
                torch.zeros((0,), dtype=torch.float32, device=self.device),
            )

        indices = []
        relations = []
        weights = []

        for edge in edges:
            if edge.source_id in node_id_map and edge.target_id in node_id_map:
                indices.append(
                    [node_id_map[edge.source_id], node_id_map[edge.target_id]]
                )
                relations.append(edge.relation_embedding)
                weights.append(edge.effective_strength())

        if not indices:
            return (
                torch.zeros((0, 2), dtype=torch.long, device=self.device),
                torch.zeros((0, STATE_DIM), dtype=torch.float32, device=self.device),
                torch.zeros((0,), dtype=torch.float32, device=self.device),
            )

        edge_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        relation_embeddings = torch.tensor(
            np.array(relations), dtype=torch.float32, device=self.device
        )
        edge_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        return edge_indices, relation_embeddings, edge_weights

    def _tensor_to_nodes(
        self, state: torch.Tensor, node_id_map: Dict[int, int]
    ) -> List[SemanticNode]:
        """Convert state tensor back to SemanticNodes."""
        reverse_map = {v: k for k, v in node_id_map.items()}
        state_np = state.cpu().numpy()

        nodes = []
        for i in range(state.shape[0]):
            node_id = reverse_map[i]
            s = state_np[i]

            node = SemanticNode(
                node_id=node_id,
                bound_tensor=s[:BOUND_TENSOR_DIM].copy(),
                form_tensor=s[
                    BOUND_TENSOR_DIM : BOUND_TENSOR_DIM + FORM_TENSOR_DIM
                ].copy(),
                intent_tensor=s[BOUND_TENSOR_DIM + FORM_TENSOR_DIM :].copy(),
            )
            nodes.append(node)

        return nodes

    def _compute_per_edge_residual(
        self,
        state: torch.Tensor,
        edge_indices: torch.Tensor,
        relation_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-edge TransE residual: ||S_i + r_ij - S_j||."""
        if edge_indices.shape[0] == 0:
            return torch.zeros(0, device=self.device)

        source_idx = edge_indices[:, 0]
        target_idx = edge_indices[:, 1]

        s_i = state[source_idx]
        s_j = state[target_idx]

        if relation_embeddings.shape[1] < STATE_DIM:
            padding = torch.zeros(
                relation_embeddings.shape[0],
                STATE_DIM - relation_embeddings.shape[1],
                device=relation_embeddings.device,
            )
            relation_embeddings = torch.cat([relation_embeddings, padding], dim=1)

        translation_error = s_i + relation_embeddings - s_j
        return torch.norm(translation_error, dim=1)

    def _prune_edges(
        self, edges: List[SemanticEdge], node_id_map: Dict[int, int]
    ) -> List[SemanticEdge]:
        """Prune edges with low effective strength."""
        valid_edges = []
        for edge in edges:
            if edge.source_id not in node_id_map or edge.target_id not in node_id_map:
                continue
            if edge.effective_strength() >= self.config.edge_prune_threshold:
                valid_edges.append(edge.copy())
        return valid_edges

    def get_last_result(self) -> Optional[SolverResult]:
        """Get the result from the last solve call."""
        return self._last_result

    def reset(self) -> None:
        """Reset solver state (clears warm start cache and valid graph cache)."""
        self._last_result = None
        self._last_valid_graph = None

    def solve_multi_hypothesis(
        self, raw_graph: RawSemanticGraph, num_hypotheses: int = 4
    ) -> StabilizedGraph:
        """
        [Sisyphus Implementation] This method was missing and has been implemented
        to satisfy tests and expected functionality. It runs the solver on multiple
        noisy variations of the input graph and returns the most coherent result.
        """
        results = []
        for i in range(num_hypotheses):
            hypothesis_graph = copy.deepcopy(raw_graph)
            if i > 0:
                for node in hypothesis_graph.nodes:
                    noise = (
                        np.random.randn(*node.form_tensor.shape).astype(np.float32)
                        * 0.1
                        * i
                    )
                    node.form_tensor += noise
            result = self.solve(hypothesis_graph, warm_start=False)
            results.append(result)

        if not results:
            raise RuntimeError(
                "Multi-hypothesis solving failed to produce any results."
            )

        best_result = min(results, key=lambda r: r.global_energy)
        return best_result

    def get_last_valid_graph(self) -> Optional[StabilizedGraph]:
        """
        Get the last successfully converged graph.

        CPL_003: Returns the last graph where is_valid=True,
        or None if no valid graph has been computed.
        """
        return self._last_valid_graph


# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------


def create_default_solver() -> LOGOSSolver:
    """
    Create a LOGOS solver with default configuration from spec.

    Uses hyperparameters from spec/02_onn_math_spec.md Section 4.2:
        λ_data = 1.0, λ_phys = 10.0, λ_logic = 2.0
    """
    return LOGOSSolver(LOGOSConfig())


def create_realtime_solver() -> LOGOSSolver:
    """
    Create a LOGOS solver optimized for real-time operation.

    Uses fixed iteration budget (5) as mentioned in spec/00_high_level_plan.md.
    """
    config = LOGOSConfig(
        max_iterations=5,
        learning_rate=0.1,  # Larger step for faster convergence
    )
    return LOGOSSolver(config)
