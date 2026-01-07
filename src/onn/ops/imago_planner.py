"""
IMAGO (Intent Modeling & Action Generation Operator) Planner.

The Planning Operator in the ONN Trinity.
Generates a ReasoningTrace by analyzing the topology of the valid graph
and identifying intent flow paths using Forman-Ricci curvature.

Core Math (from spec/02_onn_math_spec.md Section 2.3):
    R_IMAGO: G_valid × Goal → T (Reasoning Trace)

    Uses Forman-Ricci Curvature:
    F_e(v_i, v_j) = 4 - d_i - d_j + 3 * Δ_tris

    Intent flows along geodesics defined by this curvature.

Reference:
    - spec/02_onn_math_spec.md Section 2.3
    - spec/20_impl_plan.ir.yml IMPL_008
    - spec/11_interfaces.ir.yml -> ReasoningTrace

Framework-agnostic: No ROS2, DDS, or external framework dependencies.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import logging
import time

from onn.core.tensors import (
    SemanticNode,
    SemanticEdge,
    StabilizedGraph,
    ReasoningTrace,
    MissionGoal,
    BOUND_TENSOR_DIM,
    FORM_TENSOR_DIM,
    INTENT_TENSOR_DIM,
)
from onn.core.graph_utils import (
    node_lookup,
    get_edges_for_node,
    build_adjacency_set,
)
from onn.modules.curvature import (
    graph_average_curvature,
    forman_ricci_curvature,
    forman_ricci_node_curvature,
    identify_functional_clusters,
    curvature_gradient_field,
    CurvatureCluster,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class IMAGOConfig:
    """
    Configuration for the IMAGO planner.

    Reference: spec/01_constraints.md Section 3.4
    """
    # Trace horizon (lookahead time in seconds)
    trace_horizon_sec: float = 1.0

    # Valid duration for generated traces (nanoseconds)
    trace_validity_ns: int = 500_000_000  # 500ms

    # Trajectory interpolation order (polynomial degree)
    trajectory_order: int = 3  # Cubic spline

    # Curvature-based planning
    curvature_threshold: float = 0.0  # For cluster identification
    min_cluster_size: int = 2

    # Velocity clamping (spec Section 5.2)
    # ||v|| ≤ 1 / (γ_ORTSF * Δt_latency)
    max_velocity_norm: float = 10.0

    # Intent flow parameters
    intent_decay_rate: float = 0.9  # Per step
    max_path_length: int = 10  # Max nodes in path
    
    # === CPL_005: Configurable intent blend/scaling factors ===
    goal_intent_blend_alpha: float = 0.7  # Weight for goal intent vs current intent
    verb_length_scale_divisor: float = 10.0  # Divisor for verb length scaling
    min_intent_activation: float = 0.5  # Minimum activation for first dimension


# =============================================================================
# IMAGO Planner
# =============================================================================

class IMAGOPlanner:
    """
    The IMAGO Operator implementation.

    Generates ReasoningTrace from StabilizedGraph and MissionGoal by:
    1. Computing Forman-Ricci curvature to identify structure
    2. Finding functional clusters related to the goal
    3. Computing intent flow geodesics
    4. Generating trajectory coefficients for smooth motion

    Reference:
        - spec/02_onn_math_spec.md Section 2.3
        - spec/20_impl_plan.ir.yml IMPL_008
    """

    def __init__(self, config: Optional[IMAGOConfig] = None):
        """
        Initialize the IMAGO planner.

        Args:
            config: Planner configuration. Uses defaults if None.
        """
        self.config = config or IMAGOConfig()
        self._last_trace: Optional[ReasoningTrace] = None
        self._curvature_cache: Dict[Tuple[int, int], float] = {}

    def plan(
        self,
        graph: StabilizedGraph,
        goal: MissionGoal
    ) -> ReasoningTrace:
        """
        Generate a ReasoningTrace from a stabilized graph and mission goal.

        This is the main entry point for the IMAGO operator.

        Args:
            graph: Topologically valid graph from LOGOS
            goal: High-level mission intent

        Returns:
            ReasoningTrace with target state and trajectory

        Raises:
            ValueError: If graph.is_valid is False (CPL_003 compliance)

        Reference: spec/02_onn_math_spec.md Section 2.3
        """
        # === CPL_003: Gate on LOGOS convergence ===
        if not graph.is_valid:
            raise ValueError(
                f"IMAGO received invalid graph (is_valid=False). "
                f"LOGOS did not converge after {graph.iterations_used} iterations. "
                f"Cannot plan on unstable topology. Consider using get_last_valid_graph()."
            )

        logger.debug(f"Planning for goal: {goal.verb} on node {goal.target_node_id}")

        # Step 1: Compute curvature landscape
        edge_curvatures = forman_ricci_curvature(graph)
        node_curvatures = forman_ricci_node_curvature(graph, edge_curvatures)
        avg_curvature = graph_average_curvature(graph)
        self._curvature_cache = edge_curvatures

        logger.debug(f"Graph average curvature: {avg_curvature:.4f}")

        # Step 2: Identify functional clusters
        clusters = identify_functional_clusters(
            graph,
            curvature_threshold=self.config.curvature_threshold,
            min_cluster_size=self.config.min_cluster_size
        )
        logger.debug(f"Found {len(clusters)} functional clusters")

        # Step 3: Select target state based on goal
        target_state = self._select_target_state(graph, goal, clusters)

        # Step 4: Compute trajectory coefficients
        trajectory_coeffs = self._compute_trajectory(graph, target_state, goal)

        # Step 5: Build the trace
        current_time_ns = time.time_ns()
        valid_until_ns = current_time_ns + self.config.trace_validity_ns

        trace = ReasoningTrace(
            timestamp_ns=current_time_ns,
            target_state=target_state,
            trajectory_coeffs=trajectory_coeffs,
            curvature=avg_curvature,
            valid_until_ns=valid_until_ns
        )

        self._last_trace = trace
        logger.info(f"Generated trace with {len(target_state)} target nodes, "
                    f"valid for {self.config.trace_validity_ns / 1e6:.0f}ms")

        return trace

    def _select_target_state(
        self,
        graph: StabilizedGraph,
        goal: MissionGoal,
        clusters: List[CurvatureCluster]
    ) -> List[SemanticNode]:
        """
        Select target state nodes based on goal and curvature analysis.

        The target state represents the desired future configuration
        that achieves the mission goal.

        Args:
            graph: Current stabilized graph
            goal: Mission goal
            clusters: Identified functional clusters

        Returns:
            List of SemanticNode representing target state
        """
        node_dict = node_lookup(graph)
        target_nodes: List[SemanticNode] = []

        # Find the goal's target node
        target_node_id = goal.target_node_id
        target_node = node_dict.get(target_node_id)

        if target_node is None:
            logger.warning(f"Target node {target_node_id} not found in graph")
            # Return current state as target
            return [node.copy() for node in graph.nodes]

        # Find cluster containing target node
        target_cluster = None
        for cluster in clusters:
            if target_node_id in cluster.node_ids:
                target_cluster = cluster
                break

        # Select nodes for target state
        if target_cluster:
            # Include all nodes in the target cluster
            for node_id in target_cluster.node_ids:
                if node_id in node_dict:
                    modified_node = self._apply_goal_intent(
                        node_dict[node_id], goal
                    )
                    target_nodes.append(modified_node)
        else:
            # Just use the target node and its neighbors
            target_nodes.append(self._apply_goal_intent(target_node, goal))

            # Add connected nodes
            for edge in get_edges_for_node(graph, target_node_id):
                neighbor_id = (
                    edge.target_id if edge.source_id == target_node_id
                    else edge.source_id
                )
                if neighbor_id in node_dict:
                    target_nodes.append(node_dict[neighbor_id].copy())

        if not target_nodes:
            # Fallback: return copy of all nodes
            target_nodes = [node.copy() for node in graph.nodes]

        return target_nodes

    def _apply_goal_intent(
        self,
        node: SemanticNode,
        goal: MissionGoal
    ) -> SemanticNode:
        """
        Modify a node's intent tensor based on mission goal.

        Args:
            node: Source node to modify
            goal: Mission goal defining desired intent

        Returns:
            New SemanticNode with modified intent tensor
        """
        modified = node.copy()

        # Encode goal verb into intent tensor
        intent_encoding = self._encode_goal_verb(goal.verb)

        # CPL_005: Use configurable blend alpha
        alpha = self.config.goal_intent_blend_alpha
        modified.intent_tensor = (
            alpha * intent_encoding +
            (1 - alpha) * modified.intent_tensor
        ).astype(np.float32)

        # Clamp to [0, 1]
        modified.intent_tensor = np.clip(modified.intent_tensor, 0.0, 1.0)

        return modified

    def _encode_goal_verb(self, verb: str) -> np.ndarray:
        """
        Encode a goal verb into an intent tensor using CONTINUOUS encoding.

        This uses hash-based embedding instead of discrete lookup tables
        to satisfy spec/01_constraints.md (no discrete logic in critical path).

        Encoding strategy:
        - Convert verb string to deterministic feature vector using character-level encoding
        - This produces consistent embeddings without explicit verb dictionaries
        - Future: replace with learned embedding lookup from neural network

        Args:
            verb: Goal verb string (e.g., 'POUR', 'GRASP', 'MONITOR')

        Returns:
            Intent tensor encoding the verb (continuous, not discrete)

        Reference: spec/01_constraints.md Section 2.2 (No Discrete Logic in Critical Paths)
        """
        intent = np.zeros(INTENT_TENSOR_DIM, dtype=np.float32)

        # === CONTINUOUS ENCODING (no verb lookup table) ===

        # 1. Character-level hash embedding
        # This produces deterministic but continuous embeddings for any verb
        verb_normalized = verb.upper().strip()

        if len(verb_normalized) == 0:
            return intent

        # Create a pseudo-random but deterministic embedding from the verb string
        # This mimics learned embeddings without discrete branching
        char_values = [ord(c) for c in verb_normalized]

        for i, char_val in enumerate(char_values):
            # Distribute character information across intent dimensions
            for dim in range(INTENT_TENSOR_DIM):
                # Use trigonometric functions for smooth, continuous distribution
                phase = (char_val * (i + 1) * (dim + 1)) / 1000.0
                intent[dim] += np.sin(phase) * 0.3 + np.cos(phase * 1.5) * 0.2

        # 2. Normalize to [0, 1] range
        intent = (intent - intent.min()) / (max(intent.max() - intent.min(), 1e-8))

        # 3. CPL_005: Apply verb-length based scaling using config
        length_scale = min(len(verb_normalized) / self.config.verb_length_scale_divisor, 1.0)
        intent = intent * (0.5 + 0.5 * length_scale)

        # 4. CPL_005: Ensure first dimension has configurable base activation
        intent[0] = max(intent[0], self.config.min_intent_activation)

        return intent.astype(np.float32)

    def _compute_trajectory(
        self,
        graph: StabilizedGraph,
        target_state: List[SemanticNode],
        goal: MissionGoal
    ) -> np.ndarray:
        """
        Compute trajectory coefficients for smooth motion planning.

        Uses curvature gradient field to define geodesic flow.

        Args:
            graph: Current stabilized graph
            target_state: Desired target state
            goal: Mission goal

        Returns:
            Trajectory coefficients (polynomial/spline)
        """
        # Get curvature gradient field
        gradients = curvature_gradient_field(graph)

        # Build trajectory as polynomial coefficients
        # For V0: simple cubic spline with 4 coefficients per dimension
        order = self.config.trajectory_order
        n_coeffs_per_dim = order + 1  # Cubic = 4 coefficients

        # Number of controlled dimensions (bound tensor for position)
        n_dims = BOUND_TENSOR_DIM

        # Initialize coefficients
        # Shape: (n_dims, n_coeffs_per_dim)
        coeffs = np.zeros((n_dims, n_coeffs_per_dim), dtype=np.float32)

        # Get current and target bound tensors
        node_dict = node_lookup(graph)
        target_node_id = goal.target_node_id

        current_bound = None
        target_bound = None

        if target_node_id in node_dict:
            current_bound = node_dict[target_node_id].bound_tensor

        for node in target_state:
            if node.node_id == target_node_id:
                target_bound = node.bound_tensor
                break

        if current_bound is None or target_bound is None:
            # Fallback: return zeros
            return coeffs.flatten()

        # Simple cubic interpolation: p(t) = a + b*t + c*t² + d*t³
        # Boundary conditions: p(0) = current, p(1) = target, p'(0) = v0, p'(1) = v1
        # For simplicity, assume zero velocity at endpoints

        for dim in range(n_dims):
            p0 = current_bound[dim]
            p1 = target_bound[dim]
            v0 = 0.0  # Start velocity
            v1 = 0.0  # End velocity

            # Hermite interpolation coefficients
            # a = p0
            # b = v0
            # c = 3(p1-p0) - 2*v0 - v1
            # d = 2(p0-p1) + v0 + v1

            a = p0
            b = v0
            c = 3 * (p1 - p0) - 2 * v0 - v1
            d = 2 * (p0 - p1) + v0 + v1

            coeffs[dim] = [a, b, c, d]

        # Clamp velocity to satisfy small-gain theorem
        # Estimate max velocity from coefficients
        max_vel = self._estimate_max_velocity(coeffs)
        if max_vel > self.config.max_velocity_norm:
            scale = self.config.max_velocity_norm / max_vel
            coeffs[:, 1:] *= scale  # Scale derivatives

        return coeffs.flatten()

    def _estimate_max_velocity(self, coeffs: np.ndarray) -> float:
        """
        Estimate maximum velocity magnitude from trajectory coefficients.

        For cubic: p(t) = a + b*t + c*t² + d*t³
        Velocity: v(t) = b + 2*c*t + 3*d*t²

        Args:
            coeffs: Trajectory coefficients, shape (n_dims, 4)

        Returns:
            Estimated maximum velocity magnitude
        """
        # Sample velocity at several points
        max_vel = 0.0

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            vel = np.zeros(coeffs.shape[0])
            for dim in range(coeffs.shape[0]):
                a, b, c, d = coeffs[dim]
                # v(t) = b + 2*c*t + 3*d*t²
                vel[dim] = b + 2 * c * t + 3 * d * t * t

            vel_mag = np.linalg.norm(vel)
            max_vel = max(max_vel, vel_mag)

        return max_vel

    def _build_trace(
        self,
        target_state: List[SemanticNode],
        trajectory_coeffs: np.ndarray,
        curvature: float
    ) -> ReasoningTrace:
        """
        Build a ReasoningTrace from computed components.

        Args:
            target_state: Target node configuration
            trajectory_coeffs: Polynomial coefficients
            curvature: Graph average curvature

        Returns:
            Complete ReasoningTrace
        """
        current_time_ns = time.time_ns()

        return ReasoningTrace(
            timestamp_ns=current_time_ns,
            target_state=target_state,
            trajectory_coeffs=trajectory_coeffs,
            curvature=curvature,
            valid_until_ns=current_time_ns + self.config.trace_validity_ns
        )

    def get_last_trace(self) -> Optional[ReasoningTrace]:
        """Get the most recently generated trace."""
        return self._last_trace

    def get_curvature_cache(self) -> Dict[Tuple[int, int], float]:
        """Get cached edge curvatures from last planning."""
        return self._curvature_cache


# =============================================================================
# Factory Functions
# =============================================================================

def create_default_imago_planner() -> IMAGOPlanner:
    """
    Create IMAGO planner with default configuration.

    Returns:
        IMAGOPlanner with spec-compliant defaults
    """
    return IMAGOPlanner(IMAGOConfig())


def create_realtime_imago_planner() -> IMAGOPlanner:
    """
    Create IMAGO planner optimized for real-time operation.

    Uses shorter trace validity and simpler trajectory computation.
    """
    config = IMAGOConfig(
        trace_validity_ns=200_000_000,  # 200ms
        trajectory_order=2,  # Quadratic
        max_path_length=5,
    )
    return IMAGOPlanner(config)


# =============================================================================
# Utility Functions
# =============================================================================

def interpolate_trace(
    trace: ReasoningTrace,
    t: float
) -> np.ndarray:
    """
    Interpolate trajectory at time t ∈ [0, 1].

    Args:
        trace: ReasoningTrace with trajectory coefficients
        t: Normalized time (0 = start, 1 = end)

    Returns:
        Interpolated bound tensor
    """
    if trace.trajectory_coeffs.size == 0:
        return np.zeros(BOUND_TENSOR_DIM, dtype=np.float32)

    # Reshape coefficients
    n_coeffs = 4  # Cubic
    coeffs = trace.trajectory_coeffs.reshape(-1, n_coeffs)

    result = np.zeros(coeffs.shape[0], dtype=np.float32)
    for dim in range(coeffs.shape[0]):
        a, b, c, d = coeffs[dim]
        result[dim] = a + b * t + c * t * t + d * t * t * t

    return result


def validate_trace(trace: ReasoningTrace, current_time_ns: int) -> bool:
    """
    Check if a trace is valid at the given time.

    Args:
        trace: ReasoningTrace to validate
        current_time_ns: Current time in nanoseconds

    Returns:
        True if trace is still valid, False otherwise
    """
    return trace.valid_until_ns > current_time_ns
