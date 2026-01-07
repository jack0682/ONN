"""
ONN Core Tensor Data Structures

Implements the mathematical tensor objects defined in spec/11_interfaces.ir.yml:
- SemanticNode: A node S_i on the Semantic Manifold (Bound, Form, Intent fibers)
- SemanticEdge: A relationship E_ij between two nodes

Framework-agnostic: No ROS2, DDS, or external framework dependencies.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


# -----------------------------------------------------------------------------
# Constants from spec/11_interfaces.ir.yml
# -----------------------------------------------------------------------------

BOUND_TENSOR_DIM = 16  # B_i: physical occupancy and collision boundaries
FORM_TENSOR_DIM = 32  # F_i: visual identity and invariant appearance
INTENT_TENSOR_DIM = 16  # I_i: functional affordances and task relevance


# -----------------------------------------------------------------------------
# SemanticNode
# -----------------------------------------------------------------------------


@dataclass
class SemanticNode:
    """A node on the Semantic Manifold."""

    node_id: int
    bound_tensor: np.ndarray = field(
        default_factory=lambda: np.zeros(BOUND_TENSOR_DIM, dtype=np.float32)
    )
    form_tensor: np.ndarray = field(
        default_factory=lambda: np.zeros(FORM_TENSOR_DIM, dtype=np.float32)
    )
    intent_tensor: np.ndarray = field(
        default_factory=lambda: np.zeros(INTENT_TENSOR_DIM, dtype=np.float32)
    )
    uncertainty: float = 0.0

    def __post_init__(self):
        self.bound_tensor = self._validate_tensor(
            self.bound_tensor, BOUND_TENSOR_DIM, "bound_tensor"
        )
        self.form_tensor = self._validate_tensor(
            self.form_tensor, FORM_TENSOR_DIM, "form_tensor"
        )
        self.intent_tensor = self._validate_tensor(
            self.intent_tensor, INTENT_TENSOR_DIM, "intent_tensor"
        )
        if self.uncertainty < 0:
            raise ValueError("Uncertainty must be non-negative.")

    @staticmethod
    def _validate_tensor(
        tensor: np.ndarray, expected_dim: int, name: str
    ) -> np.ndarray:
        arr = np.asarray(tensor, dtype=np.float32)
        if arr.shape != (expected_dim,):
            raise ValueError(
                f"{name} must have shape ({expected_dim},), got {arr.shape}"
            )
        return arr

    @staticmethod
    def _validate_uncertainty(value: float) -> float:
        if value < 0.0:
            raise ValueError("uncertainty must be >= 0")
        return float(value)

    def combined_state(self) -> np.ndarray:
        """
        Return the full state vector [B, F, I] concatenated.
        Total dimension: 16 + 32 + 16 = 64
        """
        return np.concatenate([self.bound_tensor, self.form_tensor, self.intent_tensor])

    def copy(self) -> SemanticNode:
        """Create a deep copy of this node."""
        return SemanticNode(
            node_id=self.node_id,
            bound_tensor=self.bound_tensor.copy(),
            form_tensor=self.form_tensor.copy(),
            intent_tensor=self.intent_tensor.copy(),
            uncertainty=self.uncertainty,
        )

    def distance_to(self, other: SemanticNode) -> float:
        """
        Compute L2 distance between this node's state and another's.
        Used for constraint energy calculations.
        """
        return float(np.linalg.norm(self.combined_state() - other.combined_state()))


# -----------------------------------------------------------------------------
# SemanticEdge
# -----------------------------------------------------------------------------


@dataclass
class SemanticEdge:
    """A relationship between two nodes."""

    source_id: int
    target_id: int
    relation_embedding: np.ndarray
    weight: float = 1.0
    probability: float = 1.0
    gate: float = 1.0

    def __post_init__(self) -> None:
        self.relation_embedding = np.asarray(self.relation_embedding, dtype=np.float32)
        if self.weight < 0.0:
            raise ValueError(f"Edge weight must be >= 0, got {self.weight}")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(
                f"Edge probability must be in [0, 1], got {self.probability}"
            )
        if not 0.0 <= self.gate <= 1.0:
            raise ValueError(f"Edge gate must be in [0, 1], got {self.gate}")

    @property
    def edge_key(self) -> tuple[int, int]:
        return (self.source_id, self.target_id)

    def effective_strength(self) -> float:
        return self.weight * self.probability * self.gate

    def copy(self) -> SemanticEdge:
        return SemanticEdge(
            source_id=self.source_id,
            target_id=self.target_id,
            relation_embedding=self.relation_embedding.copy(),
            weight=self.weight,
            probability=self.probability,
            gate=self.gate,
        )


# -----------------------------------------------------------------------------
# Graph Containers (from spec/11_interfaces.ir.yml)
# -----------------------------------------------------------------------------


@dataclass
class RawSemanticGraph:
    """
    G_raw: The output of SEGO operator.

    A graph that may contain topological contradictions before
    the LOGOS solver runs.

    Reference: spec/11_interfaces.ir.yml -> data_schemas -> RawSemanticGraph
    """

    timestamp_ns: int  # Acquisition time in nanoseconds
    nodes: List[SemanticNode] = field(default_factory=list)
    edge_candidates: List[SemanticEdge] = field(default_factory=list)

    def get_node(self, node_id: int) -> Optional[SemanticNode]:
        """Retrieve node by ID, or None if not found."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def node_ids(self) -> List[int]:
        """Return list of all node IDs."""
        return [n.node_id for n in self.nodes]


@dataclass
class StabilizedGraph:
    """
    G_valid: The output of LOGOS operator.

    A graph where S and E satisfy the manifold constraints C.
    This is a topologically valid graph after the Projection-Consensus solver.

    Reference: spec/11_interfaces.ir.yml -> data_schemas -> StabilizedGraph
    """

    timestamp_ns: int  # Time of stabilization
    nodes: List[SemanticNode] = field(default_factory=list)
    edges: List[SemanticEdge] = field(
        default_factory=list
    )  # Validated edges (low-weight pruned)
    global_energy: float = 0.0  # Residual energy of the system (lower is better)

    # === CPL_003: Convergence gating for IMAGO ===
    is_valid: bool = (
        True  # True if LOGOS converged, False if max_iter hit without convergence
    )
    iterations_used: int = 0  # Number of solver iterations used

    def get_node(self, node_id: int) -> Optional[SemanticNode]:
        """Retrieve node by ID, or None if not found."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def get_edges_for_node(self, node_id: int) -> List[SemanticEdge]:
        """Get all edges connected to a given node (as source or target)."""
        return [
            e for e in self.edges if e.source_id == node_id or e.target_id == node_id
        ]

    def node_ids(self) -> List[int]:
        """Return list of all node IDs."""
        return [n.node_id for n in self.nodes]

    def adjacency_dict(self) -> dict[int, List[int]]:
        """
        Build adjacency dictionary for graph algorithms.
        Returns {node_id: [neighbor_ids]}
        """
        adj: dict[int, List[int]] = {n.node_id: [] for n in self.nodes}
        for edge in self.edges:
            if edge.source_id in adj:
                adj[edge.source_id].append(edge.target_id)
            if edge.target_id in adj:
                adj[edge.target_id].append(edge.source_id)
        return adj


# -----------------------------------------------------------------------------
# ReasoningTrace (Output of IMAGO)
# -----------------------------------------------------------------------------


@dataclass
class ReasoningTrace:
    """
    R_trace: The output of IMAGO operator.

    A continuous signal defining the intended flow of the system state.
    Used by ORTSF to execute smooth, delay-compensated actions.

    Reference: spec/11_interfaces.ir.yml -> data_schemas -> ReasoningTrace
    """

    timestamp_ns: int  # Creation time
    target_state: List[SemanticNode] = field(
        default_factory=list
    )  # S*: desired future state
    trajectory_coeffs: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    curvature: float = 0.0  # Forman-Ricci curvature scalar of active cluster
    valid_until_ns: int = 0  # Expiration time for safety

    def is_valid(self, current_time_ns: int) -> bool:
        """Check if this trace is still valid at the given time."""
        return current_time_ns <= self.valid_until_ns


# -----------------------------------------------------------------------------
# Control Schemas
# -----------------------------------------------------------------------------


class ControlMode(Enum):
    """Control mode for ActuatorCommand."""

    POSITION = "POSITION"
    VELOCITY = "VELOCITY"
    TORQUE = "TORQUE"
    IMPEDANCE = "IMPEDANCE"


@dataclass
class ActuatorCommand:
    """
    u: The output of ORTSF. Computed control actions.

    Reference: spec/11_interfaces.ir.yml -> data_schemas -> ActuatorCommand
    """

    timestamp_ns: int
    mode: ControlMode
    command_values: np.ndarray  # Values corresponding to the mode

    def __post_init__(self):
        self.command_values = np.asarray(self.command_values, dtype=np.float32)


@dataclass
class JointState:
    """
    Standard proprioceptive state.

    Reference: spec/11_interfaces.ir.yml -> data_schemas -> JointState
    """

    position: np.ndarray  # Joint angles (radians) or positions (meters)
    velocity: np.ndarray  # Joint velocities
    effort: np.ndarray  # Joint torques/forces

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float32)
        self.velocity = np.asarray(self.velocity, dtype=np.float32)
        self.effort = np.asarray(self.effort, dtype=np.float32)


@dataclass
class SensorObservation:
    """
    Z_i: The raw evidence package from HAL.

    Aggregates vision, depth, and proprioception.
    Note: rgb_images and depth_maps are represented as numpy arrays here.

    Reference: spec/11_interfaces.ir.yml -> data_schemas -> SensorObservation
    """

    timestamp_ns: int  # Acquisition time in nanoseconds
    frame_id: str  # Reference coordinate frame (e.g., 'robot_base')
    rgb_images: List[np.ndarray] = field(default_factory=list)
    depth_maps: List[np.ndarray] = field(default_factory=list)
    joint_state: Optional[JointState] = None


# -----------------------------------------------------------------------------
# Configuration Schemas
# -----------------------------------------------------------------------------


@dataclass
class MissionGoal:
    """
    High-level intent provided by the application layer.

    Reference: spec/11_interfaces.ir.yml -> data_schemas -> MissionGoal
    """

    goal_id: str  # Unique ID for this mission
    verb: str  # e.g., 'POUR', 'GRASP', 'MONITOR'
    target_node_id: int  # ID of the primary object of interest
    constraints: dict = field(default_factory=dict)  # Task-specific constraints


@dataclass
class ConstraintConfig:
    """Parameters for the LOGOS solver."""

    weights: dict[str, float] = field(default_factory=dict)
    max_iterations: int = 10
    learning_rate: float = 0.01


@dataclass
class ONNSolverConfig:
    """Brain-inspired residual dynamics hyperparameters."""

    beta_obs: float = 1.0
    beta_cons: float = 1.0
    step_size: float = 0.01
    steps: int = 10
    gate_threshold: float = 0.5
    uncertainty_damping: float = 0.1


@dataclass
class ESConfig:
    """Evolutionary Strategy configuration (CMA-ES)."""

    algorithm: str = "CMA-ES"
    population_size: int = 16
    sigma: float = 0.3
    elite_fraction: float = 0.5
    seed: int = 42
    max_generations: int = 100
    parameter_bounds: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "beta_obs": [0.1, 10.0],
            "beta_cons": [0.1, 10.0],
            "step_size": [0.001, 0.1],
            "steps": [5, 50],
            "gate_threshold": [0.1, 0.9],
            "uncertainty_damping": [0.0, 1.0],
        }
    )
    use_delta_update: bool = False
    delta_beta: float = 1.0
    optimize_w_lin: bool = False
    w_lin_shape: Optional[Tuple[int, int]] = None
    w_lin_bounds: Tuple[float, float] = (-1.0, 1.0)
    w_lin_sigma: float = 0.1
    w_lin_rank: int = 0


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics for ONN/ES."""

    violation_mean: float = 0.0
    violation_max: float = 0.0
    drift_mean: float = 0.0
    collapse_score: float = 0.0
    latency_mean: float = 0.0
    ricci_energy: float = 0.0
    smoothness: float = 0.0
    contradiction_rate: float = 0.0


@dataclass
class GateResult:
    """Individual gate evaluation result."""

    gate_id: str
    passed: bool
    value: float
    threshold: float
    description: str = ""


@dataclass
class GateReport:
    """Gate evaluation result."""

    passed: bool
    failed_gates: List[str]
    gate_results: List[GateResult]
    metrics: EvalMetrics


@dataclass
class ESReport:
    """ES training progress report."""

    generation: int
    best_fitness: float
    best_params: Dict[str, float]
    metrics: EvalMetrics
