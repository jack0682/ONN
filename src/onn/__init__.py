"""
ONN (Ontology Neural Network) Package

A framework for topological reasoning and delay-robust control.

The ONN system implements a closed-loop cognitive architecture where:
- "Existence" is defined by topological relationships (semantic manifold)
- "Action" is the delay-robust flow of continuous semantic traces

Core Components:
- onn.core: Tensor data structures (SemanticNode, SemanticEdge, graphs)
- onn.ops: Trinity of Operators (SEGO, LOGOS, IMAGO)
- onn.modules: Mathematical operations (curvature, etc.)

Framework-agnostic: No ROS2/DDS dependencies in core logic.

Reference:
    - spec/00_high_level_plan.md
    - spec/10_architecture.ir.yml
    - spec/11_interfaces.ir.yml
"""

__version__ = "0.1.0"

# Core tensors (legacy node-based)
from onn.core.tensors import (
    SemanticNode,
    SemanticEdge,
    RawSemanticGraph,
    StabilizedGraph,
    ReasoningTrace,
    SensorObservation,
    JointState,
    ActuatorCommand,
    ControlMode,
    MissionGoal,
    ConstraintConfig,
    BOUND_TENSOR_DIM,
    FORM_TENSOR_DIM,
    INTENT_TENSOR_DIM,
)

# Operators
from onn.ops import (
    ProjectionConsensusSolver,
    SEGOGaugeAnchor,
    create_default_solver,
    create_default_sego_anchor,
)

# Modules
from onn.modules import (
    forman_ricci_curvature,
    identify_functional_clusters,
    graph_average_curvature,
)

__all__ = [
    # Version
    "__version__",
    # Core tensors
    "SemanticNode",
    "SemanticEdge",
    "RawSemanticGraph",
    "StabilizedGraph",
    "ReasoningTrace",
    "SensorObservation",
    "JointState",
    "ActuatorCommand",
    "ControlMode",
    "MissionGoal",
    "ConstraintConfig",
    "BOUND_TENSOR_DIM",
    "FORM_TENSOR_DIM",
    "INTENT_TENSOR_DIM",
    # Operators
    "ProjectionConsensusSolver",
    "SEGOGaugeAnchor",
    "create_default_solver",
    "create_default_sego_anchor",
    # Modules
    "forman_ricci_curvature",
    "identify_functional_clusters",
    "graph_average_curvature",
]
