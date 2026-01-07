"""ONN Core Package - Edge Embedding Solver.

This package provides the core components for the ONN edge-embedding solver:
- EdgeGraph: Edge-centric graph representation
- CycleBasis: Cycle constraint matrix construction
- Projection: Hard and relaxed constraint projection
- Losses: Data, context, Ricci, smoothness, variance losses
- Solver: Projection-Consensus solver

Reference:
    spec/20_impl_plan.ir.yml: IMPL_015-018
"""

from onn.core.tensors import (
    SemanticNode,
    SemanticEdge,
    RawSemanticGraph,
    StabilizedGraph,
    ReasoningTrace,
    ActuatorCommand,
    ControlMode,
    JointState,
    SensorObservation,
    MissionGoal,
    ConstraintConfig,
    ONNSolverConfig,
    EvalMetrics,
    GateReport,
    GateResult,
    ESConfig,
    ESReport,
)
from onn.core.graph import (
    EdgeKey,
    EdgeGraph,
    compute_graph_laplacian,
    compute_edge_laplacian,
    get_node_degrees,
)

from onn.core.cycles import (
    CycleBasis,
    build_cycle_basis,
    cycle_matrix,
    cycle_targets,
    compute_cycle_violation,
    verify_cycle_basis,
)

from onn.core.projection import (
    project_constraints,
    relaxed_projection,
    project_with_basis,
    compute_projection_error,
    is_in_constraint_set,
)

from onn.core.losses import (
    LossConfig,
    loss_data,
    loss_context,
    loss_ricci,
    loss_smooth,
    loss_variance,
    total_loss,
    compute_gradient,
    gradient_step,
)

from onn.core.solver import (
    PCSolverConfig,
    SolverResult,
    ProjectionConsensusSolver,
    create_default_pc_solver,
    create_solver_from_dict,
    analyze_convergence,
)

from onn.core.diff_solver import (
    DiffSolverConfig,
    DifferentiableSolver,
    DifferentiableEncoder,
    EndToEndTrainer,
    create_diff_solver,
    create_diff_encoder,
    create_e2e_trainer,
)

from onn.core.relation_geometry import (
    # SE(3) primitives
    exp_so3,
    log_so3,
    exp_se3,
    log_se3,
    invert_se3,
    compose_se3,
    # Frame extraction
    frame_from_bound_tensor,
    frame_from_node,
    # Main encoder
    GeometricRelationEncoder,
    GeometricRelation,
    GeometricRelationEncoderTorch,
    create_geometric_encoder,
    # Algebra
    compose_relations,
    relation_distance,
    verify_equivariance,
    # Phase 2: Predictive Relation Model
    se3_distance,
    se3_distance_separate,
    PredictionResult,
    PredictiveRelationModel,
    create_predictive_model,
    # Temporal Prediction
    TemporalRelationState,
    TemporalPredictiveModel,
    create_temporal_model,
    # Fitness Integration
    RelationFitnessMetrics,
    compute_relation_fitness,
    # Phase 3: Contrastive Relation Learning
    ContrastivePair,
    ContrastiveRelationLearner,
    create_contrastive_learner,
    RelationCluster,
    RelationClusterer,
    create_relation_clusterer,
    ContrastiveRelationLoss,
    LearnableRelationEncoder,
)

__all__ = [
    "SemanticNode",
    "SemanticEdge",
    "RawSemanticGraph",
    "StabilizedGraph",
    "ReasoningTrace",
    "ActuatorCommand",
    "ControlMode",
    "JointState",
    "SensorObservation",
    "MissionGoal",
    "ConstraintConfig",
    "ONNSolverConfig",
    "EvalMetrics",
    "GateResult",
    "GateReport",
    "ESConfig",
    "ESReport",
    "EdgeKey",
    "EdgeGraph",
    "compute_graph_laplacian",
    "compute_edge_laplacian",
    "get_node_degrees",
    # Cycles
    "CycleBasis",
    "build_cycle_basis",
    "cycle_matrix",
    "cycle_targets",
    "compute_cycle_violation",
    "verify_cycle_basis",
    # Projection
    "project_constraints",
    "relaxed_projection",
    "project_with_basis",
    "compute_projection_error",
    "is_in_constraint_set",
    # Losses
    "LossConfig",
    "loss_data",
    "loss_context",
    "loss_ricci",
    "loss_smooth",
    "loss_variance",
    "total_loss",
    "compute_gradient",
    "gradient_step",
    # Solver
    "PCSolverConfig",
    "SolverResult",
    "ProjectionConsensusSolver",
    "create_default_pc_solver",
    "create_solver_from_dict",
    "analyze_convergence",
    # Differentiable Solver (Phase 2)
    "DiffSolverConfig",
    "DifferentiableSolver",
    "DifferentiableEncoder",
    "EndToEndTrainer",
    "create_diff_solver",
    "create_diff_encoder",
    "create_e2e_trainer",
    # Geometric Relations (Label-Free Understanding)
    "exp_so3",
    "log_so3",
    "exp_se3",
    "log_se3",
    "invert_se3",
    "compose_se3",
    "frame_from_bound_tensor",
    "frame_from_node",
    "GeometricRelationEncoder",
    "GeometricRelation",
    "GeometricRelationEncoderTorch",
    "create_geometric_encoder",
    "compose_relations",
    "relation_distance",
    "verify_equivariance",
    # Phase 2: Predictive Relation Model
    "se3_distance",
    "se3_distance_separate",
    "PredictionResult",
    "PredictiveRelationModel",
    "create_predictive_model",
    "TemporalRelationState",
    "TemporalPredictiveModel",
    "create_temporal_model",
    "RelationFitnessMetrics",
    "compute_relation_fitness",
    # Phase 3: Contrastive Relation Learning
    "ContrastivePair",
    "ContrastiveRelationLearner",
    "create_contrastive_learner",
    "RelationCluster",
    "RelationClusterer",
    "create_relation_clusterer",
    "ContrastiveRelationLoss",
    "LearnableRelationEncoder",
]
