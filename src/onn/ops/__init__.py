"""
ONN Operators Module

Implements the Trinity of Operators for the Ontology Neural Network:
- SEGO (Semantic Graph Ontology Mapper): Perception operator
- LOGOS (Logical Ontological Generator for Self-Adjustment): Correction operator
- IMAGO (Intent Modeling & Action Generation Operator): Planning operator

All operators are framework-agnostic (no ROS2/DDS dependencies).
"""

from onn.ops.logos_solver import (
    LOGOSSolver,
    LOGOSConfig,
    project_to_manifold,
    compute_obs_residual,
    compute_cons_residual,
    create_default_solver,
    create_realtime_solver,
)


from onn.ops.ortsf import ORTSFOperator, DeepDeltaPredictor


from onn.ops.sego_anchor import (
    SEGOGaugeAnchor,
    SEGOConfig,
    Detection,
    SimpleFeatureEncoder,
    create_default_sego_anchor,
)

from onn.ops.imago_planner import (
    IMAGOPlanner,
    IMAGOConfig,
    create_default_imago_planner,
    create_realtime_imago_planner,
    interpolate_trace,
    validate_trace,
)

from onn.ops.edge_stabilizer import (
    EdgeStabilizer,
    EdgeStabilizerConfig,
    EdgeStabilizerResult,
    extract_edge_features,
    stabilized_to_relation_embeddings,
    create_default_edge_stabilizer,
    create_edge_stabilizer_for_es,
)

from onn.ops.csa_pipeline import (
    CSAPipeline,
    CSAConfig,
    CSAResult,
    CSAWithES,
    ESEpisode,
    create_default_csa_pipeline,
    create_csa_with_es,
    create_synthetic_episode,
)

from onn.ops.sego_enhanced import (
    EnhancedSEGO,
    EnhancedSEGOConfig,
    MultiScaleFeatureEncoder,
    SpatialRelationAnalyzer,
    TemporalTracker,
    create_enhanced_sego,
    create_precision_sego,
)

from onn.ops.branching import (
    StagnationDetector,
    StagnationConfig,
    StagnationSignal,
    BranchFactory,
    BranchSelector,
    BranchConfig,
    BranchType,
    BranchResult,
    Branch,
    SurvivalConfig,
    boundary_gate_mutation,
    meta_param_mutation,
    topology_rewire_mutation,
    create_conservative_config,
    create_exploration_config,
    create_rewire_config,
)

ProjectionConsensusSolver = LOGOSSolver

__all__ = [
    # LOGOS
    "LOGOSSolver",
    "LOGOSConfig",
    "SolverResult",
    "ProjectionConsensusSolver",  # Alias
    "compute_data_loss",
    "compute_phys_loss",
    "compute_logic_loss",
    "compute_total_loss",
    "project_to_manifold",
    "create_default_solver",
    "create_realtime_solver",
    "STATE_DIM",
    # SEGO
    "SEGOGaugeAnchor",
    "SEGOConfig",
    "Detection",
    "SimpleFeatureEncoder",
    "create_default_sego_anchor",
    # IMAGO
    "IMAGOPlanner",
    "IMAGOConfig",
    "create_default_imago_planner",
    "create_realtime_imago_planner",
    "interpolate_trace",
    "validate_trace",
    # EdgeStabilizer (ONN-ES bridge)
    "EdgeStabilizer",
    "EdgeStabilizerConfig",
    "EdgeStabilizerResult",
    "extract_edge_features",
    "stabilized_to_relation_embeddings",
    "create_default_edge_stabilizer",
    "create_edge_stabilizer_for_es",
    # CSA Pipeline
    "CSAPipeline",
    "CSAConfig",
    "CSAResult",
    "CSAWithES",
    "ESEpisode",
    "create_default_csa_pipeline",
    "create_csa_with_es",
    "create_synthetic_episode",
    # Enhanced SEGO
    "EnhancedSEGO",
    "EnhancedSEGOConfig",
    "MultiScaleFeatureEncoder",
    "SpatialRelationAnalyzer",
    "TemporalTracker",
    "create_enhanced_sego",
    "create_precision_sego",
    # Branching / Multi-Mode Evolution
    "StagnationDetector",
    "StagnationConfig",
    "StagnationSignal",
    "BranchFactory",
    "BranchSelector",
    "BranchConfig",
    "BranchType",
    "BranchResult",
    "Branch",
    "SurvivalConfig",
    "boundary_gate_mutation",
    "meta_param_mutation",
    "topology_rewire_mutation",
    "create_conservative_config",
    "create_exploration_config",
    "create_rewire_config",
]
