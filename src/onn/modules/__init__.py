"""
ONN Modules - Mathematical Operations and Algorithms

Contains mathematical operations used by the ONN operators:
- Forman-Ricci curvature calculation (for IMAGO clustering)
- Deep Delta Learning (DDL) residual blocks
- Future: Delay predictors, interpolators for ORTSF
"""

from onn.modules.curvature import (
    forman_ricci_curvature,
    forman_ricci_node_curvature,
    graph_average_curvature,
    identify_functional_clusters,
    curvature_gradient_field,
    CurvatureCluster,
)

from onn.modules.delta import (
    # Core DDL components
    DeltaConfig,
    DeltaResidualBlock,
    DeltaResidualStack,
    DeltaLayerWrapper,
    compute_householder_matrix,
    verify_spectral_properties,
    # Delta-centric architecture components
    DeltaLinear,
    DeltaFFN,
    DeltaMLP,
    DeltaAttention,
    DeltaTransformerBlock,
    DeltaTransformer,
)

__all__ = [
    # Curvature
    "forman_ricci_curvature",
    "forman_ricci_node_curvature",
    "graph_average_curvature",
    "identify_functional_clusters",
    "curvature_gradient_field",
    "CurvatureCluster",
    # Core DDL components
    "DeltaConfig",
    "DeltaResidualBlock",
    "DeltaResidualStack",
    "DeltaLayerWrapper",
    "compute_householder_matrix",
    "verify_spectral_properties",
    # Delta-centric architecture
    "DeltaLinear",
    "DeltaFFN",
    "DeltaMLP",
    "DeltaAttention",
    "DeltaTransformerBlock",
    "DeltaTransformer",
]
