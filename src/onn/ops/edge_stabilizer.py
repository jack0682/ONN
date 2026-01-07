"""
Edge Stabilizer Module for CSA Pipeline.

Bridges ONN-ES (edge embedding optimization) with the CSA pipeline:
    SEGO → EdgeStabilizer → LOGOS → IMAGO

This module:
1. Extracts edge features (φ) from SEGO edge candidates
2. Encodes edge features to embeddings via RelationEncoder (x_obs = W_rp @ φ + W_lin @ φ)
3. Stabilizes edge embeddings using ProjectionConsensusSolver
4. Returns stabilized embeddings for use as relation_embeddings in LOGOS

The W_lin in RelationEncoder can be optimized by ES to improve pipeline performance.

Reference:
    - spec/20_impl_plan.ir.yml: ONN-ES integration
    - User roadmap: "ES가 RelationEncoder(W_lin)까지 최적화"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING

import torch
import numpy as np

from onn.core.tensors import (
    SemanticNode,
    SemanticEdge,
    RawSemanticGraph,
    BOUND_TENSOR_DIM,
    FORM_TENSOR_DIM,
    INTENT_TENSOR_DIM,
)
from onn.relation.param import RelationEncoder, RelationParamConfig
from onn.core.solver import ProjectionConsensusSolver, PCSolverConfig, SolverResult
from onn.core.graph import EdgeGraph

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EdgeStabilizerConfig:
    """Configuration for EdgeStabilizer.

    Controls how edge features are extracted and stabilized.
    """
    # Feature dimensions
    edge_feature_dim: int = 64      # φ dimension (extracted from node pairs)
    edge_embedding_dim: int = 32    # x_obs dimension (output of RelationEncoder)

    # RelationEncoder config
    use_linear_term: bool = True    # Enable W_lin for ES optimization
    random_proj_seed: int = 42      # Reproducible random projection

    # PC Solver config
    solver_steps: int = 10
    solver_step_size: float = 0.01
    solver_lambda_context: float = 1.0
    solver_lambda_ricci: float = 0.1
    solver_lambda_smooth: float = 0.1
    solver_lambda_var: float = 1.0

    # Feature extraction
    use_bound_features: bool = True
    use_form_features: bool = True
    use_intent_features: bool = True


@dataclass
class EdgeStabilizerResult:
    """Result from edge stabilization."""
    stabilized_embeddings: torch.Tensor  # Shape (num_edges, embedding_dim)
    solver_result: SolverResult
    edge_graph: EdgeGraph
    phi_features: torch.Tensor  # Original edge features


# =============================================================================
# Edge Feature Extraction
# =============================================================================

def extract_edge_features(
    edges: List[SemanticEdge],
    nodes: List[SemanticNode],
    config: EdgeStabilizerConfig,
) -> Tuple[torch.Tensor, Dict[int, int]]:
    """Extract edge features (φ) from node pairs.

    For each edge (i, j), extract features from the source and target nodes
    and combine them into a single feature vector.

    φ_ij = [bound_i - bound_j, form_i + form_j, intent_i * intent_j, ...]

    Args:
        edges: List of semantic edges
        nodes: List of semantic nodes
        config: Configuration with feature dimensions

    Returns:
        phi: Edge features tensor, shape (num_edges, edge_feature_dim)
        node_id_map: Mapping from node_id to index
    """
    if not edges or not nodes:
        return torch.zeros(0, config.edge_feature_dim), {}

    # Build node lookup
    node_dict = {node.node_id: node for node in nodes}
    node_id_map = {node.node_id: i for i, node in enumerate(nodes)}

    phi_list = []

    for edge in edges:
        source = node_dict.get(edge.source_id)
        target = node_dict.get(edge.target_id)

        if source is None or target is None:
            continue

        features = []

        # Bound tensor features (spatial/geometric)
        if config.use_bound_features:
            bound_diff = source.bound_tensor - target.bound_tensor
            bound_sum = (source.bound_tensor + target.bound_tensor) / 2
            features.extend(bound_diff[:8].tolist())  # First 8 dims of diff
            features.extend(bound_sum[:8].tolist())   # First 8 dims of mean

        # Form tensor features (appearance/semantics)
        if config.use_form_features:
            form_diff = source.form_tensor - target.form_tensor
            form_prod = source.form_tensor * target.form_tensor
            features.extend(form_diff[:8].tolist())
            features.extend(form_prod[:8].tolist())

        # Intent tensor features (functional/affordance)
        if config.use_intent_features:
            intent_sum = source.intent_tensor + target.intent_tensor
            intent_prod = source.intent_tensor * target.intent_tensor
            features.extend(intent_sum[:8].tolist())
            features.extend(intent_prod[:8].tolist())

        # Pad or truncate to feature_dim
        features = np.array(features, dtype=np.float32)
        if len(features) < config.edge_feature_dim:
            features = np.pad(features, (0, config.edge_feature_dim - len(features)))
        else:
            features = features[:config.edge_feature_dim]

        phi_list.append(features)

    if not phi_list:
        return torch.zeros(0, config.edge_feature_dim), node_id_map

    phi = torch.tensor(np.array(phi_list), dtype=torch.float32)
    return phi, node_id_map


def build_edge_graph_from_semantic(
    edges: List[SemanticEdge],
    node_id_map: Dict[int, int],
) -> EdgeGraph:
    """Build EdgeGraph structure from semantic edges.

    Args:
        edges: List of semantic edges
        node_id_map: Mapping from node_id to index

    Returns:
        EdgeGraph for cycle computation
    """
    edge_list = []

    for edge in edges:
        if edge.source_id in node_id_map and edge.target_id in node_id_map:
            src_idx = node_id_map[edge.source_id]
            tgt_idx = node_id_map[edge.target_id]
            edge_list.append((src_idx, tgt_idx))

    if not edge_list:
        # Return minimal graph
        return EdgeGraph.from_edge_list([(0, 1)])

    return EdgeGraph.from_edge_list(edge_list)


# =============================================================================
# Edge Stabilizer
# =============================================================================

class EdgeStabilizer:
    """Stabilizes edge embeddings for use in CSA pipeline.

    This module bridges ONN-ES with the CSA pipeline by:
    1. Extracting edge features from SEGO output
    2. Encoding features to embeddings via RelationEncoder
    3. Stabilizing embeddings using ProjectionConsensusSolver

    The RelationEncoder's W_lin can be optimized by ES:
        >>> stabilizer.set_w_lin(optimized_w_lin)

    Example:
        >>> stabilizer = EdgeStabilizer(config)
        >>> result = stabilizer.stabilize(raw_graph)
        >>> # Use result.stabilized_embeddings in LOGOS
    """

    def __init__(self, config: Optional[EdgeStabilizerConfig] = None):
        """Initialize the edge stabilizer.

        Args:
            config: Stabilizer configuration
        """
        self.config = config or EdgeStabilizerConfig()

        # Initialize RelationEncoder
        encoder_config = RelationParamConfig(
            input_dim=self.config.edge_feature_dim,
            output_dim=self.config.edge_embedding_dim,
            random_proj_seed=self.config.random_proj_seed,
            use_linear_term=self.config.use_linear_term,
        )
        self.encoder = RelationEncoder(encoder_config)

        # Initialize PC Solver
        solver_config = PCSolverConfig(
            step_size=self.config.solver_step_size,
            steps=self.config.solver_steps,
            lambda_context=self.config.solver_lambda_context,
            lambda_ricci=self.config.solver_lambda_ricci,
            lambda_smooth=self.config.solver_lambda_smooth,
            lambda_var=self.config.solver_lambda_var,
        )
        self.solver = ProjectionConsensusSolver(solver_config)

        # Cache for warm start
        self._last_result: Optional[EdgeStabilizerResult] = None

        logger.debug(
            f"EdgeStabilizer initialized: φ={self.config.edge_feature_dim} → "
            f"x_obs={self.config.edge_embedding_dim}"
        )

    def stabilize(
        self,
        raw_graph: RawSemanticGraph,
        x_prev: Optional[torch.Tensor] = None,
        warm_start: bool = False,
    ) -> EdgeStabilizerResult:
        """Stabilize edge embeddings from a raw semantic graph.

        Args:
            raw_graph: Raw graph from SEGO with nodes and edge_candidates
            x_prev: Previous embeddings for temporal smoothness
            warm_start: Use previous solution as starting point

        Returns:
            EdgeStabilizerResult with stabilized embeddings
        """
        # Extract edge features
        phi, node_id_map = extract_edge_features(
            raw_graph.edge_candidates,
            raw_graph.nodes,
            self.config,
        )

        if phi.shape[0] == 0:
            logger.warning("No valid edges to stabilize")
            return EdgeStabilizerResult(
                stabilized_embeddings=torch.zeros(0, self.config.edge_embedding_dim),
                solver_result=SolverResult(
                    x=torch.zeros(0, self.config.edge_embedding_dim),
                    converged=True,
                    iterations_used=0,
                    final_loss=0.0,
                ),
                edge_graph=EdgeGraph.from_edge_list([(0, 1)]),
                phi_features=phi,
            )

        # Encode to embeddings
        x_obs = self.encoder.encode(phi)

        # Build edge graph for cycle constraints
        edge_graph = build_edge_graph_from_semantic(
            raw_graph.edge_candidates,
            node_id_map,
        )

        # Check x_prev shape compatibility
        valid_x_prev = x_prev
        if x_prev is not None and x_prev.shape[0] != x_obs.shape[0]:
            logger.debug(f"x_prev shape mismatch ({x_prev.shape[0]} vs {x_obs.shape[0]}), ignoring")
            valid_x_prev = None

        # Stabilize using PC solver
        solver_result = self.solver.solve(
            x_obs,
            edge_graph,
            x_prev=valid_x_prev,
            warm_start=warm_start,
        )

        result = EdgeStabilizerResult(
            stabilized_embeddings=solver_result.x,
            solver_result=solver_result,
            edge_graph=edge_graph,
            phi_features=phi,
        )

        self._last_result = result

        logger.info(
            f"Edge stabilization: {phi.shape[0]} edges, "
            f"loss={solver_result.final_loss:.4f}, "
            f"converged={solver_result.converged}"
        )

        return result

    def set_w_lin(self, w_lin: torch.Tensor) -> None:
        """Set the learnable W_lin weights (from ES optimization).

        Args:
            w_lin: Weight matrix, shape (edge_embedding_dim, edge_feature_dim)
        """
        self.encoder.set_linear_weights(w_lin)
        logger.debug(f"W_lin updated: norm={w_lin.norm().item():.4f}")

    def get_w_lin(self) -> Optional[torch.Tensor]:
        """Get the current W_lin weights."""
        return self.encoder.get_linear_weights()

    def get_w_lin_shape(self) -> Tuple[int, int]:
        """Get the shape of W_lin for ES configuration."""
        return (self.config.edge_embedding_dim, self.config.edge_feature_dim)

    def get_last_result(self) -> Optional[EdgeStabilizerResult]:
        """Get the last stabilization result."""
        return self._last_result

    def reset(self) -> None:
        """Reset stabilizer state."""
        self.solver.reset()
        self._last_result = None


# =============================================================================
# Utility: Convert stabilized embeddings to LOGOS relation_embeddings
# =============================================================================

def stabilized_to_relation_embeddings(
    result: EdgeStabilizerResult,
    edges: List[SemanticEdge],
    target_dim: int = 64,
) -> List[SemanticEdge]:
    """Update edge relation_embeddings with stabilized values.

    Copies edges and replaces their relation_embedding with the
    stabilized embeddings from EdgeStabilizer.

    Args:
        result: Result from EdgeStabilizer
        edges: Original semantic edges
        target_dim: Target dimension for LOGOS (pad if needed)

    Returns:
        List of edges with updated relation_embeddings
    """
    stabilized = result.stabilized_embeddings.detach().numpy()

    updated_edges = []
    edge_idx = 0

    for edge in edges:
        new_edge = edge.copy()

        if edge_idx < stabilized.shape[0]:
            # Get stabilized embedding and pad to target_dim
            emb = stabilized[edge_idx]
            if len(emb) < target_dim:
                emb = np.pad(emb, (0, target_dim - len(emb)))
            else:
                emb = emb[:target_dim]

            new_edge.relation_embedding = emb.astype(np.float32)
            edge_idx += 1

        updated_edges.append(new_edge)

    return updated_edges


# =============================================================================
# Factory Functions
# =============================================================================

def create_default_edge_stabilizer() -> EdgeStabilizer:
    """Create edge stabilizer with default configuration."""
    return EdgeStabilizer(EdgeStabilizerConfig())


def create_edge_stabilizer_for_es(
    edge_feature_dim: int = 64,
    edge_embedding_dim: int = 32,
) -> EdgeStabilizer:
    """Create edge stabilizer configured for ES optimization.

    Args:
        edge_feature_dim: Dimension of edge features (φ)
        edge_embedding_dim: Dimension of edge embeddings (x_obs)

    Returns:
        EdgeStabilizer ready for W_lin optimization
    """
    config = EdgeStabilizerConfig(
        edge_feature_dim=edge_feature_dim,
        edge_embedding_dim=edge_embedding_dim,
        use_linear_term=True,  # Enable W_lin
    )
    return EdgeStabilizer(config)
