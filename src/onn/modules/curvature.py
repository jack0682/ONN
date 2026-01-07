"""
Forman-Ricci Curvature Module

Implements discrete Forman-Ricci curvature calculation for semantic graphs.
Used by IMAGO to identify functional clusters and guide intent flow.

Forman-Ricci curvature is a discrete analogue of Ricci curvature for graphs,
measuring how "curved" the network is around each edge. Negative curvature
indicates bottlenecks/bridges, positive curvature indicates dense communities.

Reference:
    - spec/00_high_level_plan.md -> Section 3.2 (IMAGO)
    - spec/10_architecture.ir.yml -> modules[imago_intent_planner]
    - Forman, R. (2003). "Bochner's method for cell complexes and combinatorial
      Ricci curvature." Discrete & Computational Geometry.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass

from onn.core.tensors import (
    SemanticNode,
    SemanticEdge,
    StabilizedGraph,
)


# -----------------------------------------------------------------------------
# Forman-Ricci Curvature
# -----------------------------------------------------------------------------

def forman_ricci_curvature(
    graph: StabilizedGraph,
    node_weights: Optional[Dict[int, float]] = None
) -> Dict[Tuple[int, int], float]:
    """
    Compute Forman-Ricci curvature for all edges in a stabilized graph.

    For an edge e = (i, j) with weight w(e), the Forman-Ricci curvature is:

        F(e) = w(e) * [ w(i)/d(i) + w(j)/d(j) - sum_{triangles} 1/max(w(e1), w(e2)) ]

    Where:
        - w(i), w(j) are node weights (default 1.0)
        - d(i), d(j) are weighted degrees of nodes i, j
        - The sum is over all triangles containing edge e
        - e1, e2 are the other two edges of each triangle

    Interpretation:
        - Positive curvature: Edge is in a dense, well-connected region
        - Negative curvature: Edge is a bottleneck/bridge between regions
        - Zero curvature: Edge connects nodes with balanced connectivity

    Args:
        graph: StabilizedGraph to compute curvature on
        node_weights: Optional node weight mapping (default: all 1.0)

    Returns:
        Dictionary mapping (source_id, target_id) tuples to curvature values

    Reference:
        - spec/10_architecture.ir.yml -> modules[imago_intent_planner]
    """
    # Build adjacency structures
    adj = _build_adjacency(graph)
    edge_weights = _build_edge_weight_map(graph)

    # Default node weights
    if node_weights is None:
        node_weights = {n.node_id: 1.0 for n in graph.nodes}

    # Compute weighted degrees
    degrees = _compute_weighted_degrees(graph, edge_weights)

    # Compute curvature for each edge
    curvatures: Dict[Tuple[int, int], float] = {}

    for edge in graph.edges:
        i, j = edge.source_id, edge.target_id
        w_e = edge.effective_strength()

        # Get node weights and degrees
        w_i = node_weights.get(i, 1.0)
        w_j = node_weights.get(j, 1.0)
        d_i = degrees.get(i, 1.0)
        d_j = degrees.get(j, 1.0)

        # Base curvature contribution from endpoints
        curvature = w_i / max(d_i, 1e-8) + w_j / max(d_j, 1e-8)

        # Subtract triangle contributions
        triangles = _find_triangles_containing_edge(i, j, adj)
        for k in triangles:
            # Get edge weights for the triangle (i, j, k)
            w_ik = edge_weights.get(_edge_key(i, k), 0.0)
            w_jk = edge_weights.get(_edge_key(j, k), 0.0)

            if w_ik > 0 and w_jk > 0:
                max_other = max(w_ik, w_jk)
                curvature -= 1.0 / max_other

        # Scale by edge weight
        curvature *= w_e

        # Store with canonical edge key
        curvatures[(i, j)] = curvature
        curvatures[(j, i)] = curvature  # Undirected

    return curvatures


def forman_ricci_node_curvature(
    graph: StabilizedGraph,
    edge_curvatures: Optional[Dict[Tuple[int, int], float]] = None
) -> Dict[int, float]:
    """
    Compute node-level curvature as average of incident edge curvatures.

    This aggregates edge curvature to nodes, useful for identifying
    nodes that are hubs (high positive) vs bridges (negative).

    Args:
        graph: StabilizedGraph
        edge_curvatures: Pre-computed edge curvatures (computed if None)

    Returns:
        Dictionary mapping node_id to average curvature
    """
    if edge_curvatures is None:
        edge_curvatures = forman_ricci_curvature(graph)

    node_curvatures: Dict[int, List[float]] = {n.node_id: [] for n in graph.nodes}

    for edge in graph.edges:
        key = (edge.source_id, edge.target_id)
        if key in edge_curvatures:
            curv = edge_curvatures[key]
            node_curvatures[edge.source_id].append(curv)
            node_curvatures[edge.target_id].append(curv)

    # Average per node
    result: Dict[int, float] = {}
    for node_id, curvs in node_curvatures.items():
        if curvs:
            result[node_id] = np.mean(curvs)
        else:
            result[node_id] = 0.0

    return result


def graph_average_curvature(graph: StabilizedGraph) -> float:
    """
    Compute the average Forman-Ricci curvature over all edges.

    This gives a global measure of the graph's "curvature state":
        - Positive: Graph has many dense clusters
        - Negative: Graph has many bottlenecks/bridges
        - Near zero: Balanced connectivity

    Used by IMAGO to assess overall graph topology.

    Args:
        graph: StabilizedGraph

    Returns:
        Average curvature scalar
    """
    if not graph.edges:
        return 0.0

    edge_curvatures = forman_ricci_curvature(graph)

    # Get unique edge curvatures (since we store both directions)
    unique_curvatures = []
    seen: Set[Tuple[int, int]] = set()

    for edge in graph.edges:
        key = _edge_key(edge.source_id, edge.target_id)
        if key not in seen:
            seen.add(key)
            if key in edge_curvatures:
                unique_curvatures.append(edge_curvatures[key])

    if not unique_curvatures:
        return 0.0

    return float(np.mean(unique_curvatures))


# -----------------------------------------------------------------------------
# Curvature-Based Clustering
# -----------------------------------------------------------------------------

@dataclass
class CurvatureCluster:
    """A cluster identified by curvature analysis."""
    cluster_id: int
    node_ids: List[int]
    internal_curvature: float  # Average curvature within cluster
    boundary_curvature: float  # Average curvature of boundary edges


def identify_functional_clusters(
    graph: StabilizedGraph,
    curvature_threshold: float = 0.0,
    min_cluster_size: int = 2
) -> List[CurvatureCluster]:
    """
    Identify functional clusters using Forman-Ricci curvature.

    Clusters are regions connected by high-curvature (positive) edges.
    Negative curvature edges act as boundaries between clusters.

    This implements part of the IMAGO functionality for identifying
    semantically meaningful groups in the topology.

    Args:
        graph: StabilizedGraph to cluster
        curvature_threshold: Edges with curvature > threshold stay in cluster
        min_cluster_size: Minimum nodes per cluster

    Returns:
        List of CurvatureCluster objects

    Reference:
        - spec/00_high_level_plan.md -> "Forman-Ricci Curvature to identify
          functional clusters"
    """
    edge_curvatures = forman_ricci_curvature(graph)

    # Build subgraph of positive-curvature edges
    positive_adj: Dict[int, Set[int]] = {n.node_id: set() for n in graph.nodes}

    for edge in graph.edges:
        key = (edge.source_id, edge.target_id)
        if edge_curvatures.get(key, 0.0) > curvature_threshold:
            positive_adj[edge.source_id].add(edge.target_id)
            positive_adj[edge.target_id].add(edge.source_id)

    # Find connected components in positive-curvature subgraph
    visited: Set[int] = set()
    clusters: List[CurvatureCluster] = []
    cluster_id = 0

    for start_node in positive_adj:
        if start_node in visited:
            continue

        # BFS to find connected component
        component: List[int] = []
        queue = [start_node]

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.append(node)

            for neighbor in positive_adj[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

        if len(component) >= min_cluster_size:
            # Compute cluster statistics
            internal_curvs = []
            boundary_curvs = []
            component_set = set(component)

            for edge in graph.edges:
                key = (edge.source_id, edge.target_id)
                curv = edge_curvatures.get(key, 0.0)

                src_in = edge.source_id in component_set
                tgt_in = edge.target_id in component_set

                if src_in and tgt_in:
                    internal_curvs.append(curv)
                elif src_in or tgt_in:
                    boundary_curvs.append(curv)

            cluster = CurvatureCluster(
                cluster_id=cluster_id,
                node_ids=component,
                internal_curvature=np.mean(internal_curvs) if internal_curvs else 0.0,
                boundary_curvature=np.mean(boundary_curvs) if boundary_curvs else 0.0
            )
            clusters.append(cluster)
            cluster_id += 1

    return clusters


# -----------------------------------------------------------------------------
# Curvature Flow (for IMAGO intent generation)
# -----------------------------------------------------------------------------

def curvature_gradient_field(
    graph: StabilizedGraph
) -> Dict[int, np.ndarray]:
    """
    Compute curvature gradient field for each node.

    The gradient points towards regions of higher curvature (denser areas).
    Used by IMAGO to guide intent flow through the semantic manifold.

    Returns:
        Dictionary mapping node_id to gradient vector (in embedding space)
    """
    edge_curvatures = forman_ricci_curvature(graph)
    node_dict = {n.node_id: n for n in graph.nodes}

    gradients: Dict[int, np.ndarray] = {}

    for node in graph.nodes:
        # Compute weighted average direction towards high-curvature neighbors
        gradient = np.zeros_like(node.combined_state())
        total_weight = 0.0

        for edge in graph.edges:
            if edge.source_id == node.node_id:
                neighbor_id = edge.target_id
            elif edge.target_id == node.node_id:
                neighbor_id = edge.source_id
            else:
                continue

            if neighbor_id not in node_dict:
                continue

            neighbor = node_dict[neighbor_id]
            curv = edge_curvatures.get((node.node_id, neighbor_id), 0.0)

            # Direction vector to neighbor
            direction = neighbor.combined_state() - node.combined_state()
            dir_norm = np.linalg.norm(direction)

            if dir_norm > 1e-8:
                # Weight by curvature (flow towards positive curvature)
                weight = max(0.0, curv + 1.0)  # Shift to ensure positivity
                gradient += weight * direction / dir_norm
                total_weight += weight

        if total_weight > 1e-8:
            gradient /= total_weight

        gradients[node.node_id] = gradient.astype(np.float32)

    return gradients


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _build_adjacency(graph: StabilizedGraph) -> Dict[int, Set[int]]:
    """Build adjacency set for each node."""
    adj: Dict[int, Set[int]] = {n.node_id: set() for n in graph.nodes}
    for edge in graph.edges:
        adj[edge.source_id].add(edge.target_id)
        adj[edge.target_id].add(edge.source_id)
    return adj


def _build_edge_weight_map(graph: StabilizedGraph) -> Dict[Tuple[int, int], float]:
    """Build edge weight lookup."""
    weights: Dict[Tuple[int, int], float] = {}
    for edge in graph.edges:
        w = edge.effective_strength()
        weights[_edge_key(edge.source_id, edge.target_id)] = w
    return weights


def _compute_weighted_degrees(
    graph: StabilizedGraph,
    edge_weights: Dict[Tuple[int, int], float]
) -> Dict[int, float]:
    """Compute weighted degree for each node."""
    degrees: Dict[int, float] = {n.node_id: 0.0 for n in graph.nodes}
    for edge in graph.edges:
        w = edge.effective_strength()
        degrees[edge.source_id] += w
        degrees[edge.target_id] += w
    return degrees


def _edge_key(i: int, j: int) -> Tuple[int, int]:
    """Canonical edge key (smaller id first)."""
    return (min(i, j), max(i, j))


def _find_triangles_containing_edge(
    i: int,
    j: int,
    adj: Dict[int, Set[int]]
) -> List[int]:
    """Find all nodes k such that (i, j, k) forms a triangle."""
    triangles = []
    neighbors_i = adj.get(i, set())
    neighbors_j = adj.get(j, set())

    # k must be neighbor of both i and j
    common = neighbors_i & neighbors_j
    triangles.extend(common)

    return list(triangles)
