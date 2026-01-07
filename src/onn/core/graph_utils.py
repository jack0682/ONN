"""
Graph Utilities for ONN Semantic Graphs.

Provides helper functions for graph operations:
- Adjacency building
- Node/edge lookup
- Graph validation
- Edge pruning

Reference:
    - spec/20_impl_plan.ir.yml IMPL_002
    - spec/11_interfaces.ir.yml (StabilizedGraph, RawSemanticGraph)

Framework-agnostic: No ROS2, DDS, or external framework dependencies.
"""

from __future__ import annotations

from typing import List, Dict, Optional, Set, Tuple, Union
import logging

from onn.core.tensors import (
    SemanticNode,
    SemanticEdge,
    RawSemanticGraph,
    StabilizedGraph,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Adjacency Building
# =============================================================================

def build_adjacency(
    graph: Union[RawSemanticGraph, StabilizedGraph],
    directed: bool = False
) -> Dict[int, List[int]]:
    """
    Build adjacency list representation of the graph.

    Args:
        graph: Either a RawSemanticGraph or StabilizedGraph
        directed: If True, only include outgoing edges per node.
                  If False (default), treat edges as undirected.

    Returns:
        Dictionary mapping node_id -> list of neighbor node_ids.
        Isolated nodes (no edges) are included with empty lists.

    Example:
        >>> adj = build_adjacency(graph)
        >>> adj[1]  # Returns list of neighbors of node 1
        [2, 3]
    """
    # Initialize adjacency for all nodes
    adj: Dict[int, List[int]] = {node.node_id: [] for node in graph.nodes}

    # Get edges from appropriate field
    edges = _get_edges(graph)

    for edge in edges:
        src, tgt = edge.source_id, edge.target_id

        # Skip edges referencing non-existent nodes
        if src not in adj or tgt not in adj:
            continue

        adj[src].append(tgt)
        if not directed:
            adj[tgt].append(src)

    return adj


def build_adjacency_set(
    graph: Union[RawSemanticGraph, StabilizedGraph],
    directed: bool = False
) -> Dict[int, Set[int]]:
    """
    Build adjacency set representation of the graph.

    Similar to build_adjacency but returns sets for O(1) lookup.

    Args:
        graph: Either a RawSemanticGraph or StabilizedGraph
        directed: If True, only include outgoing edges per node.

    Returns:
        Dictionary mapping node_id -> set of neighbor node_ids.
    """
    adj_list = build_adjacency(graph, directed)
    return {k: set(v) for k, v in adj_list.items()}


def build_weighted_adjacency(
    graph: Union[RawSemanticGraph, StabilizedGraph]
) -> Dict[int, Dict[int, float]]:
    """
    Build weighted adjacency representation using edge effective strength.

    Args:
        graph: Either a RawSemanticGraph or StabilizedGraph

    Returns:
        Nested dict: adj[source_id][target_id] = weight
    """
    adj: Dict[int, Dict[int, float]] = {node.node_id: {} for node in graph.nodes}

    edges = _get_edges(graph)

    for edge in edges:
        src, tgt = edge.source_id, edge.target_id
        weight = edge.effective_strength()

        if src in adj:
            adj[src][tgt] = weight
        if tgt in adj:
            adj[tgt][src] = weight

    return adj


# =============================================================================
# Node Lookup
# =============================================================================

def node_lookup(
    graph: Union[RawSemanticGraph, StabilizedGraph]
) -> Dict[int, SemanticNode]:
    """
    Build node lookup dictionary.

    Args:
        graph: Either a RawSemanticGraph or StabilizedGraph

    Returns:
        Dictionary mapping node_id -> SemanticNode object

    Example:
        >>> lookup = node_lookup(graph)
        >>> node = lookup[42]  # Get node with id 42
    """
    return {node.node_id: node for node in graph.nodes}


def get_node(
    graph: Union[RawSemanticGraph, StabilizedGraph],
    node_id: int
) -> Optional[SemanticNode]:
    """
    Get a single node by ID.

    Args:
        graph: Either a RawSemanticGraph or StabilizedGraph
        node_id: The node ID to look up

    Returns:
        SemanticNode if found, None otherwise
    """
    for node in graph.nodes:
        if node.node_id == node_id:
            return node
    return None


def get_node_ids(graph: Union[RawSemanticGraph, StabilizedGraph]) -> List[int]:
    """
    Get list of all node IDs in the graph.

    Args:
        graph: Either a RawSemanticGraph or StabilizedGraph

    Returns:
        List of node IDs
    """
    return [node.node_id for node in graph.nodes]


# =============================================================================
# Edge Lookup
# =============================================================================

def edge_lookup(
    graph: Union[RawSemanticGraph, StabilizedGraph]
) -> Dict[Tuple[int, int], SemanticEdge]:
    """
    Build edge lookup dictionary.

    Args:
        graph: Either a RawSemanticGraph or StabilizedGraph

    Returns:
        Dictionary mapping (source_id, target_id) -> SemanticEdge

    Example:
        >>> lookup = edge_lookup(graph)
        >>> edge = lookup[(1, 2)]  # Get edge from node 1 to node 2
    """
    edges = _get_edges(graph)
    return {(edge.source_id, edge.target_id): edge for edge in edges}


def get_edge(
    graph: Union[RawSemanticGraph, StabilizedGraph],
    source_id: int,
    target_id: int,
    check_reverse: bool = True
) -> Optional[SemanticEdge]:
    """
    Get a single edge by source and target IDs.

    Args:
        graph: Either a RawSemanticGraph or StabilizedGraph
        source_id: Source node ID
        target_id: Target node ID
        check_reverse: If True, also check (target_id, source_id)

    Returns:
        SemanticEdge if found, None otherwise
    """
    edges = _get_edges(graph)

    for edge in edges:
        if edge.source_id == source_id and edge.target_id == target_id:
            return edge
        if check_reverse and edge.source_id == target_id and edge.target_id == source_id:
            return edge

    return None


def get_edges_for_node(
    graph: Union[RawSemanticGraph, StabilizedGraph],
    node_id: int
) -> List[SemanticEdge]:
    """
    Get all edges connected to a given node (as source or target).

    Args:
        graph: Either a RawSemanticGraph or StabilizedGraph
        node_id: The node ID to find edges for

    Returns:
        List of edges connected to the node
    """
    edges = _get_edges(graph)
    return [e for e in edges if e.source_id == node_id or e.target_id == node_id]


def get_outgoing_edges(
    graph: Union[RawSemanticGraph, StabilizedGraph],
    node_id: int
) -> List[SemanticEdge]:
    """
    Get edges where node_id is the source.

    Args:
        graph: Either a RawSemanticGraph or StabilizedGraph
        node_id: The source node ID

    Returns:
        List of outgoing edges
    """
    edges = _get_edges(graph)
    return [e for e in edges if e.source_id == node_id]


def get_incoming_edges(
    graph: Union[RawSemanticGraph, StabilizedGraph],
    node_id: int
) -> List[SemanticEdge]:
    """
    Get edges where node_id is the target.

    Args:
        graph: Either a RawSemanticGraph or StabilizedGraph
        node_id: The target node ID

    Returns:
        List of incoming edges
    """
    edges = _get_edges(graph)
    return [e for e in edges if e.target_id == node_id]


# =============================================================================
# Graph Validation
# =============================================================================

class GraphValidationError(Exception):
    """Exception raised when graph validation fails."""
    pass


def validate_graph(
    graph: Union[RawSemanticGraph, StabilizedGraph],
    strict: bool = False
) -> List[str]:
    """
    Validate graph structure and constraints.

    Checks:
    1. All nodes have unique IDs
    2. All edges reference existing nodes
    3. No self-loops (edge from node to itself)
    4. Edge weights are non-negative
    5. Edge probabilities are in [0, 1]
    6. Bound tensors are normalized (strict mode only)

    Args:
        graph: Either a RawSemanticGraph or StabilizedGraph
        strict: If True, perform additional validation (e.g., tensor norms)

    Returns:
        List of warning/error messages. Empty list means valid.

    Raises:
        GraphValidationError: If strict=True and validation fails
    """
    issues: List[str] = []

    # Check 1: Unique node IDs
    node_ids = [node.node_id for node in graph.nodes]
    if len(node_ids) != len(set(node_ids)):
        duplicates = [nid for nid in node_ids if node_ids.count(nid) > 1]
        issues.append(f"Duplicate node IDs found: {set(duplicates)}")

    node_id_set = set(node_ids)
    edges = _get_edges(graph)

    for i, edge in enumerate(edges):
        # Check 2: Edges reference existing nodes
        if edge.source_id not in node_id_set:
            issues.append(f"Edge {i}: source_id {edge.source_id} not in nodes")
        if edge.target_id not in node_id_set:
            issues.append(f"Edge {i}: target_id {edge.target_id} not in nodes")

        # Check 3: No self-loops
        if edge.source_id == edge.target_id:
            issues.append(f"Edge {i}: self-loop on node {edge.source_id}")

        # Check 4: Non-negative weights
        if edge.weight < 0:
            issues.append(f"Edge {i}: negative weight {edge.weight}")

        # Check 5: Valid probability
        if not 0.0 <= edge.probability <= 1.0:
            issues.append(f"Edge {i}: invalid probability {edge.probability}")

    # Strict mode: additional checks
    if strict:
        import numpy as np

        for node in graph.nodes:
            # Check 6: Bound tensor normalized
            norm = np.linalg.norm(node.bound_tensor)
            if not np.isclose(norm, 1.0, atol=1e-4):
                issues.append(
                    f"Node {node.node_id}: bound_tensor not normalized (norm={norm:.4f})"
                )

            # Check intent tensor in [0, 1]
            if np.any(node.intent_tensor < 0) or np.any(node.intent_tensor > 1):
                issues.append(
                    f"Node {node.node_id}: intent_tensor values outside [0, 1]"
                )

    if strict and issues:
        raise GraphValidationError(f"Graph validation failed: {issues}")

    return issues


def is_connected(graph: Union[RawSemanticGraph, StabilizedGraph]) -> bool:
    """
    Check if the graph is connected (all nodes reachable from any node).

    Args:
        graph: Either a RawSemanticGraph or StabilizedGraph

    Returns:
        True if connected, False otherwise
    """
    if not graph.nodes:
        return True  # Empty graph is trivially connected

    adj = build_adjacency_set(graph, directed=False)
    start_node = graph.nodes[0].node_id
    visited: Set[int] = set()
    queue = [start_node]

    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        for neighbor in adj.get(node, set()):
            if neighbor not in visited:
                queue.append(neighbor)

    return len(visited) == len(graph.nodes)


# =============================================================================
# Edge Pruning
# =============================================================================

def prune_edges(
    graph: Union[RawSemanticGraph, StabilizedGraph],
    min_strength: float = 0.1,
    min_probability: float = 0.0
) -> List[SemanticEdge]:
    """
    Prune edges below strength or probability thresholds.

    Args:
        graph: Either a RawSemanticGraph or StabilizedGraph
        min_strength: Minimum effective strength (weight * probability)
        min_probability: Minimum edge probability

    Returns:
        List of edges that pass the thresholds
    """
    edges = _get_edges(graph)
    pruned = []

    for edge in edges:
        if edge.probability < min_probability:
            continue
        if edge.effective_strength() < min_strength:
            continue
        pruned.append(edge.copy())

    logger.debug(f"Pruned {len(edges) - len(pruned)} edges, kept {len(pruned)}")
    return pruned


def prune_edges_by_weight(
    graph: Union[RawSemanticGraph, StabilizedGraph],
    min_weight: float
) -> List[SemanticEdge]:
    """
    Prune edges below a weight threshold.

    Args:
        graph: Either a RawSemanticGraph or StabilizedGraph
        min_weight: Minimum edge weight

    Returns:
        List of edges with weight >= min_weight
    """
    edges = _get_edges(graph)
    return [edge.copy() for edge in edges if edge.weight >= min_weight]


# =============================================================================
# Graph Statistics
# =============================================================================

def compute_node_degrees(
    graph: Union[RawSemanticGraph, StabilizedGraph]
) -> Dict[int, int]:
    """
    Compute degree (number of edges) for each node.

    Args:
        graph: Either a RawSemanticGraph or StabilizedGraph

    Returns:
        Dictionary mapping node_id -> degree
    """
    degrees: Dict[int, int] = {node.node_id: 0 for node in graph.nodes}
    edges = _get_edges(graph)

    for edge in edges:
        if edge.source_id in degrees:
            degrees[edge.source_id] += 1
        if edge.target_id in degrees:
            degrees[edge.target_id] += 1

    return degrees


def compute_weighted_degrees(
    graph: Union[RawSemanticGraph, StabilizedGraph]
) -> Dict[int, float]:
    """
    Compute weighted degree (sum of edge weights) for each node.

    Args:
        graph: Either a RawSemanticGraph or StabilizedGraph

    Returns:
        Dictionary mapping node_id -> weighted degree
    """
    degrees: Dict[int, float] = {node.node_id: 0.0 for node in graph.nodes}
    edges = _get_edges(graph)

    for edge in edges:
        w = edge.effective_strength()
        if edge.source_id in degrees:
            degrees[edge.source_id] += w
        if edge.target_id in degrees:
            degrees[edge.target_id] += w

    return degrees


# =============================================================================
# Helper Functions
# =============================================================================

def _get_edges(
    graph: Union[RawSemanticGraph, StabilizedGraph]
) -> List[SemanticEdge]:
    """
    Get edges from either RawSemanticGraph or StabilizedGraph.

    RawSemanticGraph uses 'edge_candidates', StabilizedGraph uses 'edges'.
    """
    if isinstance(graph, RawSemanticGraph):
        return graph.edge_candidates
    elif isinstance(graph, StabilizedGraph):
        return graph.edges
    else:
        # Fallback: check for both attributes
        if hasattr(graph, 'edges'):
            return graph.edges
        elif hasattr(graph, 'edge_candidates'):
            return graph.edge_candidates
        else:
            return []


def copy_graph_structure(
    source: Union[RawSemanticGraph, StabilizedGraph]
) -> StabilizedGraph:
    """
    Create a deep copy of a graph as a StabilizedGraph.

    Args:
        source: Source graph to copy

    Returns:
        New StabilizedGraph with copied nodes and edges
    """
    nodes = [node.copy() for node in source.nodes]
    edges = [edge.copy() for edge in _get_edges(source)]

    return StabilizedGraph(
        timestamp_ns=source.timestamp_ns,
        nodes=nodes,
        edges=edges,
        global_energy=getattr(source, 'global_energy', 0.0)
    )


# =============================================================================
# CPL_006: Isolated Node Pruning
# =============================================================================

def prune_isolated_nodes(
    graph: StabilizedGraph,
    remove: bool = True
) -> StabilizedGraph:
    """
    Remove or flag nodes with zero edges (CPL_006 compliance).
    
    Enforces the 'existence is defined by edges' constraint from
    spec/01_constraints.md: A node without edges has no topological
    existence and should be removed.
    
    Args:
        graph: StabilizedGraph to prune
        remove: If True, return new graph without isolated nodes.
                If False, just log warnings but keep all nodes.
    
    Returns:
        New StabilizedGraph with isolated nodes removed (if remove=True)
        or the original graph (if remove=False)
    
    Reference: spec/01_constraints.md Section 2.3 (Edge-Defined Existence)
    """
    # Compute node degrees
    degrees = compute_node_degrees(graph)
    
    # Find isolated nodes
    isolated_ids = {nid for nid, degree in degrees.items() if degree == 0}
    
    if not isolated_ids:
        logger.debug("No isolated nodes found")
        return graph
    
    logger.info(f"CPL_006: Found {len(isolated_ids)} isolated nodes: {isolated_ids}")
    
    if not remove:
        logger.warning(
            f"Isolated nodes exist but remove=False. "
            f"These nodes have no topological existence: {isolated_ids}"
        )
        return graph
    
    # Filter nodes and edges
    kept_nodes = [node.copy() for node in graph.nodes if node.node_id not in isolated_ids]
    kept_edges = [edge.copy() for edge in graph.edges]  # Edges already valid
    
    logger.info(f"CPL_006: Pruned {len(isolated_ids)} isolated nodes, kept {len(kept_nodes)}")
    
    return StabilizedGraph(
        timestamp_ns=graph.timestamp_ns,
        nodes=kept_nodes,
        edges=kept_edges,
        global_energy=graph.global_energy,
        is_valid=graph.is_valid,
        iterations_used=graph.iterations_used
    )


def get_isolated_node_ids(
    graph: Union[RawSemanticGraph, StabilizedGraph]
) -> Set[int]:
    """
    Get the set of node IDs that have no edges.
    
    Args:
        graph: Either a RawSemanticGraph or StabilizedGraph
    
    Returns:
        Set of node IDs with zero edges
    """
    degrees = compute_node_degrees(graph)
    return {nid for nid, degree in degrees.items() if degree == 0}

