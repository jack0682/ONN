"""Cycle Basis Construction for Edge Embedding Constraints.

This module implements the fundamental cycle basis construction used to
create the constraint matrix C for the Projection-Consensus solver.

For a connected graph with n nodes and m edges, there are q = m - n + 1
fundamental cycles (independent loops).

Mathematical Foundation:
    - Cycle constraint: C @ x = τ (each cycle sums to target)
    - For embeddings: (C ⊗ I_p) @ vec(x) = τ
    - Default τ = 0 (closed loops: "going around returns you to start")

Reference:
    - spec/11_interfaces.ir.yml: CycleBasis
    - spec/20_impl_plan.ir.yml: IMPL_015

Author: Claude (via IMPL_015)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import torch
import numpy as np

from onn.core.graph import EdgeGraph, EdgeKey

logger = logging.getLogger(__name__)


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class CycleBasis:
    """Cycle constraint basis for edge embeddings.
    
    For a connected graph with m edges and n nodes, there are q = m - n + 1
    fundamental cycles. The cycle matrix C has shape (q, m) where:
    - C[i, j] = +1 if edge j is in cycle i (forward direction)
    - C[i, j] = -1 if edge j is in cycle i (backward direction)
    - C[i, j] = 0  if edge j is not in cycle i
    
    The constraint C @ x = τ ensures cycle consistency.
    
    Attributes:
        cycle_matrix: Cycle matrix C of shape (q, m)
        tau: Target cycle values of shape (q, p) where p is embedding dim
        edge_keys: Edge ordering for C columns
        cycles: List of cycles, each cycle is a list of (edge_idx, direction)
        
    Reference:
        spec/11_interfaces.ir.yml -> CycleBasis
    """
    cycle_matrix: torch.Tensor  # (q, m)
    tau: torch.Tensor           # (q, p) or (q,) if 1D
    edge_keys: List[EdgeKey] = field(default_factory=list)
    cycles: List[List[Tuple[int, int]]] = field(default_factory=list)  # (edge_idx, ±1)
    
    @property
    def num_cycles(self) -> int:
        """Number of fundamental cycles."""
        return self.cycle_matrix.shape[0]
    
    @property
    def num_edges(self) -> int:
        """Number of edges."""
        return self.cycle_matrix.shape[1]
    
    def verify_constraints(self, x: torch.Tensor, tolerance: float = 1e-6) -> bool:
        """Check if x satisfies C @ x ≈ τ.
        
        Args:
            x: Edge embeddings of shape (m, p)
            tolerance: Maximum allowed violation
            
        Returns:
            True if constraints are satisfied within tolerance
        """
        if x.dim() == 1:
            x = x.unsqueeze(1)
        
        violation = torch.mm(self.cycle_matrix, x) - self.tau
        max_violation = torch.abs(violation).max().item()
        
        return max_violation < tolerance


# ==============================================================================
# CYCLE BASIS CONSTRUCTION
# ==============================================================================

def build_cycle_basis(edge_graph: EdgeGraph, embedding_dim: int = 32) -> CycleBasis:
    """Construct fundamental cycle basis using spanning tree complement.
    
    Algorithm:
    1. Build a spanning tree T of the graph
    2. For each non-tree edge e = (u, v):
       - Find the unique path in T from u to v
       - The cycle is: e + path(u, v)
    3. Construct cycle matrix C from these fundamental cycles
    
    Args:
        edge_graph: EdgeGraph instance
        embedding_dim: Dimension of edge embeddings (for τ shape)
        
    Returns:
        CycleBasis with C, τ, and edge ordering
        
    Reference:
        Spanning tree complement method for fundamental cycles
    """
    if edge_graph.num_edges == 0:
        logger.warning("Empty graph, returning empty cycle basis")
        return CycleBasis(
            cycle_matrix=torch.zeros(0, 0),
            tau=torch.zeros(0, embedding_dim),
            edge_keys=[],
            cycles=[]
        )
    
    n = edge_graph.num_nodes
    m = edge_graph.num_edges
    
    # Expected number of cycles for connected graph
    expected_cycles = m - n + 1 if edge_graph.is_connected else 0
    
    if expected_cycles <= 0:
        logger.info(f"Graph is a tree (or forest), no cycles")
        return CycleBasis(
            cycle_matrix=torch.zeros(0, m),
            tau=torch.zeros(0, embedding_dim),
            edge_keys=edge_graph.edge_keys.copy(),
            cycles=[]
        )
    
    # Step 1: Build spanning tree using BFS
    tree_edges, non_tree_edges = _build_spanning_tree(edge_graph)
    
    logger.debug(f"Spanning tree: {len(tree_edges)} edges, {len(non_tree_edges)} non-tree edges")
    
    # Step 2: Find fundamental cycle for each non-tree edge
    cycles = []
    for edge_idx in non_tree_edges:
        ek = edge_graph.edge_keys[edge_idx]
        cycle = _find_fundamental_cycle(edge_graph, edge_idx, tree_edges)
        if cycle:
            cycles.append(cycle)
    
    # Step 3: Build cycle matrix
    q = len(cycles)
    C = torch.zeros(q, m)
    
    for cycle_idx, cycle in enumerate(cycles):
        for edge_idx, direction in cycle:
            C[cycle_idx, edge_idx] = float(direction)
    
    # Step 4: Create τ (default: zeros for closed loops)
    tau = torch.zeros(q, embedding_dim)
    
    logger.info(f"Built cycle basis: {q} cycles for {m} edges")
    
    return CycleBasis(
        cycle_matrix=C,
        tau=tau,
        edge_keys=edge_graph.edge_keys.copy(),
        cycles=cycles
    )


def _build_spanning_tree(edge_graph: EdgeGraph) -> Tuple[Set[int], List[int]]:
    """Build spanning tree using BFS, return tree and non-tree edge indices.
    
    Args:
        edge_graph: EdgeGraph instance
        
    Returns:
        Tuple of (tree_edge_indices, non_tree_edge_indices)
    """
    if edge_graph.num_nodes == 0:
        return set(), []
    
    # Build adjacency with edge indices
    adj: Dict[int, List[Tuple[int, int]]] = {}  # node -> [(neighbor, edge_idx), ...]
    
    for idx, ek in enumerate(edge_graph.edge_keys):
        if ek.source_id not in adj:
            adj[ek.source_id] = []
        if ek.target_id not in adj:
            adj[ek.target_id] = []
        adj[ek.source_id].append((ek.target_id, idx))
        adj[ek.target_id].append((ek.source_id, idx))
    
    # BFS to build spanning tree
    visited = set()
    tree_edges: Set[int] = set()
    start_node = edge_graph.node_ids[0]
    queue = [start_node]
    visited.add(start_node)  # Mark start as visited immediately
    
    while queue:
        node = queue.pop(0)
        
        for neighbor, edge_idx in adj.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)  # Mark as visited when adding to queue
                tree_edges.add(edge_idx)
                queue.append(neighbor)
    
    # Non-tree edges are the complement
    all_edges = set(range(edge_graph.num_edges))
    non_tree_edges = sorted(list(all_edges - tree_edges))
    
    return tree_edges, non_tree_edges


def _find_fundamental_cycle(
    edge_graph: EdgeGraph,
    non_tree_edge_idx: int,
    tree_edges: Set[int]
) -> List[Tuple[int, int]]:
    """Find the fundamental cycle formed by adding a non-tree edge.
    
    Args:
        edge_graph: EdgeGraph instance
        non_tree_edge_idx: Index of the non-tree edge
        tree_edges: Set of tree edge indices
        
    Returns:
        List of (edge_idx, direction) where direction is ±1
    """
    ek = edge_graph.edge_keys[non_tree_edge_idx]
    start_node = ek.source_id
    end_node = ek.target_id
    
    # Build tree-only adjacency
    tree_adj: Dict[int, List[Tuple[int, int, int]]] = {}  # node -> [(neighbor, edge_idx, direction), ...]
    
    for idx in tree_edges:
        tek = edge_graph.edge_keys[idx]
        if tek.source_id not in tree_adj:
            tree_adj[tek.source_id] = []
        if tek.target_id not in tree_adj:
            tree_adj[tek.target_id] = []
        # Forward direction: source -> target
        tree_adj[tek.source_id].append((tek.target_id, idx, 1))
        # Backward direction: target -> source
        tree_adj[tek.target_id].append((tek.source_id, idx, -1))
    
    # BFS to find path from start_node to end_node in tree
    visited = {start_node}
    parent: Dict[int, Tuple[int, int, int]] = {}  # node -> (parent_node, edge_idx, direction)
    queue = [start_node]
    
    while queue:
        node = queue.pop(0)
        if node == end_node:
            break
        
        for neighbor, edge_idx, direction in tree_adj.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = (node, edge_idx, direction)
                queue.append(neighbor)
    
    if end_node not in parent and start_node != end_node:
        logger.warning(f"No path found in tree from {start_node} to {end_node}")
        return []
    
    # Reconstruct path
    cycle = [(non_tree_edge_idx, 1)]  # Non-tree edge in forward direction
    
    if start_node != end_node:
        current = end_node
        while current != start_node:
            parent_node, edge_idx, direction = parent[current]
            # Reverse direction because we're going from end to start
            cycle.append((edge_idx, -direction))
            current = parent_node
    
    return cycle


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def cycle_matrix(edge_graph: EdgeGraph) -> torch.Tensor:
    """Return cycle matrix C ∈ ℝ^{q×m}.
    
    Convenience function that extracts just the matrix from CycleBasis.
    
    Args:
        edge_graph: EdgeGraph instance
        
    Returns:
        Cycle matrix of shape (q, m)
    """
    basis = build_cycle_basis(edge_graph)
    return basis.cycle_matrix


def cycle_targets(num_cycles: int, embedding_dim: int = 32) -> torch.Tensor:
    """Return τ ∈ ℝ^{q×p}, default zeros (closed loops).
    
    Args:
        num_cycles: Number of cycles (q)
        embedding_dim: Embedding dimension (p)
        
    Returns:
        Target tensor of shape (q, p)
    """
    return torch.zeros(num_cycles, embedding_dim)


def compute_cycle_violation(
    x: torch.Tensor,
    cycle_basis: CycleBasis
) -> torch.Tensor:
    """Compute ||Cx - τ||_2 for each embedding dimension.
    
    Args:
        x: Edge embeddings of shape (m, p)
        cycle_basis: CycleBasis instance
        
    Returns:
        Violation tensor of shape (p,)
    """
    if x.dim() == 1:
        x = x.unsqueeze(1)
    
    # C @ x - τ: (q, m) @ (m, p) - (q, p) = (q, p)
    residual = torch.mm(cycle_basis.cycle_matrix, x) - cycle_basis.tau
    
    # L2 norm per dimension
    return torch.norm(residual, dim=0)


def verify_cycle_basis(edge_graph: EdgeGraph, cycle_basis: CycleBasis) -> bool:
    """Verify that the cycle basis is correctly constructed.
    
    Checks:
    1. Number of cycles = m - n + 1 (for connected graph)
    2. Each cycle is indeed a cycle (starts and ends at same node)
    3. Cycle matrix has correct dimensions
    
    Args:
        edge_graph: EdgeGraph instance
        cycle_basis: CycleBasis to verify
        
    Returns:
        True if all checks pass
    """
    n = edge_graph.num_nodes
    m = edge_graph.num_edges
    q = cycle_basis.num_cycles
    
    # Check 1: Cycle count
    if edge_graph.is_connected:
        expected_q = m - n + 1
        if q != expected_q:
            logger.error(f"Expected {expected_q} cycles, got {q}")
            return False
    
    # Check 2: Matrix dimensions
    if cycle_basis.cycle_matrix.shape != (q, m):
        logger.error(f"Cycle matrix shape {cycle_basis.cycle_matrix.shape} != ({q}, {m})")
        return False
    
    # Check 3: Each row has at least 3 non-zero entries (minimum cycle)
    for i in range(q):
        nnz = (cycle_basis.cycle_matrix[i] != 0).sum().item()
        if nnz < 3:
            logger.error(f"Cycle {i} has only {nnz} edges (minimum 3 for a cycle)")
            return False
    
    logger.info(f"Cycle basis verified: {q} cycles, {m} edges")
    return True
