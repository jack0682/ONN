"""ONN Core Graph Representation for Edge Embeddings.

This module provides the EdgeGraph class for edge-centric graph representation
used by the Projection-Consensus solver.

Reference:
    - spec/11_interfaces.ir.yml: EdgeKey, EdgeEmbeddingState
    - spec/20_impl_plan.ir.yml: IMPL_015

Author: Claude (via IMPL_015)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import numpy as np

if TYPE_CHECKING:
    from onn.core.tensors import RawSemanticGraph, SemanticEdge

logger = logging.getLogger(__name__)


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class EdgeKey:
    """Canonical edge identifier.
    
    Reference: spec/11_interfaces.ir.yml -> EdgeKey
    """
    source_id: int
    target_id: int
    
    def __hash__(self) -> int:
        return hash((self.source_id, self.target_id))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EdgeKey):
            return False
        return self.source_id == other.source_id and self.target_id == other.target_id
    
    def reversed(self) -> "EdgeKey":
        """Return the reversed edge key."""
        return EdgeKey(source_id=self.target_id, target_id=self.source_id)


@dataclass
class EdgeGraph:
    """Edge-centric graph representation for ONN solver.
    
    This class represents a graph where the focus is on edges (relationships)
    rather than nodes. Each edge carries an embedding x_e ∈ ℝ^p.
    
    Attributes:
        node_ids: List of unique node identifiers (V)
        edge_keys: List of EdgeKey objects defining edges (E)
        edge_index: PyTorch tensor of shape (2, m) in COO format
        orientation: Tensor of shape (m,) with ±1 values for directed edges
        
    Reference:
        - spec/20_impl_plan.ir.yml: IMPL_015
        - Forman-Ricci curvature uses edge-centric representation
    """
    node_ids: List[int] = field(default_factory=list)
    edge_keys: List[EdgeKey] = field(default_factory=list)
    edge_index: Optional[torch.Tensor] = None  # (2, m) COO format
    orientation: Optional[torch.Tensor] = None  # (m,) ±1 values
    
    # Cached data structures
    _node_id_to_idx: Optional[dict] = field(default=None, repr=False)
    _adjacency: Optional[dict] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Build cached data structures after initialization."""
        if self.node_ids:
            self._build_node_id_map()
        if self.edge_keys and self.edge_index is None:
            self._build_edge_index()
    
    def _build_node_id_map(self) -> None:
        """Build node_id to index mapping."""
        self._node_id_to_idx = {nid: idx for idx, nid in enumerate(self.node_ids)}
    
    def _build_edge_index(self) -> None:
        """Build edge_index tensor from edge_keys."""
        if not self.edge_keys or not self._node_id_to_idx:
            return
        
        sources = []
        targets = []
        orientations = []
        
        for ek in self.edge_keys:
            src_idx = self._node_id_to_idx.get(ek.source_id)
            tgt_idx = self._node_id_to_idx.get(ek.target_id)
            
            if src_idx is None or tgt_idx is None:
                logger.warning(f"Edge {ek} references unknown node(s)")
                continue
            
            sources.append(src_idx)
            targets.append(tgt_idx)
            # Orientation: +1 for forward (source < target), -1 for backward
            orientations.append(1.0 if src_idx < tgt_idx else -1.0)
        
        if sources:
            self.edge_index = torch.tensor([sources, targets], dtype=torch.long)
            self.orientation = torch.tensor(orientations, dtype=torch.float32)
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return len(self.node_ids)
    
    @property
    def num_edges(self) -> int:
        """Number of edges in the graph."""
        return len(self.edge_keys)
    
    @property
    def is_connected(self) -> bool:
        """Check if the graph is connected using BFS."""
        if self.num_nodes <= 1:
            return True
        
        # Build adjacency list
        adj = self._get_adjacency()
        
        # BFS from first node
        visited = set()
        queue = [self.node_ids[0]]
        
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        
        return len(visited) == self.num_nodes
    
    def _get_adjacency(self) -> dict:
        """Get or build adjacency list (undirected)."""
        if self._adjacency is not None:
            return self._adjacency
        
        adj = {}
        for ek in self.edge_keys:
            if ek.source_id not in adj:
                adj[ek.source_id] = []
            if ek.target_id not in adj:
                adj[ek.target_id] = []
            adj[ek.source_id].append(ek.target_id)
            adj[ek.target_id].append(ek.source_id)
        
        self._adjacency = adj
        return adj
    
    def get_incidence_matrix(self) -> torch.Tensor:
        """Compute incidence matrix B ∈ ℝ^{n×m}.
        
        B[i, e] = +1 if node i is source of edge e
        B[i, e] = -1 if node i is target of edge e
        B[i, e] = 0  otherwise
        
        Returns:
            Incidence matrix of shape (n, m)
        """
        n = self.num_nodes
        m = self.num_edges
        
        if m == 0:
            return torch.zeros(n, 0)
        
        B = torch.zeros(n, m)
        
        for e_idx, ek in enumerate(self.edge_keys):
            src_idx = self._node_id_to_idx[ek.source_id]
            tgt_idx = self._node_id_to_idx[ek.target_id]
            B[src_idx, e_idx] = 1.0
            B[tgt_idx, e_idx] = -1.0
        
        return B
    
    @classmethod
    def from_raw_graph(cls, raw_graph: "RawSemanticGraph") -> "EdgeGraph":
        """Create EdgeGraph from RawSemanticGraph.
        
        Args:
            raw_graph: Raw semantic graph from SEGO
            
        Returns:
            EdgeGraph instance
            
        Reference:
            spec/11_interfaces.ir.yml -> RawSemanticGraph
        """
        # Extract unique node IDs
        node_ids = [node.node_id for node in raw_graph.nodes]
        
        # Extract edge keys from edge candidates
        edge_keys = []
        for edge in raw_graph.edge_candidates:
            ek = EdgeKey(
                source_id=edge.source_id,
                target_id=edge.target_id
            )
            edge_keys.append(ek)
        
        logger.debug(f"Created EdgeGraph with {len(node_ids)} nodes, {len(edge_keys)} edges")
        
        return cls(node_ids=node_ids, edge_keys=edge_keys)
    
    @classmethod
    def from_edge_list(
        cls, 
        edges: List[Tuple[int, int]],
        node_ids: Optional[List[int]] = None
    ) -> "EdgeGraph":
        """Create EdgeGraph from a list of (source, target) tuples.
        
        Args:
            edges: List of (source_id, target_id) tuples
            node_ids: Optional explicit node list. If None, inferred from edges.
            
        Returns:
            EdgeGraph instance
        """
        edge_keys = [EdgeKey(source_id=s, target_id=t) for s, t in edges]
        
        if node_ids is None:
            # Infer node IDs from edges
            all_ids = set()
            for s, t in edges:
                all_ids.add(s)
                all_ids.add(t)
            node_ids = sorted(list(all_ids))
        
        return cls(node_ids=node_ids, edge_keys=edge_keys)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def compute_graph_laplacian(edge_graph: EdgeGraph) -> torch.Tensor:
    """Compute the graph Laplacian L = B @ B^T.
    
    Args:
        edge_graph: EdgeGraph instance
        
    Returns:
        Laplacian matrix of shape (n, n)
    """
    B = edge_graph.get_incidence_matrix()
    return torch.mm(B, B.t())


def compute_edge_laplacian(edge_graph: EdgeGraph) -> torch.Tensor:
    """Compute the edge Laplacian L_1 = B^T @ B.
    
    Used for smoothness regularization on edge embeddings.
    
    Args:
        edge_graph: EdgeGraph instance
        
    Returns:
        Edge Laplacian matrix of shape (m, m)
    """
    B = edge_graph.get_incidence_matrix()
    return torch.mm(B.t(), B)


def get_node_degrees(edge_graph: EdgeGraph) -> torch.Tensor:
    """Compute node degrees (undirected).
    
    Args:
        edge_graph: EdgeGraph instance
        
    Returns:
        Degree tensor of shape (n,)
    """
    degrees = torch.zeros(edge_graph.num_nodes)
    
    for ek in edge_graph.edge_keys:
        src_idx = edge_graph._node_id_to_idx[ek.source_id]
        tgt_idx = edge_graph._node_id_to_idx[ek.target_id]
        degrees[src_idx] += 1
        degrees[tgt_idx] += 1
    
    return degrees
