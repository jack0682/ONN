"""Unit tests for ONN Core Graph module.

Reference: spec/20_impl_plan.ir.yml IMPL_015, IMPL_023
"""

import pytest
import torch

from onn.core.graph import (
    EdgeKey,
    EdgeGraph,
    compute_graph_laplacian,
    compute_edge_laplacian,
    get_node_degrees,
)


class TestEdgeKey:
    """Tests for EdgeKey dataclass."""
    
    def test_creation(self):
        """EdgeKey can be created with source and target IDs."""
        ek = EdgeKey(source_id=1, target_id=2)
        assert ek.source_id == 1
        assert ek.target_id == 2
    
    def test_hash(self):
        """EdgeKeys are hashable and can be used in sets."""
        ek1 = EdgeKey(1, 2)
        ek2 = EdgeKey(1, 2)
        ek3 = EdgeKey(2, 1)
        
        assert hash(ek1) == hash(ek2)
        assert ek1 == ek2
        assert ek1 != ek3
        
        s = {ek1, ek2, ek3}
        assert len(s) == 2
    
    def test_reversed(self):
        """Reversed creates the opposite direction edge."""
        ek = EdgeKey(1, 2)
        rev = ek.reversed()
        
        assert rev.source_id == 2
        assert rev.target_id == 1


class TestEdgeGraph:
    """Tests for EdgeGraph class."""
    
    def test_from_edge_list_simple(self):
        """Create graph from edge list."""
        edges = [(1, 2), (2, 3), (3, 1)]
        graph = EdgeGraph.from_edge_list(edges)
        
        assert graph.num_nodes == 3
        assert graph.num_edges == 3
    
    def test_from_edge_list_with_node_ids(self):
        """Create graph with explicit node IDs."""
        edges = [(1, 2)]
        graph = EdgeGraph.from_edge_list(edges, node_ids=[1, 2, 3])
        
        assert graph.num_nodes == 3
        assert graph.num_edges == 1
    
    def test_edge_index_format(self):
        """Edge index is in COO format (2, m)."""
        edges = [(0, 1), (1, 2)]
        graph = EdgeGraph.from_edge_list(edges)
        
        assert graph.edge_index.shape == (2, 2)
        assert graph.edge_index.dtype == torch.long
    
    def test_is_connected_true(self):
        """Connected graph returns True."""
        edges = [(1, 2), (2, 3)]
        graph = EdgeGraph.from_edge_list(edges)
        
        assert graph.is_connected is True
    
    def test_is_connected_false(self):
        """Disconnected graph returns False."""
        edges = [(1, 2), (3, 4)]  # Two components
        graph = EdgeGraph.from_edge_list(edges)
        
        assert graph.is_connected is False
    
    def test_incidence_matrix_triangle(self):
        """Incidence matrix for a triangle graph."""
        edges = [(0, 1), (1, 2), (2, 0)]
        graph = EdgeGraph.from_edge_list(edges)
        
        B = graph.get_incidence_matrix()
        
        assert B.shape == (3, 3)  # 3 nodes, 3 edges
        # Each row sums to 0 (for undirected interpretation)
        # Each column has exactly one +1 and one -1
        for col in range(3):
            assert B[:, col].sum() == 0
            assert (B[:, col] == 1).sum() == 1
            assert (B[:, col] == -1).sum() == 1


class TestLaplacians:
    """Tests for Laplacian computations."""
    
    def test_graph_laplacian_path(self):
        """Graph Laplacian for a path graph (0-1-2)."""
        edges = [(0, 1), (1, 2)]
        graph = EdgeGraph.from_edge_list(edges)
        
        L = compute_graph_laplacian(graph)
        
        assert L.shape == (3, 3)
        # Diagonal = degrees
        assert L[0, 0] == 1  # deg(0) = 1
        assert L[1, 1] == 2  # deg(1) = 2
        assert L[2, 2] == 1  # deg(2) = 1
    
    def test_edge_laplacian_path(self):
        """Edge Laplacian for a path graph."""
        edges = [(0, 1), (1, 2)]
        graph = EdgeGraph.from_edge_list(edges)
        
        L1 = compute_edge_laplacian(graph)
        
        assert L1.shape == (2, 2)
    
    def test_node_degrees(self):
        """Node degrees for a triangle."""
        edges = [(0, 1), (1, 2), (2, 0)]
        graph = EdgeGraph.from_edge_list(edges)
        
        degrees = get_node_degrees(graph)
        
        assert degrees.shape == (3,)
        assert (degrees == 2).all()  # All nodes have degree 2 in a triangle
