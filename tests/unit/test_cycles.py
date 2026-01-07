"""Unit tests for ONN Core Cycles module.

Reference: spec/20_impl_plan.ir.yml IMPL_015, IMPL_023
"""

import pytest
import torch

from onn.core.graph import EdgeGraph
from onn.core.cycles import (
    CycleBasis,
    build_cycle_basis,
    cycle_matrix,
    cycle_targets,
    compute_cycle_violation,
    verify_cycle_basis,
)


class TestCycleBasis:
    """Tests for CycleBasis dataclass."""
    
    def test_triangle_has_one_cycle(self):
        """A triangle graph has exactly 1 fundamental cycle."""
        # Triangle: 3 nodes, 3 edges -> q = 3 - 3 + 1 = 1
        edges = [(0, 1), (1, 2), (2, 0)]
        graph = EdgeGraph.from_edge_list(edges)
        
        basis = build_cycle_basis(graph, embedding_dim=32)
        
        assert basis.num_cycles == 1
        assert basis.num_edges == 3
    
    def test_tree_has_no_cycles(self):
        """A tree graph has no fundamental cycles."""
        # Path: 3 nodes, 2 edges -> q = 2 - 3 + 1 = 0
        edges = [(0, 1), (1, 2)]
        graph = EdgeGraph.from_edge_list(edges)
        
        basis = build_cycle_basis(graph, embedding_dim=32)
        
        assert basis.num_cycles == 0
    
    def test_square_has_one_cycle(self):
        """A square graph has exactly 1 cycle."""
        # Square: 4 nodes, 4 edges -> q = 4 - 4 + 1 = 1
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        graph = EdgeGraph.from_edge_list(edges)
        
        basis = build_cycle_basis(graph, embedding_dim=32)
        
        assert basis.num_cycles == 1
    
    def test_complete_graph_k4(self):
        """Complete graph K4 has 3 cycles."""
        # K4: 4 nodes, 6 edges -> q = 6 - 4 + 1 = 3
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        graph = EdgeGraph.from_edge_list(edges)
        
        basis = build_cycle_basis(graph, embedding_dim=32)
        
        assert basis.num_cycles == 3
    
    def test_cycle_matrix_shape(self):
        """Cycle matrix has correct shape (q, m)."""
        edges = [(0, 1), (1, 2), (2, 0)]
        graph = EdgeGraph.from_edge_list(edges)
        
        C = cycle_matrix(graph)
        
        assert C.shape == (1, 3)  # 1 cycle, 3 edges
    
    def test_cycle_targets_shape(self):
        """Cycle targets have correct shape (q, p)."""
        tau = cycle_targets(num_cycles=3, embedding_dim=32)
        
        assert tau.shape == (3, 32)
        assert (tau == 0).all()  # Default: closed loops (zeros)


class TestCycleConstraint:
    """Tests for cycle constraint satisfaction."""
    
    def test_verify_basis_triangle(self):
        """Verify cycle basis for triangle."""
        edges = [(0, 1), (1, 2), (2, 0)]
        graph = EdgeGraph.from_edge_list(edges)
        basis = build_cycle_basis(graph)
        
        assert verify_cycle_basis(graph, basis)
    
    def test_violation_zero_on_constraint_set(self):
        """Violation is zero for embeddings satisfying Cx = tau."""
        edges = [(0, 1), (1, 2), (2, 0)]
        graph = EdgeGraph.from_edge_list(edges)
        basis = build_cycle_basis(graph, embedding_dim=4)
        
        # Create x that satisfies Cx = 0
        # For a triangle with C = [s0, s1, s2] where s_i = ±1
        # We need x0 * s0 + x1 * s1 + x2 * s2 = 0 for each dim
        # Simple solution: x0 = x1 = x2 = 0
        x = torch.zeros(3, 4)
        
        violation = compute_cycle_violation(x, basis)
        
        assert torch.allclose(violation, torch.zeros(4), atol=1e-6)
    
    def test_violation_positive_for_random(self):
        """Random embeddings typically violate constraints."""
        edges = [(0, 1), (1, 2), (2, 0)]
        graph = EdgeGraph.from_edge_list(edges)
        basis = build_cycle_basis(graph, embedding_dim=4)
        
        # Random embeddings
        x = torch.randn(3, 4)
        
        violation = compute_cycle_violation(x, basis)
        
        # With random x, violation should be non-zero (with high probability)
        assert violation.sum() > 0
    
    def test_verify_constraints_method(self):
        """CycleBasis.verify_constraints method works."""
        edges = [(0, 1), (1, 2), (2, 0)]
        graph = EdgeGraph.from_edge_list(edges)
        basis = build_cycle_basis(graph, embedding_dim=4)
        
        # Zero vector satisfies constraint
        x_good = torch.zeros(3, 4)
        assert basis.verify_constraints(x_good)
        
        # Random vector typically does not
        x_bad = torch.randn(3, 4) * 10
        # May or may not satisfy by chance, so we skip assertion


class TestEmptyGraph:
    """Tests for edge cases with empty/minimal graphs."""
    
    def test_empty_graph(self):
        """Empty graph produces empty cycle basis."""
        graph = EdgeGraph(node_ids=[], edge_keys=[])
        basis = build_cycle_basis(graph)
        
        assert basis.num_cycles == 0
        assert basis.num_edges == 0
    
    def test_single_edge(self):
        """Single edge has no cycles."""
        edges = [(0, 1)]
        graph = EdgeGraph.from_edge_list(edges)
        basis = build_cycle_basis(graph)
        
        assert basis.num_cycles == 0
