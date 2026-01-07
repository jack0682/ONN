"""
Unit tests for graph utilities.

Reference: spec/20_impl_plan.ir.yml IMPL_002
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from onn.core.tensors import (
    SemanticNode,
    SemanticEdge,
    RawSemanticGraph,
    StabilizedGraph,
)
from onn.core.graph_utils import (
    build_adjacency,
    build_adjacency_set,
    build_weighted_adjacency,
    node_lookup,
    get_node,
    edge_lookup,
    get_edge,
    get_edges_for_node,
    validate_graph,
    GraphValidationError,
    is_connected,
    prune_edges,
    compute_node_degrees,
    compute_weighted_degrees,
)


@pytest.fixture
def simple_graph() -> StabilizedGraph:
    """Create a simple graph for testing."""
    nodes = [SemanticNode(node_id=i) for i in [1, 2, 3]]
    edges = [
        SemanticEdge(source_id=1, target_id=2, relation_embedding=np.zeros(16), weight=1.0, probability=0.9),
        SemanticEdge(source_id=2, target_id=3, relation_embedding=np.zeros(16), weight=0.5, probability=0.8),
    ]
    return StabilizedGraph(timestamp_ns=0, nodes=nodes, edges=edges, global_energy=0.5)


class TestBuildAdjacency:
    """Tests for adjacency building functions."""

    def test_build_adjacency_undirected(self, simple_graph):
        """Undirected adjacency should include both directions."""
        adj = build_adjacency(simple_graph, directed=False)

        assert 2 in adj[1]
        assert 1 in adj[2]
        assert 3 in adj[2]
        assert 2 in adj[3]

    def test_build_adjacency_directed(self, simple_graph):
        """Directed adjacency should only include outgoing."""
        adj = build_adjacency(simple_graph, directed=True)

        assert 2 in adj[1]
        assert 1 not in adj[2]  # Only outgoing

    def test_build_adjacency_set(self, simple_graph):
        """Adjacency set should use sets."""
        adj = build_adjacency_set(simple_graph)

        assert isinstance(adj[1], set)
        assert 2 in adj[1]

    def test_build_weighted_adjacency(self, simple_graph):
        """Weighted adjacency should include edge weights."""
        adj = build_weighted_adjacency(simple_graph)

        # Weight from 1->2 is 1.0 * 0.9 = 0.9
        assert adj[1][2] == pytest.approx(0.9)

    def test_isolated_nodes_included(self):
        """Isolated nodes should have empty adjacency lists."""
        nodes = [SemanticNode(node_id=1), SemanticNode(node_id=2)]
        graph = StabilizedGraph(timestamp_ns=0, nodes=nodes, edges=[])

        adj = build_adjacency(graph)
        assert 1 in adj
        assert 2 in adj
        assert adj[1] == []
        assert adj[2] == []


class TestNodeLookup:
    """Tests for node lookup functions."""

    def test_node_lookup(self, simple_graph):
        """node_lookup should create dict by ID."""
        lookup = node_lookup(simple_graph)

        assert 1 in lookup
        assert lookup[1].node_id == 1

    def test_get_node(self, simple_graph):
        """get_node should return node by ID."""
        node = get_node(simple_graph, 2)
        assert node.node_id == 2

        missing = get_node(simple_graph, 999)
        assert missing is None


class TestEdgeLookup:
    """Tests for edge lookup functions."""

    def test_edge_lookup(self, simple_graph):
        """edge_lookup should create dict by (source, target)."""
        lookup = edge_lookup(simple_graph)

        assert (1, 2) in lookup
        assert lookup[(1, 2)].source_id == 1

    def test_get_edge(self, simple_graph):
        """get_edge should return edge by endpoints."""
        edge = get_edge(simple_graph, 1, 2)
        assert edge.source_id == 1

        # Also works with reversed order
        edge_rev = get_edge(simple_graph, 2, 1, check_reverse=True)
        assert edge_rev is not None

    def test_get_edges_for_node(self, simple_graph):
        """get_edges_for_node should return all connected edges."""
        edges = get_edges_for_node(simple_graph, 2)
        assert len(edges) == 2  # Connected to 1 and 3


class TestValidateGraph:
    """Tests for graph validation."""

    def test_valid_graph_passes(self, simple_graph):
        """Valid graph should return empty issues list."""
        issues = validate_graph(simple_graph)
        assert issues == []

    def test_duplicate_node_ids(self):
        """Duplicate node IDs should be flagged."""
        nodes = [SemanticNode(node_id=1), SemanticNode(node_id=1)]
        graph = StabilizedGraph(timestamp_ns=0, nodes=nodes, edges=[])

        issues = validate_graph(graph)
        assert len(issues) > 0
        assert "Duplicate" in issues[0]

    def test_invalid_edge_reference(self):
        """Edges referencing non-existent nodes should be flagged."""
        nodes = [SemanticNode(node_id=1)]
        edges = [SemanticEdge(source_id=1, target_id=999, relation_embedding=np.zeros(16))]
        graph = StabilizedGraph(timestamp_ns=0, nodes=nodes, edges=edges)

        issues = validate_graph(graph)
        assert len(issues) > 0

    def test_self_loop(self):
        """Self-loops should be flagged."""
        nodes = [SemanticNode(node_id=1)]
        edges = [SemanticEdge(source_id=1, target_id=1, relation_embedding=np.zeros(16))]
        graph = StabilizedGraph(timestamp_ns=0, nodes=nodes, edges=edges)

        issues = validate_graph(graph)
        assert any("self-loop" in i for i in issues)

    def test_strict_mode_raises(self):
        """Strict mode should raise on issues."""
        nodes = [SemanticNode(node_id=1), SemanticNode(node_id=1)]
        graph = StabilizedGraph(timestamp_ns=0, nodes=nodes, edges=[])

        with pytest.raises(GraphValidationError):
            validate_graph(graph, strict=True)


class TestIsConnected:
    """Tests for connectivity checking."""

    def test_connected_graph(self, simple_graph):
        """Connected graph should return True."""
        assert is_connected(simple_graph) is True

    def test_disconnected_graph(self):
        """Disconnected graph should return False."""
        nodes = [SemanticNode(node_id=i) for i in [1, 2, 3, 4]]
        edges = [
            SemanticEdge(source_id=1, target_id=2, relation_embedding=np.zeros(16)),
            # 3 and 4 are isolated
        ]
        graph = StabilizedGraph(timestamp_ns=0, nodes=nodes, edges=edges)

        assert is_connected(graph) is False

    def test_empty_graph_connected(self):
        """Empty graph is trivially connected."""
        graph = StabilizedGraph(timestamp_ns=0, nodes=[], edges=[])
        assert is_connected(graph) is True


class TestPruneEdges:
    """Tests for edge pruning."""

    def test_prune_low_strength(self, simple_graph):
        """Should prune edges below strength threshold."""
        # Edge 2->3 has strength 0.5 * 0.8 = 0.4
        pruned = prune_edges(simple_graph, min_strength=0.5)

        # Only edge 1->2 (strength 0.9) should remain
        assert len(pruned) == 1
        assert pruned[0].source_id == 1

    def test_prune_low_probability(self, simple_graph):
        """Should prune edges below probability threshold."""
        pruned = prune_edges(simple_graph, min_probability=0.85)

        # Only edge 1->2 (prob 0.9) should remain
        assert len(pruned) == 1


class TestDegreeComputation:
    """Tests for degree computation."""

    def test_compute_node_degrees(self, simple_graph):
        """Should compute correct degrees."""
        degrees = compute_node_degrees(simple_graph)

        assert degrees[1] == 1  # Connected to 2
        assert degrees[2] == 2  # Connected to 1 and 3
        assert degrees[3] == 1  # Connected to 2

    def test_compute_weighted_degrees(self, simple_graph):
        """Should compute correct weighted degrees."""
        degrees = compute_weighted_degrees(simple_graph)

        # Node 1: edge to 2 with strength 0.9
        assert degrees[1] == pytest.approx(0.9)

        # Node 2: edges with strengths 0.9 and 0.4
        assert degrees[2] == pytest.approx(1.3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
