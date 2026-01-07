import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from onn.core.tensors import (
    SemanticNode,
    SemanticEdge,
    RawSemanticGraph,
    StabilizedGraph,
    BOUND_TENSOR_DIM,
    FORM_TENSOR_DIM,
    INTENT_TENSOR_DIM,
)


class TestSemanticNode:
    def test_node_creation_defaults(self):
        """Test node creation with default uncertainty."""
        node = SemanticNode(node_id=1)
        assert node.node_id == 1
        assert node.bound_tensor.shape == (BOUND_TENSOR_DIM,)
        assert node.form_tensor.shape == (FORM_TENSOR_DIM,)
        assert node.intent_tensor.shape == (INTENT_TENSOR_DIM,)
        assert node.uncertainty == pytest.approx(0.0)

    def test_node_creation_custom_uncertainty(self):
        """Test node creation with custom uncertainty."""
        node = SemanticNode(node_id=2, uncertainty=0.5)
        assert node.uncertainty == pytest.approx(0.5)

    def test_node_invalid_uncertainty(self):
        """Test that negative uncertainty raises ValueError."""
        with pytest.raises(ValueError, match="uncertainty must be >= 0"):
            SemanticNode(node_id=3, uncertainty=-0.1)

    def test_node_creation_custom_tensors(self):
        bound = np.random.randn(BOUND_TENSOR_DIM).astype(np.float32)
        form = np.random.randn(FORM_TENSOR_DIM).astype(np.float32)
        intent = np.random.rand(INTENT_TENSOR_DIM).astype(np.float32)

        node = SemanticNode(
            node_id=42,
            bound_tensor=bound,
            form_tensor=form,
            intent_tensor=intent,
            uncertainty=0.7,
        )

        assert node.node_id == 42
        assert np.allclose(node.bound_tensor, bound)
        assert np.allclose(node.form_tensor, form)
        assert np.allclose(node.intent_tensor, intent)
        assert node.uncertainty == pytest.approx(0.7)

    def test_node_combined_state(self):
        node = SemanticNode(node_id=1)
        state = node.combined_state()
        expected_dim = BOUND_TENSOR_DIM + FORM_TENSOR_DIM + INTENT_TENSOR_DIM
        assert state.shape == (expected_dim,)

    def test_node_copy(self):
        node = SemanticNode(node_id=1, bound_tensor=np.ones(BOUND_TENSOR_DIM))
        copy = node.copy()
        node.bound_tensor[0] = 99.0
        assert copy.bound_tensor[0] != 99.0

    def test_node_distance(self):
        node1 = SemanticNode(node_id=1)
        node2 = SemanticNode(node_id=2)
        dist = node1.distance_to(node2)
        assert dist >= 0.0
        assert node1.distance_to(node1) < 1e-6

    def test_node_invalid_tensor_shape(self):
        with pytest.raises(ValueError):
            SemanticNode(node_id=1, bound_tensor=np.zeros(10))

    def test_node_invalid_uncertainty(self):
        with pytest.raises(ValueError):
            SemanticNode(node_id=1, uncertainty=-0.1)


class TestSemanticEdge:
    def test_edge_creation_defaults(self):
        """Test edge creation with default gate."""
        relation = np.random.randn(16).astype(np.float32)
        edge = SemanticEdge(source_id=1, target_id=2, relation_embedding=relation)
        assert edge.gate == pytest.approx(1.0)

    def test_edge_creation_custom_gate(self):
        """Test edge creation with custom gate."""
        relation = np.random.randn(16).astype(np.float32)
        edge = SemanticEdge(
            source_id=1, target_id=2, relation_embedding=relation, gate=0.7
        )
        assert edge.gate == pytest.approx(0.7)

    def test_edge_invalid_gate(self):
        """Test that gate value outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="Edge gate must be in"):
            SemanticEdge(
                source_id=1, target_id=2, relation_embedding=np.zeros(16), gate=1.1
            )
        with pytest.raises(ValueError, match="Edge gate must be in"):
            SemanticEdge(
                source_id=1, target_id=2, relation_embedding=np.zeros(16), gate=-0.1
            )

    def test_edge_effective_strength_uses_gate(self):
        """Test that effective_strength correctly multiplies by gate."""
        edge = SemanticEdge(
            source_id=1,
            target_id=2,
            relation_embedding=np.zeros(16),
            weight=2.0,
            probability=0.5,
            gate=0.25,
        )
        assert edge.effective_strength() == pytest.approx(0.25)


class TestRawSemanticGraph:
    def test_graph_creation(self):
        nodes = [SemanticNode(node_id=i) for i in range(3)]
        edges = [
            SemanticEdge(source_id=0, target_id=1, relation_embedding=np.zeros(16))
        ]
        graph = RawSemanticGraph(timestamp_ns=1000, nodes=nodes, edge_candidates=edges)
        assert graph.timestamp_ns == 1000
        assert len(graph.nodes) == 3
        assert len(graph.edge_candidates) == 1

    def test_graph_get_node_and_ids(self):
        nodes = [SemanticNode(node_id=i) for i in [1, 5, 10]]
        graph = RawSemanticGraph(timestamp_ns=0, nodes=nodes)
        assert graph.get_node(5).node_id == 5
        assert graph.get_node(999) is None
        assert graph.node_ids() == [1, 5, 10]


class TestStabilizedGraph:
    def test_adjacency_dict(self):
        nodes = [SemanticNode(node_id=i) for i in [1, 2, 3]]
        edges = [
            SemanticEdge(source_id=1, target_id=2, relation_embedding=np.zeros(16)),
            SemanticEdge(source_id=2, target_id=3, relation_embedding=np.zeros(16)),
        ]
        graph = StabilizedGraph(
            timestamp_ns=0, nodes=nodes, edges=edges, global_energy=0.5
        )
        adj = graph.adjacency_dict()
        assert 2 in adj[1]
        assert 1 in adj[2]
        assert 3 in adj[2]
