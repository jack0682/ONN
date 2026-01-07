"""
Smoke test for LOGOS Solver.

Verifies basic solver functionality and energy minimization.

Reference: spec/20_impl_plan.ir.yml IMPL_006
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from onn.ops.logos_solver import (
    LOGOSSolver,
    LOGOSConfig,
    create_default_solver,
)
from onn.core.tensors import (
    SemanticNode,
    SemanticEdge,
    RawSemanticGraph,
    StabilizedGraph,
    BOUND_TENSOR_DIM,
    FORM_TENSOR_DIM,
    INTENT_TENSOR_DIM,
)


@pytest.fixture
def simple_raw_graph() -> RawSemanticGraph:
    """Create a simple raw graph for testing."""
    nodes = [
        SemanticNode(
            node_id=1,
            bound_tensor=np.random.randn(BOUND_TENSOR_DIM).astype(np.float32),
            form_tensor=np.random.randn(FORM_TENSOR_DIM).astype(np.float32),
            intent_tensor=np.random.rand(INTENT_TENSOR_DIM).astype(np.float32),
        ),
        SemanticNode(
            node_id=2,
            bound_tensor=np.random.randn(BOUND_TENSOR_DIM).astype(np.float32),
            form_tensor=np.random.randn(FORM_TENSOR_DIM).astype(np.float32),
            intent_tensor=np.random.rand(INTENT_TENSOR_DIM).astype(np.float32),
        ),
        SemanticNode(
            node_id=3,
            bound_tensor=np.random.randn(BOUND_TENSOR_DIM).astype(np.float32),
            form_tensor=np.random.randn(FORM_TENSOR_DIM).astype(np.float32),
            intent_tensor=np.random.rand(INTENT_TENSOR_DIM).astype(np.float32),
        ),
    ]

    edges = [
        SemanticEdge(
            source_id=1, target_id=2,
            relation_embedding=np.random.randn(16).astype(np.float32),
            weight=1.0, probability=0.9
        ),
        SemanticEdge(
            source_id=2, target_id=3,
            relation_embedding=np.random.randn(16).astype(np.float32),
            weight=0.8, probability=0.8
        ),
    ]

    return RawSemanticGraph(timestamp_ns=1000, nodes=nodes, edge_candidates=edges)


class TestLOGOSSolverSmoke:
    """Smoke tests for LOGOS Solver."""

    def test_create_default_solver(self):
        """Default solver should be created with spec hyperparameters."""
        solver = create_default_solver()

        assert solver is not None
        assert solver.config.lambda_data == 1.0
        assert solver.config.lambda_phys == 10.0
        assert solver.config.lambda_logic == 2.0

    def test_solve_returns_stabilized_graph(self, simple_raw_graph):
        """Solve should return a StabilizedGraph."""
        solver = create_default_solver()
        result = solver.solve(simple_raw_graph)

        assert isinstance(result, StabilizedGraph)
        assert len(result.nodes) == len(simple_raw_graph.nodes)

    def test_output_nodes_have_normalized_bounds(self, simple_raw_graph):
        """Output nodes should have normalized bound tensors."""
        solver = create_default_solver()
        result = solver.solve(simple_raw_graph)

        for node in result.nodes:
            norm = np.linalg.norm(node.bound_tensor)
            assert np.isclose(norm, 1.0, atol=1e-4), \
                f"Node {node.node_id} bound tensor norm={norm}"

    def test_output_nodes_have_valid_intent(self, simple_raw_graph):
        """Output nodes should have intent tensors in [0, 1]."""
        solver = create_default_solver()
        result = solver.solve(simple_raw_graph)

        for node in result.nodes:
            assert np.all(node.intent_tensor >= 0), \
                f"Node {node.node_id} has negative intent"
            assert np.all(node.intent_tensor <= 1), \
                f"Node {node.node_id} has intent > 1"

    def test_global_energy_is_finite(self, simple_raw_graph):
        """Output global energy should be finite."""
        solver = create_default_solver()
        result = solver.solve(simple_raw_graph)

        assert np.isfinite(result.global_energy)
        assert result.global_energy >= 0

    def test_energy_decreases(self, simple_raw_graph):
        """H-01: Energy should decrease during iteration."""
        config = LOGOSConfig(max_iterations=10, learning_rate=0.05)
        solver = LOGOSSolver(config)

        solver.solve(simple_raw_graph)
        last_result = solver.get_last_result()

        assert last_result is not None
        assert len(last_result.energy_history) > 1

        # Final energy should be <= initial energy
        assert last_result.energy_history[-1] <= last_result.energy_history[0] + 1e-4

    def test_empty_graph_raises(self):
        """Solver should raise on empty graph."""
        solver = create_default_solver()
        empty_graph = RawSemanticGraph(timestamp_ns=0, nodes=[], edge_candidates=[])

        with pytest.raises(ValueError, match="empty"):
            solver.solve(empty_graph)

    def test_single_node_works(self):
        """Solver should handle single node graphs."""
        solver = create_default_solver()
        single_node = RawSemanticGraph(
            timestamp_ns=0,
            nodes=[SemanticNode(node_id=1)],
            edge_candidates=[]
        )

        result = solver.solve(single_node)
        assert len(result.nodes) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
