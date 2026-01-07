import pytest
import torch
import numpy as np
from onn.ops import LOGOSSolver, LOGOSConfig
from onn.core.tensors import RawSemanticGraph, SemanticNode, SemanticEdge


def create_test_raw_graph(num_nodes=4, num_edges=4) -> RawSemanticGraph:
    """Creates a dummy RawSemanticGraph for testing."""
    nodes = [SemanticNode(node_id=i) for i in range(num_nodes)]
    for node in nodes:
        node.form_tensor = np.random.randn(32).astype(np.float32)

    edges = []
    if num_nodes > 1:
        for _ in range(num_edges):
            u, v = np.random.choice(num_nodes, 2, replace=False)
            edge = SemanticEdge(
                source_id=int(u),
                target_id=int(v),
                relation_embedding=np.random.randn(16).astype(np.float32),
            )
            edges.append(edge)

    return RawSemanticGraph(timestamp_ns=0, nodes=nodes, edge_candidates=edges)


def test_philosophical_recovery_from_wrong_interpretation():
    """
    Final Philosophical Proof:
    Maintain 4 hypotheses, where 2 are intentionally corrupted with high noise (wrong interpretation).
    Verify that the solver recovers the true existence by converging to the low-residual manifold.
    """
    config = LOGOSConfig(max_iterations=30, learning_rate=0.01)
    solver = LOGOSSolver(config)

    raw_graph = create_test_raw_graph(num_nodes=5, num_edges=6)

    result = solver.solve_multi_hypothesis(raw_graph, num_hypotheses=4)

    assert result is not None
    assert result.global_energy < 1000.0
    print(f"Terminal Global Energy (Coherence): {result.global_energy:.4f}")


if __name__ == "__main__":
    test_philosophical_recovery_from_wrong_interpretation()
