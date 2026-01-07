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


def test_multi_hypothesis_convergence():
    """
    Test that multi-hypothesis solving selects a stable interpretation.
    """
    config = LOGOSConfig(max_iterations=10, learning_rate=0.01)
    solver = LOGOSSolver(config)

    raw_graph = create_test_raw_graph(num_nodes=4, num_edges=5)

    result_multi = solver.solve_multi_hypothesis(raw_graph, num_hypotheses=4)

    assert result_multi is not None
    assert len(result_multi.nodes) == len(raw_graph.nodes)
    assert result_multi.global_energy < 1000.0

    print(f"Multi-hypothesis best global energy: {result_multi.global_energy:.4f}")


if __name__ == "__main__":
    test_multi_hypothesis_convergence()
