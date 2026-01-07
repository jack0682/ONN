"""
Visualization script for ONN relationship graph evolution.
Plots residual convergence and state dynamics.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from onn.ops import LOGOSSolver, LOGOSConfig
from onn.core.tensors import (
    SemanticNode,
    SemanticEdge,
    RawSemanticGraph,
    BOUND_TENSOR_DIM,
    FORM_TENSOR_DIM,
    INTENT_TENSOR_DIM,
)


def create_sample_graph(num_nodes: int = 3) -> RawSemanticGraph:
    nodes = []
    for i in range(num_nodes):
        bound = np.random.randn(BOUND_TENSOR_DIM).astype(np.float32)
        bound[11] = abs(bound[11]) + 0.1
        node = SemanticNode(
            node_id=i,
            bound_tensor=bound,
            form_tensor=np.random.randn(FORM_TENSOR_DIM).astype(np.float32),
            intent_tensor=np.random.rand(INTENT_TENSOR_DIM).astype(np.float32),
        )
        nodes.append(node)

    edges = []
    for i in range(num_nodes - 1):
        edge = SemanticEdge(
            source_id=i,
            target_id=i + 1,
            relation_embedding=np.random.randn(16).astype(np.float32),
            weight=1.0,
            probability=0.9,
        )
        edges.append(edge)

    return RawSemanticGraph(timestamp_ns=1000, nodes=nodes, edge_candidates=edges)


def plot_convergence():
    print("Running LOGOSSolver and capturing dynamics...")
    graph = create_sample_graph(num_nodes=5)

    config = LOGOSConfig(max_iterations=50, learning_rate=0.01)
    solver = LOGOSSolver(config)

    solver.solve(graph)
    result = solver.get_last_result()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(result.residual_norm_history, "r-o", label="Combined Residual")
    plt.yscale("log")
    plt.title("Residual Convergence (Log Scale)")
    plt.xlabel("Iteration")
    plt.ylabel("Norm")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(result.energy_history, "b-o", label="Total Energy (L)")
    plt.title("Energy Landscape Trajectory")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    output_path = os.path.join(
        os.path.dirname(__file__), "../outputs/onn_convergence.png"
    )
    plt.savefig(output_path)
    print(f"Convergence plot saved to {output_path}")


if __name__ == "__main__":
    plot_convergence()
