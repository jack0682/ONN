"""
Integration tests for ONN-ES system on real-world benchmark graphs.
"""

import numpy as np
import networkx as nx
import pytest

from onn.topo.filtration import compute_topo_summary, compute_betti_numbers
from onn.core.graph import EdgeGraph
from onn.ops.branching import Branch


def test_zachary_karate_club_meaningful_topology():
    """
    Tests filtration on the ZKC graph using edge betweenness centrality as weights.

    This test ensures that:
    1. The filtration is non-degenerate (i.e., τ* carries a meaningful signal).
    2. The β0 profile is monotonically non-increasing.
    3. The computed τ* is stable under small perturbations.
    """
    # 1. Load graph and compute meaningful edge weights (betweenness centrality)
    zkc_nx = nx.karate_club_graph()
    num_nodes = zkc_nx.number_of_nodes()

    # The from_edge_list method creates a sorted list of edge_keys
    # We must ensure our centrality-based gates align with this order.
    onn_graph = EdgeGraph.from_edge_list(list(zkc_nx.edges()))
    edge_indices_np = onn_graph.edge_index.T.numpy()

    centrality = nx.edge_betweenness_centrality(zkc_nx, normalized=True)

    # Create a mapping from EdgeKey to centrality to ensure order
    centrality_map = {
        tuple(sorted((u, v))): score for (u, v), score in centrality.items()
    }

    # Build the gates array in the same order as onn_graph.edge_keys
    gates = np.array(
        [
            centrality_map[tuple(sorted((ek.source_id, ek.target_id)))]
            for ek in onn_graph.edge_keys
        ]
    )

    # 2. Compute topological summary
    topo_summary = compute_topo_summary(
        num_nodes=num_nodes,
        edge_indices=edge_indices_np,
        gates=gates,
    )

    # 3. [Task 1.1] Assert filtration is non-degenerate
    delta_beta0 = np.diff(topo_summary.beta0_profile)
    delta_beta1 = np.diff(topo_summary.beta1_profile)
    is_degenerate = not (np.any(delta_beta0 != 0) or np.any(delta_beta1 != 0))

    assert not is_degenerate, (
        "Filtration is degenerate; τ* carries no signal. Edge weights may be uniform."
    )

    # 4. [Task 1.3] Assert β0 monotonicity
    beta0_profile = topo_summary.beta0_profile
    is_monotonic = np.all(
        beta0_profile[:-1] >= beta0_profile[1:]
    )  # tau decreases, so beta0 should be non-increasing
    assert is_monotonic, (
        f"β0 profile is not monotonically non-increasing: {beta0_profile}"
    )

    # 5. [Task 1.3] Assert τ* stability under noise
    tau_stars = []
    rng = np.random.RandomState(42)
    for _ in range(10):
        noise = rng.normal(0, 0.01, size=gates.shape)
        noisy_gates = np.clip(gates + noise, 0, 1)

        noisy_summary = compute_topo_summary(
            num_nodes=num_nodes,
            edge_indices=edge_indices_np,
            gates=noisy_gates,
        )
        tau_stars.append(noisy_summary.tau0_star)

    tau_variance = np.var(tau_stars)
    assert tau_variance < 0.05, f"τ* is unstable under noise. Variance: {tau_variance}"
    print(f"\nτ* stability check passed. Variance: {tau_variance:.6f}")
