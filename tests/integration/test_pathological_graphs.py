import pytest
import numpy as np
from typing import Tuple

from onn.topo.filtration import compute_topo_summary, compute_betti_numbers
from onn.ops.branching import (
    boundary_gate_mutation,
    topology_rewire_mutation,
    BranchFactory,
)


def create_complete_graph(n: int) -> Tuple[int, np.ndarray, np.ndarray]:
    """K_n: Complete graph with all nodes connected."""
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_list.append([i, j])
    edge_indices = np.array(edge_list)
    gates = np.full(len(edge_indices), 0.7)
    return n, edge_indices, gates


def create_star_graph(n: int) -> Tuple[int, np.ndarray, np.ndarray]:
    """Star graph: Central hub connected to n-1 leaves."""
    edge_list = [[0, i] for i in range(1, n)]
    edge_indices = np.array(edge_list)
    gates = np.full(len(edge_indices), 0.6)
    return n, edge_indices, gates


def create_tree_graph(n: int) -> Tuple[int, np.ndarray, np.ndarray]:
    """Binary tree with n nodes (no cycles)."""
    edge_list = []
    for i in range(n):
        left_child = 2 * i + 1
        right_child = 2 * i + 2
        if left_child < n:
            edge_list.append([i, left_child])
        if right_child < n:
            edge_list.append([i, right_child])
    edge_indices = np.array(edge_list) if edge_list else np.array([]).reshape(0, 2)
    gates = np.full(len(edge_indices), 0.5) if len(edge_list) > 0 else np.array([])
    return n, edge_indices, gates


def create_isolated_nodes(n: int) -> Tuple[int, np.ndarray, np.ndarray]:
    """n isolated nodes with no edges."""
    edge_indices = np.array([]).reshape(0, 2)
    gates = np.array([])
    return n, edge_indices, gates


def create_biconnected_graph() -> Tuple[int, np.ndarray, np.ndarray]:
    """Graph with multiple biconnected components (designed cycles)."""
    edge_list = [
        [0, 1], [1, 2], [2, 0],
        [2, 3], [3, 4], [4, 5], [5, 3],
        [5, 6], [6, 7], [7, 8], [8, 6],
    ]
    edge_indices = np.array(edge_list)
    gates = np.array([0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.7, 0.65, 0.6, 0.55])
    return 9, edge_indices, gates


class TestPathologicalGraphs:

    def test_complete_graph_betti_numbers(self):
        """Complete graph K_n: β₀=1, β₁=n(n-1)/2 - (n-1)."""
        for n in [4, 5, 6]:
            num_nodes, edge_indices, gates = create_complete_graph(n)

            beta0, beta1 = compute_betti_numbers(num_nodes, edge_indices, gates, tau=0.3)
            expected_beta1 = (n - 1) * (n - 2) // 2

            print(f"K_{n}: β₀={beta0}, β₁={beta1} (expected β₁={expected_beta1})")

            assert beta0 == 1, f"K_{n} should have β₀=1, got {beta0}"
            assert beta1 == expected_beta1, f"K_{n} should have β₁={expected_beta1}, got {beta1}"

    def test_complete_graph_no_crash(self):
        """Ensure compute_topo_summary doesn't crash on complete graph."""
        num_nodes, edge_indices, gates = create_complete_graph(6)

        topo = compute_topo_summary(num_nodes, edge_indices, gates)

        assert np.isfinite(topo.tau0_star), "τ₀* is not finite"
        assert np.isfinite(topo.tau1_star), "τ₁* is not finite"
        assert not np.any(np.isnan(topo.beta0_profile)), "β₀ profile has NaN"
        assert not np.any(np.isnan(topo.beta1_profile)), "β₁ profile has NaN"

    def test_star_graph_betti_numbers(self):
        """Star graph: β₀=1 (connected), β₁=0 (no cycles)."""
        for n in [5, 8, 10]:
            num_nodes, edge_indices, gates = create_star_graph(n)

            beta0, beta1 = compute_betti_numbers(num_nodes, edge_indices, gates, tau=0.3)

            print(f"Star_{n}: β₀={beta0}, β₁={beta1}")

            assert beta0 == 1, f"Star should have β₀=1, got {beta0}"
            assert beta1 == 0, f"Star should have β₁=0, got {beta1}"

    def test_star_graph_mutation(self):
        """Mutations shouldn't crash on star graph."""
        num_nodes, edge_indices, gates = create_star_graph(8)
        rng = np.random.RandomState(42)

        mutated = boundary_gate_mutation(gates.copy(), tau_star=0.5, delta=0.2, sigma=0.2, rng=rng)

        assert len(mutated) == len(gates), "Mutation changed gate count"
        assert np.all(np.isfinite(mutated)), "Mutation produced non-finite gates"
        assert np.all(mutated >= 0) and np.all(mutated <= 1), "Gates out of [0,1] range"

    def test_tree_graph_betti_numbers(self):
        """Tree: β₀=1, β₁=0 (no cycles)."""
        for n in [7, 15, 31]:
            num_nodes, edge_indices, gates = create_tree_graph(n)

            if len(gates) == 0:
                continue

            beta0, beta1 = compute_betti_numbers(num_nodes, edge_indices, gates, tau=0.3)

            print(f"Tree_{n}: β₀={beta0}, β₁={beta1}")

            assert beta0 == 1, f"Tree should have β₀=1, got {beta0}"
            assert beta1 == 0, f"Tree should have β₁=0, got {beta1}"

    def test_isolated_nodes_betti_numbers(self):
        """Isolated nodes: β₀=n, β₁=0."""
        for n in [1, 5, 10]:
            num_nodes, edge_indices, gates = create_isolated_nodes(n)

            beta0, beta1 = compute_betti_numbers(num_nodes, edge_indices, gates, tau=0.3)

            print(f"Isolated_{n}: β₀={beta0}, β₁={beta1}")

            assert beta0 == n, f"Isolated should have β₀={n}, got {beta0}"
            assert beta1 == 0, f"Isolated should have β₁=0, got {beta1}"

    def test_isolated_nodes_topo_summary(self):
        """Topo summary shouldn't crash on empty graph."""
        num_nodes, edge_indices, gates = create_isolated_nodes(5)

        topo = compute_topo_summary(num_nodes, edge_indices, gates)

        assert topo.beta0_profile[-1] == num_nodes, f"Expected β₀={num_nodes} at τ=0"
        assert topo.beta1_profile[-1] == 0, "Expected β₁=0 at τ=0"

    def test_biconnected_graph_betti_numbers(self):
        """Biconnected graph with 3 cycles: β₁ should be 3."""
        num_nodes, edge_indices, gates = create_biconnected_graph()

        beta0, beta1 = compute_betti_numbers(num_nodes, edge_indices, gates, tau=0.3)

        print(f"Biconnected: β₀={beta0}, β₁={beta1}")

        assert beta0 == 1, f"Biconnected should have β₀=1, got {beta0}"
        assert beta1 == 3, f"Biconnected should have β₁=3, got {beta1}"

    def test_rewire_mutation_on_complete_graph(self):
        """Rewire mutation on complete graph shouldn't crash."""
        num_nodes, edge_indices, gates = create_complete_graph(5)
        rng = np.random.RandomState(42)

        mutated = topology_rewire_mutation(
            gates=gates.copy(),
            edge_indices=edge_indices,
            num_nodes=num_nodes,
            beta0_current=1,
            beta1_current=6,
            rng=rng,
            target_beta0=1,
            boost_cycles=True,
        )

        assert len(mutated) == len(gates), "Mutation changed gate count"
        assert np.all(np.isfinite(mutated)), "Mutation produced non-finite gates"

    def test_rewire_mutation_on_isolated_nodes(self):
        """Rewire mutation on isolated nodes should be no-op."""
        num_nodes, edge_indices, gates = create_isolated_nodes(5)
        rng = np.random.RandomState(42)

        mutated = topology_rewire_mutation(
            gates=gates.copy(),
            edge_indices=edge_indices,
            num_nodes=num_nodes,
            beta0_current=5,
            beta1_current=0,
            rng=rng,
            target_beta0=1,
            boost_cycles=False,
        )

        assert len(mutated) == 0, "Isolated nodes should have no gates to mutate"

    def test_branch_factory_on_pathological_graphs(self):
        """BranchFactory shouldn't crash on any pathological graph."""
        graphs = [
            ("complete", create_complete_graph(5)),
            ("star", create_star_graph(6)),
            ("tree", create_tree_graph(7)),
            ("isolated", create_isolated_nodes(4)),
            ("biconnected", create_biconnected_graph()),
        ]

        factory = BranchFactory(base_seed=42)

        for name, (num_nodes, edge_indices, gates) in graphs:
            if len(gates) == 0:
                print(f"{name}: skipped (no edges)")
                continue

            try:
                branches = factory.make_branches(
                    parent_gates=gates,
                    parent_meta_params={"gate_lr": 0.05},
                    tau_star=0.5,
                    edge_indices=edge_indices,
                    num_nodes=num_nodes,
                )

                print(f"{name}: created {len(branches)} branches ✓")

                for branch in branches:
                    assert np.all(np.isfinite(branch.gates)), f"{name} branch has non-finite gates"
                    assert len(branch.gates) == len(gates), f"{name} branch has wrong gate count"

            except Exception as e:
                pytest.fail(f"BranchFactory crashed on {name}: {e}")

    def test_all_pathological_graphs_no_nan(self):
        """All pathological graphs should produce finite values."""
        graphs = [
            ("complete_4", create_complete_graph(4)),
            ("complete_6", create_complete_graph(6)),
            ("star_5", create_star_graph(5)),
            ("star_10", create_star_graph(10)),
            ("tree_7", create_tree_graph(7)),
            ("tree_15", create_tree_graph(15)),
            ("isolated_1", create_isolated_nodes(1)),
            ("isolated_5", create_isolated_nodes(5)),
            ("biconnected", create_biconnected_graph()),
        ]

        print(f"\n=== Pathological Graph Summary ===")
        print(f"{'Graph':>15} {'Nodes':>6} {'Edges':>6} {'β₀':>4} {'β₁':>4} {'τ₀*':>8} {'τ₁*':>8}")
        print("-" * 60)

        all_finite = True

        for name, (num_nodes, edge_indices, gates) in graphs:
            topo = compute_topo_summary(num_nodes, edge_indices, gates)

            beta0 = topo.beta0_profile[-1] if len(topo.beta0_profile) > 0 else num_nodes
            beta1 = topo.beta1_profile[-1] if len(topo.beta1_profile) > 0 else 0

            print(f"{name:>15} {num_nodes:>6} {len(gates):>6} {beta0:>4} {beta1:>4} {topo.tau0_star:>8.4f} {topo.tau1_star:>8.4f}")

            if not np.isfinite(topo.tau0_star) or not np.isfinite(topo.tau1_star):
                all_finite = False

        assert all_finite, "Some graphs produced non-finite τ* values"
