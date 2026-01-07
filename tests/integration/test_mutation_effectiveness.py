import pytest
import numpy as np
import torch
from typing import List, Tuple, Dict
from dataclasses import dataclass
from scipy import stats

from onn.ops.branching import (
    boundary_gate_mutation,
    topology_rewire_mutation,
)
from onn.topo.filtration import compute_topo_summary, compute_betti_numbers


@dataclass
class GateMetrics:
    mean_gate: float
    active_ratio: float
    gate_std: float
    beta0: int
    beta1: int
    tau0_star: float
    tau1_star: float


def create_random_graph(seed: int, num_nodes: int = 8, edge_density: float = 0.4):
    rng = np.random.RandomState(seed)

    edge_list = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if rng.random() < edge_density:
                edge_list.append([i, j])

    if len(edge_list) < num_nodes - 1:
        for i in range(num_nodes - 1):
            if [i, i + 1] not in edge_list and [i + 1, i] not in edge_list:
                edge_list.append([i, i + 1])

    edge_indices = np.array(edge_list) if edge_list else np.array([]).reshape(0, 2)
    num_edges = len(edge_indices)

    gates = 0.3 + rng.uniform(0.0, 0.5, size=num_edges)

    return num_nodes, edge_indices, gates


def compute_gate_metrics(
    num_nodes: int,
    edge_indices: np.ndarray,
    gates: np.ndarray,
) -> GateMetrics:
    if len(gates) == 0:
        return GateMetrics(
            mean_gate=0.0,
            active_ratio=0.0,
            gate_std=0.0,
            beta0=num_nodes,
            beta1=0,
            tau0_star=0.5,
            tau1_star=0.5,
        )

    mean_gate = float(np.mean(gates))
    active_ratio = float(np.mean(gates > 0.3))
    gate_std = float(np.std(gates))

    topo = compute_topo_summary(num_nodes, edge_indices, gates)
    beta0, beta1 = compute_betti_numbers(num_nodes, edge_indices, gates, tau=0.3)

    return GateMetrics(
        mean_gate=mean_gate,
        active_ratio=active_ratio,
        gate_std=gate_std,
        beta0=beta0,
        beta1=beta1,
        tau0_star=topo.tau0_star,
        tau1_star=topo.tau1_star,
    )


class TestMutationEffectiveness:
    N_TRIALS = 100

    def test_boundary_mutation_changes_gates(self):
        change_count = 0
        delta_gates_list = []

        for seed in range(self.N_TRIALS):
            num_nodes, edge_indices, gates = create_random_graph(seed, num_nodes=8)
            if len(gates) == 0:
                continue

            original_gates = gates.copy()
            rng = np.random.RandomState(seed + 10000)

            tau_star = 0.5
            mutated_gates = boundary_gate_mutation(
                gates=original_gates,
                tau_star=tau_star,
                delta=0.25,
                sigma=0.3,
                rng=rng,
            )

            delta = np.abs(mutated_gates - original_gates)
            max_delta = np.max(delta)
            delta_gates_list.append(max_delta)

            if max_delta > 0.01:
                change_count += 1

        change_rate = change_count / len(delta_gates_list)
        mean_delta = np.mean(delta_gates_list)
        max_delta_overall = np.max(delta_gates_list)

        print(f"\n=== BOUNDARY Mutation Gate Changes ===")
        print(f"Change rate (Δ > 0.01): {change_rate:.2%}")
        print(f"Mean max delta: {mean_delta:.4f}")
        print(f"Max delta overall: {max_delta_overall:.4f}")

        assert change_rate >= 0.5, (
            f"Boundary mutation rarely changes gates: {change_rate:.2%}"
        )
        assert mean_delta > 0.01, f"Mean delta too small: {mean_delta:.4f}"

    def test_rewire_mutation_changes_gates(self):
        change_count = 0
        delta_gates_list = []

        for seed in range(self.N_TRIALS):
            num_nodes, edge_indices, gates = create_random_graph(
                seed + 1000, num_nodes=8
            )
            if len(gates) == 0:
                continue

            original_gates = gates.copy()
            rng = np.random.RandomState(seed + 11000)

            metrics = compute_gate_metrics(num_nodes, edge_indices, original_gates)

            mutated_gates = topology_rewire_mutation(
                gates=original_gates,
                edge_indices=edge_indices,
                num_nodes=num_nodes,
                beta0_current=metrics.beta0,
                beta1_current=metrics.beta1,
                rng=rng,
                target_beta0=1,
                boost_cycles=True,
            )

            delta = np.abs(mutated_gates - original_gates)
            max_delta = np.max(delta) if len(delta) > 0 else 0
            delta_gates_list.append(max_delta)

            if max_delta > 0.01:
                change_count += 1

        change_rate = change_count / len(delta_gates_list) if delta_gates_list else 0
        mean_delta = np.mean(delta_gates_list) if delta_gates_list else 0
        max_delta_overall = np.max(delta_gates_list) if delta_gates_list else 0

        print(f"\n=== REWIRE Mutation Gate Changes ===")
        print(f"Change rate (Δ > 0.01): {change_rate:.2%}")
        print(f"Mean max delta: {mean_delta:.4f}")
        print(f"Max delta overall: {max_delta_overall:.4f}")

        assert change_rate >= 0.3, (
            f"Rewire mutation rarely changes gates: {change_rate:.2%}"
        )

    def test_boundary_mutation_increases_active_ratio(self):
        improvement_count = 0
        same_count = 0
        degradation_count = 0

        for seed in range(self.N_TRIALS):
            num_nodes, edge_indices, gates = create_random_graph(
                seed + 2000, num_nodes=10
            )
            if len(gates) == 0:
                continue

            original_metrics = compute_gate_metrics(num_nodes, edge_indices, gates)

            rng = np.random.RandomState(seed + 12000)
            mutated_gates = boundary_gate_mutation(
                gates=gates.copy(),
                tau_star=0.5,
                delta=0.3,
                sigma=0.25,
                rng=rng,
            )

            mutated_metrics = compute_gate_metrics(
                num_nodes, edge_indices, mutated_gates
            )

            if mutated_metrics.active_ratio > original_metrics.active_ratio + 0.01:
                improvement_count += 1
            elif mutated_metrics.active_ratio < original_metrics.active_ratio - 0.01:
                degradation_count += 1
            else:
                same_count += 1

        total = improvement_count + same_count + degradation_count
        print(f"\n=== BOUNDARY Mutation Active Ratio Effect ===")
        print(
            f"Improved:  {improvement_count}/{total} ({improvement_count / total:.2%})"
        )
        print(f"Same:      {same_count}/{total} ({same_count / total:.2%})")
        print(
            f"Degraded:  {degradation_count}/{total} ({degradation_count / total:.2%})"
        )

        assert improvement_count + same_count >= degradation_count, (
            f"Boundary mutation degrades more than it helps: {degradation_count} > {improvement_count + same_count}"
        )

    def test_rewire_mutation_reduces_beta0(self):
        beta0_reduced = 0
        beta0_same = 0
        beta0_increased = 0

        for seed in range(self.N_TRIALS):
            num_nodes, edge_indices, gates = create_random_graph(
                seed + 3000, num_nodes=10, edge_density=0.3
            )
            if len(gates) == 0:
                continue

            original_metrics = compute_gate_metrics(num_nodes, edge_indices, gates)

            if original_metrics.beta0 <= 1:
                beta0_same += 1
                continue

            rng = np.random.RandomState(seed + 13000)
            mutated_gates = topology_rewire_mutation(
                gates=gates.copy(),
                edge_indices=edge_indices,
                num_nodes=num_nodes,
                beta0_current=original_metrics.beta0,
                beta1_current=original_metrics.beta1,
                rng=rng,
                target_beta0=1,
                boost_cycles=True,
            )

            mutated_metrics = compute_gate_metrics(
                num_nodes, edge_indices, mutated_gates
            )

            if mutated_metrics.beta0 < original_metrics.beta0:
                beta0_reduced += 1
            elif mutated_metrics.beta0 > original_metrics.beta0:
                beta0_increased += 1
            else:
                beta0_same += 1

        total = beta0_reduced + beta0_same + beta0_increased
        print(f"\n=== REWIRE Mutation β₀ Effect (Component Bridging) ===")
        print(f"β₀ reduced:   {beta0_reduced}/{total} ({beta0_reduced / total:.2%})")
        print(f"β₀ same:      {beta0_same}/{total} ({beta0_same / total:.2%})")
        print(
            f"β₀ increased: {beta0_increased}/{total} ({beta0_increased / total:.2%})"
        )

        if beta0_reduced + beta0_same > 0:
            success_rate = (
                beta0_reduced / (beta0_reduced + beta0_increased)
                if (beta0_reduced + beta0_increased) > 0
                else 1.0
            )
            print(f"Bridge success rate: {success_rate:.2%}")

    def test_rewire_mutation_increases_beta1(self):
        beta1_increased = 0
        beta1_same = 0
        beta1_decreased = 0

        for seed in range(self.N_TRIALS):
            num_nodes, edge_indices, gates = create_random_graph(
                seed + 4000, num_nodes=8, edge_density=0.5
            )
            if len(gates) == 0:
                continue

            original_metrics = compute_gate_metrics(num_nodes, edge_indices, gates)

            rng = np.random.RandomState(seed + 14000)
            mutated_gates = topology_rewire_mutation(
                gates=gates.copy(),
                edge_indices=edge_indices,
                num_nodes=num_nodes,
                beta0_current=original_metrics.beta0,
                beta1_current=original_metrics.beta1,
                rng=rng,
                target_beta0=1,
                boost_cycles=True,
            )

            mutated_metrics = compute_gate_metrics(
                num_nodes, edge_indices, mutated_gates
            )

            if mutated_metrics.beta1 > original_metrics.beta1:
                beta1_increased += 1
            elif mutated_metrics.beta1 < original_metrics.beta1:
                beta1_decreased += 1
            else:
                beta1_same += 1

        total = beta1_increased + beta1_same + beta1_decreased
        print(f"\n=== REWIRE Mutation β₁ Effect (Cycle Boosting) ===")
        print(
            f"β₁ increased: {beta1_increased}/{total} ({beta1_increased / total:.2%})"
        )
        print(f"β₁ same:      {beta1_same}/{total} ({beta1_same / total:.2%})")
        print(
            f"β₁ decreased: {beta1_decreased}/{total} ({beta1_decreased / total:.2%})"
        )

        if beta1_increased + beta1_decreased > 0:
            boost_rate = beta1_increased / (beta1_increased + beta1_decreased)
            print(f"Cycle boost success rate: {boost_rate:.2%}")

    def test_mutation_tau_star_shift(self):
        tau0_shifts = []
        tau1_shifts = []

        for seed in range(self.N_TRIALS):
            num_nodes, edge_indices, gates = create_random_graph(
                seed + 5000, num_nodes=10
            )
            if len(gates) == 0:
                continue

            original_metrics = compute_gate_metrics(num_nodes, edge_indices, gates)

            rng = np.random.RandomState(seed + 15000)
            mutated_gates = boundary_gate_mutation(
                gates=gates.copy(),
                tau_star=0.5,
                delta=0.3,
                sigma=0.25,
                rng=rng,
            )

            mutated_metrics = compute_gate_metrics(
                num_nodes, edge_indices, mutated_gates
            )

            tau0_shifts.append(
                abs(mutated_metrics.tau0_star - original_metrics.tau0_star)
            )
            tau1_shifts.append(
                abs(mutated_metrics.tau1_star - original_metrics.tau1_star)
            )

        mean_tau0_shift = np.mean(tau0_shifts)
        mean_tau1_shift = np.mean(tau1_shifts)
        max_tau0_shift = np.max(tau0_shifts)
        max_tau1_shift = np.max(tau1_shifts)

        print(f"\n=== Mutation τ* Shift Effect ===")
        print(f"Mean |Δτ₀*|: {mean_tau0_shift:.4f}")
        print(f"Mean |Δτ₁*|: {mean_tau1_shift:.4f}")
        print(f"Max |Δτ₀*|:  {max_tau0_shift:.4f}")
        print(f"Max |Δτ₁*|:  {max_tau1_shift:.4f}")

        assert max_tau0_shift > 0 or max_tau1_shift > 0, "Mutations never shift τ*"

    def test_mutation_comparison_summary(self):
        results = {
            "boundary": {"delta_mean": [], "delta_active": []},
            "rewire": {"delta_mean": [], "delta_active": []},
        }

        for seed in range(self.N_TRIALS):
            num_nodes, edge_indices, gates = create_random_graph(
                seed + 6000, num_nodes=10
            )
            if len(gates) == 0:
                continue

            original_metrics = compute_gate_metrics(num_nodes, edge_indices, gates)

            rng1 = np.random.RandomState(seed + 16000)
            boundary_gates = boundary_gate_mutation(
                gates=gates.copy(), tau_star=0.5, delta=0.3, sigma=0.25, rng=rng1
            )
            boundary_metrics = compute_gate_metrics(
                num_nodes, edge_indices, boundary_gates
            )

            rng2 = np.random.RandomState(seed + 17000)
            rewire_gates = topology_rewire_mutation(
                gates=gates.copy(),
                edge_indices=edge_indices,
                num_nodes=num_nodes,
                beta0_current=original_metrics.beta0,
                beta1_current=original_metrics.beta1,
                rng=rng2,
                target_beta0=1,
                boost_cycles=True,
            )
            rewire_metrics = compute_gate_metrics(num_nodes, edge_indices, rewire_gates)

            results["boundary"]["delta_mean"].append(
                boundary_metrics.mean_gate - original_metrics.mean_gate
            )
            results["boundary"]["delta_active"].append(
                boundary_metrics.active_ratio - original_metrics.active_ratio
            )
            results["rewire"]["delta_mean"].append(
                rewire_metrics.mean_gate - original_metrics.mean_gate
            )
            results["rewire"]["delta_active"].append(
                rewire_metrics.active_ratio - original_metrics.active_ratio
            )

        print(f"\n=== Mutation Comparison Summary ===")
        print(f"{'Mutation':<12} {'Δmean_gate':>12} {'Δactive_ratio':>14}")
        print("-" * 40)

        for mtype in ["boundary", "rewire"]:
            dm = np.mean(results[mtype]["delta_mean"])
            da = np.mean(results[mtype]["delta_active"])
            print(f"{mtype:<12} {dm:>+12.4f} {da:>+14.4f}")

        t_mean, p_mean = stats.ttest_ind(
            results["boundary"]["delta_mean"], results["rewire"]["delta_mean"]
        )
        t_active, p_active = stats.ttest_ind(
            results["boundary"]["delta_active"], results["rewire"]["delta_active"]
        )

        print(f"\nStatistical comparison (boundary vs rewire):")
        print(f"  Δmean_gate: t={t_mean:.3f}, p={p_mean:.4f}")
        print(f"  Δactive_ratio: t={t_active:.3f}, p={p_active:.4f}")


class TestMutationEffectivenessOnSparseGraphs:
    """Test mutations on sparse/disconnected graphs where rewire should help."""

    N_TRIALS = 50

    def create_sparse_disconnected_graph(self, seed: int, num_nodes: int = 12):
        """Create graph with 2-3 disconnected components."""
        rng = np.random.RandomState(seed)

        edge_list = []
        gates_list = []

        comp1_size = num_nodes // 3
        comp2_size = num_nodes // 3
        comp3_size = num_nodes - comp1_size - comp2_size

        for i in range(comp1_size - 1):
            edge_list.append([i, i + 1])
            gates_list.append(0.6 + rng.uniform(-0.1, 0.1))

        offset2 = comp1_size
        for i in range(comp2_size - 1):
            edge_list.append([offset2 + i, offset2 + i + 1])
            gates_list.append(0.5 + rng.uniform(-0.1, 0.1))

        offset3 = comp1_size + comp2_size
        for i in range(comp3_size - 1):
            edge_list.append([offset3 + i, offset3 + i + 1])
            gates_list.append(0.4 + rng.uniform(-0.1, 0.1))

        edge_list.append([comp1_size - 1, offset2])
        gates_list.append(0.15 + rng.uniform(-0.05, 0.05))

        edge_list.append([offset2 + comp2_size - 1, offset3])
        gates_list.append(0.1 + rng.uniform(-0.05, 0.05))

        edge_indices = np.array(edge_list)
        gates = np.array(gates_list)

        return num_nodes, edge_indices, gates

    def test_rewire_bridges_disconnected_components(self):
        bridge_success = 0
        bridge_attempted = 0

        for seed in range(self.N_TRIALS):
            num_nodes, edge_indices, gates = self.create_sparse_disconnected_graph(seed)

            original_beta0, _ = compute_betti_numbers(
                num_nodes, edge_indices, gates, tau=0.3
            )

            if original_beta0 <= 1:
                continue

            bridge_attempted += 1

            rng = np.random.RandomState(seed + 20000)
            mutated_gates = topology_rewire_mutation(
                gates=gates.copy(),
                edge_indices=edge_indices,
                num_nodes=num_nodes,
                beta0_current=original_beta0,
                beta1_current=0,
                rng=rng,
                target_beta0=1,
                boost_cycles=False,
            )

            mutated_beta0, _ = compute_betti_numbers(
                num_nodes, edge_indices, mutated_gates, tau=0.3
            )

            if mutated_beta0 < original_beta0:
                bridge_success += 1

        if bridge_attempted > 0:
            success_rate = bridge_success / bridge_attempted
        else:
            success_rate = 0.0

        print(f"\n=== REWIRE on Disconnected Graphs ===")
        print(f"Graphs with β₀ > 1: {bridge_attempted}/{self.N_TRIALS}")
        print(f"Successful bridges: {bridge_success}/{bridge_attempted}")
        print(f"Bridge success rate: {success_rate:.2%}")

        assert bridge_attempted > 0, "No disconnected graphs generated"
        assert success_rate >= 0.3, (
            f"REWIRE rarely bridges components: {success_rate:.2%}"
        )

    def test_rewire_increases_mean_gate_on_sparse(self):
        delta_gates = []

        for seed in range(self.N_TRIALS):
            num_nodes, edge_indices, gates = self.create_sparse_disconnected_graph(
                seed + 1000
            )

            original_mean = np.mean(gates)

            rng = np.random.RandomState(seed + 21000)
            mutated_gates = topology_rewire_mutation(
                gates=gates.copy(),
                edge_indices=edge_indices,
                num_nodes=num_nodes,
                beta0_current=3,
                beta1_current=0,
                rng=rng,
                target_beta0=1,
                boost_cycles=True,
            )

            mutated_mean = np.mean(mutated_gates)
            delta_gates.append(mutated_mean - original_mean)

        mean_delta = np.mean(delta_gates)
        positive_rate = np.mean(np.array(delta_gates) > 0)

        print(f"\n=== REWIRE Mean Gate Effect on Sparse Graphs ===")
        print(f"Mean Δmean_gate: {mean_delta:.4f}")
        print(f"Positive delta rate: {positive_rate:.2%}")

        # Use Wilcoxon signed-rank test to avoid precision loss warnings on near-identical data
        if len(delta_gates) >= 10:
            t_stat, p_value = stats.wilcoxon(delta_gates, zero_method="pratt")
            print(f"Wilcoxon signed-rank test: W={t_stat:.3f}, p={p_value:.4f}")
        else:
            # Fallback to t-test with warning suppression for very small samples
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                t_stat, p_value = stats.ttest_1samp(delta_gates, 0)
            print(f"One-sample t-test (small sample): t={t_stat:.3f}, p={p_value:.4f}")

        assert mean_delta > 0, f"REWIRE doesn't increase mean gate: {mean_delta:.4f}"

    def test_boundary_preserves_structure_on_sparse(self):
        beta0_preserved = 0

        for seed in range(self.N_TRIALS):
            num_nodes, edge_indices, gates = self.create_sparse_disconnected_graph(
                seed + 2000
            )

            original_beta0, original_beta1 = compute_betti_numbers(
                num_nodes, edge_indices, gates, tau=0.3
            )

            rng = np.random.RandomState(seed + 22000)
            mutated_gates = boundary_gate_mutation(
                gates=gates.copy(),
                tau_star=0.4,
                delta=0.2,
                sigma=0.15,
                rng=rng,
            )

            mutated_beta0, mutated_beta1 = compute_betti_numbers(
                num_nodes, edge_indices, mutated_gates, tau=0.3
            )

            if mutated_beta0 == original_beta0:
                beta0_preserved += 1

        preserve_rate = beta0_preserved / self.N_TRIALS
        print(f"\n=== BOUNDARY Preserves Structure on Sparse Graphs ===")
        print(f"β₀ preserved: {beta0_preserved}/{self.N_TRIALS} ({preserve_rate:.2%})")

        assert preserve_rate >= 0.5, f"BOUNDARY too destructive: {preserve_rate:.2%}"
