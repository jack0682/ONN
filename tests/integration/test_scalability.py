import pytest
import numpy as np
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass

from onn.topo.filtration import compute_topo_summary, compute_filtration_profiles
from onn.ops.branching import topology_rewire_mutation


@dataclass
class ScalabilityMetrics:
    num_nodes: int
    num_edges: int
    topo_time: float
    mutation_time: float
    tau0_star: float
    tau1_star: float


def create_scalable_graph(num_nodes: int, edge_density: float = 0.2, seed: int = 42):
    rng = np.random.RandomState(seed)

    edge_list = []
    for i in range(num_nodes - 1):
        edge_list.append([i, i + 1])

    for i in range(num_nodes):
        for j in range(i + 2, num_nodes):
            if rng.random() < edge_density:
                edge_list.append([i, j])

    edge_indices = np.array(edge_list)
    num_edges = len(edge_indices)

    gates = rng.uniform(0.3, 0.8, size=num_edges)

    return num_nodes, edge_indices, gates


class TestScalability:
    GRAPH_SIZES = [10, 25, 50, 100]
    EDGE_DENSITY = 0.15

    def test_topo_summary_runtime(self):
        results: Dict[int, List[float]] = {n: [] for n in self.GRAPH_SIZES}

        for n in self.GRAPH_SIZES:
            for trial in range(3):
                num_nodes, edge_indices, gates = create_scalable_graph(
                    n, edge_density=self.EDGE_DENSITY, seed=trial * 100 + n
                )

                start = time.perf_counter()
                topo = compute_topo_summary(num_nodes, edge_indices, gates)
                elapsed = time.perf_counter() - start

                results[n].append(elapsed)

        print(f"\n=== Topological Summary Runtime ===")
        print(f"{'Nodes':>8} {'Edges':>8} {'Mean (ms)':>12} {'Max (ms)':>12}")
        print("-" * 44)

        for n in self.GRAPH_SIZES:
            num_nodes, edge_indices, _ = create_scalable_graph(n, self.EDGE_DENSITY, seed=0)
            mean_time = np.mean(results[n]) * 1000
            max_time = np.max(results[n]) * 1000
            print(f"{n:>8} {len(edge_indices):>8} {mean_time:>12.2f} {max_time:>12.2f}")

        time_100 = np.mean(results[100]) * 1000
        assert time_100 < 5000, f"100-node graph too slow: {time_100:.2f}ms (limit: 5000ms)"

    def test_filtration_profile_runtime(self):
        results: Dict[int, List[float]] = {n: [] for n in self.GRAPH_SIZES}

        for n in self.GRAPH_SIZES:
            for trial in range(3):
                num_nodes, edge_indices, gates = create_scalable_graph(
                    n, edge_density=self.EDGE_DENSITY, seed=trial * 100 + n
                )

                start = time.perf_counter()
                profiles = compute_filtration_profiles(num_nodes, edge_indices, gates, num_tau_steps=101)
                elapsed = time.perf_counter() - start

                results[n].append(elapsed)

        print(f"\n=== Filtration Profile Runtime (101 steps) ===")
        print(f"{'Nodes':>8} {'Mean (ms)':>12} {'Max (ms)':>12}")
        print("-" * 36)

        for n in self.GRAPH_SIZES:
            mean_time = np.mean(results[n]) * 1000
            max_time = np.max(results[n]) * 1000
            print(f"{n:>8} {mean_time:>12.2f} {max_time:>12.2f}")

    def test_mutation_runtime(self):
        results: Dict[int, List[float]] = {n: [] for n in self.GRAPH_SIZES}

        for n in self.GRAPH_SIZES:
            for trial in range(3):
                num_nodes, edge_indices, gates = create_scalable_graph(
                    n, edge_density=self.EDGE_DENSITY, seed=trial * 100 + n
                )

                rng = np.random.RandomState(trial + n)

                start = time.perf_counter()
                mutated = topology_rewire_mutation(
                    gates=gates.copy(),
                    edge_indices=edge_indices,
                    num_nodes=num_nodes,
                    beta0_current=1,
                    beta1_current=0,
                    rng=rng,
                    target_beta0=1,
                    boost_cycles=True,
                )
                elapsed = time.perf_counter() - start

                results[n].append(elapsed)

        print(f"\n=== Topology Rewire Mutation Runtime ===")
        print(f"{'Nodes':>8} {'Mean (ms)':>12} {'Max (ms)':>12}")
        print("-" * 36)

        for n in self.GRAPH_SIZES:
            mean_time = np.mean(results[n]) * 1000
            max_time = np.max(results[n]) * 1000
            print(f"{n:>8} {mean_time:>12.2f} {max_time:>12.2f}")

    def test_tau_star_stability_at_scale(self):
        tau0_std_by_size = {}
        tau1_std_by_size = {}

        for n in self.GRAPH_SIZES:
            tau0_values = []
            tau1_values = []

            for trial in range(10):
                num_nodes, edge_indices, gates = create_scalable_graph(
                    n, edge_density=self.EDGE_DENSITY, seed=trial * 100 + n
                )

                topo = compute_topo_summary(num_nodes, edge_indices, gates)
                tau0_values.append(topo.tau0_star)
                tau1_values.append(topo.tau1_star)

            tau0_std_by_size[n] = np.std(tau0_values)
            tau1_std_by_size[n] = np.std(tau1_values)

        print(f"\n=== τ* Stability at Scale ===")
        print(f"{'Nodes':>8} {'τ₀* std':>12} {'τ₁* std':>12}")
        print("-" * 36)

        for n in self.GRAPH_SIZES:
            print(f"{n:>8} {tau0_std_by_size[n]:>12.4f} {tau1_std_by_size[n]:>12.4f}")

        assert tau0_std_by_size[100] < 0.3, f"τ₀* too variable at 100 nodes"
        assert tau1_std_by_size[100] < 0.3, f"τ₁* too variable at 100 nodes"

    def test_scaling_exponent(self):
        times = {}

        for n in self.GRAPH_SIZES:
            trial_times = []
            for trial in range(5):
                num_nodes, edge_indices, gates = create_scalable_graph(
                    n, edge_density=self.EDGE_DENSITY, seed=trial * 100 + n
                )

                start = time.perf_counter()
                topo = compute_topo_summary(num_nodes, edge_indices, gates)
                elapsed = time.perf_counter() - start

                trial_times.append(elapsed)

            times[n] = np.mean(trial_times)

        nodes = np.array(list(times.keys()))
        runtimes = np.array(list(times.values()))

        log_nodes = np.log(nodes)
        log_times = np.log(runtimes + 1e-10)
        slope, intercept = np.polyfit(log_nodes, log_times, 1)

        print(f"\n=== Scaling Exponent Analysis ===")
        print(f"Time(n) ≈ O(n^{slope:.2f})")
        print(f"Nodes vs Runtime:")
        for n in self.GRAPH_SIZES:
            print(f"  n={n}: {times[n]*1000:.2f}ms")

        assert slope < 4.0, f"Scaling too steep: O(n^{slope:.2f})"

    def test_edge_count_vs_runtime(self):
        densities = [0.1, 0.2, 0.3, 0.5]
        times_by_density = {}

        n = 50

        for density in densities:
            trial_times = []
            for trial in range(3):
                num_nodes, edge_indices, gates = create_scalable_graph(
                    n, edge_density=density, seed=trial * 100
                )

                start = time.perf_counter()
                topo = compute_topo_summary(num_nodes, edge_indices, gates)
                elapsed = time.perf_counter() - start

                trial_times.append((len(edge_indices), elapsed))

            avg_edges = np.mean([t[0] for t in trial_times])
            avg_time = np.mean([t[1] for t in trial_times])
            times_by_density[density] = (avg_edges, avg_time)

        print(f"\n=== Edge Count vs Runtime (n={n}) ===")
        print(f"{'Density':>10} {'Edges':>8} {'Time (ms)':>12}")
        print("-" * 34)

        for density in densities:
            edges, t = times_by_density[density]
            print(f"{density:>10.1%} {edges:>8.0f} {t*1000:>12.2f}")

    def test_memory_reasonable(self):
        import sys

        for n in [10, 50, 100]:
            num_nodes, edge_indices, gates = create_scalable_graph(
                n, edge_density=self.EDGE_DENSITY, seed=42
            )

            edge_size = sys.getsizeof(edge_indices) + edge_indices.nbytes
            gate_size = sys.getsizeof(gates) + gates.nbytes

            topo = compute_topo_summary(num_nodes, edge_indices, gates)

            profile_size = (
                sys.getsizeof(topo.beta0_profile) + topo.beta0_profile.nbytes +
                sys.getsizeof(topo.beta1_profile) + topo.beta1_profile.nbytes +
                sys.getsizeof(topo.tau_grid) + topo.tau_grid.nbytes
            )

            total_kb = (edge_size + gate_size + profile_size) / 1024

            print(f"n={n}: ~{total_kb:.1f} KB")

        assert total_kb < 1024, f"Memory usage too high: {total_kb:.1f} KB > 1 MB"
