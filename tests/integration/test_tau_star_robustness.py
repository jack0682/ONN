import pytest
import numpy as np
from typing import List, Tuple
from scipy import stats

from onn.topo.filtration import (
    compute_topo_summary,
    compute_betti_numbers,
    compute_tau_star,
    compute_tau_star_v2,
    compute_filtration_profiles,
)


def create_random_graph(seed: int, num_nodes: int = 10, edge_density: float = 0.4):
    rng = np.random.RandomState(seed)

    edge_list = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if rng.random() < edge_density:
                edge_list.append([i, j])

    if len(edge_list) < num_nodes - 1:
        for i in range(num_nodes - 1):
            if [i, i + 1] not in edge_list:
                edge_list.append([i, i + 1])

    edge_indices = np.array(edge_list) if edge_list else np.array([]).reshape(0, 2)
    num_edges = len(edge_indices)

    gates = rng.uniform(0.2, 0.8, size=num_edges)

    return num_nodes, edge_indices, gates


class TestTauStarRobustness:
    N_TRIALS = 50
    NOISE_LEVELS = [0.001, 0.005, 0.01, 0.02, 0.05]

    def test_tau_star_stability_under_small_perturbation(self):
        """τ* should be stable under ±0.01 gate noise."""
        tau0_variances = []
        tau1_variances = []

        for seed in range(self.N_TRIALS):
            num_nodes, edge_indices, gates = create_random_graph(seed, num_nodes=10)
            if len(gates) == 0:
                continue

            topo_original = compute_topo_summary(num_nodes, edge_indices, gates)
            tau0_original = topo_original.tau0_star
            tau1_original = topo_original.tau1_star

            tau0_perturbed = []
            tau1_perturbed = []

            for perturb_seed in range(20):
                rng = np.random.RandomState(seed * 1000 + perturb_seed)
                noise = rng.uniform(-0.01, 0.01, size=gates.shape)
                perturbed_gates = np.clip(gates + noise, 0.0, 1.0)

                topo_perturbed = compute_topo_summary(
                    num_nodes, edge_indices, perturbed_gates
                )
                tau0_perturbed.append(topo_perturbed.tau0_star)
                tau1_perturbed.append(topo_perturbed.tau1_star)

            tau0_variances.append(np.var(tau0_perturbed))
            tau1_variances.append(np.var(tau1_perturbed))

        mean_tau0_var = np.mean(tau0_variances)
        mean_tau1_var = np.mean(tau1_variances)
        max_tau0_var = np.max(tau0_variances)
        max_tau1_var = np.max(tau1_variances)

        print(f"\n=== τ* Stability Under ±0.01 Perturbation ===")
        print(f"Mean variance τ₀*: {mean_tau0_var:.6f}")
        print(f"Mean variance τ₁*: {mean_tau1_var:.6f}")
        print(f"Max variance τ₀*:  {max_tau0_var:.6f}")
        print(f"Max variance τ₁*:  {max_tau1_var:.6f}")

        assert mean_tau0_var < 0.01, f"τ₀* too unstable: variance = {mean_tau0_var:.6f}"
        assert mean_tau1_var < 0.02, f"τ₁* too unstable: variance = {mean_tau1_var:.6f}"

    def test_tau_star_consistency_across_seeds(self):
        """Same graph should produce consistent τ* across different random initializations."""
        num_nodes = 12
        edge_indices = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [0, 5],
                [1, 4],
                [2, 6],
                [6, 7],
                [7, 8],
                [8, 9],
                [9, 10],
                [10, 11],
                [6, 11],
            ]
        )
        gates = np.array(
            [0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.7, 0.65, 0.6, 0.55, 0.5, 0.3]
        )

        tau0_values = []
        tau1_values = []

        for trial in range(30):
            rng = np.random.RandomState(trial)
            small_noise = rng.uniform(-0.001, 0.001, size=gates.shape)
            noisy_gates = np.clip(gates + small_noise, 0.0, 1.0)

            topo = compute_topo_summary(num_nodes, edge_indices, noisy_gates)
            tau0_values.append(topo.tau0_star)
            tau1_values.append(topo.tau1_star)

        tau0_std = np.std(tau0_values)
        tau1_std = np.std(tau1_values)

        print(f"\n=== τ* Consistency Across Seeds ===")
        print(f"τ₀* mean: {np.mean(tau0_values):.4f}, std: {tau0_std:.6f}")
        print(f"τ₁* mean: {np.mean(tau1_values):.4f}, std: {tau1_std:.6f}")

        assert tau0_std < 0.05, f"τ₀* inconsistent across seeds: std = {tau0_std:.6f}"
        assert tau1_std < 0.05, f"τ₁* inconsistent across seeds: std = {tau1_std:.6f}"

    def test_betti_false_positives_under_noise(self):
        """Count how often β₀/β₁ flip spuriously under noise."""
        false_positive_count = 0
        total_tests = 0

        for seed in range(self.N_TRIALS):
            num_nodes, edge_indices, gates = create_random_graph(
                seed + 100, num_nodes=10
            )
            if len(gates) == 0:
                continue

            beta0_original, beta1_original = compute_betti_numbers(
                num_nodes, edge_indices, gates, tau=0.4
            )

            for perturb_seed in range(10):
                rng = np.random.RandomState(seed * 100 + perturb_seed)
                noise = rng.uniform(-0.02, 0.02, size=gates.shape)
                perturbed_gates = np.clip(gates + noise, 0.0, 1.0)

                beta0_perturbed, beta1_perturbed = compute_betti_numbers(
                    num_nodes, edge_indices, perturbed_gates, tau=0.4
                )

                total_tests += 1
                if (
                    beta0_perturbed != beta0_original
                    or beta1_perturbed != beta1_original
                ):
                    false_positive_count += 1

        false_positive_rate = (
            false_positive_count / total_tests if total_tests > 0 else 0
        )

        print(f"\n=== Betti Number False Positives Under ±0.02 Noise ===")
        print(f"Total tests: {total_tests}")
        print(f"False positives: {false_positive_count}")
        print(f"False positive rate: {false_positive_rate:.2%}")

        assert false_positive_rate < 0.30, (
            f"Too many false positives: {false_positive_rate:.2%}"
        )

    def test_tau_star_on_tree_graph(self):
        """Tree graph (no cycles): β₁ = 0 always."""
        num_nodes = 8
        edge_indices = np.array(
            [[0, 1], [1, 2], [1, 3], [2, 4], [2, 5], [3, 6], [3, 7]]
        )

        for gate_level in [0.3, 0.5, 0.7]:
            gates = np.full(len(edge_indices), gate_level)

            topo = compute_topo_summary(num_nodes, edge_indices, gates)

            print(f"\n=== Tree Graph at gate={gate_level} ===")
            print(f"τ₀*: {topo.tau0_star:.4f}, τ₁*: {topo.tau1_star:.4f}")
            print(
                f"β₀(τ=0): {topo.beta0_profile[-1]}, β₁(τ=0): {topo.beta1_profile[-1]}"
            )

            assert topo.beta1_profile[-1] == 0, (
                f"Tree should have β₁=0, got {topo.beta1_profile[-1]}"
            )

    def test_tau_star_on_cycle_graph(self):
        """Simple cycle graph: β₁ should be 1 when all edges active."""
        num_nodes = 6
        edge_indices = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]])
        gates = np.array([0.8, 0.75, 0.7, 0.65, 0.6, 0.55])

        topo = compute_topo_summary(num_nodes, edge_indices, gates)

        print(f"\n=== Cycle Graph (n=6) ===")
        print(f"τ₀*: {topo.tau0_star:.4f}, τ₁*: {topo.tau1_star:.4f}")
        print(f"β₀(τ=0): {topo.beta0_profile[-1]}, β₁(τ=0): {topo.beta1_profile[-1]}")

        assert topo.beta0_profile[-1] == 1, (
            f"Cycle should have β₀=1, got {topo.beta0_profile[-1]}"
        )
        assert topo.beta1_profile[-1] == 1, (
            f"Cycle should have β₁=1, got {topo.beta1_profile[-1]}"
        )

    def test_tau_star_on_complete_graph(self):
        """Complete graph K_n: β₁ = n(n-1)/2 - (n-1) = (n-1)(n-2)/2."""
        num_nodes = 5
        edge_list = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edge_list.append([i, j])
        edge_indices = np.array(edge_list)
        gates = np.full(len(edge_indices), 0.7)

        topo = compute_topo_summary(num_nodes, edge_indices, gates)

        expected_beta1 = (num_nodes - 1) * (num_nodes - 2) // 2

        print(f"\n=== Complete Graph K_{num_nodes} ===")
        print(f"τ₀*: {topo.tau0_star:.4f}, τ₁*: {topo.tau1_star:.4f}")
        print(f"β₀(τ=0): {topo.beta0_profile[-1]}, β₁(τ=0): {topo.beta1_profile[-1]}")
        print(f"Expected β₁: {expected_beta1}")

        assert topo.beta0_profile[-1] == 1, (
            f"K_n should have β₀=1, got {topo.beta0_profile[-1]}"
        )
        assert topo.beta1_profile[-1] == expected_beta1, (
            f"K_n should have β₁={expected_beta1}, got {topo.beta1_profile[-1]}"
        )

    def test_tau_star_on_disconnected_graph(self):
        """Disconnected graph: β₀ should equal number of components."""
        num_nodes = 9
        edge_indices = np.array([[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8]])
        gates = np.full(len(edge_indices), 0.8)

        topo = compute_topo_summary(num_nodes, edge_indices, gates)

        print(f"\n=== Disconnected Graph (3 components) ===")
        print(f"τ₀*: {topo.tau0_star:.4f}, τ₁*: {topo.tau1_star:.4f}")
        print(f"β₀(τ=0): {topo.beta0_profile[-1]}, β₁(τ=0): {topo.beta1_profile[-1]}")

        assert topo.beta0_profile[-1] == 3, (
            f"Should have β₀=3, got {topo.beta0_profile[-1]}"
        )
        assert topo.beta1_profile[-1] == 0, (
            f"Should have β₁=0, got {topo.beta1_profile[-1]}"
        )

    def test_tau_star_noise_sensitivity_analysis(self):
        """Analyze τ* sensitivity across different noise levels."""
        sensitivities_tau0 = {level: [] for level in self.NOISE_LEVELS}
        sensitivities_tau1 = {level: [] for level in self.NOISE_LEVELS}

        for seed in range(30):
            num_nodes, edge_indices, gates = create_random_graph(
                seed + 200, num_nodes=12
            )
            if len(gates) == 0:
                continue

            topo_original = compute_topo_summary(num_nodes, edge_indices, gates)

            for noise_level in self.NOISE_LEVELS:
                tau0_deviations = []
                tau1_deviations = []

                for perturb_seed in range(10):
                    rng = np.random.RandomState(seed * 1000 + perturb_seed)
                    noise = rng.uniform(-noise_level, noise_level, size=gates.shape)
                    perturbed_gates = np.clip(gates + noise, 0.0, 1.0)

                    topo_perturbed = compute_topo_summary(
                        num_nodes, edge_indices, perturbed_gates
                    )

                    tau0_deviations.append(
                        abs(topo_perturbed.tau0_star - topo_original.tau0_star)
                    )
                    tau1_deviations.append(
                        abs(topo_perturbed.tau1_star - topo_original.tau1_star)
                    )

                sensitivities_tau0[noise_level].append(np.mean(tau0_deviations))
                sensitivities_tau1[noise_level].append(np.mean(tau1_deviations))

        print(f"\n=== τ* Noise Sensitivity Analysis ===")
        print(f"{'Noise Level':>12} {'Mean |Δτ₀*|':>14} {'Mean |Δτ₁*|':>14}")
        print("-" * 42)

        for noise_level in self.NOISE_LEVELS:
            mean_tau0 = np.mean(sensitivities_tau0[noise_level])
            mean_tau1 = np.mean(sensitivities_tau1[noise_level])
            print(f"{noise_level:>12.3f} {mean_tau0:>14.6f} {mean_tau1:>14.6f}")

        noise_001 = np.mean(sensitivities_tau0[0.01]) + np.mean(
            sensitivities_tau1[0.01]
        )
        assert noise_001 < 0.2, f"τ* too sensitive to ±0.01 noise: {noise_001:.4f}"

    def test_tau_grid_resolution_effect(self):
        """Compare τ* at different grid resolutions."""
        num_nodes = 10
        edge_indices = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 8],
                [8, 9],
                [0, 9],
                [2, 7],
            ]
        )
        gates = np.array(
            [0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25]
        )

        resolutions = [21, 51, 101, 201, 401]
        tau0_by_res = {}
        tau1_by_res = {}

        for num_steps in resolutions:
            tau_grid = np.linspace(1.0, 0.0, num_steps)
            profiles = compute_filtration_profiles(
                num_nodes, edge_indices, gates, tau_grid
            )
            tau0_by_res[num_steps] = compute_tau_star(profiles.beta0, profiles.tau_grid)
            tau1_by_res[num_steps] = compute_tau_star(profiles.beta1, profiles.tau_grid)

        print(f"\n=== τ* vs Grid Resolution ===")
        print(f"{'Resolution':>10} {'τ₀*':>10} {'τ₁*':>10}")
        print("-" * 32)
        for res in resolutions:
            print(f"{res:>10} {tau0_by_res[res]:>10.4f} {tau1_by_res[res]:>10.4f}")

        tau0_std = np.std(list(tau0_by_res.values()))
        tau1_std = np.std(list(tau1_by_res.values()))
        print(f"\nStd across resolutions: τ₀*={tau0_std:.4f}, τ₁*={tau1_std:.4f}")

        assert tau0_std < 0.1, f"τ₀* varies too much with resolution: {tau0_std:.4f}"
        assert tau1_std < 0.1, f"τ₁* varies too much with resolution: {tau1_std:.4f}"

    def test_tau_star_v2_behavior_and_stability(self):
        """
        Tests the center-of-mass τ* v2 calculation for correctness, robustness
        to multiple peaks, and stability under noise.
        """
        print(f"\n=== τ* v2 Advanced Behavior and Stability ===")

        # 1. Two-peak profile
        tau_grid = np.linspace(1.0, 0.0, 101)
        beta_profile = np.zeros(101)
        # Large peak at tau=0.7 (index 30)
        beta_profile[30:] += 10
        # Smaller peak at tau=0.3 (index 70)
        beta_profile[70:] += 5

        tau_v1 = compute_tau_star(beta_profile, tau_grid)
        tau_v2_full = compute_tau_star_v2(
            beta_profile, tau_grid, window_ratio=1.0
        )  # Full support
        tau_v2_filtered = compute_tau_star_v2(
            beta_profile, tau_grid, window_ratio=0.2
        )  # 20% window

        print(
            f"Two-peak profile: v1 (argmax)={tau_v1:.4f}, v2 (full)={tau_v2_full:.4f}, v2 (filtered)={tau_v2_filtered:.4f}"
        )

        # v1 (argmax) should find the larger peak at ~0.7
        assert 0.65 < tau_v1 < 0.75
        # v2 (full support) should be between the two peaks, pulled toward the larger one
        assert 0.5 < tau_v2_full < 0.7
        # v2 (filtered window) should ignore the smaller peak and be very close to the main one
        assert 0.65 < tau_v2_filtered < 0.75
        assert abs(tau_v2_filtered - tau_v1) < 0.05

        # 2. Noise stability test
        rng = np.random.RandomState(42)
        noisy_taus = []
        for _ in range(20):
            noise = rng.normal(0, 0.5, size=beta_profile.shape)
            noisy_profile = beta_profile + noise
            noisy_taus.append(
                compute_tau_star_v2(noisy_profile, tau_grid, window_ratio=0.2)
            )

        tau_variance = np.var(noisy_taus)
        print(f"τ* v2 stability under noise: variance = {tau_variance:.6f}")
        assert tau_variance < 0.01, "τ* v2 is too unstable under noise"
