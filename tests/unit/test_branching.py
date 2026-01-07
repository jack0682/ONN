import pytest
import numpy as np
import torch

from onn.ops.branching import (
    StagnationDetector,
    StagnationConfig,
    StagnationSignal,
    BranchFactory,
    BranchSelector,
    BranchResult,
    BranchConfig,
    BranchType,
    SurvivalConfig,
    boundary_gate_mutation,
    create_conservative_config,
    create_exploration_config,
    create_rewire_config,
)
from onn.topo.filtration import (
    compute_betti_numbers,
    compute_filtration_profiles,
    compute_tau_star,
    compute_topo_summary,
)


class TestFiltration:
    def test_betti_numbers_disconnected(self):
        num_nodes = 4
        edge_indices = np.array([[0, 1], [2, 3]])
        gates = np.array([1.0, 1.0])

        b0, b1 = compute_betti_numbers(num_nodes, edge_indices, gates, tau=0.5)
        assert b0 == 2
        assert b1 == 0

    def test_betti_numbers_connected(self):
        num_nodes = 3
        edge_indices = np.array([[0, 1], [1, 2], [2, 0]])
        gates = np.array([1.0, 1.0, 1.0])

        b0, b1 = compute_betti_numbers(num_nodes, edge_indices, gates, tau=0.5)
        assert b0 == 1
        assert b1 == 1

    def test_filtration_profiles_decreasing_beta0(self):
        num_nodes = 4
        edge_indices = np.array([[0, 1], [1, 2], [2, 3]])
        gates = np.array([0.9, 0.5, 0.2])

        profiles = compute_filtration_profiles(num_nodes, edge_indices, gates)

        assert profiles.beta0[0] == 4
        assert profiles.beta0[-1] <= 1

    def test_tau_star_detection(self):
        beta_profile = np.array([4, 4, 3, 2, 1, 1, 1])
        tau_grid = np.linspace(1.0, 0.0, 7)

        tau_star = compute_tau_star(beta_profile, tau_grid)
        assert 0.0 <= tau_star <= 1.0


class TestStagnationDetector:
    def test_plateau_detection(self):
        detector = StagnationDetector(
            StagnationConfig(
                window_size=5,
                eps_slope=0.01,
                min_history_len=5,
            )
        )

        residual_history = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        gate_history = [0.5] * 7
        uncertainty_history = [1.0] * 7

        signal = detector.detect(residual_history, gate_history, uncertainty_history)

        assert signal.stable == True
        assert signal.reasons["plateau"] == True

    def test_no_plateau_during_improvement(self):
        detector = StagnationDetector(
            StagnationConfig(
                window_size=5,
                eps_slope=0.01,
                min_history_len=5,
            )
        )

        residual_history = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1]
        gate_history = [0.5 + i * 0.05 for i in range(7)]
        uncertainty_history = [1.0 - i * 0.1 for i in range(7)]

        signal = detector.detect(residual_history, gate_history, uncertainty_history)

        assert signal.stable == False
        assert signal.reasons["plateau"] == False

    def test_saturation_detection(self):
        detector = StagnationDetector(
            StagnationConfig(
                window_size=5,
                eps_slope=0.5,
                eps_gate_std=0.01,
                eps_u_std=0.01,
                min_history_len=5,
            )
        )

        residual_history = [1.0, 0.9, 0.85, 0.82, 0.80, 0.79, 0.78]
        gate_history = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        uncertainty_history = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        signal = detector.detect(residual_history, gate_history, uncertainty_history)

        assert signal.reasons["saturation"] == True

    def test_topo_lock_detection(self):
        detector = StagnationDetector(
            StagnationConfig(
                eps_tau_var=0.001,
                min_history_len=3,
            )
        )

        residual_history = [1.0] * 10
        gate_history = [0.5] * 10
        uncertainty_history = [1.0] * 10
        tau0_star_history = [0.5, 0.5, 0.5, 0.5, 0.5]
        tau1_star_history = [0.3, 0.3, 0.3, 0.3, 0.3]

        signal = detector.detect(
            residual_history,
            gate_history,
            uncertainty_history,
            tau0_star_history,
            tau1_star_history,
        )

        assert signal.reasons["topo_lock"] == True


class TestConditionalMutation:
    def test_mutation_only_when_stable(self):
        np.random.seed(42)

        parent_gates = np.array([0.5, 0.5, 0.5, 0.5])
        parent_meta = {"beta_obs": 1.0, "beta_cons": 1.0}
        tau_star = 0.5

        factory = BranchFactory(base_seed=42)
        branches = factory.make_branches(parent_gates, parent_meta, tau_star)

        conservative = [
            b for b in branches if b.config.branch_type == BranchType.CONSERVATIVE
        ][0]
        exploration = [
            b for b in branches if b.config.branch_type == BranchType.EXPLORATION
        ][0]

        assert np.allclose(conservative.gates, parent_gates)
        assert not np.allclose(exploration.gates, parent_gates)

    def test_boundary_mutation_targets_near_tau_star(self):
        np.random.seed(123)
        rng = np.random.RandomState(123)

        gates = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        tau_star = 0.5
        delta = 0.15
        sigma = 0.3

        mutated = boundary_gate_mutation(gates, tau_star, delta, sigma, rng)

        assert np.allclose(mutated[0], gates[0], atol=1e-6)
        assert np.allclose(mutated[4], gates[4], atol=1e-6)
        assert not np.allclose(mutated[2], gates[2])


class TestBranchSurvivalSelection:
    def test_collapse_branch_filtered(self):
        selector = BranchSelector(
            SurvivalConfig(
                min_active_edge_ratio=0.2,
                min_mean_gate=0.1,
            )
        )

        collapsed = BranchResult(
            branch=None,
            final_gates=np.array([0.0, 0.0, 0.0]),
            final_uncertainty=1.0,
            active_edge_ratio=0.0,
            mean_gate=0.0,
            fitness=0.5,
            converged=False,
            has_nan=False,
            tau0_star=0.5,
            tau1_star=0.3,
            beta0_final=3,
            beta1_final=0,
        )

        surviving = BranchResult(
            branch=None,
            final_gates=np.array([0.8, 0.7, 0.6]),
            final_uncertainty=1.0,
            active_edge_ratio=1.0,
            mean_gate=0.7,
            fitness=0.3,
            converged=True,
            has_nan=False,
            tau0_star=0.5,
            tau1_star=0.3,
            beta0_final=1,
            beta1_final=1,
        )

        winner, backup = selector.select([collapsed, surviving])

        assert winner is not None
        assert winner.mean_gate == 0.7
        assert backup is None

    def test_nan_branch_filtered(self):
        selector = BranchSelector()

        nan_result = BranchResult(
            branch=None,
            final_gates=np.array([0.5, 0.5]),
            final_uncertainty=1.0,
            active_edge_ratio=0.5,
            mean_gate=0.5,
            fitness=1.0,
            converged=True,
            has_nan=True,
            tau0_star=0.5,
            tau1_star=0.3,
            beta0_final=1,
            beta1_final=0,
        )

        winner, backup = selector.select([nan_result])

        assert winner is None

    def test_winner_has_highest_fitness(self):
        selector = BranchSelector()

        results = [
            BranchResult(
                branch=None,
                final_gates=np.array([0.5, 0.5]),
                final_uncertainty=1.0,
                active_edge_ratio=0.5,
                mean_gate=0.5,
                fitness=0.3,
                converged=True,
                has_nan=False,
                tau0_star=0.5,
                tau1_star=0.3,
                beta0_final=1,
                beta1_final=0,
            ),
            BranchResult(
                branch=None,
                final_gates=np.array([0.6, 0.6]),
                final_uncertainty=1.0,
                active_edge_ratio=0.6,
                mean_gate=0.6,
                fitness=0.9,
                converged=True,
                has_nan=False,
                tau0_star=0.4,
                tau1_star=0.2,
                beta0_final=1,
                beta1_final=1,
            ),
        ]

        winner, backup = selector.select(results)

        assert winner is not None
        assert winner.fitness == 0.9
        assert backup is not None
        assert backup.fitness == 0.3


class TestGateCollapsePrevention:
    def test_conservative_branch_maintains_gates(self):
        np.random.seed(42)

        parent_gates = np.array([0.8, 0.7, 0.6, 0.5])
        parent_meta = {}

        factory = BranchFactory(base_seed=42)
        branches = factory.make_branches(parent_gates, parent_meta, tau_star=0.5)

        conservative = [
            b for b in branches if b.config.branch_type == BranchType.CONSERVATIVE
        ][0]

        assert conservative.config.min_active_edge_ratio >= 0.3
        assert conservative.config.min_mean_gate >= 0.2
        assert conservative.config.apply_mutation is False
