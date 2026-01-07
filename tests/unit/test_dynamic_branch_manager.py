import pytest
import numpy as np
from typing import List, Dict

from onn.ops.branching import (
    Branch,
    BranchConfig,
    BranchType,
    BranchResult,
    DynamicBranchManager,
    DynamicBranchManagerConfig,
    StagnationConfig,
)


# Helper to create mock branches and results
def create_mock_branch(
    seed: int, branch_type: BranchType = BranchType.CONSERVATIVE
) -> Branch:
    config = BranchConfig(branch_type=branch_type, seed=seed)
    rng = np.random.RandomState(seed)
    # Use small gate array for simplicity
    gates = rng.rand(10)
    meta_params = {"gate_lr": 0.1}
    return Branch(config=config, gates=gates, meta_params=meta_params, rng=rng)


def create_mock_result(
    branch: Branch,
    fitness: float,
    mean_gate: float = 0.5,
    uncertainty: float = 1.0,
    tau0_star: float = 0.5,
    tau1_star: float = 0.5,
) -> BranchResult:
    return BranchResult(
        branch=branch,
        final_gates=branch.gates,
        final_uncertainty=uncertainty,
        active_edge_ratio=0.5,
        mean_gate=mean_gate,
        fitness=fitness,
        converged=False,
        has_nan=False,
        tau0_star=tau0_star,
        tau1_star=tau1_star,
        beta0_final=1,
        beta1_final=1,
    )


class TestDynamicBranchManager:
    def setup_method(self):
        self.initial_branches = [create_mock_branch(seed=i) for i in range(3)]
        self.manager_config = DynamicBranchManagerConfig(
            stagnation_config=StagnationConfig(min_history_len=3, window_size=3)
        )
        self.manager = DynamicBranchManager(
            initial_branches=self.initial_branches,
            config=self.manager_config,
            base_seed=42,
        )
        # Mock parent info needed for spawning
        self.parent_gates = np.random.rand(10)
        self.parent_meta_params = {"gate_lr": 0.1}
        self.parent_tau_star = 0.5
        self.edge_indices = np.array([[0, 1], [1, 2]])
        self.num_nodes = 3

    def test_initialization(self):
        """Test that the manager initializes correctly."""
        assert len(self.manager.branches) == 3
        assert self.manager.next_branch_id == 3
        assert len(self.manager.history) == 3

    def test_no_spawning_when_improving(self):
        """Should not spawn new branches if fitness is improving."""
        # Use exponential growth and wide variance to ensure no stagnation is detected
        for i in range(5):
            results = [
                create_mock_result(
                    b,
                    fitness=1.5**i,
                    mean_gate=0.5 + np.sin(i) * 0.2,
                    uncertainty=1.0 + np.cos(i) * 0.2,
                    tau0_star=0.5 + np.sin(i) * 0.2,
                    tau1_star=0.5 + np.cos(i) * 0.2,
                )
                for b in self.initial_branches
            ]
            self.manager._update_histories(results)

        # Now, check for spawning
        final_results = [
            create_mock_result(
                b,
                fitness=1.5**5,
                mean_gate=0.6,
                uncertainty=0.9,
                tau0_star=0.7,
                tau1_star=0.3,
            )
            for b in self.initial_branches
        ]
        spawned = self.manager.step(
            final_results,
            self.parent_gates,
            self.parent_meta_params,
            self.parent_tau_star,
            self.edge_indices,
            self.num_nodes,
        )

        assert len(spawned) == 0, (
            "No branches should be spawned when fitness is improving"
        )
        assert len(self.manager.branches) == 3, "Branch count should not change"

    def test_stagnation_triggers_spawning(self):
        """Should spawn a new branch when most branches are stagnating."""
        # Simulate a few steps of stagnating fitness for all branches
        for i in range(5):
            # Use low variance values to trigger stagnation detector
            results = [
                create_mock_result(b, fitness=1.0, mean_gate=0.5, uncertainty=1.0)
                for b in self.initial_branches
            ]
            self.manager._update_histories(results)

        # Now, the step call should detect stagnation and spawn
        final_results = [
            create_mock_result(b, fitness=1.0) for b in self.initial_branches
        ]
        spawned = self.manager.step(
            final_results,
            self.parent_gates,
            self.parent_meta_params,
            self.parent_tau_star,
            self.edge_indices,
            self.num_nodes,
        )

        assert len(spawned) == 1, "Exactly one branch should be spawned on stagnation"
        assert len(self.manager.branches) == 4, (
            "Total branch count should increase to 4"
        )

        new_branch = spawned[0]
        assert new_branch.config.branch_type == BranchType.EXPLORATION, (
            "Spawned branch should be of exploration type"
        )
        assert self.manager.branches[3] is new_branch, (
            "New branch should be added to the manager's list"
        )

    def test_spawning_respects_max_branches(self):
        """Should not spawn branches if max_branches limit is reached."""
        self.manager.config.max_branches = 3  # Set max to current number

        # Simulate stagnation
        for i in range(5):
            results = [
                create_mock_result(b, fitness=1.0) for b in self.initial_branches
            ]
            self.manager._update_histories(results)

        final_results = [
            create_mock_result(b, fitness=1.0) for b in self.initial_branches
        ]
        spawned = self.manager.step(
            final_results,
            self.parent_gates,
            self.parent_meta_params,
            self.parent_tau_star,
            self.edge_indices,
            self.num_nodes,
        )

        assert len(spawned) == 0, (
            "No branches should be spawned if max_branches is reached"
        )
        assert len(self.manager.branches) == 3, "Branch count should remain at max"

    def test_get_branch_by_config(self):
        """Test the utility method to retrieve a branch by its config."""
        target_branch = self.initial_branches[1]
        target_config = target_branch.config

        branch_id, found_branch = self.manager.get_branch_by_config(target_config)

        assert branch_id == 1
        assert found_branch is target_branch

    def test_detector_handles_slow_improvement(self):
        """StagnationDetector should not fire on slow but steady improvement."""
        # This slope is negative, but smaller in magnitude than the default eps_slope (0.01)
        # log_slope = (log(0.4) - log(0.5)) / 5 = (-0.916 - (-0.693)) / 5 = -0.044
        # Since -0.044 < -0.01, this should NOT be considered a plateau.
        slow_improving_residuals = np.linspace(0.5, 0.4, 5).tolist()

        # Ensure other conditions don't trigger
        varying_gates = (np.sin(np.arange(5)) * 0.2 + 0.5).tolist()
        varying_uncertainty = (np.cos(np.arange(5)) * 0.2 + 1.0).tolist()
        tau0_history = (np.sin(np.arange(5)) * 0.2 + 0.5).tolist()
        tau1_history = (np.cos(np.arange(5)) * 0.2 + 0.5).tolist()

        signal = self.manager.stagnation_detector.detect(
            residual_history=slow_improving_residuals,
            gate_history=varying_gates,
            uncertainty_history=varying_uncertainty,
            tau0_star_history=tau0_history,
            tau1_star_history=tau1_history,
        )

        assert not signal.stable, "Detector fired prematurely on slow improvement"
        assert not signal.reasons["plateau"], (
            "Plateau should not be detected on slow improvement"
        )
