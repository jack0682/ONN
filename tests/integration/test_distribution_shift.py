import pytest
import numpy as np
import torch

from onn.ops.logos_solver import LOGOSSolver, LOGOSConfig
from onn.ops.branching import (
    BranchFactory,
    BranchSelector,
    BranchResult,
    SurvivalConfig,
    BranchType,
)
from onn.topo.filtration import compute_topo_summary
from onn.core.tensors import (
    SemanticNode,
    SemanticEdge,
    RawSemanticGraph,
    BOUND_TENSOR_DIM,
)


def create_test_graph(seed: int, noise_scale: float = 0.0) -> RawSemanticGraph:
    rng = np.random.RandomState(seed)
    num_nodes = 4

    nodes = []
    for i in range(num_nodes):
        bound = rng.randn(BOUND_TENSOR_DIM).astype(np.float32)
        bound[11] = abs(bound[11]) + 0.5
        if noise_scale > 0:
            bound += rng.randn(BOUND_TENSOR_DIM).astype(np.float32) * noise_scale
        node = SemanticNode(node_id=i, bound_tensor=bound, uncertainty=0.5)
        nodes.append(node)

    edges = []
    for i in range(num_nodes - 1):
        edge = SemanticEdge(
            source_id=i,
            target_id=i + 1,
            relation_embedding=rng.randn(16).astype(np.float32),
            gate=0.7,
        )
        edges.append(edge)

    return RawSemanticGraph(timestamp_ns=0, nodes=nodes, edge_candidates=edges)


def run_branch_under_condition(
    branch_type: BranchType,
    seed: int,
    noise_scale: float = 0.0,
    solver_steps: int = 50,
    delay_factor: int = 1,
) -> BranchResult:
    graph = create_test_graph(seed, noise_scale)

    config = LOGOSConfig(
        max_iterations=5 * delay_factor,
        learning_rate=0.01,
        uncertainty_lr=0.02,
        gate_lr=0.05,
    )
    solver = LOGOSSolver(config)

    factory = BranchFactory(base_seed=seed)
    parent_gates = np.array([e.gate for e in graph.edge_candidates])
    parent_meta = {"gate_lr": 0.05, "uncertainty_lr": 0.02}

    edge_indices = np.array([[e.source_id, e.target_id] for e in graph.edge_candidates])
    num_nodes = len(graph.nodes)

    branches = factory.make_branches(
        parent_gates=parent_gates,
        parent_meta_params=parent_meta,
        tau_star=0.5,
        edge_indices=edge_indices,
        num_nodes=num_nodes,
    )

    target_branch = next(
        (b for b in branches if b.config.branch_type == branch_type), branches[0]
    )

    final_gates = target_branch.gates.copy()
    final_uncertainty = 0.5
    residual_norms = []
    converged = False
    has_nan = False

    for _ in range(solver_steps):
        result_graph = solver.solve(graph)
        result = solver.get_last_result()

        if result and result.final_gates is not None:
            fg = result.final_gates
            if hasattr(fg, "detach"):
                fg = fg.detach().cpu().numpy()
            final_gates = fg

            fu = result.final_uncertainties
            if hasattr(fu, "detach"):
                fu = fu.detach().cpu().numpy()
            final_uncertainty = float(np.mean(fu))

            if result.residual_norm_history:
                residual_norms.append(result.residual_norm_history[-1])

            if np.any(np.isnan(fg)) or np.any(np.isnan(fu)):
                has_nan = True
                break

            if result.converged:
                converged = True

    active_ratio = float(np.mean(final_gates > 0.1)) if len(final_gates) > 0 else 0.0
    mean_gate = float(np.mean(final_gates)) if len(final_gates) > 0 else 0.0

    fitness = active_ratio * 0.5 + mean_gate * 0.3 + (1.0 if converged else 0.0) * 0.2

    topo = compute_topo_summary(
        num_nodes=num_nodes, edge_indices=edge_indices, gates=final_gates
    )

    return BranchResult(
        branch=target_branch,
        final_gates=final_gates,
        final_uncertainty=final_uncertainty,
        active_edge_ratio=active_ratio,
        mean_gate=mean_gate,
        fitness=fitness,
        converged=converged,
        has_nan=has_nan,
        tau0_star=topo.tau0_star,
        tau1_star=topo.tau1_star,
        beta0_final=int(topo.beta0_profile[-1]) if len(topo.beta0_profile) > 0 else 1,
        beta1_final=int(topo.beta1_profile[-1]) if len(topo.beta1_profile) > 0 else 0,
    )


class TestDistributionShiftRobustness:
    N_TRIALS = 30
    SURVIVAL_THRESHOLD = 0.80

    def test_conservative_survives_noise_low(self):
        survival_config = SurvivalConfig(
            min_active_edge_ratio=0.15, min_mean_gate=0.1, u_max_threshold=10.0
        )
        selector = BranchSelector(survival_config)

        survival_count = 0
        for seed in range(self.N_TRIALS):
            result = run_branch_under_condition(
                branch_type=BranchType.CONSERVATIVE,
                seed=seed,
                noise_scale=0.1,
                solver_steps=30,
            )
            if selector._passes_survival_filter(result):
                survival_count += 1

        survival_rate = survival_count / self.N_TRIALS
        assert survival_rate >= self.SURVIVAL_THRESHOLD, (
            f"CONSERVATIVE survival under low noise: {survival_rate:.2f} < {self.SURVIVAL_THRESHOLD}"
        )

    def test_rewire_survives_noise_high(self):
        survival_config = SurvivalConfig(
            min_active_edge_ratio=0.1, min_mean_gate=0.08, u_max_threshold=15.0
        )
        selector = BranchSelector(survival_config)

        survival_count = 0
        for seed in range(self.N_TRIALS):
            result = run_branch_under_condition(
                branch_type=BranchType.REWIRE,
                seed=seed + 1000,
                noise_scale=0.5,
                solver_steps=40,
            )
            if selector._passes_survival_filter(result):
                survival_count += 1

        survival_rate = survival_count / self.N_TRIALS
        min_threshold = 0.70
        assert survival_rate >= min_threshold, (
            f"REWIRE survival under high noise: {survival_rate:.2f} < {min_threshold}"
        )

    def test_exploration_survives_delay_high(self):
        survival_config = SurvivalConfig(
            min_active_edge_ratio=0.1, min_mean_gate=0.08, u_max_threshold=15.0
        )
        selector = BranchSelector(survival_config)

        survival_count = 0
        for seed in range(self.N_TRIALS):
            result = run_branch_under_condition(
                branch_type=BranchType.EXPLORATION,
                seed=seed + 2000,
                noise_scale=0.0,
                solver_steps=30,
                delay_factor=3,
            )
            if selector._passes_survival_filter(result):
                survival_count += 1

        survival_rate = survival_count / self.N_TRIALS
        min_threshold = 0.70
        assert survival_rate >= min_threshold, (
            f"EXPLORATION survival under high delay: {survival_rate:.2f} < {min_threshold}"
        )
