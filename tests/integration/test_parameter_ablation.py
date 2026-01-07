import pytest
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from onn.ops.logos_solver import LOGOSSolver, LOGOSConfig
from onn.ops.branching import (
    BranchConfig,
    BranchType,
    BranchFactory,
    BranchSelector,
    SurvivalConfig,
)
from onn.topo.filtration import compute_topo_summary
from onn.core.tensors import (
    SemanticNode,
    SemanticEdge,
    RawSemanticGraph,
    BOUND_TENSOR_DIM,
)


def create_test_graph(seed: int, num_nodes: int = 6):
    rng = np.random.RandomState(seed)

    nodes = []
    for i in range(num_nodes):
        bound = rng.randn(BOUND_TENSOR_DIM).astype(np.float32)
        bound[11] = abs(bound[11]) + 0.5
        node = SemanticNode(node_id=i, bound_tensor=bound, uncertainty=0.5)
        nodes.append(node)

    edges = []
    for i in range(num_nodes - 1):
        edge = SemanticEdge(
            source_id=i,
            target_id=i + 1,
            relation_embedding=rng.randn(16).astype(np.float32),
            gate=0.5 + rng.uniform(-0.15, 0.15),
        )
        edges.append(edge)

    if num_nodes >= 4:
        edge = SemanticEdge(
            source_id=0,
            target_id=num_nodes - 1,
            relation_embedding=rng.randn(16).astype(np.float32),
            gate=0.4,
        )
        edges.append(edge)

    return RawSemanticGraph(timestamp_ns=0, nodes=nodes, edge_candidates=edges)


@dataclass
class TrialResult:
    final_mean_gate: float
    final_active_ratio: float
    final_uncertainty: float
    converged: bool
    collapsed: bool


def run_solver_trial(
    graph: RawSemanticGraph,
    gate_lr: float = 0.05,
    gate_threshold: float = 0.5,
    uncertainty_lr: float = 0.02,
    uncertainty_max: float = 5.0,
    num_steps: int = 30,
) -> TrialResult:
    config = LOGOSConfig(
        max_iterations=8,
        learning_rate=0.01,
        gate_lr=gate_lr,
        gate_threshold=gate_threshold,
        uncertainty_lr=uncertainty_lr,
        uncertainty_max=uncertainty_max,
    )
    solver = LOGOSSolver(config)

    final_gates = None
    final_uncertainty = 0.5
    converged = False

    for _ in range(num_steps):
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

            if result.converged:
                converged = True

    if final_gates is None:
        return TrialResult(
            final_mean_gate=0.5,
            final_active_ratio=0.5,
            final_uncertainty=0.5,
            converged=False,
            collapsed=True,
        )

    mean_gate = float(np.mean(final_gates))
    active_ratio = float(np.mean(final_gates > 0.1))
    collapsed = mean_gate < 0.1 or active_ratio < 0.2

    return TrialResult(
        final_mean_gate=mean_gate,
        final_active_ratio=active_ratio,
        final_uncertainty=final_uncertainty,
        converged=converged,
        collapsed=collapsed,
    )


class TestParameterAblation:
    N_TRIALS = 20

    def test_gate_lr_sensitivity(self):
        gate_lr_values = [0.01, 0.05, 0.1, 0.2]
        results: Dict[float, List[TrialResult]] = {v: [] for v in gate_lr_values}

        for seed in range(self.N_TRIALS):
            graph = create_test_graph(seed)

            for gate_lr in gate_lr_values:
                trial = run_solver_trial(graph, gate_lr=gate_lr)
                results[gate_lr].append(trial)

        print(f"\n=== gate_lr Sensitivity ===")
        print(f"{'gate_lr':>10} {'mean_gate':>12} {'active_ratio':>14} {'collapsed%':>12}")
        print("-" * 52)

        for gate_lr in gate_lr_values:
            trials = results[gate_lr]
            mean_gate = np.mean([t.final_mean_gate for t in trials])
            active_ratio = np.mean([t.final_active_ratio for t in trials])
            collapse_rate = np.mean([t.collapsed for t in trials])
            print(f"{gate_lr:>10.2f} {mean_gate:>12.4f} {active_ratio:>14.4f} {collapse_rate:>12.2%}")

        collapse_rates = [np.mean([t.collapsed for t in results[v]]) for v in gate_lr_values]
        assert min(collapse_rates) < 0.5, "All gate_lr values cause high collapse"

    def test_gate_threshold_sensitivity(self):
        threshold_values = [0.2, 0.4, 0.6, 0.8]
        results: Dict[float, List[TrialResult]] = {v: [] for v in threshold_values}

        for seed in range(self.N_TRIALS):
            graph = create_test_graph(seed + 100)

            for threshold in threshold_values:
                trial = run_solver_trial(graph, gate_threshold=threshold)
                results[threshold].append(trial)

        print(f"\n=== gate_threshold Sensitivity ===")
        print(f"{'threshold':>10} {'mean_gate':>12} {'active_ratio':>14} {'collapsed%':>12}")
        print("-" * 52)

        for threshold in threshold_values:
            trials = results[threshold]
            mean_gate = np.mean([t.final_mean_gate for t in trials])
            active_ratio = np.mean([t.final_active_ratio for t in trials])
            collapse_rate = np.mean([t.collapsed for t in trials])
            print(f"{threshold:>10.2f} {mean_gate:>12.4f} {active_ratio:>14.4f} {collapse_rate:>12.2%}")

    def test_uncertainty_lr_sensitivity(self):
        u_lr_values = [0.01, 0.05, 0.1, 0.2]
        results: Dict[float, List[TrialResult]] = {v: [] for v in u_lr_values}

        for seed in range(self.N_TRIALS):
            graph = create_test_graph(seed + 200)

            for u_lr in u_lr_values:
                trial = run_solver_trial(graph, uncertainty_lr=u_lr)
                results[u_lr].append(trial)

        print(f"\n=== uncertainty_lr Sensitivity ===")
        print(f"{'u_lr':>10} {'uncertainty':>12} {'mean_gate':>12} {'collapsed%':>12}")
        print("-" * 52)

        for u_lr in u_lr_values:
            trials = results[u_lr]
            uncertainty = np.mean([t.final_uncertainty for t in trials])
            mean_gate = np.mean([t.final_mean_gate for t in trials])
            collapse_rate = np.mean([t.collapsed for t in trials])
            print(f"{u_lr:>10.2f} {uncertainty:>12.4f} {mean_gate:>12.4f} {collapse_rate:>12.2%}")

    def test_uncertainty_max_sensitivity(self):
        u_max_values = [2.0, 5.0, 10.0, 20.0]
        results: Dict[float, List[TrialResult]] = {v: [] for v in u_max_values}

        for seed in range(self.N_TRIALS):
            graph = create_test_graph(seed + 300)

            for u_max in u_max_values:
                trial = run_solver_trial(graph, uncertainty_max=u_max)
                results[u_max].append(trial)

        print(f"\n=== uncertainty_max Sensitivity ===")
        print(f"{'u_max':>10} {'uncertainty':>12} {'mean_gate':>12} {'collapsed%':>12}")
        print("-" * 52)

        for u_max in u_max_values:
            trials = results[u_max]
            uncertainty = np.mean([t.final_uncertainty for t in trials])
            mean_gate = np.mean([t.final_mean_gate for t in trials])
            collapse_rate = np.mean([t.collapsed for t in trials])
            print(f"{u_max:>10.1f} {uncertainty:>12.4f} {mean_gate:>12.4f} {collapse_rate:>12.2%}")

    def test_parameter_importance_ranking(self):
        baseline_graph = create_test_graph(seed=999)
        baseline = run_solver_trial(baseline_graph)
        baseline_score = baseline.final_mean_gate + baseline.final_active_ratio

        param_effects = {}

        gate_lr_low = run_solver_trial(baseline_graph, gate_lr=0.01)
        gate_lr_high = run_solver_trial(baseline_graph, gate_lr=0.2)
        gate_lr_effect = abs(
            (gate_lr_high.final_mean_gate + gate_lr_high.final_active_ratio) -
            (gate_lr_low.final_mean_gate + gate_lr_low.final_active_ratio)
        )
        param_effects["gate_lr"] = gate_lr_effect

        thresh_low = run_solver_trial(baseline_graph, gate_threshold=0.2)
        thresh_high = run_solver_trial(baseline_graph, gate_threshold=0.8)
        thresh_effect = abs(
            (thresh_high.final_mean_gate + thresh_high.final_active_ratio) -
            (thresh_low.final_mean_gate + thresh_low.final_active_ratio)
        )
        param_effects["gate_threshold"] = thresh_effect

        u_lr_low = run_solver_trial(baseline_graph, uncertainty_lr=0.01)
        u_lr_high = run_solver_trial(baseline_graph, uncertainty_lr=0.2)
        u_lr_effect = abs(
            (u_lr_high.final_mean_gate + u_lr_high.final_active_ratio) -
            (u_lr_low.final_mean_gate + u_lr_low.final_active_ratio)
        )
        param_effects["uncertainty_lr"] = u_lr_effect

        u_max_low = run_solver_trial(baseline_graph, uncertainty_max=2.0)
        u_max_high = run_solver_trial(baseline_graph, uncertainty_max=20.0)
        u_max_effect = abs(
            (u_max_high.final_mean_gate + u_max_high.final_active_ratio) -
            (u_max_low.final_mean_gate + u_max_low.final_active_ratio)
        )
        param_effects["uncertainty_max"] = u_max_effect

        sorted_params = sorted(param_effects.items(), key=lambda x: x[1], reverse=True)

        print(f"\n=== Parameter Importance Ranking ===")
        print(f"{'Parameter':>20} {'Effect Size':>15}")
        print("-" * 38)

        for param, effect in sorted_params:
            print(f"{param:>20} {effect:>15.4f}")

        assert len(sorted_params) > 0, "No parameters analyzed"

    def test_optimal_parameter_range(self):
        best_configs = []

        for seed in range(10):
            graph = create_test_graph(seed + 500)
            best_score = -1
            best_config = None

            for gate_lr in [0.01, 0.05, 0.1]:
                for u_lr in [0.01, 0.05, 0.1]:
                    trial = run_solver_trial(graph, gate_lr=gate_lr, uncertainty_lr=u_lr, num_steps=20)

                    score = trial.final_mean_gate + trial.final_active_ratio
                    if not trial.collapsed and score > best_score:
                        best_score = score
                        best_config = {"gate_lr": gate_lr, "uncertainty_lr": u_lr}

            if best_config:
                best_configs.append(best_config)

        print(f"\n=== Optimal Parameter Ranges ===")

        if best_configs:
            gate_lrs = [c["gate_lr"] for c in best_configs]
            u_lrs = [c["uncertainty_lr"] for c in best_configs]

            from collections import Counter
            gate_lr_counts = Counter(gate_lrs)
            u_lr_counts = Counter(u_lrs)

            print(f"Best gate_lr distribution: {dict(gate_lr_counts)}")
            print(f"Best uncertainty_lr distribution: {dict(u_lr_counts)}")
        else:
            print("No optimal configs found (all collapsed)")
