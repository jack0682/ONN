import pytest
import numpy as np
import torch
from typing import List, Tuple

from onn.ops.logos_solver import LOGOSSolver, LOGOSConfig
from onn.ops.branching import (
    BranchFactory,
    BranchType,
    create_conservative_config,
    create_exploration_config,
    create_rewire_config,
)
from onn.topo.filtration import compute_topo_summary
from onn.core.tensors import (
    SemanticNode,
    SemanticEdge,
    RawSemanticGraph,
    BOUND_TENSOR_DIM,
)


def cliffs_delta(sample1: List[float], sample2: List[float]) -> float:
    """
    Compute Cliff's delta effect size between two samples.

    Cliff's delta = (n_greater - n_less) / (n1 * n2)
    where n_greater = count of sample1[i] > sample2[j]
          n_less = count of sample1[i] < sample2[j]

    Returns value in [-1, 1]:
    - |d| < 0.147: negligible
    - 0.147 <= |d| < 0.33: small
    - 0.33 <= |d| < 0.474: medium
    - |d| >= 0.474: large
    """
    n1, n2 = len(sample1), len(sample2)
    if n1 == 0 or n2 == 0:
        return 0.0

    n_greater = 0
    n_less = 0

    for x in sample1:
        for y in sample2:
            if x > y:
                n_greater += 1
            elif x < y:
                n_less += 1

    return (n_greater - n_less) / (n1 * n2)


def create_varied_graph(seed: int, variation: str = "base") -> RawSemanticGraph:
    rng = np.random.RandomState(seed)
    num_nodes = 5

    nodes = []
    for i in range(num_nodes):
        bound = rng.randn(BOUND_TENSOR_DIM).astype(np.float32)
        bound[11] = abs(bound[11]) + 0.3

        if variation == "sparse":
            bound *= 0.5
        elif variation == "dense":
            bound *= 1.5

        node = SemanticNode(node_id=i, bound_tensor=bound, uncertainty=0.5)
        nodes.append(node)

    edges = []
    base_gate = 0.6 if variation == "base" else (0.4 if variation == "sparse" else 0.8)

    for i in range(num_nodes - 1):
        edge = SemanticEdge(
            source_id=i,
            target_id=i + 1,
            relation_embedding=rng.randn(16).astype(np.float32),
            gate=base_gate + rng.uniform(-0.1, 0.1),
        )
        edges.append(edge)

    if num_nodes >= 3:
        edges.append(
            SemanticEdge(
                source_id=0,
                target_id=num_nodes - 1,
                relation_embedding=rng.randn(16).astype(np.float32),
                gate=base_gate * 0.8,
            )
        )

    return RawSemanticGraph(timestamp_ns=0, nodes=nodes, edge_candidates=edges)


def collect_tau_star_for_branch(
    branch_type: BranchType,
    n_trials: int,
    seed_offset: int = 0,
) -> Tuple[List[float], List[float]]:
    tau0_samples = []
    tau1_samples = []

    for trial in range(n_trials):
        seed = seed_offset + trial
        variation = ["base", "sparse", "dense"][trial % 3]
        graph = create_varied_graph(seed, variation)

        config = LOGOSConfig(
            max_iterations=8,
            learning_rate=0.01,
            gate_lr=0.08,
            uncertainty_lr=0.03,
        )
        solver = LOGOSSolver(config)

        factory = BranchFactory(base_seed=seed)
        parent_gates = np.array([e.gate for e in graph.edge_candidates])
        parent_meta = {"gate_lr": 0.08}

        edge_indices = np.array(
            [[e.source_id, e.target_id] for e in graph.edge_candidates]
        )
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

        gates = target_branch.gates.copy()

        for _ in range(20):
            result_graph = solver.solve(graph)
            result = solver.get_last_result()
            if result and result.final_gates is not None:
                fg = result.final_gates
                if hasattr(fg, "detach"):
                    fg = fg.detach().cpu().numpy()
                gates = fg

        topo = compute_topo_summary(
            num_nodes=num_nodes, edge_indices=edge_indices, gates=gates
        )

        tau0_samples.append(topo.tau0_star)
        tau1_samples.append(topo.tau1_star)

    return tau0_samples, tau1_samples


class TestTauStarDistribution:
    N_TRIALS = 50
    TAU0_EFFECT_THRESHOLD = 0.05
    TAU1_EFFECT_THRESHOLD = 0.01

    def test_tau0_star_separation_rewire_vs_conservative(self):
        tau0_conservative, _ = collect_tau_star_for_branch(
            BranchType.CONSERVATIVE, self.N_TRIALS, seed_offset=0
        )
        tau0_rewire, _ = collect_tau_star_for_branch(
            BranchType.REWIRE, self.N_TRIALS, seed_offset=5000
        )

        delta = cliffs_delta(tau0_conservative, tau0_rewire)

        assert abs(delta) >= self.TAU0_EFFECT_THRESHOLD, (
            f"τ0* separation too weak between CONSERVATIVE and REWIRE: "
            f"Cliff's δ = {delta:.3f} (threshold: {self.TAU0_EFFECT_THRESHOLD})"
        )

    def test_tau1_star_separation_rewire_vs_conservative(self):
        _, tau1_conservative = collect_tau_star_for_branch(
            BranchType.CONSERVATIVE, self.N_TRIALS, seed_offset=10000
        )
        _, tau1_rewire = collect_tau_star_for_branch(
            BranchType.REWIRE, self.N_TRIALS, seed_offset=15000
        )

        delta = cliffs_delta(tau1_conservative, tau1_rewire)

        assert abs(delta) >= self.TAU1_EFFECT_THRESHOLD, (
            f"τ1* separation too weak between CONSERVATIVE and REWIRE: "
            f"Cliff's δ = {delta:.3f} (threshold: {self.TAU1_EFFECT_THRESHOLD})"
        )

    def test_tau_star_distributions_not_degenerate(self):
        tau0_samples, tau1_samples = collect_tau_star_for_branch(
            BranchType.REWIRE, self.N_TRIALS, seed_offset=20000
        )

        tau0_std = np.std(tau0_samples)
        tau1_std = np.std(tau1_samples)

        min_std = 0.01
        assert tau0_std >= min_std, f"τ0* distribution degenerate: std = {tau0_std:.4f}"
        assert tau1_std >= min_std, f"τ1* distribution degenerate: std = {tau1_std:.4f}"

        assert 0.0 <= np.mean(tau0_samples) <= 1.0
        assert 0.0 <= np.mean(tau1_samples) <= 1.0
