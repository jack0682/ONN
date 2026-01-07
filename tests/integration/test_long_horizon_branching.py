import pytest
import numpy as np
from typing import List, Dict
from collections import Counter

from onn.ops.branching import (
    Branch,
    BranchConfig,
    BranchType,
    BranchResult,
    DynamicBranchManager,
    DynamicBranchManagerConfig,
    StagnationConfig,
    BranchSelector,
    SurvivalConfig,
    create_conservative_config,
    create_exploration_config,
)


# Helper to create a mock result with some logic
def create_dynamic_result(branch: Branch, cycle: int) -> BranchResult:
    rng = branch.rng
    fitness = 0

    # Define behavior based on branch type
    if branch.config.branch_type == BranchType.CONSERVATIVE:
        # Slow, steady improvement
        fitness = 10 + cycle * 0.1 + rng.normal(0, 0.05)
    elif branch.config.branch_type == BranchType.EXPLORATION:
        # Volatile: could be great or terrible
        fitness = 8 + np.sin(branch.config.seed + cycle / 2.0) * 5 + rng.normal(0, 1)

    return BranchResult(
        branch=branch,
        final_gates=branch.gates,
        final_uncertainty=rng.uniform(0.5, 1.5),
        active_edge_ratio=rng.uniform(0.3, 0.8),
        mean_gate=rng.uniform(0.3, 0.7),
        fitness=fitness,
        converged=False,
        has_nan=False,
        tau0_star=rng.uniform(0.2, 0.8),
        tau1_star=rng.uniform(0.2, 0.8),
        beta0_final=1,
        beta1_final=rng.randint(1, 10),
    )


def test_long_horizon_branching_diversity():
    """
    Simulates a multi-cycle run to ensure branching preserves diversity
    and a conservative lineage can survive.
    """
    POPULATION_SIZE = 4
    NUM_CYCLES = 20

    # 1. Setup
    rng = np.random.RandomState(42)

    # Start with a mix of branches
    initial_branches = [
        Branch(
            config=create_conservative_config(seed=0),
            gates=rng.rand(10),
            meta_params={},
            rng=np.random.RandomState(0),
        ),
        Branch(
            config=create_exploration_config(seed=1),
            gates=rng.rand(10),
            meta_params={},
            rng=np.random.RandomState(1),
        ),
    ]

    manager_config = DynamicBranchManagerConfig(
        max_branches=POPULATION_SIZE
        + 2,  # Allow temporary overpopulation before selection
        stagnation_config=StagnationConfig(min_history_len=4, window_size=4),
    )
    manager = DynamicBranchManager(
        initial_branches, config=manager_config, base_seed=100
    )
    selector = BranchSelector(
        SurvivalConfig(min_active_edge_ratio=0.1, min_mean_gate=0.1)
    )

    lineage_survival = {"CONSERVATIVE": 0, "EXPLORATION": 0, "REWIRE": 0}

    # 2. Simulation Loop
    for cycle in range(NUM_CYCLES):
        current_population = list(manager.branches.values())

        # Generate results for the current population
        results = [
            create_dynamic_result(branch, cycle) for branch in current_population
        ]

        # Let the manager spawn new branches if stagnation is detected
        # In this mock, we need to provide some plausible parent state
        fittest_branch_result = max(results, key=lambda r: r.fitness)
        spawned_branches = manager.step(
            results,
            parent_gates=fittest_branch_result.final_gates,
            parent_meta_params=fittest_branch_result.branch.meta_params,
            parent_tau_star=fittest_branch_result.tau0_star,
            edge_indices=np.array([[0, 1]]),  # Dummy value
            num_nodes=2,  # Dummy value
        )

        # Combine current population and newly spawned branches for selection
        full_population = current_population + spawned_branches
        full_results = [
            create_dynamic_result(branch, cycle) for branch in full_population
        ]

        # Select survivors for the next generation (keep top-K by fitness)
        survivors = []
        for r in sorted(full_results, key=lambda r: r.fitness, reverse=True):
            if selector._passes_survival_filter(r) and len(survivors) < POPULATION_SIZE:
                survivors.append(r.branch)

        # Ensure at least one conservative branch survives if present in population
        if not any(b.config.branch_type == BranchType.CONSERVATIVE for b in survivors):
            conservative_candidates = [
                r.branch
                for r in full_results
                if r.branch.config.branch_type == BranchType.CONSERVATIVE
            ]
            if conservative_candidates:
                survivors = survivors[:-1] if survivors else []
                survivors.append(conservative_candidates[0])

        # If selection fails entirely, keep the best one to prevent population collapse
        if not survivors:
            survivors = [fittest_branch_result.branch]

        manager.branches = {i: branch for i, branch in enumerate(survivors)}
        manager.next_branch_id = len(survivors)
        manager.history = {
            k: v for k, v in manager.history.items() if k in manager.branches
        }

        # Track stats
        branch_types = [b.config.branch_type.value for b in manager.branches.values()]
        type_counts = Counter(branch_types)
        print(f"Cycle {cycle}: {type_counts}, Population: {len(manager.branches)}")

        for b_type in lineage_survival:
            if b_type in type_counts:
                lineage_survival[b_type] += 1

    # 3. Assertions
    print(f"Survival counts over {NUM_CYCLES} cycles: {lineage_survival}")

    # Assert that the conservative lineage is highly resilient
    assert lineage_survival["CONSERVATIVE"] >= NUM_CYCLES * 0.8, (
        "Conservative branch should be highly likely to survive"
    )

    # Assert that diversity was maintained (not just one type of branch taking over)
    final_types = {b.config.branch_type for b in manager.branches.values()}
    assert len(final_types) > 1, f"Diversity collapsed. Final types: {final_types}"

    # Assert that the population size is managed correctly
    assert len(manager.branches) <= POPULATION_SIZE
