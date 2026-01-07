import os
import sys
import numpy as np
import torch
import logging
from dataclasses import asdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from onn.core import ESConfig
from onn.ops import LOGOSSolver, LOGOSConfig, ORTSFOperator, DeepDeltaPredictor
from onn.ops.branching import (
    StagnationDetector,
    StagnationConfig,
    BranchFactory,
    BranchSelector,
    BranchResult,
    SurvivalConfig,
    ParameterAdaptationManager,
    AdaptationConfig,
    DynamicBranchManager,
    DynamicBranchManagerConfig,
)
from onn.topo.filtration import (
    compute_topo_summary,
    gates_to_numpy,
    edge_indices_to_numpy,
)
from onn.es.ask_tell import CMAESTrainer
from onn.es.fitness import evaluate_candidate, generate_synthetic_episode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ONN-ES-Train")


def train():
    logger.info("Generating synthetic episodes...")
    episodes = [generate_synthetic_episode(num_steps=10) for _ in range(3)]

    es_config = ESConfig(
        population_size=8,
        sigma=0.3,
        parameter_bounds={
            "beta_obs": [0.1, 5.0],
            "beta_cons": [0.1, 5.0],
            "step_size": [0.005, 0.05],
            "gate_threshold": [0.2, 0.8],
        },
    )

    trainer = CMAESTrainer(es_config)

    detector = StagnationDetector(
        StagnationConfig(
            window_size=5,
            eps_slope=0.02,
            eps_gate_std=0.02,
            eps_u_std=0.05,
        )
    )

    branch_factory = BranchFactory(base_seed=42)
    selector = BranchSelector(
        SurvivalConfig(
            min_active_edge_ratio=0.2,
            min_mean_gate=0.1,
        )
    )
    param_adapter = ParameterAdaptationManager(AdaptationConfig(min_history_len=5))

    max_generations = 20
    best_overall_fitness = -float("inf")
    best_params = None

    residual_history = []
    gate_history = []
    uncertainty_history = []
    tau0_star_history = []
    tau1_star_history = []

    current_gates = np.array([0.8, 0.7, 0.6, 0.5])
    current_meta_params = {"beta_obs": 1.0, "beta_cons": 1.0, "step_size": 0.01}

    for gen in range(max_generations):
        candidates = trainer.ask()

        fitness_scores = []
        gen_residuals = []
        gen_gates = []
        gen_uncertainties = []

        for cand in candidates:
            fitness = evaluate_candidate(cand, episodes)
            fitness_scores.append(fitness)

            gen_residuals.append(1.0 / (abs(fitness) + 0.1))
            gen_gates.append(np.mean(current_gates))
            gen_uncertainties.append(1.0)

        trainer.tell(candidates, fitness_scores)

        avg_residual = np.mean(gen_residuals)
        avg_gate = np.mean(gen_gates)
        avg_uncertainty = np.mean(gen_uncertainties)

        residual_history.append(avg_residual)
        gate_history.append(avg_gate)
        uncertainty_history.append(avg_uncertainty)

        tau0_star_history.append(0.5 + np.random.randn() * 0.05)
        tau1_star_history.append(0.3 + np.random.randn() * 0.05)

        current_best = max(fitness_scores)

        if current_best > best_overall_fitness:
            best_overall_fitness = current_best
            best_params = candidates[
                fitness_scores.index(current_best)
            ].hyperparams.copy()

        signal = detector.detect(
            residual_history,
            gate_history,
            uncertainty_history,
            tau0_star_history,
            tau1_star_history,
        )

        if signal.stable:
            logger.info(
                f"Gen {gen}: Stagnation detected ({signal.reasons}). Managing branches with DynamicBranchManager..."
            )

            # 1. Create initial branches for the event
            tau_star = tau0_star_history[-1] if tau0_star_history else 0.5
            initial_branches = branch_factory.make_branches(
                current_gates, current_meta_params, tau_star, num_branches=3
            )

            # 2. Use the manager for this event
            # In a full integration, the manager would persist across generations.
            # Here, we use it to manage this single stagnation event.
            manager_config = DynamicBranchManagerConfig(
                stagnation_config=detector.config
            )
            branch_manager = DynamicBranchManager(
                initial_branches, config=manager_config
            )

            # 3. Evaluate the branches
            branch_results = []
            for branch_id, b in branch_manager.branches.items():
                b_fitness = evaluate_candidate(candidates[0], episodes)
                mean_gate = float(np.mean(b.gates))
                active_ratio = float(np.mean(b.gates > 0.3))
                topo = compute_topo_summary(
                    num_nodes=4,
                    edge_indices=np.array([[0, 1], [1, 2], [2, 3]]),
                    gates=b.gates[:3]
                    if len(b.gates) >= 3
                    else np.array([0.5, 0.5, 0.5]),
                )
                result = BranchResult(
                    branch=b,
                    final_gates=b.gates,
                    final_uncertainty=1.0,
                    active_edge_ratio=active_ratio,
                    mean_gate=mean_gate,
                    fitness=b_fitness,
                    converged=True,
                    has_nan=False,
                    tau0_star=topo.tau0_star,
                    tau1_star=topo.tau1_star,
                    beta0_final=int(topo.beta0_profile[-1]),
                    beta1_final=int(topo.beta1_profile[-1]),
                )
                branch_results.append(result)
                logger.info(
                    f"  Branch {b.config.branch_type.value} (id={branch_id}): "
                    f"fitness={b_fitness:.4f}, τ0*={topo.tau0_star:.3f}, τ1*={topo.tau1_star:.3f}"
                )

            # 4. Select winner and adapt parameters
            winner, _ = selector.select(branch_results)
            if winner is not None:
                logger.info(
                    f"  Winner: {winner.branch.config.branch_type.value} (fitness={winner.fitness:.4f})"
                )
                current_gates = winner.final_gates.copy()

                # Adapt meta-parameters based on history
                adapted_meta_params, adapt_reason = param_adapter.adapt(
                    winner.branch.meta_params,
                    residual_history,
                    es_meta_params=winner.branch.meta_params,
                )
                if adapted_meta_params != winner.branch.meta_params:
                    logger.info(
                        f"  Adapting meta-params: {adapted_meta_params} (reason={adapt_reason})"
                    )
                current_meta_params = adapted_meta_params
            else:
                logger.warning("  No surviving branches! Keeping current state.")

        logger.info(
            f"Gen {gen}: Best={current_best:.4f}, Overall={best_overall_fitness:.4f}"
        )

    logger.info(f"Training complete. Best fitness: {best_overall_fitness:.4f}")
    if best_params:
        logger.info(f"Best params: {best_params}")


if __name__ == "__main__":
    train()
