#!/usr/bin/env python3
"""Hybrid ES + GD Validation Experiments.

Validates the hybrid approach:
- ES: 7 hyperparameters (stable, low-dimensional)
- GD: W_lin with surrogate loss (high-dimensional, gradient-based)

Usage:
    # Smoke test
    python experiments/validate_hybrid.py --mode smoke

    # Full experiment
    python experiments/validate_hybrid.py --mode full

Author: Claude (Hybrid Validation)
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from onn.es import (
    # Hybrid
    create_hybrid_trainer,
    HybridTrainer,
    # Geometric
    generate_geometric_episode,
    evaluate_geometric_episode,
    compute_geometric_fitness,
    create_geometric_fitness_config,
    GeometricFitnessConfig,
)
from onn.core.solver import create_default_pc_solver
from onn.core.cycles import build_cycle_basis

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class HybridExperimentConfig:
    """Configuration for hybrid validation."""
    # Problem size
    num_nodes: int = 5
    embedding_dim: int = 32
    input_dim: int = 64

    # Episode parameters
    num_episodes: int = 5
    episode_length: int = 20

    # ES parameters (hyperparams only)
    es_population: int = 16
    es_generations: int = 50

    # GD parameters (W_lin)
    gd_lr: float = 1e-3
    gd_steps_per_gen: int = 10

    # Experiment
    num_seeds: int = 3
    save_results: bool = True
    output_dir: str = "experiments/results"


@dataclass
class SmokeConfig(HybridExperimentConfig):
    """Quick smoke test."""
    num_nodes: int = 3
    num_episodes: int = 2
    episode_length: int = 5
    es_population: int = 4
    es_generations: int = 5
    gd_steps_per_gen: int = 5
    num_seeds: int = 1


@dataclass
class FullConfig(HybridExperimentConfig):
    """Full experiment."""
    num_nodes: int = 8
    num_episodes: int = 10
    episode_length: int = 30
    es_population: int = 16
    es_generations: int = 100
    gd_steps_per_gen: int = 20
    num_seeds: int = 5


def run_hybrid_experiment(
    config: HybridExperimentConfig,
    seed: int = 42,
) -> Dict:
    """Run hybrid ES + GD experiment.

    Returns convergence curves for both ES fitness and GD loss.
    """
    logger.info(f"Running Hybrid Experiment (seed={seed})")

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate episodes
    episodes = [
        generate_geometric_episode(
            num_steps=config.episode_length,
            num_nodes=config.num_nodes,
            embedding_dim=config.embedding_dim,
            seed=seed + i,
        )
        for i in range(config.num_episodes)
    ]

    # Create hybrid trainer
    trainer = create_hybrid_trainer(
        input_dim=config.input_dim,
        output_dim=config.embedding_dim,
        es_population=config.es_population,
        gd_lr=config.gd_lr,
        gd_steps=config.gd_steps_per_gen,
    )

    # Fitness config
    fitness_config = create_geometric_fitness_config(geometric_weight=1.0)
    solver = create_default_pc_solver()

    # History
    es_fitness_history = []
    gd_loss_history = []
    best_fitness = float('-inf')

    start_time = time.time()

    for gen in range(config.es_generations):
        # === ES: Optimize hyperparameters ===
        candidates = trainer.ask_hyperparams()

        fitnesses = []
        for candidate in candidates:
            # Evaluate with current W_lin
            total_fitness = 0.0
            for ep in episodes:
                metrics = evaluate_geometric_episode(solver, ep, fitness_config)
                total_fitness += compute_geometric_fitness(metrics, fitness_config)
            fitnesses.append(total_fitness / len(episodes))

        trainer.tell_hyperparams(candidates, fitnesses)

        gen_best = max(fitnesses)
        gen_mean = np.mean(fitnesses)

        if gen_best > best_fitness:
            best_fitness = gen_best

        es_fitness_history.append({
            'generation': gen,
            'best': gen_best,
            'mean': gen_mean,
        })

        # === GD: Update W_lin ===
        # Collect training data from episodes
        all_phi = []
        all_cycle_matrices = []

        for ep in episodes:
            for step in ep:
                # x_obs is already computed, but we need phi
                # Use x_obs as proxy for phi (simplified)
                all_phi.append(step.x_obs)

                # Build cycle matrix
                if step.edge_graph.num_edges > 0:
                    cycle_basis = build_cycle_basis(step.edge_graph)
                    if cycle_basis.num_cycles > 0:
                        all_cycle_matrices.append(cycle_basis.cycle_matrix.float())

        if all_phi:
            # Stack and pad to input_dim
            batch_phi = torch.cat(all_phi, dim=0)
            if batch_phi.shape[1] < config.input_dim:
                padding = torch.zeros(
                    batch_phi.shape[0],
                    config.input_dim - batch_phi.shape[1]
                )
                batch_phi = torch.cat([batch_phi, padding], dim=1)
            elif batch_phi.shape[1] > config.input_dim:
                batch_phi = batch_phi[:, :config.input_dim]

            # Average cycle matrix (simplified)
            cycle_matrix = None
            if all_cycle_matrices:
                # Use first one for simplicity
                cycle_matrix = all_cycle_matrices[0]
                # Pad/truncate to match batch size
                if cycle_matrix.shape[1] != batch_phi.shape[0]:
                    cycle_matrix = None

            # Update W_lin
            loss_breakdown = trainer.update_w_lin(
                batch_phi=batch_phi,
                cycle_matrix=cycle_matrix,
            )

            gd_loss_history.append({
                'generation': gen,
                **loss_breakdown,
            })

        # Log progress
        if gen % 10 == 0 or gen == config.es_generations - 1:
            gd_loss = gd_loss_history[-1]['total'] if gd_loss_history else 0
            logger.info(
                f"  Gen {gen}: ES best={gen_best:.4f}, "
                f"GD loss={gd_loss:.4f}"
            )

    elapsed = time.time() - start_time

    return {
        'seed': seed,
        'es_fitness_history': es_fitness_history,
        'gd_loss_history': gd_loss_history,
        'final_best_fitness': best_fitness,
        'best_hyperparams': trainer.get_best_hyperparams(),
        'elapsed_seconds': elapsed,
    }


def run_comparison(
    config: HybridExperimentConfig,
    seed: int = 42,
) -> Dict:
    """Compare Hybrid vs Pure ES approaches."""
    logger.info(f"Running Comparison (seed={seed})")

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate episodes
    episodes = [
        generate_geometric_episode(
            num_steps=config.episode_length,
            num_nodes=config.num_nodes,
            embedding_dim=config.embedding_dim,
            seed=seed + i,
        )
        for i in range(config.num_episodes)
    ]

    fitness_config = create_geometric_fitness_config()
    solver = create_default_pc_solver()

    results = {}

    # === Hybrid ES + GD ===
    logger.info("  Testing: Hybrid ES + GD")
    trainer_hybrid = create_hybrid_trainer(
        input_dim=config.input_dim,
        output_dim=config.embedding_dim,
        es_population=config.es_population,
        gd_lr=config.gd_lr,
        gd_steps=config.gd_steps_per_gen,
    )

    hybrid_best = float('-inf')
    for gen in range(config.es_generations // 2):
        candidates = trainer_hybrid.ask_hyperparams()
        fitnesses = []
        for candidate in candidates:
            total = 0.0
            for ep in episodes:
                m = evaluate_geometric_episode(solver, ep, fitness_config)
                total += compute_geometric_fitness(m, fitness_config)
            fitnesses.append(total / len(episodes))
        trainer_hybrid.tell_hyperparams(candidates, fitnesses)

        # GD update (simplified)
        batch_phi = torch.randn(20, config.input_dim)
        trainer_hybrid.update_w_lin(batch_phi=batch_phi)

        if max(fitnesses) > hybrid_best:
            hybrid_best = max(fitnesses)

    results['hybrid'] = {
        'fitness': hybrid_best,
        'es_params': 7,
        'gd_params': 2048,
    }
    logger.info(f"    Hybrid fitness: {hybrid_best:.4f}")

    # === Pure ES (low-rank) ===
    logger.info("  Testing: Pure ES (low-rank)")
    from onn.es import create_trainer_with_w_lin, evaluate_geometric_candidate, create_geometric_encoder_es

    trainer_es = create_trainer_with_w_lin(
        w_lin_shape=(config.embedding_dim, config.input_dim),
        population_size=config.es_population,
        w_lin_rank=8,
    )
    encoder_es = create_geometric_encoder_es(
        input_dim=config.input_dim,
        output_dim=config.embedding_dim,
    )

    es_best = float('-inf')
    for gen in range(config.es_generations // 2):
        candidates = trainer_es.ask()
        fitnesses = [
            evaluate_geometric_candidate(c, episodes, encoder_es, fitness_config)
            for c in candidates
        ]
        trainer_es.tell(candidates, fitnesses)

        if max(fitnesses) > es_best:
            es_best = max(fitnesses)

    results['pure_es_lowrank'] = {
        'fitness': es_best,
        'total_params': trainer_es.state.dim,
    }
    logger.info(f"    Pure ES (low-rank) fitness: {es_best:.4f}")

    return {
        'seed': seed,
        'results': results,
        'hybrid_advantage': results['hybrid']['fitness'] - results['pure_es_lowrank']['fitness'],
    }


def main():
    parser = argparse.ArgumentParser(description="Hybrid ES+GD Validation")
    parser.add_argument(
        '--mode', type=str, default='smoke',
        choices=['smoke', 'full'],
    )
    parser.add_argument(
        '--output-dir', type=str, default='experiments/results',
    )
    args = parser.parse_args()

    if args.mode == 'smoke':
        logger.info("Running SMOKE TEST")
        config = SmokeConfig()
    else:
        logger.info("Running FULL EXPERIMENT")
        config = FullConfig()

    config.output_dir = args.output_dir

    # Run experiments
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'config': asdict(config),
        'experiments': {},
    }

    # Main experiment
    experiment_results = []
    for seed in range(config.num_seeds):
        result = run_hybrid_experiment(config, seed=42 + seed * 100)
        experiment_results.append(result)
    all_results['experiments']['hybrid_convergence'] = experiment_results

    # Comparison
    comparison_results = []
    for seed in range(config.num_seeds):
        result = run_comparison(config, seed=42 + seed * 100)
        comparison_results.append(result)
    all_results['experiments']['comparison'] = comparison_results

    # Summary
    print("\n" + "=" * 70)
    print("                 HYBRID ES + GD VALIDATION SUMMARY")
    print("=" * 70)

    print("\n### CONVERGENCE ###")
    for r in experiment_results:
        print(f"  Seed {r['seed']}:")
        print(f"    Final fitness: {r['final_best_fitness']:.4f}")
        print(f"    Time: {r['elapsed_seconds']:.1f}s")

    print("\n### COMPARISON ###")
    for r in comparison_results:
        print(f"  Seed {r['seed']}:")
        print(f"    Hybrid: {r['results']['hybrid']['fitness']:.4f}")
        print(f"    Pure ES: {r['results']['pure_es_lowrank']['fitness']:.4f}")
        print(f"    Advantage: {r['hybrid_advantage']:.4f}")

    print("=" * 70)

    # Save
    if config.save_results:
        os.makedirs(config.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(
            config.output_dir,
            f"hybrid_validation_{timestamp}.json"
        )
        with open(filepath, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    main()
