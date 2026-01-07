#!/usr/bin/env python3
"""ONN-ES Validation Experiments.

This script validates whether ONN-ES with geometric relations is truly
a groundbreaking architecture.

Experiments:
1. Convergence Test: Does ES find better solutions over generations?
2. Comparison Test: How does it compare to baselines?
3. Robustness Test: How robust is it to perturbations?
4. Ablation Study: What is the contribution of each component?
5. Scaling Test: How does it scale with problem size?

Usage:
    # Smoke test (quick validation)
    python experiments/validate_onn_es.py --mode smoke

    # Full experiment (comprehensive)
    python experiments/validate_onn_es.py --mode full

    # Specific experiment
    python experiments/validate_onn_es.py --experiment convergence

Author: Claude (ONN-ES Validation)
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from onn.es import (
    CMAESTrainer,
    create_default_trainer,
    create_trainer_with_w_lin,
    Candidate,
    GeometricFitnessConfig,
    GeometricEvalMetrics,
    GeometricRelationEncoderES,
    evaluate_geometric_episode,
    compute_geometric_fitness,
    evaluate_geometric_candidate,
    generate_geometric_episode,
    create_geometric_encoder_es,
    create_geometric_fitness_config,
)
from onn.core.solver import create_default_pc_solver
from onn.core.relation_geometry import (
    create_geometric_encoder,
    create_predictive_model,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for validation experiments."""
    # Problem size
    num_nodes: int = 5
    num_edges: int = 20  # Approximate (depends on connectivity)
    embedding_dim: int = 32

    # Episode parameters
    num_episodes: int = 5
    episode_length: int = 20
    event_prob: float = 0.1
    noise_std: float = 0.1

    # ES parameters
    population_size: int = 16
    num_generations: int = 50
    w_lin_input_dim: int = 64
    w_lin_output_dim: int = 32
    w_lin_rank: int = 8  # Low-rank factorization (0=full, >0=low-rank)

    # Experiment settings
    num_seeds: int = 3  # Number of random seeds for averaging
    save_results: bool = True
    output_dir: str = "experiments/results"


@dataclass
class SmokeTestConfig(ExperimentConfig):
    """Quick smoke test configuration."""
    num_nodes: int = 3
    num_episodes: int = 2
    episode_length: int = 5
    population_size: int = 4
    num_generations: int = 5
    num_seeds: int = 1


@dataclass
class FullExperimentConfig(ExperimentConfig):
    """Full experiment configuration (user will run)."""
    num_nodes: int = 10
    num_episodes: int = 10
    episode_length: int = 50
    population_size: int = 32
    num_generations: int = 100
    num_seeds: int = 5


# ==============================================================================
# EXPERIMENT: CONVERGENCE TEST
# ==============================================================================

def run_convergence_test(
    config: ExperimentConfig,
    seed: int = 42,
) -> Dict:
    """Test if ES converges to better solutions.

    Measures:
    - Fitness over generations
    - Best fitness achieved
    - Convergence speed (generations to 90% of final fitness)
    """
    logger.info(f"Running Convergence Test (seed={seed})")

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate episodes
    episodes = [
        generate_geometric_episode(
            num_steps=config.episode_length,
            num_nodes=config.num_nodes,
            embedding_dim=config.embedding_dim,
            event_prob=config.event_prob,
            noise_std=config.noise_std,
            motion_type="circular",
            seed=seed + i,
        )
        for i in range(config.num_episodes)
    ]

    # Create trainer and encoder
    trainer = create_trainer_with_w_lin(
        w_lin_shape=(config.w_lin_output_dim, config.w_lin_input_dim),
        population_size=config.population_size,
        w_lin_rank=config.w_lin_rank,
    )
    encoder = create_geometric_encoder_es(
        input_dim=config.w_lin_input_dim,
        output_dim=config.w_lin_output_dim,
        use_geometric=True,
    )
    fitness_config = create_geometric_fitness_config(geometric_weight=1.0)

    # Run ES optimization
    fitness_history = []
    best_fitness_history = []
    best_fitness = float('-inf')

    start_time = time.time()

    for gen in range(config.num_generations):
        candidates = trainer.ask()

        fitnesses = []
        for candidate in candidates:
            fitness = evaluate_geometric_candidate(
                candidate, episodes, encoder, fitness_config
            )
            fitnesses.append(fitness)

        trainer.tell(candidates, fitnesses)

        gen_best = max(fitnesses)
        gen_mean = np.mean(fitnesses)

        if gen_best > best_fitness:
            best_fitness = gen_best

        fitness_history.append({
            'generation': gen,
            'best': gen_best,
            'mean': gen_mean,
            'std': np.std(fitnesses),
        })
        best_fitness_history.append(best_fitness)

        if gen % 10 == 0 or gen == config.num_generations - 1:
            logger.info(f"  Gen {gen}: best={gen_best:.4f}, mean={gen_mean:.4f}")

    elapsed_time = time.time() - start_time

    # Compute convergence metrics
    final_fitness = best_fitness_history[-1]
    threshold_90 = 0.9 * final_fitness if final_fitness > 0 else final_fitness * 1.1

    convergence_gen = config.num_generations
    for i, f in enumerate(best_fitness_history):
        if f >= threshold_90:
            convergence_gen = i
            break

    # Improvement ratio
    initial_fitness = fitness_history[0]['best']
    improvement = (final_fitness - initial_fitness) / (abs(initial_fitness) + 1e-6)

    return {
        'test': 'convergence',
        'seed': seed,
        'config': asdict(config),
        'fitness_history': fitness_history,
        'best_fitness_history': best_fitness_history,
        'final_best_fitness': final_fitness,
        'initial_fitness': initial_fitness,
        'improvement_ratio': improvement,
        'convergence_generation': convergence_gen,
        'elapsed_time_seconds': elapsed_time,
    }


# ==============================================================================
# EXPERIMENT: COMPARISON TEST
# ==============================================================================

def run_comparison_test(
    config: ExperimentConfig,
    seed: int = 42,
) -> Dict:
    """Compare different configurations.

    Baselines:
    1. Random projection only (no learning)
    2. ES without geometric encoding
    3. ES with geometric encoding (full system)
    """
    logger.info(f"Running Comparison Test (seed={seed})")

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

    results = {}

    # === Baseline 1: Random Projection Only ===
    logger.info("  Testing: Random Projection (no learning)")
    encoder_rp = create_geometric_encoder_es(
        input_dim=config.w_lin_input_dim,
        output_dim=config.w_lin_output_dim,
        use_geometric=False,
        alpha=1.0, beta=0.0, gamma=0.0,  # Only random projection
    )
    fitness_config = create_geometric_fitness_config()

    # Evaluate with zero W_lin
    solver = create_default_pc_solver()
    rp_metrics = []
    for ep in episodes:
        metrics = evaluate_geometric_episode(solver, ep, fitness_config)
        rp_metrics.append(metrics)

    rp_fitness = np.mean([compute_geometric_fitness(m, fitness_config) for m in rp_metrics])
    results['random_projection'] = {
        'fitness': rp_fitness,
        'violation_mean': np.mean([m.violation_mean for m in rp_metrics]),
        'prediction_accuracy': np.mean([m.prediction_accuracy for m in rp_metrics]),
    }
    logger.info(f"    Fitness: {rp_fitness:.4f}")

    # === Baseline 2: ES without Geometric ===
    logger.info("  Testing: ES (no geometric)")
    encoder_no_geo = create_geometric_encoder_es(
        input_dim=config.w_lin_input_dim,
        output_dim=config.w_lin_output_dim,
        use_geometric=False,
    )
    trainer_no_geo = create_trainer_with_w_lin(
        w_lin_shape=(config.w_lin_output_dim, config.w_lin_input_dim),
        population_size=config.population_size,
        w_lin_rank=config.w_lin_rank,
    )

    # Quick training
    half_gens = config.num_generations // 2
    for gen in range(half_gens):
        candidates = trainer_no_geo.ask()
        fitnesses = [
            evaluate_geometric_candidate(c, episodes, encoder_no_geo, fitness_config)
            for c in candidates
        ]
        trainer_no_geo.tell(candidates, fitnesses)

    # Get best candidate
    candidates = trainer_no_geo.ask()
    fitnesses = [
        evaluate_geometric_candidate(c, episodes, encoder_no_geo, fitness_config)
        for c in candidates
    ]
    best_idx = np.argmax(fitnesses)

    results['es_no_geometric'] = {
        'fitness': fitnesses[best_idx],
        'generations': half_gens,
    }
    logger.info(f"    Fitness: {fitnesses[best_idx]:.4f} (after {half_gens} gens)")

    # === Full System: ES + Geometric ===
    logger.info("  Testing: ES + Geometric (full system)")
    encoder_full = create_geometric_encoder_es(
        input_dim=config.w_lin_input_dim,
        output_dim=config.w_lin_output_dim,
        use_geometric=True,
    )
    trainer_full = create_trainer_with_w_lin(
        w_lin_shape=(config.w_lin_output_dim, config.w_lin_input_dim),
        population_size=config.population_size,
        w_lin_rank=config.w_lin_rank,
    )

    fitness_config_geo = create_geometric_fitness_config(geometric_weight=1.0)

    for gen in range(half_gens):
        candidates = trainer_full.ask()
        fitnesses = [
            evaluate_geometric_candidate(c, episodes, encoder_full, fitness_config_geo)
            for c in candidates
        ]
        trainer_full.tell(candidates, fitnesses)

    candidates = trainer_full.ask()
    fitnesses = [
        evaluate_geometric_candidate(c, episodes, encoder_full, fitness_config_geo)
        for c in candidates
    ]
    best_idx = np.argmax(fitnesses)

    results['es_with_geometric'] = {
        'fitness': fitnesses[best_idx],
        'generations': half_gens,
    }
    logger.info(f"    Fitness: {fitnesses[best_idx]:.4f} (after {half_gens} gens)")

    # Compute improvement
    baseline = results['random_projection']['fitness']
    es_only_improvement = (results['es_no_geometric']['fitness'] - baseline) / (abs(baseline) + 1e-6)
    full_improvement = (results['es_with_geometric']['fitness'] - baseline) / (abs(baseline) + 1e-6)

    return {
        'test': 'comparison',
        'seed': seed,
        'results': results,
        'es_only_improvement_percent': es_only_improvement * 100,
        'full_system_improvement_percent': full_improvement * 100,
        'geometric_contribution_percent': (full_improvement - es_only_improvement) * 100,
    }


# ==============================================================================
# EXPERIMENT: ROBUSTNESS TEST
# ==============================================================================

def run_robustness_test(
    config: ExperimentConfig,
    seed: int = 42,
) -> Dict:
    """Test robustness to perturbations and noise."""
    logger.info(f"Running Robustness Test (seed={seed})")

    np.random.seed(seed)
    torch.manual_seed(seed)

    results = {}

    # Test different noise levels
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]

    encoder = create_geometric_encoder_es(
        input_dim=config.w_lin_input_dim,
        output_dim=config.w_lin_output_dim,
        use_geometric=True,
    )
    fitness_config = create_geometric_fitness_config()
    solver = create_default_pc_solver()

    noise_results = []
    for noise in noise_levels:
        episodes = [
            generate_geometric_episode(
                num_steps=config.episode_length,
                num_nodes=config.num_nodes,
                embedding_dim=config.embedding_dim,
                noise_std=noise,
                seed=seed,
            )
            for _ in range(config.num_episodes)
        ]

        metrics_list = [
            evaluate_geometric_episode(solver, ep, fitness_config)
            for ep in episodes
        ]

        avg_fitness = np.mean([
            compute_geometric_fitness(m, fitness_config) for m in metrics_list
        ])
        avg_violation = np.mean([m.violation_mean for m in metrics_list])
        avg_prediction = np.mean([m.prediction_accuracy for m in metrics_list])

        noise_results.append({
            'noise_std': noise,
            'fitness': avg_fitness,
            'violation': avg_violation,
            'prediction_accuracy': avg_prediction,
        })
        logger.info(f"  Noise={noise:.2f}: fitness={avg_fitness:.4f}, pred_acc={avg_prediction:.4f}")

    results['noise_sensitivity'] = noise_results

    # Test event recovery
    event_probs = [0.0, 0.1, 0.2, 0.3]
    event_results = []

    for event_prob in event_probs:
        episodes = [
            generate_geometric_episode(
                num_steps=config.episode_length,
                num_nodes=config.num_nodes,
                embedding_dim=config.embedding_dim,
                event_prob=event_prob,
                seed=seed,
            )
            for _ in range(config.num_episodes)
        ]

        metrics_list = [
            evaluate_geometric_episode(solver, ep, fitness_config)
            for ep in episodes
        ]

        avg_latency = np.mean([m.latency_mean for m in metrics_list])
        avg_fitness = np.mean([
            compute_geometric_fitness(m, fitness_config) for m in metrics_list
        ])

        event_results.append({
            'event_prob': event_prob,
            'fitness': avg_fitness,
            'recovery_latency': avg_latency,
        })
        logger.info(f"  EventProb={event_prob:.1f}: fitness={avg_fitness:.4f}, latency={avg_latency:.2f}")

    results['event_recovery'] = event_results

    return {
        'test': 'robustness',
        'seed': seed,
        'results': results,
    }


# ==============================================================================
# EXPERIMENT: ABLATION STUDY
# ==============================================================================

def run_ablation_study(
    config: ExperimentConfig,
    seed: int = 42,
) -> Dict:
    """Ablation study: contribution of each component."""
    logger.info(f"Running Ablation Study (seed={seed})")

    np.random.seed(seed)
    torch.manual_seed(seed)

    episodes = [
        generate_geometric_episode(
            num_steps=config.episode_length,
            num_nodes=config.num_nodes,
            embedding_dim=config.embedding_dim,
            seed=seed + i,
        )
        for i in range(config.num_episodes)
    ]

    solver = create_default_pc_solver()
    results = {}

    # Configuration variations
    ablation_configs = [
        {
            'name': 'base_only',
            'description': 'Standard ONN-ES metrics only',
            'config': GeometricFitnessConfig(
                alpha_prediction=0, alpha_bidirectional=0,
                alpha_temporal=0, alpha_cycle=0, alpha_contrastive=0,
            ),
        },
        {
            'name': 'with_prediction',
            'description': '+ Prediction accuracy',
            'config': GeometricFitnessConfig(
                alpha_prediction=0.5, alpha_bidirectional=0,
                alpha_temporal=0, alpha_cycle=0, alpha_contrastive=0,
            ),
        },
        {
            'name': 'with_bidirectional',
            'description': '+ Bidirectional consistency',
            'config': GeometricFitnessConfig(
                alpha_prediction=0.5, alpha_bidirectional=0.3,
                alpha_temporal=0, alpha_cycle=0, alpha_contrastive=0,
            ),
        },
        {
            'name': 'with_cycle',
            'description': '+ Cycle consistency',
            'config': GeometricFitnessConfig(
                alpha_prediction=0.5, alpha_bidirectional=0.3,
                alpha_temporal=0, alpha_cycle=0.5, alpha_contrastive=0,
            ),
        },
        {
            'name': 'full_system',
            'description': 'All components',
            'config': create_geometric_fitness_config(geometric_weight=1.0),
        },
    ]

    for ablation in ablation_configs:
        fitness_config = ablation['config']

        metrics_list = [
            evaluate_geometric_episode(solver, ep, fitness_config)
            for ep in episodes
        ]

        avg_fitness = np.mean([
            compute_geometric_fitness(m, fitness_config) for m in metrics_list
        ])

        results[ablation['name']] = {
            'description': ablation['description'],
            'fitness': avg_fitness,
            'violation_mean': np.mean([m.violation_mean for m in metrics_list]),
            'prediction_accuracy': np.mean([m.prediction_accuracy for m in metrics_list]),
            'cycle_consistency': np.mean([m.cycle_consistency for m in metrics_list]),
        }
        logger.info(f"  {ablation['name']}: fitness={avg_fitness:.4f}")

    return {
        'test': 'ablation',
        'seed': seed,
        'results': results,
    }


# ==============================================================================
# EXPERIMENT: SCALING TEST
# ==============================================================================

def run_scaling_test(
    config: ExperimentConfig,
    seed: int = 42,
) -> Dict:
    """Test how the system scales with problem size."""
    logger.info(f"Running Scaling Test (seed={seed})")

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Node count scaling
    node_counts = [3, 5, 8, 10]
    if config.num_nodes >= 15:
        node_counts.append(15)
    if config.num_nodes >= 20:
        node_counts.append(20)

    results = {'node_scaling': [], 'episode_scaling': []}

    solver = create_default_pc_solver()
    fitness_config = create_geometric_fitness_config()

    for num_nodes in node_counts:
        start_time = time.time()

        episodes = [
            generate_geometric_episode(
                num_steps=config.episode_length,
                num_nodes=num_nodes,
                embedding_dim=config.embedding_dim,
                seed=seed,
            )
            for _ in range(min(3, config.num_episodes))
        ]

        metrics_list = [
            evaluate_geometric_episode(solver, ep, fitness_config)
            for ep in episodes
        ]

        elapsed = time.time() - start_time
        avg_fitness = np.mean([compute_geometric_fitness(m, fitness_config) for m in metrics_list])

        results['node_scaling'].append({
            'num_nodes': num_nodes,
            'fitness': avg_fitness,
            'time_seconds': elapsed,
            'time_per_step': elapsed / (len(episodes) * config.episode_length),
        })
        logger.info(f"  Nodes={num_nodes}: fitness={avg_fitness:.4f}, time={elapsed:.2f}s")

    # Episode length scaling
    episode_lengths = [5, 10, 20, 50]
    if config.episode_length >= 100:
        episode_lengths.append(100)

    for ep_len in episode_lengths:
        start_time = time.time()

        episodes = [
            generate_geometric_episode(
                num_steps=ep_len,
                num_nodes=config.num_nodes,
                embedding_dim=config.embedding_dim,
                seed=seed,
            )
            for _ in range(min(3, config.num_episodes))
        ]

        metrics_list = [
            evaluate_geometric_episode(solver, ep, fitness_config)
            for ep in episodes
        ]

        elapsed = time.time() - start_time
        avg_fitness = np.mean([compute_geometric_fitness(m, fitness_config) for m in metrics_list])

        results['episode_scaling'].append({
            'episode_length': ep_len,
            'fitness': avg_fitness,
            'time_seconds': elapsed,
        })
        logger.info(f"  EpLen={ep_len}: fitness={avg_fitness:.4f}, time={elapsed:.2f}s")

    return {
        'test': 'scaling',
        'seed': seed,
        'results': results,
    }


# ==============================================================================
# MAIN RUNNER
# ==============================================================================

def run_all_experiments(
    config: ExperimentConfig,
    experiments: Optional[List[str]] = None,
) -> Dict:
    """Run all or selected experiments."""

    all_experiments = {
        'convergence': run_convergence_test,
        'comparison': run_comparison_test,
        'robustness': run_robustness_test,
        'ablation': run_ablation_study,
        'scaling': run_scaling_test,
    }

    if experiments is None:
        experiments = list(all_experiments.keys())

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': asdict(config),
        'experiments': {},
    }

    for exp_name in experiments:
        if exp_name not in all_experiments:
            logger.warning(f"Unknown experiment: {exp_name}")
            continue

        exp_fn = all_experiments[exp_name]

        # Run with multiple seeds
        seed_results = []
        for seed in range(config.num_seeds):
            try:
                result = exp_fn(config, seed=42 + seed * 100)
                seed_results.append(result)
            except Exception as e:
                logger.error(f"Error in {exp_name} (seed={seed}): {e}")
                import traceback
                traceback.print_exc()

        results['experiments'][exp_name] = seed_results

    return results


def print_summary(results: Dict):
    """Print a summary of the results."""
    print("\n" + "=" * 70)
    print("                      ONN-ES VALIDATION SUMMARY")
    print("=" * 70)

    for exp_name, exp_results in results['experiments'].items():
        print(f"\n### {exp_name.upper()} ###")

        if exp_name == 'convergence':
            for r in exp_results:
                print(f"  Seed {r['seed']}:")
                print(f"    Initial fitness: {r['initial_fitness']:.4f}")
                print(f"    Final fitness:   {r['final_best_fitness']:.4f}")
                print(f"    Improvement:     {r['improvement_ratio']*100:.1f}%")
                print(f"    Converged at:    Gen {r['convergence_generation']}")

        elif exp_name == 'comparison':
            for r in exp_results:
                print(f"  ES-only improvement:    {r['es_only_improvement_percent']:.1f}%")
                print(f"  Full system improvement: {r['full_system_improvement_percent']:.1f}%")
                print(f"  Geometric contribution: {r['geometric_contribution_percent']:.1f}%")

        elif exp_name == 'ablation':
            for r in exp_results:
                print("  Component contributions:")
                for name, data in r['results'].items():
                    print(f"    {name}: fitness={data['fitness']:.4f}")

    print("\n" + "=" * 70)


def save_results(results: Dict, output_dir: str):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"onn_es_validation_{timestamp}.json")

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="ONN-ES Validation Experiments")
    parser.add_argument(
        '--mode', type=str, default='smoke',
        choices=['smoke', 'full'],
        help='Experiment mode: smoke (quick) or full (comprehensive)'
    )
    parser.add_argument(
        '--experiment', type=str, default=None,
        help='Run specific experiment: convergence, comparison, robustness, ablation, scaling'
    )
    parser.add_argument(
        '--output-dir', type=str, default='experiments/results',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Select configuration
    if args.mode == 'smoke':
        logger.info("Running SMOKE TEST (quick validation)")
        config = SmokeTestConfig()
    else:
        logger.info("Running FULL EXPERIMENT")
        config = FullExperimentConfig()

    config.output_dir = args.output_dir

    # Select experiments
    experiments = None
    if args.experiment:
        experiments = [args.experiment]

    # Run experiments
    logger.info(f"Configuration: {asdict(config)}")
    results = run_all_experiments(config, experiments)

    # Print summary
    print_summary(results)

    # Save results
    if config.save_results:
        filepath = save_results(results, config.output_dir)
        print(f"\nResults saved to: {filepath}")

    return results


if __name__ == "__main__":
    main()
