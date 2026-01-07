#!/usr/bin/env python3
"""Delta vs Euler Training Benchmark.

Compares ES-based training with:
1. Standard Euler gradient step: x_new = x - η * grad
2. Delta ODE gradient step: x_new = x - β * η * k * (k^T grad)

Saves training results to .pt files for later generalization experiments.

Usage:
    # Train both variants
    python benchmarks/delta_vs_euler_training.py --mode both --generations 50 --output results/

    # Train only Euler
    python benchmarks/delta_vs_euler_training.py --mode euler --generations 50

    # Train only Delta with ES-optimized beta
    python benchmarks/delta_vs_euler_training.py --mode delta --optimize-beta --generations 50

    # Train Delta with fixed beta
    python benchmarks/delta_vs_euler_training.py --mode delta --delta-beta 1.5 --generations 50

Author: Claude
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, 'src')

from onn.core.solver import PCSolverConfig, ProjectionConsensusSolver, create_solver_from_dict
from onn.core.graph import EdgeGraph, EdgeKey
from onn.core.cycles import build_cycle_basis
from onn.es.ask_tell import ESConfig, CMAESTrainer, Candidate


# ==============================================================================
# DATA GENERATION
# ==============================================================================

def generate_synthetic_graph(
    num_nodes: int = 10,
    edge_prob: float = 0.3,
    embedding_dim: int = 32,
    seed: int = 42,
) -> Tuple[EdgeGraph, torch.Tensor]:
    """Generate a synthetic graph with random embeddings.

    Args:
        num_nodes: Number of nodes
        edge_prob: Probability of edge between any two nodes
        embedding_dim: Dimension of edge embeddings
        seed: Random seed

    Returns:
        Tuple of (EdgeGraph, x_obs tensor)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate random edges
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.random() < edge_prob:
                edges.append((i, j))

    if len(edges) < 3:
        # Ensure minimum connectivity
        edges = [(0, 1), (1, 2), (0, 2)]

    num_edges = len(edges)

    # Create EdgeGraph with node_ids and edge_keys
    node_ids = list(range(num_nodes))
    edge_keys = [EdgeKey(source_id=src, target_id=tgt) for src, tgt in edges]

    graph = EdgeGraph(
        node_ids=node_ids,
        edge_keys=edge_keys,
    )

    # Generate random observations
    x_obs = torch.randn(num_edges, embedding_dim) * 0.1

    return graph, x_obs


def generate_episode_batch(
    num_episodes: int = 5,
    steps_per_episode: int = 10,
    num_nodes: int = 10,
    embedding_dim: int = 32,
    seed: int = 42,
) -> List[List[Dict]]:
    """Generate a batch of episodes for training.

    Each episode is a sequence of steps with the SAME graph structure
    but different observations (simulating temporal evolution).

    Returns:
        List of episodes, each episode is a list of step dicts
    """
    episodes = []

    for ep_idx in range(num_episodes):
        episode = []
        ep_seed = seed + ep_idx * 1000

        # Same graph structure for entire episode (for x_prev compatibility)
        graph, _ = generate_synthetic_graph(
            num_nodes=num_nodes,
            edge_prob=0.3,
            embedding_dim=embedding_dim,
            seed=ep_seed,
        )
        num_edges = graph.num_edges

        for step_idx in range(steps_per_episode):
            # Different observations at each step (simulating sensor noise)
            torch.manual_seed(ep_seed + step_idx)
            x_obs = torch.randn(num_edges, embedding_dim) * 0.1

            episode.append({
                'graph': graph,
                'x_obs': x_obs,
                'x_prev': episode[-1]['x_final'] if episode else None,
            })

            # Placeholder for x_final (will be set during evaluation)
            episode[-1]['x_final'] = x_obs.clone()

        episodes.append(episode)

    return episodes


# ==============================================================================
# EVALUATION
# ==============================================================================

@dataclass
class EvalMetrics:
    """Metrics from evaluating a candidate."""
    violation_mean: float = 0.0
    violation_std: float = 0.0
    loss_mean: float = 0.0
    loss_std: float = 0.0
    convergence_rate: float = 0.0
    iterations_mean: float = 0.0
    drift_mean: float = 0.0  # ||x_t - x_{t-1}||


def evaluate_candidate(
    candidate: Candidate,
    episodes: List[List[Dict]],
    use_delta: bool = False,
    fixed_delta_beta: Optional[float] = None,
) -> EvalMetrics:
    """Evaluate a candidate on episodes.

    Args:
        candidate: ES candidate with hyperparameters
        episodes: List of episodes
        use_delta: Whether to use delta update
        fixed_delta_beta: Fixed beta (if None, use from candidate params)

    Returns:
        EvalMetrics
    """
    params = candidate.hyperparams.copy()

    # Set delta options
    params['use_delta_update'] = use_delta
    if use_delta:
        if fixed_delta_beta is not None:
            params['delta_beta'] = fixed_delta_beta
        elif 'delta_beta' not in params:
            params['delta_beta'] = 1.0

    solver = create_solver_from_dict(params)

    all_violations = []
    all_losses = []
    all_converged = []
    all_iterations = []
    all_drifts = []

    for episode in episodes:
        x_prev = None

        for step in episode:
            graph = step['graph']
            x_obs = step['x_obs']

            result = solver.solve(x_obs, graph, x_prev=x_prev)

            all_violations.append(result.breakdown.get('context', 0.0))
            all_losses.append(result.final_loss)
            all_converged.append(float(result.converged))
            all_iterations.append(result.iterations_used)

            if x_prev is not None:
                drift = torch.norm(result.x - x_prev).item()
                all_drifts.append(drift)

            x_prev = result.x.clone()
            step['x_final'] = result.x

    return EvalMetrics(
        violation_mean=np.mean(all_violations),
        violation_std=np.std(all_violations),
        loss_mean=np.mean(all_losses),
        loss_std=np.std(all_losses),
        convergence_rate=np.mean(all_converged),
        iterations_mean=np.mean(all_iterations),
        drift_mean=np.mean(all_drifts) if all_drifts else 0.0,
    )


def compute_fitness(metrics: EvalMetrics) -> float:
    """Compute scalar fitness from metrics (higher is better)."""
    # Negative loss and violation (want to minimize)
    # Positive convergence rate (want to maximize)
    fitness = (
        - 1.0 * metrics.loss_mean
        - 10.0 * metrics.violation_mean
        + 0.5 * metrics.convergence_rate
        - 0.1 * metrics.drift_mean
    )
    return fitness


# ==============================================================================
# TRAINING
# ==============================================================================

@dataclass
class TrainingResult:
    """Result of a training run."""
    mode: str  # 'euler' or 'delta'
    best_params: Dict[str, float]
    best_fitness: float
    best_metrics: Dict[str, float]
    fitness_history: List[float]
    generations: int
    total_time: float
    config: Dict


def train_es(
    mode: str,
    generations: int = 50,
    population_size: int = 16,
    num_episodes: int = 5,
    steps_per_episode: int = 10,
    embedding_dim: int = 32,
    optimize_beta: bool = False,
    fixed_delta_beta: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
) -> TrainingResult:
    """Train using ES with either Euler or Delta update.

    Args:
        mode: 'euler' or 'delta'
        generations: Number of ES generations
        population_size: ES population size
        num_episodes: Number of training episodes
        steps_per_episode: Steps per episode
        embedding_dim: Embedding dimension
        optimize_beta: If True, ES optimizes delta_beta (delta mode only)
        fixed_delta_beta: Fixed beta value when not optimizing
        seed: Random seed
        verbose: Print progress

    Returns:
        TrainingResult
    """
    use_delta = (mode == 'delta')

    # Configure parameter bounds
    param_bounds = {
        "step_size": [0.001, 0.1],
        "steps": [5, 30],
        "projection_alpha": [0.5, 1.0],
        "lambda_context": [0.1, 10.0],
        "lambda_ricci": [0.01, 1.0],
        "lambda_smooth": [0.01, 1.0],
        "lambda_var": [0.1, 10.0],
    }

    if use_delta and optimize_beta:
        param_bounds["delta_beta"] = [0.1, 2.0]

    es_config = ESConfig(
        population_size=population_size,
        sigma=0.3,
        seed=seed,
        parameter_bounds=param_bounds,
        use_delta_update=use_delta,
        delta_beta=fixed_delta_beta,
    )

    trainer = CMAESTrainer(es_config)

    # Generate training data
    episodes = generate_episode_batch(
        num_episodes=num_episodes,
        steps_per_episode=steps_per_episode,
        embedding_dim=embedding_dim,
        seed=seed,
    )

    fitness_history = []
    best_fitness = float('-inf')
    best_candidate = None
    best_metrics = None

    start_time = time.time()

    for gen in range(generations):
        # Ask for candidates
        candidates = trainer.ask()

        # Evaluate each candidate
        fitnesses = []
        for candidate in candidates:
            beta = None if (use_delta and optimize_beta) else fixed_delta_beta
            metrics = evaluate_candidate(
                candidate, episodes,
                use_delta=use_delta,
                fixed_delta_beta=beta,
            )
            fitness = compute_fitness(metrics)
            fitnesses.append(fitness)

            if fitness > best_fitness:
                best_fitness = fitness
                best_candidate = candidate
                best_metrics = metrics

        # Tell ES the results
        trainer.tell(candidates, fitnesses)

        gen_best = max(fitnesses)
        gen_mean = np.mean(fitnesses)
        fitness_history.append(gen_best)

        if verbose and (gen % 10 == 0 or gen == generations - 1):
            print(f"[{mode.upper()}] Gen {gen:3d}: best={gen_best:.4f}, mean={gen_mean:.4f}, "
                  f"violation={best_metrics.violation_mean:.4f}")

    total_time = time.time() - start_time

    return TrainingResult(
        mode=mode,
        best_params=best_candidate.hyperparams if best_candidate else {},
        best_fitness=best_fitness,
        best_metrics=asdict(best_metrics) if best_metrics else {},
        fitness_history=fitness_history,
        generations=generations,
        total_time=total_time,
        config={
            'population_size': population_size,
            'num_episodes': num_episodes,
            'steps_per_episode': steps_per_episode,
            'embedding_dim': embedding_dim,
            'optimize_beta': optimize_beta,
            'fixed_delta_beta': fixed_delta_beta,
            'seed': seed,
        }
    )


# ==============================================================================
# SAVE/LOAD
# ==============================================================================

def save_result(result: TrainingResult, output_dir: str) -> str:
    """Save training result to .pt file.

    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{result.mode}_training_{timestamp}.pt"
    filepath = os.path.join(output_dir, filename)

    # Convert to serializable dict
    data = {
        'mode': result.mode,
        'best_params': result.best_params,
        'best_fitness': result.best_fitness,
        'best_metrics': result.best_metrics,
        'fitness_history': result.fitness_history,
        'generations': result.generations,
        'total_time': result.total_time,
        'config': result.config,
        'timestamp': timestamp,
    }

    torch.save(data, filepath)
    print(f"Saved: {filepath}")

    # Also save JSON summary
    json_path = filepath.replace('.pt', '.json')
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    return filepath


def load_result(filepath: str) -> Dict:
    """Load training result from .pt file."""
    return torch.load(filepath, weights_only=False)


# ==============================================================================
# GENERALIZATION TEST
# ==============================================================================

def test_generalization(
    euler_result_path: str,
    delta_result_path: str,
    num_test_episodes: int = 10,
    test_seeds: List[int] = None,
    verbose: bool = True,
) -> Dict:
    """Test generalization of trained models on unseen data.

    Args:
        euler_result_path: Path to Euler training result .pt
        delta_result_path: Path to Delta training result .pt
        num_test_episodes: Number of test episodes
        test_seeds: Seeds for test data (should differ from training)
        verbose: Print progress

    Returns:
        Comparison results dict
    """
    if test_seeds is None:
        test_seeds = [1000, 2000, 3000, 4000, 5000]

    euler_data = load_result(euler_result_path)
    delta_data = load_result(delta_result_path)

    results = {
        'euler': {'violations': [], 'losses': [], 'convergence': []},
        'delta': {'violations': [], 'losses': [], 'convergence': []},
    }

    for seed in test_seeds:
        # Generate test episodes
        test_episodes = generate_episode_batch(
            num_episodes=num_test_episodes,
            steps_per_episode=10,
            embedding_dim=euler_data['config']['embedding_dim'],
            seed=seed,
        )

        # Test Euler
        euler_candidate = Candidate(hyperparams=euler_data['best_params'])
        euler_metrics = evaluate_candidate(
            euler_candidate, test_episodes,
            use_delta=False,
        )
        results['euler']['violations'].append(euler_metrics.violation_mean)
        results['euler']['losses'].append(euler_metrics.loss_mean)
        results['euler']['convergence'].append(euler_metrics.convergence_rate)

        # Test Delta
        delta_candidate = Candidate(hyperparams=delta_data['best_params'])
        delta_beta = delta_data['best_params'].get('delta_beta', delta_data['config']['fixed_delta_beta'])
        delta_metrics = evaluate_candidate(
            delta_candidate, test_episodes,
            use_delta=True,
            fixed_delta_beta=delta_beta,
        )
        results['delta']['violations'].append(delta_metrics.violation_mean)
        results['delta']['losses'].append(delta_metrics.loss_mean)
        results['delta']['convergence'].append(delta_metrics.convergence_rate)

    # Compute summary
    summary = {}
    for mode in ['euler', 'delta']:
        summary[mode] = {
            'violation_mean': np.mean(results[mode]['violations']),
            'violation_std': np.std(results[mode]['violations']),
            'loss_mean': np.mean(results[mode]['losses']),
            'loss_std': np.std(results[mode]['losses']),
            'convergence_mean': np.mean(results[mode]['convergence']),
        }

    if verbose:
        print("\n" + "=" * 60)
        print("GENERALIZATION TEST RESULTS")
        print("=" * 60)
        print(f"{'Metric':<20} {'Euler':>15} {'Delta':>15} {'Δ (Delta-Euler)':>15}")
        print("-" * 60)

        for metric in ['violation_mean', 'loss_mean', 'convergence_mean']:
            euler_val = summary['euler'][metric]
            delta_val = summary['delta'][metric]
            diff = delta_val - euler_val
            sign = '+' if diff > 0 else ''
            print(f"{metric:<20} {euler_val:>15.4f} {delta_val:>15.4f} {sign}{diff:>14.4f}")

        print("=" * 60)

        # Winner determination
        euler_score = -summary['euler']['violation_mean'] - summary['euler']['loss_mean']
        delta_score = -summary['delta']['violation_mean'] - summary['delta']['loss_mean']

        if delta_score > euler_score * 1.05:
            print("Winner: DELTA (>5% better)")
        elif euler_score > delta_score * 1.05:
            print("Winner: EULER (>5% better)")
        else:
            print("Result: TIE (within 5%)")

    return {'results': results, 'summary': summary}


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Delta vs Euler Training Benchmark")
    parser.add_argument('--mode', choices=['euler', 'delta', 'both'], default='both',
                        help="Training mode")
    parser.add_argument('--generations', type=int, default=50,
                        help="Number of ES generations")
    parser.add_argument('--population', type=int, default=16,
                        help="ES population size")
    parser.add_argument('--episodes', type=int, default=5,
                        help="Number of training episodes")
    parser.add_argument('--steps', type=int, default=10,
                        help="Steps per episode")
    parser.add_argument('--embedding-dim', type=int, default=32,
                        help="Embedding dimension")
    parser.add_argument('--optimize-beta', action='store_true',
                        help="Let ES optimize delta_beta (delta mode)")
    parser.add_argument('--delta-beta', type=float, default=1.0,
                        help="Fixed delta beta value")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed")
    parser.add_argument('--output', type=str, default='results/delta_benchmark',
                        help="Output directory")
    parser.add_argument('--test', action='store_true',
                        help="Run generalization test after training")

    args = parser.parse_args()

    print("=" * 60)
    print("DELTA VS EULER TRAINING BENCHMARK")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Generations: {args.generations}")
    print(f"Population: {args.population}")
    print(f"Episodes: {args.episodes}")
    print(f"Embedding dim: {args.embedding_dim}")
    if args.mode in ['delta', 'both']:
        print(f"Optimize beta: {args.optimize_beta}")
        print(f"Fixed delta_beta: {args.delta_beta}")
    print("=" * 60)

    results = {}

    if args.mode in ['euler', 'both']:
        print("\n>>> Training EULER variant...")
        euler_result = train_es(
            mode='euler',
            generations=args.generations,
            population_size=args.population,
            num_episodes=args.episodes,
            steps_per_episode=args.steps,
            embedding_dim=args.embedding_dim,
            seed=args.seed,
        )
        euler_path = save_result(euler_result, args.output)
        results['euler'] = euler_result
        results['euler_path'] = euler_path

    if args.mode in ['delta', 'both']:
        print("\n>>> Training DELTA variant...")
        delta_result = train_es(
            mode='delta',
            generations=args.generations,
            population_size=args.population,
            num_episodes=args.episodes,
            steps_per_episode=args.steps,
            embedding_dim=args.embedding_dim,
            optimize_beta=args.optimize_beta,
            fixed_delta_beta=args.delta_beta,
            seed=args.seed,
        )
        delta_path = save_result(delta_result, args.output)
        results['delta'] = delta_result
        results['delta_path'] = delta_path

    # Summary comparison
    if args.mode == 'both':
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"{'Metric':<25} {'Euler':>15} {'Delta':>15}")
        print("-" * 60)
        print(f"{'Best Fitness':<25} {results['euler'].best_fitness:>15.4f} {results['delta'].best_fitness:>15.4f}")
        print(f"{'Violation (mean)':<25} {results['euler'].best_metrics['violation_mean']:>15.4f} {results['delta'].best_metrics['violation_mean']:>15.4f}")
        print(f"{'Loss (mean)':<25} {results['euler'].best_metrics['loss_mean']:>15.4f} {results['delta'].best_metrics['loss_mean']:>15.4f}")
        print(f"{'Convergence Rate':<25} {results['euler'].best_metrics['convergence_rate']:>15.4f} {results['delta'].best_metrics['convergence_rate']:>15.4f}")
        print(f"{'Training Time (s)':<25} {results['euler'].total_time:>15.2f} {results['delta'].total_time:>15.2f}")
        print("=" * 60)

        if args.test:
            print("\n>>> Running generalization test...")
            test_generalization(
                results['euler_path'],
                results['delta_path'],
                num_test_episodes=10,
                test_seeds=[1000, 2000, 3000, 4000, 5000],
            )

    print("\nDone!")


if __name__ == '__main__':
    main()
