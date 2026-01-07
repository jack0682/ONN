#!/usr/bin/env python3
"""Phase 2 Validation: End-to-End Differentiable Training.

Tests K-step unroll solver with gradient flow to W_lin.

Key experiments:
1. Gradient Flow: Verify gradients reach W_lin
2. Learning Curve: Does W_lin improve with training?
3. K-step Ablation: Effect of unroll depth K
4. Comparison: Phase 1 (surrogate) vs Phase 2 (E2E)

Usage:
    python experiments/validate_phase2.py --mode smoke
    python experiments/validate_phase2.py --mode full

Author: Claude (Phase 2 Validation)
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
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from onn.core import (
    EdgeGraph,
    create_diff_solver,
    create_diff_encoder,
    create_e2e_trainer,
    EndToEndTrainer,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Phase2Config:
    """Configuration for Phase 2 validation."""
    input_dim: int = 64
    output_dim: int = 32

    # Graph
    num_nodes: int = 6
    num_edges: int = 12

    # Training
    num_steps: int = 100
    lr: float = 1e-3
    K: int = 10  # Unroll steps

    # Experiment
    num_seeds: int = 3
    save_results: bool = True
    output_dir: str = "experiments/results"


@dataclass
class SmokeConfig(Phase2Config):
    num_steps: int = 20
    K: int = 5
    num_seeds: int = 1


@dataclass
class FullConfig(Phase2Config):
    num_steps: int = 200
    K: int = 15
    num_seeds: int = 5


def create_random_graph(num_nodes: int, num_edges: int, seed: int = 42) -> EdgeGraph:
    """Create a random connected graph."""
    rng = np.random.RandomState(seed)

    # Start with spanning tree to ensure connectivity
    edges = []
    for i in range(1, num_nodes):
        j = rng.randint(0, i)
        edges.append((j, i))

    # Add random edges
    all_possible = [(i, j) for i in range(num_nodes) for j in range(i+1, num_nodes)]
    existing = set(edges)
    remaining = [e for e in all_possible if e not in existing and (e[1], e[0]) not in existing]

    extra_needed = num_edges - len(edges)
    if extra_needed > 0 and remaining:
        extra = rng.choice(len(remaining), min(extra_needed, len(remaining)), replace=False)
        for idx in extra:
            edges.append(remaining[idx])

    return EdgeGraph.from_edge_list(edges)


def test_gradient_flow(config: Phase2Config, seed: int = 42) -> Dict:
    """Test 1: Verify gradient flows to W_lin."""
    logger.info(f"Test 1: Gradient Flow (seed={seed})")

    torch.manual_seed(seed)

    encoder = create_diff_encoder(config.input_dim, config.output_dim)
    solver = create_diff_solver(K=config.K)

    graph = create_random_graph(config.num_nodes, config.num_edges, seed)
    phi = torch.randn(graph.num_edges, config.input_dim, requires_grad=True)

    # Forward pass
    x_obs = encoder(phi)
    x_final, info = solver(x_obs, graph)
    loss = x_final.sum()

    # Backward pass
    loss.backward()

    grad_exists = encoder.W_lin.grad is not None
    grad_norm = encoder.W_lin.grad.norm().item() if grad_exists else 0.0
    grad_nonzero = grad_norm > 1e-10

    logger.info(f"  Grad exists: {grad_exists}, norm: {grad_norm:.6f}")

    return {
        'seed': seed,
        'grad_exists': grad_exists,
        'grad_norm': grad_norm,
        'grad_nonzero': grad_nonzero,
        'solver_loss': info['final_loss'],
    }


def test_learning_curve(config: Phase2Config, seed: int = 42) -> Dict:
    """Test 2: Does W_lin improve with training?"""
    logger.info(f"Test 2: Learning Curve (seed={seed})")

    torch.manual_seed(seed)
    np.random.seed(seed)

    trainer = create_e2e_trainer(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        K=config.K,
        lr=config.lr,
    )

    graph = create_random_graph(config.num_nodes, config.num_edges, seed)

    loss_history = []
    w_lin_norm_history = []

    for step in range(config.num_steps):
        # Random batch each step
        phi = torch.randn(graph.num_edges, config.input_dim)

        result = trainer.train_step(phi, graph)
        loss_history.append(result['loss'])

        w_lin_norm = trainer.encoder.W_lin.data.norm().item()
        w_lin_norm_history.append(w_lin_norm)

        if step % 20 == 0 or step == config.num_steps - 1:
            logger.info(f"  Step {step}: loss={result['loss']:.4f}, W_lin norm={w_lin_norm:.4f}")

    # Check if loss decreased
    initial_loss = np.mean(loss_history[:5])
    final_loss = np.mean(loss_history[-5:])
    improved = final_loss < initial_loss
    improvement = (initial_loss - final_loss) / (abs(initial_loss) + 1e-6)

    logger.info(f"  Initial loss: {initial_loss:.4f}, Final: {final_loss:.4f}, Improved: {improved}")

    return {
        'seed': seed,
        'loss_history': loss_history,
        'w_lin_norm_history': w_lin_norm_history,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'improved': improved,
        'improvement_ratio': improvement,
    }


def test_k_ablation(config: Phase2Config, seed: int = 42) -> Dict:
    """Test 3: Effect of unroll depth K."""
    logger.info(f"Test 3: K-step Ablation (seed={seed})")

    torch.manual_seed(seed)

    K_values = [1, 3, 5, 10, 15]
    results = {}

    graph = create_random_graph(config.num_nodes, config.num_edges, seed)

    for K in K_values:
        if K > config.K * 2:  # Skip very large K in smoke test
            continue

        trainer = create_e2e_trainer(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            K=K,
            lr=config.lr,
        )

        # Quick training
        losses = []
        for step in range(min(50, config.num_steps)):
            phi = torch.randn(graph.num_edges, config.input_dim)
            result = trainer.train_step(phi, graph)
            losses.append(result['loss'])

        final_loss = np.mean(losses[-10:])

        results[f'K={K}'] = {
            'K': K,
            'final_loss': final_loss,
            'num_steps': len(losses),
        }
        logger.info(f"  K={K}: final_loss={final_loss:.4f}")

    return {
        'seed': seed,
        'results': results,
    }


def test_phase1_vs_phase2(config: Phase2Config, seed: int = 42) -> Dict:
    """Test 4: Compare Phase 1 (surrogate) vs Phase 2 (E2E)."""
    logger.info(f"Test 4: Phase 1 vs Phase 2 (seed={seed})")

    torch.manual_seed(seed)

    graph = create_random_graph(config.num_nodes, config.num_edges, seed)
    num_train_steps = min(50, config.num_steps)

    # Phase 2: End-to-End
    logger.info("  Training Phase 2 (E2E)...")
    trainer_e2e = create_e2e_trainer(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        K=config.K,
        lr=config.lr,
    )

    e2e_losses = []
    for step in range(num_train_steps):
        phi = torch.randn(graph.num_edges, config.input_dim)
        result = trainer_e2e.train_step(phi, graph)
        e2e_losses.append(result['loss'])

    # Phase 1: Surrogate (from hybrid_trainer)
    logger.info("  Training Phase 1 (Surrogate)...")
    from onn.es.hybrid_trainer import LearnableEncoder, SurrogateLoss, SurrogateLossConfig

    encoder_surrogate = LearnableEncoder(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
    )
    surrogate_loss = SurrogateLoss(SurrogateLossConfig())
    optimizer = torch.optim.AdamW(encoder_surrogate.parameters(), lr=config.lr)

    surrogate_losses = []
    for step in range(num_train_steps):
        phi = torch.randn(graph.num_edges, config.input_dim)

        optimizer.zero_grad()
        x_obs = encoder_surrogate(phi)
        loss, _ = surrogate_loss(x_obs, phi)
        loss.backward()
        optimizer.step()

        surrogate_losses.append(loss.item())

    e2e_final = np.mean(e2e_losses[-10:])
    surrogate_final = np.mean(surrogate_losses[-10:])

    logger.info(f"  E2E final loss: {e2e_final:.4f}")
    logger.info(f"  Surrogate final loss: {surrogate_final:.4f}")

    return {
        'seed': seed,
        'e2e_loss_history': e2e_losses,
        'surrogate_loss_history': surrogate_losses,
        'e2e_final': e2e_final,
        'surrogate_final': surrogate_final,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Validation")
    parser.add_argument('--mode', type=str, default='smoke', choices=['smoke', 'full'])
    parser.add_argument('--output-dir', type=str, default='experiments/results')
    args = parser.parse_args()

    if args.mode == 'smoke':
        logger.info("Running SMOKE TEST")
        config = SmokeConfig()
    else:
        logger.info("Running FULL EXPERIMENT")
        config = FullConfig()

    config.output_dir = args.output_dir

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'config': asdict(config),
        'experiments': {},
    }

    # Run tests
    for seed in range(config.num_seeds):
        actual_seed = 42 + seed * 100

        # Test 1: Gradient Flow
        grad_result = test_gradient_flow(config, actual_seed)
        all_results['experiments'].setdefault('gradient_flow', []).append(grad_result)

        # Test 2: Learning Curve
        learn_result = test_learning_curve(config, actual_seed)
        all_results['experiments'].setdefault('learning_curve', []).append(learn_result)

        # Test 3: K Ablation
        k_result = test_k_ablation(config, actual_seed)
        all_results['experiments'].setdefault('k_ablation', []).append(k_result)

        # Test 4: Phase 1 vs Phase 2
        compare_result = test_phase1_vs_phase2(config, actual_seed)
        all_results['experiments'].setdefault('phase_comparison', []).append(compare_result)

    # Summary
    print("\n" + "=" * 70)
    print("            PHASE 2 (END-TO-END) VALIDATION SUMMARY")
    print("=" * 70)

    print("\n### GRADIENT FLOW ###")
    for r in all_results['experiments']['gradient_flow']:
        print(f"  Seed {r['seed']}: grad_exists={r['grad_exists']}, norm={r['grad_norm']:.6f}")

    print("\n### LEARNING CURVE ###")
    for r in all_results['experiments']['learning_curve']:
        print(f"  Seed {r['seed']}: {r['initial_loss']:.4f} -> {r['final_loss']:.4f} "
              f"(improved={r['improved']}, {r['improvement_ratio']*100:.1f}%)")

    print("\n### K-STEP ABLATION ###")
    for r in all_results['experiments']['k_ablation']:
        print(f"  Seed {r['seed']}:")
        for k, v in r['results'].items():
            print(f"    {k}: loss={v['final_loss']:.4f}")

    print("\n### PHASE 1 vs PHASE 2 ###")
    for r in all_results['experiments']['phase_comparison']:
        print(f"  Seed {r['seed']}:")
        print(f"    E2E (Phase 2):      {r['e2e_final']:.4f}")
        print(f"    Surrogate (Phase 1): {r['surrogate_final']:.4f}")

    print("=" * 70)

    # Save
    if config.save_results:
        os.makedirs(config.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(config.output_dir, f"phase2_validation_{timestamp}.json")
        with open(filepath, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    main()
