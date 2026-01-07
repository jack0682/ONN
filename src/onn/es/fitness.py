"""Episodic Fitness Evaluation for ES.

This module implements fitness evaluation for CMA-ES by running
episodes with the ONN solver and aggregating metrics.

Fitness is computed as:
    F(θ) = -α₁*violation - α₂*drift + α₃*variance - α₄*latency

(Negative because CMA-ES maximizes fitness)

Reference:
    - spec/20_impl_plan.ir.yml: IMPL_021
    - User roadmap: "core metrics: violation, drift, anti-collapse, latency"

Author: Claude (via IMPL_021)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Any

import torch
import numpy as np

from onn.core.tensors import EvalMetrics

if TYPE_CHECKING:
    from onn.core.solver import ProjectionConsensusSolver, SolverResult
    from onn.core.graph import EdgeGraph
    from onn.core.cycles import CycleBasis
    from onn.relation.param import RelationEncoder
    from onn.es.ask_tell import Candidate
    from onn.core.tensors import EvalMetrics

logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================


@dataclass
class FitnessConfig:
    """Fitness function weights. Higher weights = more important."""

    w_violation: float = 10.0
    w_drift: float = 1.0
    w_latency: float = 2.0
    w_collapse: float = 1.0
    w_smoothness: float = -0.1
    w_ricci: float = -0.05

    recovery_threshold: float = 0.1  # Violation threshold for "recovered"


@dataclass
class EpisodeStep:
    """A single step in an episode.

    Contains the graph state and optional event markers.
    """

    x_obs: torch.Tensor  # Observed embeddings (or fallback if no encoder)
    edge_graph: "EdgeGraph"  # Graph structure
    is_event: bool = False  # Whether an event occurred
    phi: Optional[torch.Tensor] = None  # Pair features for RelationEncoder


# ==============================================================================
# EPISODE EVALUATION
# ==============================================================================


def evaluate_episode(
    solver: "ProjectionConsensusSolver",
    episode: List[EpisodeStep],
    config: Optional[FitnessConfig] = None,
) -> EvalMetrics:
    """Run solver on episode and compute metrics.

    Args:
        solver: Configured ONN solver
        episode: List of EpisodeStep with observations
        config: Fitness configuration

    Returns:
        EvalMetrics summarizing the episode
    """
    if config is None:
        config = FitnessConfig()

    if not episode:
        return EvalMetrics()

    violations = []
    drifts = []
    ricci_energies = []
    smoothness_vals = []
    variance_vals = []

    x_prev = None
    last_event_step = -1
    latencies = []

    for t, step in enumerate(episode):
        # Solve
        result = solver.solve(step.x_obs, step.edge_graph, x_prev=x_prev)
        x_hat = result.x

        # Violation (from solver breakdown)
        violation = result.breakdown.get("context", 0.0)
        violations.append(violation)

        # Drift from observation
        drift = torch.norm(x_hat - step.x_obs).item()
        drifts.append(drift)

        # Ricci energy
        ricci = result.breakdown.get("ricci", 0.0)
        ricci_energies.append(ricci)

        # Smoothness
        if x_prev is not None:
            smooth = torch.norm(x_hat - x_prev).item()
            smoothness_vals.append(smooth)

        # Variance (collapse score)
        var = torch.var(x_hat).item()
        variance_vals.append(var)

        # Event handling
        if step.is_event:
            last_event_step = t
        elif last_event_step >= 0 and violation < config.recovery_threshold:
            latency = t - last_event_step
            latencies.append(latency)
            last_event_step = -1  # Reset

        x_prev = x_hat

    # Aggregate metrics
    return EvalMetrics(
        violation_mean=np.mean(violations) if violations else 0.0,
        violation_max=np.max(violations) if violations else 0.0,
        drift_mean=np.mean(drifts) if drifts else 0.0,
        ricci_energy=np.mean(ricci_energies) if ricci_energies else 0.0,
        smoothness=np.mean(smoothness_vals) if smoothness_vals else 0.0,
        latency_mean=np.mean(latencies) if latencies else 0.0,
        collapse_score=np.mean(variance_vals) if variance_vals else 0.0,
    )


def aggregate_metrics(metrics_list: List[EvalMetrics]) -> float:
    """Aggregate metrics to scalar fitness (higher = better).

    Args:
        metrics_list: List of EvalMetrics from multiple episodes

    Returns:
        Scalar fitness value
    """
    if not metrics_list:
        return float("-inf")

    # Average across episodes
    avg = EvalMetrics(
        violation_mean=np.mean([m.violation_mean for m in metrics_list]),
        violation_max=np.max([m.violation_max for m in metrics_list]),
        drift_mean=np.mean([m.drift_mean for m in metrics_list]),
        ricci_energy=np.mean([m.ricci_energy for m in metrics_list]),
        smoothness=np.mean([m.smoothness for m in metrics_list]),
        latency_mean=np.mean([m.latency_mean for m in metrics_list]),
        collapse_score=np.mean([m.collapse_score for m in metrics_list]),
    )

    return compute_fitness(avg)


def compute_fitness(
    metrics: EvalMetrics,
    config: Optional[FitnessConfig] = None,
) -> float:
    """Compute fitness from metrics. Higher is better."""
    if config is None:
        config = FitnessConfig()

    fitness = (
        -config.w_violation * metrics.violation_mean
        - config.w_drift * metrics.drift_mean
        - config.w_latency * metrics.latency_mean
        + config.w_collapse * metrics.collapse_score
        + config.w_smoothness * metrics.smoothness
        + config.w_ricci * metrics.ricci_energy
    )

    return fitness


# ==============================================================================
# SYNTHETIC EPISODE GENERATION
# ==============================================================================


def evaluate_candidate(
    candidate: "Candidate",
    episodes: List[List[EpisodeStep]],
    relation_encoder: Optional["RelationEncoder"] = None,
    fitness_config: Optional[FitnessConfig] = None,
) -> float:
    """Evaluate a candidate (hyperparams + W_lin) on multiple episodes.

    This is the main fitness function for CMA-ES optimization.

    Args:
        candidate: Candidate with hyperparams and optional W_lin
        episodes: List of episodes to evaluate on
        relation_encoder: RelationEncoder to use (required if candidate has W_lin)
        fitness_config: Fitness configuration

    Returns:
        Scalar fitness value (higher = better)

    Example:
        >>> trainer = create_trainer_with_w_lin(w_lin_shape=(32, 64))
        >>> encoder = RelationEncoder(RelationParamConfig(input_dim=64, output_dim=32))
        >>> candidates = trainer.ask()
        >>> fitnesses = [evaluate_candidate(c, episodes, encoder) for c in candidates]
        >>> trainer.tell(candidates, fitnesses)
    """
    from onn.core.solver import create_solver_from_dict
    from onn.es.ask_tell import Candidate

    if fitness_config is None:
        fitness_config = FitnessConfig()

    # Set W_lin if provided
    if candidate.w_lin is not None:
        if relation_encoder is None:
            raise ValueError("relation_encoder required when candidate has W_lin")
        w_lin_tensor = torch.tensor(candidate.w_lin, dtype=torch.float32)
        relation_encoder.set_linear_weights(w_lin_tensor)

    # Create solver from hyperparameters
    solver = create_solver_from_dict(candidate.hyperparams)

    # Evaluate on all episodes
    metrics_list = []
    for episode in episodes:
        # If phi is provided, use encoder to generate x_obs
        processed_episode = []
        for step in episode:
            if step.phi is not None and relation_encoder is not None:
                x_obs = relation_encoder.encode(step.phi)
                processed_step = EpisodeStep(
                    x_obs=x_obs,
                    edge_graph=step.edge_graph,
                    is_event=step.is_event,
                    phi=step.phi,
                )
            else:
                processed_step = step
            processed_episode.append(processed_step)

        metrics = evaluate_episode(solver, processed_episode, fitness_config)
        metrics_list.append(metrics)

    return aggregate_metrics(metrics_list)


def generate_synthetic_episode(
    num_steps: int = 10,
    num_edges: int = 6,
    embedding_dim: int = 32,
    event_prob: float = 0.1,
    noise_std: float = 0.1,
    seed: Optional[int] = None,
) -> List[EpisodeStep]:
    """Generate a synthetic episode for testing.

    Args:
        num_steps: Number of steps in the episode
        num_edges: Number of edges in the graph
        embedding_dim: Embedding dimension
        event_prob: Probability of an event at each step
        noise_std: Standard deviation of observation noise
        seed: Random seed

    Returns:
        List of EpisodeStep
    """
    from onn.core.graph import EdgeGraph

    rng = np.random.RandomState(seed)

    # Create a simple graph (triangle + extra edges)
    n_nodes = max(3, num_edges // 2 + 1)
    edges = []
    for i in range(n_nodes - 1):
        edges.append((i, i + 1))
    edges.append((n_nodes - 1, 0))  # Close the cycle

    # Add extra edges if needed
    while len(edges) < num_edges:
        i, j = rng.randint(0, n_nodes, 2)
        if i != j and (i, j) not in edges and (j, i) not in edges:
            edges.append((i, j))

    graph = EdgeGraph.from_edge_list(edges[:num_edges])

    # Generate base embeddings
    base_x = torch.randn(num_edges, embedding_dim)

    episode = []
    for t in range(num_steps):
        # Add noise
        noise = torch.randn(num_edges, embedding_dim) * noise_std
        x_obs = base_x + noise

        # Random event
        is_event = rng.random() < event_prob
        if is_event:
            # Simulate event: random perturbation
            base_x = base_x + torch.randn(num_edges, embedding_dim) * 0.5

        episode.append(
            EpisodeStep(
                x_obs=x_obs,
                edge_graph=graph,
                is_event=is_event,
            )
        )

    return episode
