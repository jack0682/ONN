#!/usr/bin/env python3
"""Inverted Pendulum / Oscillator Prediction Benchmark.

Physics-based benchmark comparing Delta ODE vs Euler for:
1. Inverted Pendulum: θ̈ = (g/l)sin(θ) - (b/ml²)θ̇ + (1/ml²)u
2. Damped Oscillator: ẍ = -ω²x - γẋ

Graph Structure:
- Nodes: Physical quantities (θ, θ̇, x, ẋ, etc.)
- Edges: Physical relationships (coupling, derivatives)
- Embeddings: State evolution prediction

This tests whether Delta ODE provides better:
- Temporal stability (smooth predictions)
- Constraint satisfaction (energy conservation, limits)
- Generalization to unseen initial conditions

Usage:
    # Train and compare
    python benchmarks/pendulum_benchmark.py --system pendulum --generations 50

    # Test on oscillator
    python benchmarks/pendulum_benchmark.py --system oscillator --generations 50

    # Full comparison
    python benchmarks/pendulum_benchmark.py --system both --test

Author: Claude
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, 'src')

from onn.core.solver import PCSolverConfig, ProjectionConsensusSolver, create_solver_from_dict
from onn.core.graph import EdgeGraph, EdgeKey
from onn.es.ask_tell import ESConfig, CMAESTrainer, Candidate


# ==============================================================================
# PHYSICS SIMULATION
# ==============================================================================

@dataclass
class PendulumParams:
    """Inverted pendulum physical parameters."""
    g: float = 9.81       # Gravity (m/s²)
    l: float = 1.0        # Length (m)
    m: float = 1.0        # Mass (kg)
    b: float = 0.1        # Damping coefficient
    dt: float = 0.02      # Time step (s)


@dataclass
class OscillatorParams:
    """Damped harmonic oscillator parameters."""
    omega: float = 2.0    # Natural frequency (rad/s)
    gamma: float = 0.1    # Damping ratio
    dt: float = 0.02      # Time step (s)


@dataclass
class DuffingParams:
    """Duffing oscillator parameters (nonlinear).

    ẍ = -δẋ - αx - βx³

    Exhibits:
    - Hardening spring (β > 0): frequency increases with amplitude
    - Softening spring (β < 0): frequency decreases with amplitude
    - Chaotic behavior possible with forcing
    """
    alpha: float = 1.0    # Linear stiffness
    beta: float = 0.5     # Cubic nonlinearity (hardening)
    delta: float = 0.1    # Damping
    dt: float = 0.01      # Time step (smaller for stability)


def simulate_pendulum(
    theta0: float,
    theta_dot0: float,
    num_steps: int,
    params: PendulumParams = None,
    control: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate inverted pendulum dynamics.

    θ̈ = (g/l)sin(θ) - (b/ml²)θ̇ + (1/ml²)u

    Args:
        theta0: Initial angle (rad)
        theta_dot0: Initial angular velocity (rad/s)
        num_steps: Number of simulation steps
        params: Physical parameters
        control: Control input sequence (optional)

    Returns:
        theta: Angle trajectory (num_steps,)
        theta_dot: Angular velocity trajectory (num_steps,)
    """
    if params is None:
        params = PendulumParams()

    theta = np.zeros(num_steps)
    theta_dot = np.zeros(num_steps)

    theta[0] = theta0
    theta_dot[0] = theta_dot0

    for t in range(1, num_steps):
        u = control[t-1] if control is not None else 0.0

        # Euler integration (simple for now)
        theta_ddot = (
            (params.g / params.l) * np.sin(theta[t-1])
            - (params.b / (params.m * params.l**2)) * theta_dot[t-1]
            + (1 / (params.m * params.l**2)) * u
        )

        theta_dot[t] = theta_dot[t-1] + params.dt * theta_ddot
        theta[t] = theta[t-1] + params.dt * theta_dot[t]

    return theta, theta_dot


def simulate_oscillator(
    x0: float,
    v0: float,
    num_steps: int,
    params: OscillatorParams = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate damped harmonic oscillator.

    ẍ = -ω²x - γẋ

    Args:
        x0: Initial position
        v0: Initial velocity
        num_steps: Number of steps
        params: Physical parameters

    Returns:
        x: Position trajectory (num_steps,)
        v: Velocity trajectory (num_steps,)
    """
    if params is None:
        params = OscillatorParams()

    x = np.zeros(num_steps)
    v = np.zeros(num_steps)

    x[0] = x0
    v[0] = v0

    for t in range(1, num_steps):
        a = -params.omega**2 * x[t-1] - params.gamma * v[t-1]
        v[t] = v[t-1] + params.dt * a
        x[t] = x[t-1] + params.dt * v[t]

    return x, v


def simulate_duffing(
    x0: float,
    v0: float,
    num_steps: int,
    params: DuffingParams = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate Duffing oscillator (nonlinear).

    ẍ = -δẋ - αx - βx³

    The cubic term creates nonlinear dynamics:
    - Amplitude-dependent frequency
    - Multiple equilibria possible
    - Can exhibit chaos with forcing

    Args:
        x0: Initial position
        v0: Initial velocity
        num_steps: Number of steps
        params: Physical parameters

    Returns:
        x: Position trajectory (num_steps,)
        v: Velocity trajectory (num_steps,)
    """
    if params is None:
        params = DuffingParams()

    x = np.zeros(num_steps)
    v = np.zeros(num_steps)

    x[0] = x0
    v[0] = v0

    for t in range(1, num_steps):
        # Duffing equation: ẍ = -δẋ - αx - βx³
        a = -params.delta * v[t-1] - params.alpha * x[t-1] - params.beta * x[t-1]**3

        # RK4 integration for better accuracy with nonlinear dynamics
        k1_v = a
        k1_x = v[t-1]

        x_mid = x[t-1] + 0.5 * params.dt * k1_x
        v_mid = v[t-1] + 0.5 * params.dt * k1_v
        a_mid = -params.delta * v_mid - params.alpha * x_mid - params.beta * x_mid**3

        k2_v = a_mid
        k2_x = v_mid

        x_mid2 = x[t-1] + 0.5 * params.dt * k2_x
        v_mid2 = v[t-1] + 0.5 * params.dt * k2_v
        a_mid2 = -params.delta * v_mid2 - params.alpha * x_mid2 - params.beta * x_mid2**3

        k3_v = a_mid2
        k3_x = v_mid2

        x_end = x[t-1] + params.dt * k3_x
        v_end = v[t-1] + params.dt * k3_v
        a_end = -params.delta * v_end - params.alpha * x_end - params.beta * x_end**3

        k4_v = a_end
        k4_x = v_end

        v[t] = v[t-1] + (params.dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        x[t] = x[t-1] + (params.dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)

    return x, v


def compute_energy_duffing(x: np.ndarray, v: np.ndarray, params: DuffingParams = None) -> np.ndarray:
    """Compute total energy of Duffing oscillator.

    E = (1/2)v² + (1/2)αx² + (1/4)βx⁴
    """
    if params is None:
        params = DuffingParams()

    KE = 0.5 * v**2
    PE = 0.5 * params.alpha * x**2 + 0.25 * params.beta * x**4

    return KE + PE


def compute_energy_pendulum(theta: np.ndarray, theta_dot: np.ndarray, params: PendulumParams = None) -> np.ndarray:
    """Compute total energy of pendulum (should be approximately conserved)."""
    if params is None:
        params = PendulumParams()

    # Kinetic energy: (1/2)ml²θ̇²
    KE = 0.5 * params.m * params.l**2 * theta_dot**2
    # Potential energy: mgl(1 - cos(θ))
    PE = params.m * params.g * params.l * (1 - np.cos(theta))

    return KE + PE


def compute_energy_oscillator(x: np.ndarray, v: np.ndarray, params: OscillatorParams = None) -> np.ndarray:
    """Compute total energy of oscillator."""
    if params is None:
        params = OscillatorParams()

    # Kinetic: (1/2)v²
    KE = 0.5 * v**2
    # Potential: (1/2)ω²x²
    PE = 0.5 * params.omega**2 * x**2

    return KE + PE


# ==============================================================================
# GRAPH CONSTRUCTION
# ==============================================================================

def create_physics_graph(system: str = 'pendulum') -> EdgeGraph:
    """Create graph structure for physical system.

    Pendulum graph:
        Node 0: θ (angle)
        Node 1: θ̇ (angular velocity)
        Node 2: sin(θ) (nonlinearity)
        Node 3: Energy

    Duffing graph:
        Node 0: x (position)
        Node 1: v (velocity)
        Node 2: x³ (cubic nonlinearity)
        Node 3: Energy

    Oscillator graph:
        Node 0: x (position)
        Node 1: v (velocity)
        Node 2: Energy
    """
    if system == 'pendulum':
        node_ids = [0, 1, 2, 3]  # θ, θ̇, sin(θ), E
        edge_keys = [
            EdgeKey(0, 1),  # θ → θ̇ (derivative relation)
            EdgeKey(1, 0),  # θ̇ → θ (integration)
            EdgeKey(0, 2),  # θ → sin(θ) (nonlinearity)
            EdgeKey(2, 1),  # sin(θ) → θ̇ (dynamics)
            EdgeKey(0, 3),  # θ → E (potential)
            EdgeKey(1, 3),  # θ̇ → E (kinetic)
        ]
    elif system == 'duffing':
        node_ids = [0, 1, 2, 3]  # x, v, x³, E
        edge_keys = [
            EdgeKey(0, 1),  # x → v (derivative)
            EdgeKey(1, 0),  # v → x (integration)
            EdgeKey(0, 2),  # x → x³ (cubic nonlinearity)
            EdgeKey(2, 1),  # x³ → v (nonlinear dynamics)
            EdgeKey(0, 3),  # x → E (potential: αx² + βx⁴)
            EdgeKey(1, 3),  # v → E (kinetic)
            EdgeKey(2, 3),  # x³ → E (quartic potential)
        ]
    else:  # oscillator (linear)
        node_ids = [0, 1, 2]  # x, v, E
        edge_keys = [
            EdgeKey(0, 1),  # x → v (derivative)
            EdgeKey(1, 0),  # v → x (integration)
            EdgeKey(0, 2),  # x → E (potential)
            EdgeKey(1, 2),  # v → E (kinetic)
        ]

    return EdgeGraph(node_ids=node_ids, edge_keys=edge_keys)


def state_to_embedding(
    state: np.ndarray,
    embedding_dim: int = 32,
    system: str = 'pendulum',
) -> torch.Tensor:
    """Convert physical state to edge embeddings.

    Uses Fourier features for better representation of periodic dynamics.

    Args:
        state: Physical state [θ, θ̇] or [x, v]
        embedding_dim: Target embedding dimension
        system: 'pendulum', 'duffing', or 'oscillator'

    Returns:
        x_obs: Edge embeddings (num_edges, embedding_dim)
    """
    if system == 'pendulum':
        theta, theta_dot = state[0], state[1]

        # Compute derived quantities
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        energy = 0.5 * theta_dot**2 + 9.81 * (1 - cos_theta)  # Normalized

        # Base features for each edge
        edge_features = [
            [theta, theta_dot],           # θ → θ̇
            [theta_dot, theta],           # θ̇ → θ
            [theta, sin_theta],           # θ → sin(θ)
            [sin_theta, theta_dot],       # sin(θ) → θ̇
            [theta, energy],              # θ → E
            [theta_dot, energy],          # θ̇ → E
        ]
        num_edges = 6

    elif system == 'duffing':
        x, v = state[0], state[1]

        # Duffing nonlinearity and energy
        x_cubed = x**3
        # E = (1/2)v² + (1/2)αx² + (1/4)βx⁴ (α=1, β=0.5)
        energy = 0.5 * v**2 + 0.5 * x**2 + 0.125 * x**4

        edge_features = [
            [x, v],              # x → v
            [v, x],              # v → x
            [x, x_cubed],        # x → x³ (nonlinearity)
            [x_cubed, v],        # x³ → v (nonlinear dynamics)
            [x, energy],         # x → E
            [v, energy],         # v → E
            [x_cubed, energy],   # x³ → E
        ]
        num_edges = 7

    else:  # oscillator (linear)
        x, v = state[0], state[1]
        energy = 0.5 * v**2 + 0.5 * 4.0 * x**2  # ω²=4

        edge_features = [
            [x, v],          # x → v
            [v, x],          # v → x
            [x, energy],     # x → E
            [v, energy],     # v → E
        ]
        num_edges = 4

    # Expand to embedding_dim using Fourier features
    embeddings = np.zeros((num_edges, embedding_dim))

    for i, feat in enumerate(edge_features):
        # Fourier expansion
        for j in range(embedding_dim // 4):
            freq = (j + 1) * np.pi
            embeddings[i, 4*j] = np.sin(freq * feat[0])
            embeddings[i, 4*j + 1] = np.cos(freq * feat[0])
            embeddings[i, 4*j + 2] = np.sin(freq * feat[1])
            embeddings[i, 4*j + 3] = np.cos(freq * feat[1])

    return torch.tensor(embeddings, dtype=torch.float32)


# ==============================================================================
# EPISODE GENERATION
# ==============================================================================

def generate_physics_episodes(
    system: str = 'pendulum',
    num_episodes: int = 10,
    steps_per_episode: int = 50,
    embedding_dim: int = 32,
    seed: int = 42,
) -> List[List[Dict]]:
    """Generate episodes from physical simulations.

    Each episode is a trajectory with different initial conditions.
    """
    np.random.seed(seed)
    episodes = []

    for ep_idx in range(num_episodes):
        episode = []
        graph = create_physics_graph(system)

        if system == 'pendulum':
            # Random initial conditions (small angles)
            theta0 = np.random.uniform(-0.5, 0.5)
            theta_dot0 = np.random.uniform(-1.0, 1.0)

            theta, theta_dot = simulate_pendulum(
                theta0, theta_dot0, steps_per_episode + 1
            )

            for t in range(steps_per_episode):
                state_t = np.array([theta[t], theta_dot[t]])
                state_t1 = np.array([theta[t+1], theta_dot[t+1]])

                x_obs = state_to_embedding(state_t, embedding_dim, system)
                x_target = state_to_embedding(state_t1, embedding_dim, system)

                episode.append({
                    'graph': graph,
                    'x_obs': x_obs,
                    'x_target': x_target,  # Ground truth next state
                    'x_prev': episode[-1]['x_final'] if episode else None,
                    'state': state_t,
                    'state_next': state_t1,
                    'time': t,
                })
                episode[-1]['x_final'] = x_obs.clone()

        elif system == 'duffing':
            # Duffing oscillator - wider range for nonlinear effects
            x0 = np.random.uniform(-1.5, 1.5)
            v0 = np.random.uniform(-1.5, 1.5)

            x, v = simulate_duffing(x0, v0, steps_per_episode + 1)

            for t in range(steps_per_episode):
                state_t = np.array([x[t], v[t]])
                state_t1 = np.array([x[t+1], v[t+1]])

                x_obs = state_to_embedding(state_t, embedding_dim, system)
                x_target = state_to_embedding(state_t1, embedding_dim, system)

                episode.append({
                    'graph': graph,
                    'x_obs': x_obs,
                    'x_target': x_target,
                    'x_prev': episode[-1]['x_final'] if episode else None,
                    'state': state_t,
                    'state_next': state_t1,
                    'time': t,
                })
                episode[-1]['x_final'] = x_obs.clone()

        else:  # oscillator (linear)
            x0 = np.random.uniform(-2.0, 2.0)
            v0 = np.random.uniform(-2.0, 2.0)

            x, v = simulate_oscillator(x0, v0, steps_per_episode + 1)

            for t in range(steps_per_episode):
                state_t = np.array([x[t], v[t]])
                state_t1 = np.array([x[t+1], v[t+1]])

                x_obs = state_to_embedding(state_t, embedding_dim, system)
                x_target = state_to_embedding(state_t1, embedding_dim, system)

                episode.append({
                    'graph': graph,
                    'x_obs': x_obs,
                    'x_target': x_target,
                    'x_prev': episode[-1]['x_final'] if episode else None,
                    'state': state_t,
                    'state_next': state_t1,
                    'time': t,
                })
                episode[-1]['x_final'] = x_obs.clone()

        episodes.append(episode)

    return episodes


# ==============================================================================
# EVALUATION
# ==============================================================================

@dataclass
class PhysicsMetrics:
    """Metrics for physics prediction."""
    prediction_error: float = 0.0       # ||x_pred - x_target||
    state_error: float = 0.0            # Error in physical state
    energy_violation: float = 0.0       # Energy conservation error
    smoothness: float = 0.0             # Temporal smoothness
    constraint_violation: float = 0.0   # Cycle constraint violation
    convergence_rate: float = 0.0


def evaluate_physics_candidate(
    candidate: Candidate,
    episodes: List[List[Dict]],
    system: str = 'pendulum',
    use_delta: bool = False,
    delta_beta: float = 1.0,
) -> PhysicsMetrics:
    """Evaluate candidate on physics prediction task."""
    params = candidate.hyperparams.copy()
    params['use_delta_update'] = use_delta
    if use_delta:
        params['delta_beta'] = params.get('delta_beta', delta_beta)

    solver = create_solver_from_dict(params)

    all_pred_errors = []
    all_state_errors = []
    all_energy_viols = []
    all_smoothness = []
    all_constraint_viols = []
    all_converged = []

    for episode in episodes:
        x_prev = None
        prev_energy = None

        for step in episode:
            graph = step['graph']
            x_obs = step['x_obs']
            x_target = step['x_target']

            result = solver.solve(x_obs, graph, x_prev=x_prev)
            x_pred = result.x

            # Prediction error
            pred_error = torch.norm(x_pred - x_target).item()
            all_pred_errors.append(pred_error)

            # Constraint violation
            all_constraint_viols.append(result.breakdown.get('context', 0.0))
            all_converged.append(float(result.converged))

            # Smoothness (if we have previous prediction)
            if x_prev is not None:
                smoothness = torch.norm(x_pred - x_prev).item()
                all_smoothness.append(smoothness)

            # Energy conservation (approximate)
            state = step['state']
            if system == 'pendulum':
                current_energy = compute_energy_pendulum(
                    np.array([state[0]]), np.array([state[1]])
                )[0]
            elif system == 'duffing':
                current_energy = compute_energy_duffing(
                    np.array([state[0]]), np.array([state[1]])
                )[0]
            else:
                current_energy = compute_energy_oscillator(
                    np.array([state[0]]), np.array([state[1]])
                )[0]

            if prev_energy is not None:
                energy_viol = abs(current_energy - prev_energy) / (prev_energy + 1e-6)
                all_energy_viols.append(energy_viol)
            prev_energy = current_energy

            x_prev = x_pred.clone()
            step['x_final'] = x_pred

    return PhysicsMetrics(
        prediction_error=np.mean(all_pred_errors),
        energy_violation=np.mean(all_energy_viols) if all_energy_viols else 0.0,
        smoothness=np.mean(all_smoothness) if all_smoothness else 0.0,
        constraint_violation=np.mean(all_constraint_viols),
        convergence_rate=np.mean(all_converged),
    )


def compute_physics_fitness(metrics: PhysicsMetrics) -> float:
    """Compute fitness for physics task."""
    fitness = (
        - 10.0 * metrics.prediction_error
        - 5.0 * metrics.constraint_violation
        - 2.0 * metrics.energy_violation
        - 1.0 * metrics.smoothness
        + 0.5 * metrics.convergence_rate
    )
    return fitness


# ==============================================================================
# TRAINING
# ==============================================================================

@dataclass
class PhysicsTrainingResult:
    """Result of physics training."""
    mode: str
    system: str
    best_params: Dict[str, float]
    best_fitness: float
    best_metrics: Dict[str, float]
    fitness_history: List[float]
    generations: int
    total_time: float


def train_physics(
    system: str = 'pendulum',
    mode: str = 'euler',
    generations: int = 50,
    population_size: int = 16,
    num_episodes: int = 10,
    steps_per_episode: int = 50,
    embedding_dim: int = 32,
    optimize_beta: bool = False,
    delta_beta: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
) -> PhysicsTrainingResult:
    """Train on physics prediction task."""
    use_delta = (mode == 'delta')

    # Parameter bounds
    param_bounds = {
        "step_size": [0.001, 0.1],
        "steps": [5, 30],
        "projection_alpha": [0.5, 1.0],
        "lambda_context": [0.1, 10.0],
        "lambda_ricci": [0.01, 1.0],
        "lambda_smooth": [0.1, 5.0],  # Important for physics!
        "lambda_var": [0.1, 5.0],
    }

    if use_delta and optimize_beta:
        param_bounds["delta_beta"] = [0.1, 2.0]

    es_config = ESConfig(
        population_size=population_size,
        sigma=0.3,
        seed=seed,
        parameter_bounds=param_bounds,
    )

    trainer = CMAESTrainer(es_config)

    # Generate training data
    if verbose:
        print(f"Generating {system} episodes...")
    episodes = generate_physics_episodes(
        system=system,
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
        candidates = trainer.ask()
        fitnesses = []

        for candidate in candidates:
            beta = candidate.hyperparams.get('delta_beta', delta_beta) if use_delta else 1.0
            metrics = evaluate_physics_candidate(
                candidate, episodes,
                system=system,
                use_delta=use_delta,
                delta_beta=beta,
            )
            fitness = compute_physics_fitness(metrics)
            fitnesses.append(fitness)

            if fitness > best_fitness:
                best_fitness = fitness
                best_candidate = candidate
                best_metrics = metrics

        trainer.tell(candidates, fitnesses)

        gen_best = max(fitnesses)
        fitness_history.append(gen_best)

        if verbose and (gen % 10 == 0 or gen == generations - 1):
            print(f"[{mode.upper()}/{system.upper()}] Gen {gen:3d}: "
                  f"fitness={gen_best:.4f}, pred_err={best_metrics.prediction_error:.4f}")

    total_time = time.time() - start_time

    return PhysicsTrainingResult(
        mode=mode,
        system=system,
        best_params=best_candidate.hyperparams if best_candidate else {},
        best_fitness=best_fitness,
        best_metrics=asdict(best_metrics) if best_metrics else {},
        fitness_history=fitness_history,
        generations=generations,
        total_time=total_time,
    )


def save_physics_result(result: PhysicsTrainingResult, output_dir: str) -> str:
    """Save physics training result."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{result.system}_{result.mode}_{timestamp}.pt"
    filepath = os.path.join(output_dir, filename)

    data = asdict(result)
    data['timestamp'] = timestamp

    torch.save(data, filepath)
    print(f"Saved: {filepath}")

    # JSON summary
    json_path = filepath.replace('.pt', '.json')
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    return filepath


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Physics Prediction Benchmark")
    parser.add_argument('--system', choices=['pendulum', 'oscillator', 'duffing', 'all'],
                        default='duffing', help="Physical system (duffing=nonlinear)")
    parser.add_argument('--mode', choices=['euler', 'delta', 'both'],
                        default='both', help="Update mode")
    parser.add_argument('--generations', type=int, default=50)
    parser.add_argument('--population', type=int, default=16)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--optimize-beta', action='store_true')
    parser.add_argument('--delta-beta', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results/physics_benchmark')
    parser.add_argument('--test', action='store_true', help="Run generalization test")

    args = parser.parse_args()

    if args.system == 'all':
        systems = ['pendulum', 'duffing', 'oscillator']
    else:
        systems = [args.system]
    modes = ['euler', 'delta'] if args.mode == 'both' else [args.mode]

    print("=" * 70)
    print("PHYSICS PREDICTION BENCHMARK: Delta vs Euler")
    print("=" * 70)
    print(f"Systems: {systems}")
    print(f"Modes: {modes}")
    print(f"Generations: {args.generations}")
    print(f"Episodes: {args.episodes}, Steps: {args.steps}")
    print("=" * 70)

    all_results = {}

    for system in systems:
        print(f"\n{'='*70}")
        print(f"SYSTEM: {system.upper()}")
        print(f"{'='*70}")

        for mode in modes:
            print(f"\n>>> Training {mode.upper()} on {system}...")

            result = train_physics(
                system=system,
                mode=mode,
                generations=args.generations,
                population_size=args.population,
                num_episodes=args.episodes,
                steps_per_episode=args.steps,
                optimize_beta=args.optimize_beta,
                delta_beta=args.delta_beta,
                seed=args.seed,
            )

            save_physics_result(result, args.output)
            all_results[f"{system}_{mode}"] = result

    # Summary comparison
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("SUMMARY COMPARISON")
        print("=" * 70)
        print(f"{'System':<12} {'Mode':<8} {'Fitness':>12} {'Pred Err':>12} {'Energy Viol':>12}")
        print("-" * 70)

        for key, result in all_results.items():
            system, mode = key.rsplit('_', 1)
            print(f"{system:<12} {mode:<8} {result.best_fitness:>12.4f} "
                  f"{result.best_metrics['prediction_error']:>12.4f} "
                  f"{result.best_metrics['energy_violation']:>12.4f}")

        print("=" * 70)

        # Determine winner per system
        for system in systems:
            euler_key = f"{system}_euler"
            delta_key = f"{system}_delta"

            if euler_key in all_results and delta_key in all_results:
                euler_fit = all_results[euler_key].best_fitness
                delta_fit = all_results[delta_key].best_fitness

                if delta_fit > euler_fit * 1.05:
                    print(f"{system}: DELTA wins (+{(delta_fit/euler_fit - 1)*100:.1f}%)")
                elif euler_fit > delta_fit * 1.05:
                    print(f"{system}: EULER wins (+{(euler_fit/delta_fit - 1)*100:.1f}%)")
                else:
                    print(f"{system}: TIE (within 5%)")

    print("\nDone!")


if __name__ == '__main__':
    main()
