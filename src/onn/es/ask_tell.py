"""CMA-ES Ask/Tell Implementation.

This module implements the CMA-ES (Covariance Matrix Adaptation Evolution
Strategy) algorithm for optimizing ONN solver hyperparameters.

CMA-ES is well-suited for:
- Low-to-medium dimensional optimization (10-50 parameters)
- Non-convex, noisy objective functions
- Black-box optimization (no gradient required)

Reference:
    - spec/11_interfaces.ir.yml: ESConfig, ESReport
    - spec/20_impl_plan.ir.yml: IMPL_020

Author: Claude (via IMPL_020)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================


@dataclass
class ESConfig:
    """Evolutionary Strategy configuration (CMA-ES).

    Reference: spec/11_interfaces.ir.yml -> ESConfig
    """

    algorithm: str = "CMA-ES"
    population_size: int = 16  # Number of candidates per generation
    sigma: float = 0.3  # Initial step size
    elite_fraction: float = 0.5  # Fraction of top candidates to use
    seed: int = 42  # Random seed
    max_generations: int = 100  # Maximum generations

    # Parameter bounds: {param_name: [min, max]}
    # Note: Add 'delta_beta' to enable ES search over delta update strength
    parameter_bounds: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "beta_obs": [0.1, 10.0],
            "beta_cons": [0.1, 10.0],
            "step_size": [0.001, 0.1],
            "steps": [5, 50],
            "gate_threshold": [0.1, 0.9],
            "uncertainty_damping": [0.0, 1.0],
        }
    )

    # Delta update options (for ES-based delta integration)
    use_delta_update: bool = False  # Enable delta ODE in solver
    # If True and 'delta_beta' not in parameter_bounds, uses fixed delta_beta below
    delta_beta: float = 1.0  # Fixed β when not optimized by ES

    # Neural network parameter optimization (W_lin)
    optimize_w_lin: bool = False  # Whether to optimize W_lin
    w_lin_shape: Optional[Tuple[int, int]] = None  # (output_dim, input_dim)
    w_lin_bounds: Tuple[float, float] = (-1.0, 1.0)  # Min/max for W_lin elements
    w_lin_sigma: float = 0.1  # Initial sigma for W_lin (typically smaller)

    # Low-rank factorization for W_lin (NEW - reduces parameter count)
    # W_lin = U @ V where U is (output_dim, rank), V is (rank, input_dim)
    # Set rank > 0 to enable. Total params = output_dim*rank + rank*input_dim
    w_lin_rank: int = 0  # 0 = full rank, >0 = low-rank factorization


@dataclass
class CMAESState:
    """CMA-ES optimizer state.

    Stores the current state of the CMA-ES algorithm including
    mean, covariance, and evolution paths.
    """

    mean: np.ndarray  # Current mean (θ)
    sigma: float  # Current step size
    C: np.ndarray  # Covariance matrix
    pc: np.ndarray  # Evolution path for C
    ps: np.ndarray  # Evolution path for σ
    generation: int = 0

    # CMA-ES constants (computed from dimension)
    dim: int = 0
    mu: int = 0  # Number of parents
    weights: Optional[np.ndarray] = None
    mueff: float = 0.0  # Variance effective selection mass
    cc: float = 0.0  # Learning rate for cumulation
    cs: float = 0.0  # Learning rate for σ path
    c1: float = 0.0  # Learning rate for rank-1 update
    cmu: float = 0.0  # Learning rate for rank-μ update
    damps: float = 0.0  # Damping for σ
    chiN: float = 0.0  # Expected norm of N(0,I)


@dataclass
class Candidate:
    """A candidate solution containing hyperparameters and optional W_lin.

    This is the return type of CMAESTrainer.ask() when W_lin optimization is enabled.
    """

    hyperparams: Dict[str, float]  # Solver hyperparameters
    w_lin: Optional[np.ndarray] = None  # W_lin matrix (output_dim, input_dim)
    _raw_params: Optional[np.ndarray] = (
        None  # Raw parameters before reconstruction (internal)
    )


# ==============================================================================
# CMA-ES TRAINER
# ==============================================================================


class CMAESTrainer:
    """CMA-ES trainer for ONN hyperparameter and neural parameter optimization.

    Implements the ask-tell interface for evolutionary optimization.
    Supports optimization of both solver hyperparameters and RelationEncoder W_lin.

    Example:
        >>> config = ESConfig(optimize_w_lin=True, w_lin_shape=(32, 64))
        >>> trainer = CMAESTrainer(config)
        >>> candidates = trainer.ask(n=16)
        >>> fitnesses = [evaluate(c) for c in candidates]
        >>> trainer.tell(candidates, fitnesses)

    Reference:
        - Hansen, N. (2016). The CMA Evolution Strategy: A Tutorial.
        - spec/20_impl_plan.ir.yml: IMPL_020
    """

    def __init__(self, config: ESConfig):
        """Initialize the CMA-ES trainer.

        Args:
            config: ES configuration
        """
        self.config = config
        self.rng = np.random.RandomState(config.seed)

        # Get parameter names and bounds for hyperparameters
        self.param_names = list(config.parameter_bounds.keys())
        self.bounds = config.parameter_bounds
        self.hyperparam_dim = len(self.param_names)

        # W_lin dimensions
        self.optimize_w_lin = config.optimize_w_lin
        self.w_lin_rank = config.w_lin_rank
        if self.optimize_w_lin:
            if config.w_lin_shape is None:
                raise ValueError("w_lin_shape required when optimize_w_lin=True")
            self.w_lin_shape = config.w_lin_shape
            out_dim, in_dim = config.w_lin_shape

            # Low-rank factorization: W = U @ V
            if self.w_lin_rank > 0:
                rank = min(self.w_lin_rank, out_dim, in_dim)
                self.w_lin_dim = out_dim * rank + rank * in_dim
                self._u_shape = (out_dim, rank)
                self._v_shape = (rank, in_dim)
                logger.info(
                    f"Low-rank W_lin: rank={rank}, params={self.w_lin_dim} "
                    f"(vs full={out_dim * in_dim})"
                )
            else:
                self.w_lin_dim = out_dim * in_dim
                self._u_shape = None
                self._v_shape = None
        else:
            self.w_lin_shape = None
            self.w_lin_dim = 0
            self._u_shape = None
            self._v_shape = None

        # Total dimension
        self.dim = self.hyperparam_dim + self.w_lin_dim

        # Initialize state
        self.state = self._init_state()

        # History tracking
        self.best_fitness = float("-inf")
        self.best_candidate: Optional[Candidate] = None
        self.best_params: Optional[Dict[str, float]] = None  # Legacy compatibility
        self.fitness_history: List[float] = []

        logger.info(
            f"CMA-ES initialized: {self.hyperparam_dim} hyperparams + "
            f"{self.w_lin_dim} W_lin params = {self.dim} total, σ={config.sigma}"
        )

    def _init_state(self) -> CMAESState:
        """Initialize CMA-ES state with proper constants."""
        dim = self.dim

        # Initialize mean: hyperparams at center of bounds, W_lin at zero
        hyperparam_mean = np.array(
            [(self.bounds[p][0] + self.bounds[p][1]) / 2 for p in self.param_names]
        )

        if self.optimize_w_lin:
            w_lin_mean = np.zeros(self.w_lin_dim)
            mean = np.concatenate([hyperparam_mean, w_lin_mean])
        else:
            mean = hyperparam_mean

        # Initialize covariance as identity
        C = np.eye(dim)

        # Evolution paths
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # CMA-ES constants (following Hansen's tutorial)
        n = dim
        lambda_ = self.config.population_size
        mu = int(lambda_ * self.config.elite_fraction)

        # Recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / (weights**2).sum()

        # Learning rates
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

        # Expected norm of N(0, I)
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

        return CMAESState(
            mean=mean,
            sigma=self.config.sigma,
            C=C,
            pc=pc,
            ps=ps,
            generation=0,
            dim=dim,
            mu=mu,
            weights=weights,
            mueff=mueff,
            cc=cc,
            cs=cs,
            c1=c1,
            cmu=cmu,
            damps=damps,
            chiN=chiN,
        )

    def ask(self, n: Optional[int] = None) -> List[Candidate]:
        """Sample n candidate parameter vectors.

        Args:
            n: Number of candidates. Default: population_size.

        Returns:
            List of Candidate objects (hyperparams + optional W_lin)
        """
        if n is None:
            n = self.config.population_size

        state = self.state

        # Sample from N(mean, σ² * C)
        # Eigendecomposition for sampling
        D, B = np.linalg.eigh(state.C)
        D = np.sqrt(np.maximum(D, 1e-20))

        candidates = []
        for _ in range(n):
            z = self.rng.randn(state.dim)
            y = B @ (D * z)  # y ~ N(0, C)
            x = state.mean + state.sigma * y

            # Extract hyperparameters (first hyperparam_dim elements)
            params = {}
            for i, name in enumerate(self.param_names):
                lo, hi = self.bounds[name]
                val = np.clip(x[i], lo, hi)

                # Integer parameters
                if name == "steps":
                    val = int(round(val))

                params[name] = float(val)

            # Extract W_lin (remaining elements)
            w_lin = None
            raw_params = None
            if self.optimize_w_lin:
                w_lin_flat = x[self.hyperparam_dim :]
                lo, hi = self.config.w_lin_bounds
                w_lin_flat = np.clip(w_lin_flat, lo, hi)
                raw_params = w_lin_flat.copy()  # Store raw for tell()

                # Low-rank reconstruction: W = U @ V
                if self._u_shape is not None:
                    u_size = self._u_shape[0] * self._u_shape[1]
                    U = w_lin_flat[:u_size].reshape(self._u_shape)
                    V = w_lin_flat[u_size:].reshape(self._v_shape)
                    w_lin = U @ V  # Reconstruct full W_lin
                else:
                    w_lin = w_lin_flat.reshape(self.w_lin_shape)

            candidates.append(
                Candidate(hyperparams=params, w_lin=w_lin, _raw_params=raw_params)
            )

        return candidates

    def tell(
        self,
        candidates: List[Candidate],
        fitnesses: List[float],
    ) -> None:
        """Update mean/covariance from fitness ranking.

        Args:
            candidates: List of Candidate objects (from ask)
            fitnesses: Fitness values for each candidate (higher = better)
        """
        state = self.state
        n = len(candidates)
        mu = state.mu

        # Convert candidates to numpy (full vectors including W_lin)
        xs_list = []
        for c in candidates:
            # Hyperparameters
            hp_vec = np.array([c.hyperparams[name] for name in self.param_names])

            # W_lin (if optimizing) - use raw params for low-rank consistency
            if self.optimize_w_lin and c._raw_params is not None:
                full_vec = np.concatenate([hp_vec, c._raw_params])
            elif self.optimize_w_lin and c.w_lin is not None:
                # Fallback for legacy candidates without _raw_params
                w_lin_flat = c.w_lin.flatten()
                full_vec = np.concatenate([hp_vec, w_lin_flat])
            else:
                full_vec = hp_vec

            xs_list.append(full_vec)

        xs = np.array(xs_list)

        # Sort by fitness (descending)
        indices = np.argsort(fitnesses)[::-1]
        xs_sorted = xs[indices]
        fitnesses_sorted = [fitnesses[i] for i in indices]

        # Track best
        if fitnesses_sorted[0] > self.best_fitness:
            self.best_fitness = fitnesses_sorted[0]
            self.best_candidate = candidates[indices[0]]
            self.best_params = candidates[indices[0]].hyperparams  # Legacy
            logger.info(f"New best fitness: {self.best_fitness:.4f}")

        self.fitness_history.append(fitnesses_sorted[0])

        # Select elite
        xs_elite = xs_sorted[:mu]

        # Weighted recombination
        old_mean = state.mean.copy()
        state.mean = np.sum(state.weights[:, None] * xs_elite, axis=0)

        # Evolution path for σ
        Cinvsqrt = np.linalg.inv(np.linalg.cholesky(state.C + 1e-8 * np.eye(state.dim)))
        state.ps = (1 - state.cs) * state.ps + np.sqrt(
            state.cs * (2 - state.cs) * state.mueff
        ) * Cinvsqrt @ (state.mean - old_mean) / state.sigma

        # Heaviside function for stalling
        hsig = np.linalg.norm(state.ps) / np.sqrt(
            1 - (1 - state.cs) ** (2 * (state.generation + 1))
        ) / state.chiN < 1.4 + 2 / (state.dim + 1)

        # Evolution path for C
        state.pc = (1 - state.cc) * state.pc + hsig * np.sqrt(
            state.cc * (2 - state.cc) * state.mueff
        ) * (state.mean - old_mean) / state.sigma

        # Covariance matrix update
        artmp = (xs_elite - old_mean) / state.sigma  # (mu, dim)
        state.C = (
            (1 - state.c1 - state.cmu) * state.C
            + state.c1
            * (
                np.outer(state.pc, state.pc)
                + (1 - hsig) * state.cc * (2 - state.cc) * state.C
            )
            + state.cmu * (artmp.T @ np.diag(state.weights) @ artmp)
        )

        # Symmetrize C
        state.C = (state.C + state.C.T) / 2

        # Step size update
        state.sigma *= np.exp(
            (state.cs / state.damps) * (np.linalg.norm(state.ps) / state.chiN - 1)
        )

        state.generation += 1
        logger.debug(
            f"Generation {state.generation}: best={fitnesses_sorted[0]:.4f}, σ={state.sigma:.4f}"
        )

    def get_best(self) -> Tuple[Candidate, float]:
        """Get the best candidate found so far.

        Returns:
            Tuple of (best_candidate, best_fitness)
        """
        if self.best_candidate is None:
            return Candidate(hyperparams={}, w_lin=None), self.best_fitness
        return self.best_candidate, self.best_fitness

    def get_best_hyperparams(self) -> Tuple[Dict[str, float], float]:
        """Get the best hyperparameters found so far (legacy compatibility).

        Returns:
            Tuple of (best_hyperparams, best_fitness)
        """
        return self.best_params or {}, self.best_fitness

    def get_report(self) -> Dict:
        """Get a report of the current state.

        Returns:
            Dictionary with generation, best_fitness, best_params, etc.
        """
        report = {
            "generation": self.state.generation,
            "best_fitness": self.best_fitness,
            "best_params": self.best_params or {},
            "sigma": self.state.sigma,
            "optimize_w_lin": self.optimize_w_lin,
        }

        if self.optimize_w_lin and self.best_candidate is not None:
            report["has_best_w_lin"] = self.best_candidate.w_lin is not None
            if self.best_candidate.w_lin is not None:
                report["w_lin_norm"] = float(np.linalg.norm(self.best_candidate.w_lin))

        return report

    def reset(self) -> None:
        """Reset the trainer to initial state."""
        self.state = self._init_state()
        self.best_fitness = float("-inf")
        self.best_candidate = None
        self.best_params = None
        self.fitness_history = []


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================


def create_default_trainer() -> CMAESTrainer:
    """Create a CMA-ES trainer with default configuration.

    Returns:
        Configured CMAESTrainer
    """
    return CMAESTrainer(ESConfig())


def create_trainer_for_params(
    param_bounds: Dict[str, List[float]],
    population_size: int = 16,
    sigma: float = 0.3,
) -> CMAESTrainer:
    """Create a trainer for specific parameters.

    Args:
        param_bounds: Parameter bounds dictionary
        population_size: Population size
        sigma: Initial step size

    Returns:
        Configured CMAESTrainer
    """
    config = ESConfig(
        parameter_bounds=param_bounds,
        population_size=population_size,
        sigma=sigma,
    )
    return CMAESTrainer(config)


def create_trainer_with_w_lin(
    w_lin_shape: Tuple[int, int],
    param_bounds: Optional[Dict[str, List[float]]] = None,
    population_size: int = 16,
    sigma: float = 0.3,
    w_lin_sigma: float = 0.1,
    w_lin_bounds: Tuple[float, float] = (-1.0, 1.0),
    w_lin_rank: int = 0,
    seed: int = 42,
) -> CMAESTrainer:
    """Create a trainer that optimizes both hyperparameters and W_lin.

    Args:
        w_lin_shape: Shape of W_lin matrix (output_dim, input_dim)
        param_bounds: Hyperparameter bounds dictionary (uses defaults if None)
        population_size: Population size
        sigma: Initial step size for hyperparameters
        w_lin_sigma: Initial step size for W_lin (typically smaller)
        w_lin_bounds: Min/max for W_lin elements
        w_lin_rank: Low-rank factorization rank (0=full, >0=low-rank)
            With rank=8 for 32x64 W_lin: 768 params vs 2048 full
        seed: Random seed

    Returns:
        Configured CMAESTrainer for joint optimization

    Example:
        >>> # Full W_lin (2048 params for 32x64)
        >>> trainer = create_trainer_with_w_lin(w_lin_shape=(32, 64))
        >>> # Low-rank W_lin (768 params for rank=8)
        >>> trainer = create_trainer_with_w_lin(w_lin_shape=(32, 64), w_lin_rank=8)
    """
    config = ESConfig(
        population_size=population_size,
        sigma=sigma,
        seed=seed,
        optimize_w_lin=True,
        w_lin_shape=w_lin_shape,
        w_lin_bounds=w_lin_bounds,
        w_lin_sigma=w_lin_sigma,
        w_lin_rank=w_lin_rank,
    )

    if param_bounds is not None:
        config.parameter_bounds = param_bounds

    return CMAESTrainer(config)
