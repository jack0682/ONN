"""Hybrid ES + Gradient Descent Trainer.

Phase 1 Implementation:
- ES: Optimizes 7 solver hyperparameters (stable, low-dimensional)
- GD: Trains W_lin with surrogate loss (no solver unroll needed)

The surrogate loss operates directly on x_obs without backprop through solver:
- L_cycle: Cycle consistency
- L_smooth: Temporal smoothness
- L_collapse: Anti-collapse (variance)
- L_contrastive: Contrastive learning signal

Philosophy:
    "ES는 탐색, GD는 학습" - 각 알고리즘이 잘하는 것에 집중

Reference:
    - Phase 2 will add K-step unroll for end-to-end W_lin gradients

Author: Claude (Hybrid ES+GD)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from onn.es.ask_tell import (
    ESConfig,
    CMAESTrainer,
    Candidate,
    create_default_trainer,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# SURROGATE LOSS FOR W_LIN (No Solver Unroll)
# ==============================================================================

@dataclass
class SurrogateLossConfig:
    """Configuration for W_lin surrogate loss.

    These losses operate directly on x_obs = encode(phi) without
    backpropagating through the solver.
    """
    # Loss weights
    lambda_cycle: float = 1.0       # Cycle consistency: C @ x ≈ 0
    lambda_smooth: float = 0.5      # Temporal smoothness: ||x_t - x_{t-1}||
    lambda_collapse: float = 1.0    # Anti-collapse: variance(x) > threshold
    lambda_contrastive: float = 0.3 # Contrastive: similar relations → similar embeddings
    lambda_reconstruction: float = 0.5  # Reconstruction: decode(x) ≈ phi

    # Thresholds
    min_variance: float = 0.01      # Minimum variance for anti-collapse
    contrastive_temperature: float = 0.1  # Temperature for InfoNCE

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


class LearnableEncoder(nn.Module):
    """Learnable W_lin encoder for surrogate loss training.

    x_obs = α * normalize(W_rp @ φ) + β * W_lin @ φ + γ * W_geo @ ξ

    Only W_lin is learnable; W_rp and W_geo are fixed.
    """

    def __init__(
        self,
        input_dim: int = 64,
        output_dim: int = 32,
        geometric_dim: int = 6,
        alpha: float = 0.3,
        beta: float = 0.5,
        gamma: float = 0.2,
        random_seed: int = 42,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Fixed random projection
        rng = np.random.RandomState(random_seed)
        W_rp = rng.randn(output_dim, input_dim) / np.sqrt(input_dim)
        self.register_buffer('W_rp', torch.tensor(W_rp, dtype=torch.float32))

        # Learnable linear (this is what GD optimizes)
        self.W_lin = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.xavier_normal_(self.W_lin.weight)

        # Fixed geometric projection
        W_geo = rng.randn(output_dim, geometric_dim) / np.sqrt(geometric_dim)
        self.register_buffer('W_geo', torch.tensor(W_geo, dtype=torch.float32))

        logger.info(
            f"LearnableEncoder: input={input_dim}, output={output_dim}, "
            f"W_lin params={input_dim * output_dim}"
        )

    def forward(
        self,
        phi: torch.Tensor,
        xi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode pair features to edge embeddings.

        Args:
            phi: Pair features (batch, input_dim)
            xi: Geometric features (batch, 6) - optional

        Returns:
            x_obs: Edge embeddings (batch, output_dim)
        """
        # Random projection (fixed)
        x_rp = torch.mm(phi, self.W_rp.t())
        x_rp = torch.nn.functional.normalize(x_rp, dim=1)

        # Learnable linear
        x_lin = self.W_lin(phi)

        # Geometric (if available)
        if xi is not None:
            x_geo = torch.mm(xi, self.W_geo.t())
        else:
            x_geo = torch.zeros_like(x_lin)

        # Combine
        x_obs = self.alpha * x_rp + self.beta * x_lin + self.gamma * x_geo

        return x_obs

    def get_w_lin(self) -> np.ndarray:
        """Get W_lin as numpy array for ES candidate."""
        return self.W_lin.weight.detach().cpu().numpy()

    def set_w_lin(self, w_lin: np.ndarray):
        """Set W_lin from numpy array."""
        with torch.no_grad():
            self.W_lin.weight.copy_(torch.tensor(w_lin, dtype=torch.float32))


class SurrogateLoss(nn.Module):
    """Surrogate loss for W_lin training without solver unroll.

    Computes losses directly on x_obs:
    L_total = λ_cycle * L_cycle + λ_smooth * L_smooth +
              λ_collapse * L_collapse + λ_contrastive * L_contrastive
    """

    def __init__(self, config: SurrogateLossConfig):
        super().__init__()
        self.config = config

    def cycle_loss(
        self,
        x_obs: torch.Tensor,
        cycle_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Cycle consistency loss: C @ x should be close to zero.

        Args:
            x_obs: Edge embeddings (m, p)
            cycle_matrix: Cycle basis matrix (k, m)

        Returns:
            L_cycle: Mean squared cycle violation
        """
        if cycle_matrix is None or cycle_matrix.shape[0] == 0:
            return torch.tensor(0.0, device=x_obs.device)

        # C @ x for each dimension
        # cycle_matrix: (k, m), x_obs: (m, p) -> (k, p)
        cycle_sum = torch.mm(cycle_matrix, x_obs)

        # Mean squared violation
        loss = torch.mean(cycle_sum ** 2)
        return loss

    def smoothness_loss(
        self,
        x_obs_t: torch.Tensor,
        x_obs_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Temporal smoothness loss.

        Args:
            x_obs_t: Current embeddings (m, p)
            x_obs_prev: Previous embeddings (m, p)

        Returns:
            L_smooth: Mean squared difference
        """
        if x_obs_prev is None:
            return torch.tensor(0.0, device=x_obs_t.device)

        diff = x_obs_t - x_obs_prev
        loss = torch.mean(diff ** 2)
        return loss

    def collapse_loss(self, x_obs: torch.Tensor) -> torch.Tensor:
        """Anti-collapse loss: maintain minimum variance.

        Args:
            x_obs: Edge embeddings (m, p)

        Returns:
            L_collapse: Penalty for low variance
        """
        variance = torch.var(x_obs, dim=0).mean()

        # Penalize when variance falls below threshold
        loss = torch.relu(self.config.min_variance - variance)
        return loss

    def contrastive_loss(
        self,
        x_obs: torch.Tensor,
        positive_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> torch.Tensor:
        """Contrastive loss: similar relations should have similar embeddings.

        Uses InfoNCE-style loss if positive pairs provided,
        otherwise uses self-similarity regularization.

        Args:
            x_obs: Edge embeddings (m, p)
            positive_pairs: List of (i, j) indices that should be similar

        Returns:
            L_contrastive: Contrastive loss
        """
        # Check for valid input
        if x_obs.shape[0] < 2:
            return torch.tensor(0.0, device=x_obs.device)

        # Check for zero/nan embeddings
        norms = torch.norm(x_obs, dim=1, keepdim=True)
        if (norms < 1e-8).any() or torch.isnan(x_obs).any():
            return torch.tensor(0.0, device=x_obs.device)

        if positive_pairs is None or len(positive_pairs) == 0:
            # Self-similarity: encourage diverse embeddings
            x_norm = torch.nn.functional.normalize(x_obs, dim=1, eps=1e-8)
            sim_matrix = torch.mm(x_norm, x_norm.t())

            # Off-diagonal should be low
            mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=x_obs.device)
            off_diag = sim_matrix[mask]

            # Penalize high similarity
            loss = torch.mean(torch.relu(off_diag - 0.5))
            return loss

        # InfoNCE loss for positive pairs
        temperature = self.config.contrastive_temperature
        x_norm = torch.nn.functional.normalize(x_obs, dim=1, eps=1e-8)

        total_loss = 0.0
        for i, j in positive_pairs:
            # Positive similarity
            pos_sim = torch.sum(x_norm[i] * x_norm[j]) / temperature

            # All similarities (negatives)
            all_sim = torch.mm(x_norm[i:i+1], x_norm.t()) / temperature

            # InfoNCE
            loss_ij = -pos_sim + torch.logsumexp(all_sim, dim=1)
            total_loss = total_loss + loss_ij

        return total_loss / len(positive_pairs)

    def reconstruction_loss(
        self,
        x_obs: torch.Tensor,
        phi: torch.Tensor,
        decoder: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Reconstruction loss: encourage information preservation.

        Args:
            x_obs: Edge embeddings (m, p)
            phi: Original pair features (m, d)
            decoder: Optional decoder network

        Returns:
            L_recon: Reconstruction loss
        """
        if decoder is None:
            # Simple variance-based loss (safer than corrcoef which can produce NaN)
            if x_obs.shape[0] > 1:
                # Encourage x_obs to preserve variance from phi
                x_var = torch.var(x_obs, dim=0).mean()
                phi_var = torch.var(phi, dim=0).mean()

                # Penalize if x has much lower variance than phi
                var_ratio = x_var / (phi_var + 1e-8)
                loss = torch.relu(0.5 - var_ratio)  # Want var_ratio >= 0.5
            else:
                loss = torch.tensor(0.0, device=x_obs.device)

            return loss

        # Use decoder
        phi_recon = decoder(x_obs)
        loss = torch.mean((phi_recon - phi) ** 2)
        return loss

    def forward(
        self,
        x_obs: torch.Tensor,
        phi: torch.Tensor,
        cycle_matrix: Optional[torch.Tensor] = None,
        x_obs_prev: Optional[torch.Tensor] = None,
        positive_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total surrogate loss.

        Args:
            x_obs: Edge embeddings (m, p)
            phi: Original pair features (m, d)
            cycle_matrix: Cycle basis matrix (k, m)
            x_obs_prev: Previous timestep embeddings
            positive_pairs: Positive pairs for contrastive loss

        Returns:
            total_loss: Combined loss
            breakdown: Individual loss values
        """
        c = self.config

        l_cycle = self.cycle_loss(x_obs, cycle_matrix)
        l_smooth = self.smoothness_loss(x_obs, x_obs_prev)
        l_collapse = self.collapse_loss(x_obs)
        l_contrastive = self.contrastive_loss(x_obs, positive_pairs)
        l_recon = self.reconstruction_loss(x_obs, phi)

        total = (
            c.lambda_cycle * l_cycle +
            c.lambda_smooth * l_smooth +
            c.lambda_collapse * l_collapse +
            c.lambda_contrastive * l_contrastive +
            c.lambda_reconstruction * l_recon
        )

        breakdown = {
            'cycle': l_cycle.item(),
            'smooth': l_smooth.item(),
            'collapse': l_collapse.item(),
            'contrastive': l_contrastive.item(),
            'reconstruction': l_recon.item(),
            'total': total.item(),
        }

        return total, breakdown


# ==============================================================================
# HYBRID TRAINER
# ==============================================================================

@dataclass
class HybridTrainerConfig:
    """Configuration for Hybrid ES + GD trainer."""
    # ES config (hyperparameters only)
    es_population_size: int = 16
    es_sigma: float = 0.3
    es_seed: int = 42

    # GD config (W_lin)
    gd_learning_rate: float = 1e-3
    gd_weight_decay: float = 1e-4
    gd_steps_per_es_gen: int = 10  # GD steps per ES generation

    # Encoder config
    input_dim: int = 64
    output_dim: int = 32
    alpha: float = 0.3  # Random projection weight
    beta: float = 0.5   # Learnable linear weight
    gamma: float = 0.2  # Geometric weight

    # Surrogate loss
    surrogate_config: SurrogateLossConfig = field(
        default_factory=SurrogateLossConfig
    )


class HybridTrainer:
    """Hybrid ES + GD trainer.

    ES optimizes 7 solver hyperparameters.
    GD optimizes W_lin using surrogate loss.

    Usage:
        >>> trainer = HybridTrainer(config)
        >>> # Each generation:
        >>> hp_candidates = trainer.ask_hyperparams()
        >>> # Evaluate with your fitness function
        >>> trainer.tell_hyperparams(hp_candidates, fitnesses)
        >>> # Update W_lin with surrogate loss
        >>> trainer.update_w_lin(episodes)
    """

    def __init__(self, config: HybridTrainerConfig):
        self.config = config

        # ES for hyperparameters only (7 params)
        es_config = ESConfig(
            population_size=config.es_population_size,
            sigma=config.es_sigma,
            seed=config.es_seed,
            optimize_w_lin=False,  # ES does NOT optimize W_lin
        )
        self.es_trainer = CMAESTrainer(es_config)

        # Learnable encoder (GD optimizes this)
        self.encoder = LearnableEncoder(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
        )

        # GD optimizer for W_lin
        self.optimizer = optim.AdamW(
            self.encoder.parameters(),
            lr=config.gd_learning_rate,
            weight_decay=config.gd_weight_decay,
        )

        # Surrogate loss
        self.surrogate_loss = SurrogateLoss(config.surrogate_config)

        # State
        self.generation = 0
        self.gd_step = 0
        self.loss_history: List[Dict[str, float]] = []

        logger.info(
            f"HybridTrainer initialized: "
            f"ES={self.es_trainer.dim} params, "
            f"GD={sum(p.numel() for p in self.encoder.parameters())} params"
        )

    def ask_hyperparams(self, n: Optional[int] = None) -> List[Candidate]:
        """Sample hyperparameter candidates from ES.

        Returns:
            List of Candidate with only hyperparams (no W_lin)
        """
        candidates = self.es_trainer.ask(n)

        # Attach current W_lin to all candidates for evaluation
        w_lin = self.encoder.get_w_lin()
        for c in candidates:
            c.w_lin = w_lin

        return candidates

    def tell_hyperparams(
        self,
        candidates: List[Candidate],
        fitnesses: List[float],
    ):
        """Update ES with fitness values.

        Args:
            candidates: Candidates from ask_hyperparams()
            fitnesses: Fitness values (higher = better)
        """
        # Remove W_lin before telling ES (it only tracks hyperparams)
        for c in candidates:
            c.w_lin = None
            c._raw_params = None

        self.es_trainer.tell(candidates, fitnesses)
        self.generation += 1

    def update_w_lin(
        self,
        batch_phi: torch.Tensor,
        batch_xi: Optional[torch.Tensor] = None,
        cycle_matrix: Optional[torch.Tensor] = None,
        x_obs_prev: Optional[torch.Tensor] = None,
        positive_pairs: Optional[List[Tuple[int, int]]] = None,
        num_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """Update W_lin using surrogate loss (GD).

        Args:
            batch_phi: Pair features (batch, input_dim)
            batch_xi: Geometric features (batch, 6)
            cycle_matrix: Cycle basis matrix
            x_obs_prev: Previous embeddings
            positive_pairs: Positive pairs for contrastive loss
            num_steps: Number of GD steps (default: gd_steps_per_es_gen)

        Returns:
            Loss breakdown dictionary
        """
        if num_steps is None:
            num_steps = self.config.gd_steps_per_es_gen

        self.encoder.train()
        total_breakdown = {}

        for _ in range(num_steps):
            self.optimizer.zero_grad()

            # Forward
            x_obs = self.encoder(batch_phi, batch_xi)

            # Compute surrogate loss
            loss, breakdown = self.surrogate_loss(
                x_obs=x_obs,
                phi=batch_phi,
                cycle_matrix=cycle_matrix,
                x_obs_prev=x_obs_prev,
                positive_pairs=positive_pairs,
            )

            # Backward
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)

            # Update
            self.optimizer.step()

            self.gd_step += 1

            # Accumulate for reporting
            for k, v in breakdown.items():
                total_breakdown[k] = total_breakdown.get(k, 0) + v / num_steps

        self.loss_history.append(total_breakdown)
        self.encoder.eval()

        return total_breakdown

    def get_best_hyperparams(self) -> Dict[str, float]:
        """Get best hyperparameters found by ES."""
        if self.es_trainer.best_candidate:
            return self.es_trainer.best_candidate.hyperparams
        return {}

    def get_w_lin(self) -> np.ndarray:
        """Get current W_lin."""
        return self.encoder.get_w_lin()

    def encode(
        self,
        phi: torch.Tensor,
        xi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode using current W_lin."""
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(phi, xi)

    @property
    def state(self):
        """ES state (for compatibility)."""
        return self.es_trainer.state


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

def create_hybrid_trainer(
    input_dim: int = 64,
    output_dim: int = 32,
    es_population: int = 16,
    gd_lr: float = 1e-3,
    gd_steps: int = 10,
) -> HybridTrainer:
    """Create a hybrid ES + GD trainer.

    Args:
        input_dim: Pair feature dimension
        output_dim: Embedding dimension
        es_population: ES population size
        gd_lr: Learning rate for W_lin
        gd_steps: GD steps per ES generation

    Returns:
        Configured HybridTrainer

    Example:
        >>> trainer = create_hybrid_trainer()
        >>> candidates = trainer.ask_hyperparams()
        >>> fitnesses = [evaluate(c) for c in candidates]
        >>> trainer.tell_hyperparams(candidates, fitnesses)
        >>> trainer.update_w_lin(batch_phi)
    """
    config = HybridTrainerConfig(
        es_population_size=es_population,
        gd_learning_rate=gd_lr,
        gd_steps_per_es_gen=gd_steps,
        input_dim=input_dim,
        output_dim=output_dim,
    )
    return HybridTrainer(config)


def create_surrogate_loss(
    lambda_cycle: float = 1.0,
    lambda_smooth: float = 0.5,
    lambda_collapse: float = 1.0,
    lambda_contrastive: float = 0.3,
) -> SurrogateLoss:
    """Create surrogate loss for W_lin training."""
    config = SurrogateLossConfig(
        lambda_cycle=lambda_cycle,
        lambda_smooth=lambda_smooth,
        lambda_collapse=lambda_collapse,
        lambda_contrastive=lambda_contrastive,
    )
    return SurrogateLoss(config)
