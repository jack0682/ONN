"""Hyperparameter Schedules for ES Training.

This module provides schedules for dynamically adjusting hyperparameters
during ES training, such as:
- ρ schedule: soft → hard penalty strength
- K schedule: increase inner steps over time
- λ schedules: adjust loss weights

Reference:
    - spec/20_impl_plan.ir.yml: IMPL_021
    - User roadmap: "ρ schedule (soft→hard penalty)"

Author: Claude (via IMPL_021)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class ScheduleConfig:
    """Hyperparameter schedule configuration."""
    warmup_generations: int = 10    # Warmup period
    total_generations: int = 100    # Expected total generations
    
    # ρ (penalty strength) schedule
    rho_start: float = 0.1          # Initial soft penalty
    rho_end: float = 10.0           # Final hard penalty
    rho_schedule: str = "linear"    # "linear", "exponential", "cosine"
    
    # K (inner steps) schedule
    k_start: int = 5
    k_end: int = 20
    k_schedule: str = "linear"
    
    # η (step size) schedule
    eta_start: float = 0.01
    eta_end: float = 0.001
    eta_schedule: str = "cosine"
    
    # λ schedules (usually fixed, but can decay)
    lambda_context_decay: float = 1.0  # Multiply per generation
    lambda_ricci_decay: float = 1.0


# ==============================================================================
# SCHEDULE FUNCTIONS
# ==============================================================================

def linear_schedule(start: float, end: float, progress: float) -> float:
    """Linear interpolation between start and end.
    
    Args:
        start: Starting value
        end: Ending value
        progress: Progress in [0, 1]
        
    Returns:
        Interpolated value
    """
    return start + (end - start) * progress


def exponential_schedule(start: float, end: float, progress: float) -> float:
    """Exponential interpolation (log-linear).
    
    Args:
        start: Starting value (must be > 0)
        end: Ending value (must be > 0)
        progress: Progress in [0, 1]
        
    Returns:
        Interpolated value
    """
    if start <= 0 or end <= 0:
        raise ValueError("exponential_schedule requires positive values")
    
    log_start = np.log(start)
    log_end = np.log(end)
    return np.exp(log_start + (log_end - log_start) * progress)


def cosine_schedule(start: float, end: float, progress: float) -> float:
    """Cosine annealing schedule.
    
    Starts slow, accelerates in middle, slows at end.
    
    Args:
        start: Starting value
        end: Ending value
        progress: Progress in [0, 1]
        
    Returns:
        Interpolated value
    """
    return end + (start - end) * (1 + np.cos(np.pi * progress)) / 2


def step_schedule(start: float, end: float, progress: float, steps: int = 5) -> float:
    """Step schedule (piecewise constant).
    
    Args:
        start: Starting value
        end: Ending value
        progress: Progress in [0, 1]
        steps: Number of discrete steps
        
    Returns:
        Step value
    """
    step_idx = min(int(progress * steps), steps - 1)
    return linear_schedule(start, end, step_idx / (steps - 1))


def get_schedule_fn(name: str):
    """Get schedule function by name.
    
    Args:
        name: Schedule name ("linear", "exponential", "cosine", "step")
        
    Returns:
        Schedule function
    """
    schedules = {
        "linear": linear_schedule,
        "exponential": exponential_schedule,
        "cosine": cosine_schedule,
        "step": step_schedule,
    }
    
    if name not in schedules:
        logger.warning(f"Unknown schedule '{name}', using linear")
        return linear_schedule
    
    return schedules[name]


# ==============================================================================
# SCHEDULE APPLICATION
# ==============================================================================

def apply_schedule(
    params: Dict[str, float],
    generation: int,
    config: ScheduleConfig,
) -> Dict[str, float]:
    """Apply scheduled changes to solver parameters.
    
    Args:
        params: Current parameter dictionary
        generation: Current generation number
        config: Schedule configuration
        
    Returns:
        Updated parameter dictionary
    """
    # Compute progress (clamped to [0, 1])
    if generation < config.warmup_generations:
        progress = 0.0
    else:
        effective_gen = generation - config.warmup_generations
        effective_total = max(1, config.total_generations - config.warmup_generations)
        progress = min(1.0, effective_gen / effective_total)
    
    updated = params.copy()
    
    # Apply ρ schedule (maps to lambda_context in our implementation)
    rho_fn = get_schedule_fn(config.rho_schedule)
    updated["lambda_context"] = rho_fn(config.rho_start, config.rho_end, progress)
    
    # Apply K schedule
    k_fn = get_schedule_fn(config.k_schedule)
    updated["steps"] = int(round(k_fn(config.k_start, config.k_end, progress)))
    
    # Apply η schedule
    eta_fn = get_schedule_fn(config.eta_schedule)
    updated["step_size"] = eta_fn(config.eta_start, config.eta_end, progress)
    
    # Apply λ decays (if configured)
    if "lambda_context" in updated and config.lambda_context_decay != 1.0:
        updated["lambda_context"] *= config.lambda_context_decay ** generation
    
    if "lambda_ricci" in updated and config.lambda_ricci_decay != 1.0:
        updated["lambda_ricci"] *= config.lambda_ricci_decay ** generation
    
    logger.debug(
        f"Schedule gen={generation}: ρ={updated.get('lambda_context', 0):.3f}, "
        f"K={updated.get('steps', 0)}, η={updated.get('step_size', 0):.4f}"
    )
    
    return updated


def create_warmup_schedule(
    base_params: Dict[str, float],
    warmup_generations: int = 10,
) -> ScheduleConfig:
    """Create a schedule that warms up to the base parameters.
    
    Args:
        base_params: Target parameters after warmup
        warmup_generations: Warmup duration
        
    Returns:
        ScheduleConfig
    """
    return ScheduleConfig(
        warmup_generations=warmup_generations,
        rho_start=base_params.get("lambda_context", 1.0) * 0.1,
        rho_end=base_params.get("lambda_context", 1.0),
        k_start=max(1, int(base_params.get("steps", 10) * 0.5)),
        k_end=int(base_params.get("steps", 10)),
        eta_start=base_params.get("step_size", 0.01) * 10,
        eta_end=base_params.get("step_size", 0.01),
    )
