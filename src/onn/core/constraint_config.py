"""
Constraint Configuration for LOGOS Solver.

Provides loading and merging of constraint configuration from YAML files
and programmatic overrides.

Reference:
    - spec/20_impl_plan.ir.yml IMPL_003
    - spec/02_onn_math_spec.md Section 4.2 (Hyperparameters)
    - config/constraint_defaults.yaml

Framework-agnostic: No ROS2, DDS, or external framework dependencies.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class WeightsConfig:
    """
    Loss function weights from spec/02_onn_math_spec.md Section 4.2.

    L_total = λ_data * L_data + λ_phys * L_phys + λ_logic * L_logic
    """
    lambda_data: float = 1.0   # Data fidelity weight
    lambda_phys: float = 10.0  # Physical validity weight
    lambda_logic: float = 2.0  # Logical consistency weight


@dataclass
class SolverConfig:
    """Solver parameters from spec/02_onn_math_spec.md Section 5."""
    max_iterations: int = 10      # Max iterations per solve
    learning_rate: float = 0.01   # Gradient descent step size (η)
    tolerance: float = 1e-6       # Convergence tolerance
    device: str = "cpu"           # PyTorch device


@dataclass
class PhysicsConfig:
    """Physical constraint parameters from spec/02_onn_math_spec.md Section 3.1."""
    overlap_threshold: float = 0.5  # Threshold for L_phys
    min_separation: float = 0.01    # Minimum separation (meters)


@dataclass
class PruningConfig:
    """Edge pruning parameters."""
    edge_prune_threshold: float = 0.1  # Min effective strength
    min_probability: float = 0.1       # Min edge probability


@dataclass
class TopologyConfig:
    """Topological constraint parameters from spec/02_onn_math_spec.md Section 3.3."""
    max_weighted_degree: float = 10.0  # Σ_j w_ij ≤ K_max
    max_edges: int = 1000              # Maximum edges in graph


@dataclass
class StabilityConfig:
    """Stability parameters from spec/02_onn_math_spec.md Section 5.2."""
    gamma_onn: float = 0.5       # ONN Lipschitz constant
    gamma_ortsf: float = 0.5     # ORTSF Lipschitz constant
    max_velocity_norm: float = 10.0  # Max velocity in tangent space


@dataclass
class ConstraintConfigFull:
    """
    Complete constraint configuration aggregating all subsections.

    This is the preferred structure for passing complete configurations.
    """
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested dictionary."""
        return {
            "weights": asdict(self.weights),
            "solver": asdict(self.solver),
            "physics": asdict(self.physics),
            "pruning": asdict(self.pruning),
            "topology": asdict(self.topology),
            "stability": asdict(self.stability),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConstraintConfigFull:
        """Create from nested dictionary."""
        return cls(
            weights=WeightsConfig(**data.get("weights", {})),
            solver=SolverConfig(**data.get("solver", {})),
            physics=PhysicsConfig(**data.get("physics", {})),
            pruning=PruningConfig(**data.get("pruning", {})),
            topology=TopologyConfig(**data.get("topology", {})),
            stability=StabilityConfig(**data.get("stability", {})),
        )


# =============================================================================
# Default Configuration Path
# =============================================================================

def get_default_config_path() -> Path:
    """
    Get path to default constraint configuration file.

    Searches in order:
    1. CSA_CONSTRAINT_CONFIG environment variable
    2. config/constraint_defaults.yaml relative to project root
    3. Package-relative path
    """
    # Check environment variable
    env_path = os.environ.get("CSA_CONSTRAINT_CONFIG")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # Try relative to current working directory
    cwd_path = Path("config/constraint_defaults.yaml")
    if cwd_path.exists():
        return cwd_path

    # Try relative to this file (src/onn/core/ -> config/)
    this_file = Path(__file__)
    project_root = this_file.parent.parent.parent.parent
    project_path = project_root / "config" / "constraint_defaults.yaml"
    if project_path.exists():
        return project_path

    # Return default (may not exist)
    return Path("config/constraint_defaults.yaml")


# =============================================================================
# Loading Functions
# =============================================================================

def load_constraint_config(
    path: Optional[Union[str, Path]] = None,
    use_defaults_on_error: bool = True
) -> ConstraintConfigFull:
    """
    Load constraint configuration from YAML file.

    Args:
        path: Path to YAML config file. Uses default if None.
        use_defaults_on_error: If True, return defaults on load error.
                               If False, raise exception.

    Returns:
        ConstraintConfigFull with loaded values

    Raises:
        FileNotFoundError: If file not found and use_defaults_on_error=False
        ValueError: If YAML parsing fails and use_defaults_on_error=False

    Example:
        >>> config = load_constraint_config()
        >>> config.weights.lambda_phys
        10.0
    """
    if path is None:
        path = get_default_config_path()
    else:
        path = Path(path)

    # Try to load YAML
    try:
        import yaml

        if not path.exists():
            logger.warning(f"Config file not found: {path}. Using defaults.")
            if use_defaults_on_error:
                return ConstraintConfigFull()
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        logger.info(f"Loaded constraint config from {path}")
        return ConstraintConfigFull.from_dict(data)

    except ImportError:
        logger.warning("PyYAML not installed. Using default config.")
        if use_defaults_on_error:
            return ConstraintConfigFull()
        raise

    except Exception as e:
        logger.error(f"Error loading config from {path}: {e}")
        if use_defaults_on_error:
            return ConstraintConfigFull()
        raise


def load_constraint_config_dict(
    path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Load constraint configuration as a flat dictionary.

    Useful for backward compatibility with code expecting dict format.

    Args:
        path: Path to YAML config file. Uses default if None.

    Returns:
        Dictionary with all configuration values
    """
    config = load_constraint_config(path)
    return config.to_dict()


# =============================================================================
# Merging Functions
# =============================================================================

def merge_constraint_config(
    base: ConstraintConfigFull,
    overrides: Dict[str, Any]
) -> ConstraintConfigFull:
    """
    Merge override values into a base configuration.

    Overrides can be nested (e.g., {"weights": {"lambda_phys": 5.0}})
    or flat (e.g., {"lambda_phys": 5.0} - will update weights.lambda_phys).

    Args:
        base: Base configuration to start from
        overrides: Dictionary of values to override

    Returns:
        New ConstraintConfigFull with merged values

    Example:
        >>> base = load_constraint_config()
        >>> merged = merge_constraint_config(base, {"weights": {"lambda_phys": 20.0}})
        >>> merged.weights.lambda_phys
        20.0
    """
    # Convert base to dict for merging
    base_dict = base.to_dict()

    # Merge overrides
    merged = _deep_merge(base_dict, overrides)

    # Also handle flat overrides (e.g., "lambda_phys" -> weights.lambda_phys)
    flat_mappings = {
        "lambda_data": ("weights", "lambda_data"),
        "lambda_phys": ("weights", "lambda_phys"),
        "lambda_logic": ("weights", "lambda_logic"),
        "max_iterations": ("solver", "max_iterations"),
        "learning_rate": ("solver", "learning_rate"),
        "tolerance": ("solver", "tolerance"),
        "overlap_threshold": ("physics", "overlap_threshold"),
        "edge_prune_threshold": ("pruning", "edge_prune_threshold"),
    }

    for key, (section, subkey) in flat_mappings.items():
        if key in overrides and section in merged:
            merged[section][subkey] = overrides[key]

    return ConstraintConfigFull.from_dict(merged)


def merge_from_environment(
    config: ConstraintConfigFull
) -> ConstraintConfigFull:
    """
    Merge environment variable overrides into configuration.

    Supported environment variables:
    - CSA_LAMBDA_DATA: weights.lambda_data
    - CSA_LAMBDA_PHYS: weights.lambda_phys
    - CSA_LAMBDA_LOGIC: weights.lambda_logic
    - CSA_MAX_ITERATIONS: solver.max_iterations
    - CSA_LEARNING_RATE: solver.learning_rate
    - CSA_DEVICE: solver.device

    Args:
        config: Base configuration

    Returns:
        New configuration with environment overrides applied
    """
    overrides: Dict[str, Any] = {}

    env_mappings = {
        "CSA_LAMBDA_DATA": ("weights", "lambda_data", float),
        "CSA_LAMBDA_PHYS": ("weights", "lambda_phys", float),
        "CSA_LAMBDA_LOGIC": ("weights", "lambda_logic", float),
        "CSA_MAX_ITERATIONS": ("solver", "max_iterations", int),
        "CSA_LEARNING_RATE": ("solver", "learning_rate", float),
        "CSA_TOLERANCE": ("solver", "tolerance", float),
        "CSA_DEVICE": ("solver", "device", str),
        "CSA_OVERLAP_THRESHOLD": ("physics", "overlap_threshold", float),
    }

    for env_var, (section, key, converter) in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            try:
                if section not in overrides:
                    overrides[section] = {}
                overrides[section][key] = converter(value)
                logger.debug(f"Applied env override: {env_var}={value}")
            except ValueError as e:
                logger.warning(f"Invalid value for {env_var}: {value}. Error: {e}")

    if overrides:
        return merge_constraint_config(config, overrides)
    return config


# =============================================================================
# Validation
# =============================================================================

def validate_constraint_config(config: ConstraintConfigFull) -> list[str]:
    """
    Validate constraint configuration values.

    Args:
        config: Configuration to validate

    Returns:
        List of validation error messages. Empty if valid.
    """
    errors: list[str] = []

    # Weights must be non-negative
    if config.weights.lambda_data < 0:
        errors.append("weights.lambda_data must be non-negative")
    if config.weights.lambda_phys < 0:
        errors.append("weights.lambda_phys must be non-negative")
    if config.weights.lambda_logic < 0:
        errors.append("weights.lambda_logic must be non-negative")

    # Solver parameters
    if config.solver.max_iterations <= 0:
        errors.append("solver.max_iterations must be positive")
    if config.solver.learning_rate <= 0:
        errors.append("solver.learning_rate must be positive")
    if config.solver.tolerance <= 0:
        errors.append("solver.tolerance must be positive")

    # Physics parameters
    if not 0 <= config.physics.overlap_threshold <= 1:
        errors.append("physics.overlap_threshold must be in [0, 1]")

    # Small-gain theorem
    if config.stability.gamma_onn * config.stability.gamma_ortsf >= 1:
        errors.append(
            f"Small-gain theorem violated: γ_ONN * γ_ORTSF = "
            f"{config.stability.gamma_onn * config.stability.gamma_ortsf} >= 1"
        )

    return errors


# =============================================================================
# Helper Functions
# =============================================================================

def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()

    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def get_default_constraint_config() -> ConstraintConfigFull:
    """
    Get default constraint configuration.

    Attempts to load from file, falls back to hardcoded defaults.
    Environment variable overrides are applied.

    Returns:
        ConstraintConfigFull with defaults
    """
    config = load_constraint_config(use_defaults_on_error=True)
    config = merge_from_environment(config)
    return config


# =============================================================================
# Convenience: Convert to LOGOS Config format
# =============================================================================

def to_logos_config(config: ConstraintConfigFull):
    """
    Convert ConstraintConfigFull to LOGOSConfig format.

    This bridges between the generic constraint config and the
    specific LOGOS solver config.

    Args:
        config: Full constraint configuration

    Returns:
        LOGOSConfig instance (imported from onn.ops.logos_solver)
    """
    # Delay import to avoid circular dependency
    from onn.ops.logos_solver import LOGOSConfig

    return LOGOSConfig(
        lambda_data=config.weights.lambda_data,
        lambda_phys=config.weights.lambda_phys,
        lambda_logic=config.weights.lambda_logic,
        overlap_threshold=config.physics.overlap_threshold,
        max_iterations=config.solver.max_iterations,
        learning_rate=config.solver.learning_rate,
        tolerance=config.solver.tolerance,
        edge_prune_threshold=config.pruning.edge_prune_threshold,
        device=config.solver.device,
    )
