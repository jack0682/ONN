"""
Unit tests for constraint configuration.

Reference: spec/20_impl_plan.ir.yml IMPL_003
"""

import pytest
import numpy as np
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# Check if PyYAML is available
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from onn.core.constraint_config import (
    WeightsConfig,
    SolverConfig,
    PhysicsConfig,
    ConstraintConfigFull,
    load_constraint_config,
    merge_constraint_config,
    validate_constraint_config,
    get_default_constraint_config,
)


class TestWeightsConfig:
    """Tests for WeightsConfig dataclass."""

    def test_default_values(self):
        """Defaults should match spec/02_onn_math_spec.md Section 4.2."""
        config = WeightsConfig()

        assert config.lambda_data == 1.0
        assert config.lambda_phys == 10.0
        assert config.lambda_logic == 2.0

    def test_custom_values(self):
        """Should accept custom values."""
        config = WeightsConfig(lambda_data=2.0, lambda_phys=5.0, lambda_logic=3.0)

        assert config.lambda_data == 2.0
        assert config.lambda_phys == 5.0


class TestSolverConfig:
    """Tests for SolverConfig dataclass."""

    def test_default_values(self):
        """Defaults should be reasonable."""
        config = SolverConfig()

        assert config.max_iterations == 10
        assert config.learning_rate == 0.01
        assert config.tolerance > 0


class TestConstraintConfigFull:
    """Tests for ConstraintConfigFull dataclass."""

    def test_default_creation(self):
        """Should create with all default sub-configs."""
        config = ConstraintConfigFull()

        assert config.weights is not None
        assert config.solver is not None
        assert config.physics is not None

    def test_to_dict(self):
        """to_dict should produce nested dictionary."""
        config = ConstraintConfigFull()
        d = config.to_dict()

        assert "weights" in d
        assert "lambda_phys" in d["weights"]
        assert d["weights"]["lambda_phys"] == 10.0

    def test_from_dict(self):
        """from_dict should reconstruct config."""
        data = {
            "weights": {"lambda_data": 2.0},
            "solver": {"max_iterations": 20}
        }
        config = ConstraintConfigFull.from_dict(data)

        assert config.weights.lambda_data == 2.0
        assert config.solver.max_iterations == 20


class TestLoadConstraintConfig:
    """Tests for load_constraint_config function."""

    def test_load_returns_config(self):
        """Should return a ConstraintConfigFull."""
        config = load_constraint_config()
        assert isinstance(config, ConstraintConfigFull)

    def test_load_missing_file_uses_defaults(self):
        """Missing file with use_defaults_on_error=True should use defaults."""
        config = load_constraint_config(
            path="/nonexistent/path.yaml",
            use_defaults_on_error=True
        )
        assert config.weights.lambda_phys == 10.0

    @pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
    def test_load_missing_file_raises(self):
        """Missing file with use_defaults_on_error=False should raise."""
        with pytest.raises(FileNotFoundError):
            load_constraint_config(
                path="/nonexistent/path.yaml",
                use_defaults_on_error=False
            )

    @pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
    def test_load_from_yaml_file(self):
        """Should load from actual YAML file."""
        yaml_content = """
weights:
  lambda_data: 5.0
  lambda_phys: 20.0
solver:
  max_iterations: 15
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = load_constraint_config(path=f.name)

            assert config.weights.lambda_data == 5.0
            assert config.weights.lambda_phys == 20.0
            assert config.solver.max_iterations == 15

        os.unlink(f.name)


class TestMergeConstraintConfig:
    """Tests for merge_constraint_config function."""

    def test_merge_nested_override(self):
        """Should merge nested overrides."""
        base = ConstraintConfigFull()
        overrides = {"weights": {"lambda_phys": 20.0}}

        merged = merge_constraint_config(base, overrides)

        assert merged.weights.lambda_phys == 20.0
        # Other values unchanged
        assert merged.weights.lambda_data == 1.0

    def test_merge_flat_override(self):
        """Should handle flat overrides."""
        base = ConstraintConfigFull()
        overrides = {"lambda_phys": 15.0}  # Flat key

        merged = merge_constraint_config(base, overrides)

        assert merged.weights.lambda_phys == 15.0

    def test_merge_multiple_sections(self):
        """Should merge across multiple sections."""
        base = ConstraintConfigFull()
        overrides = {
            "weights": {"lambda_data": 2.0},
            "solver": {"learning_rate": 0.05}
        }

        merged = merge_constraint_config(base, overrides)

        assert merged.weights.lambda_data == 2.0
        assert merged.solver.learning_rate == 0.05


class TestValidateConstraintConfig:
    """Tests for validate_constraint_config function."""

    def test_valid_config_no_errors(self):
        """Valid config should return empty list."""
        config = ConstraintConfigFull()
        errors = validate_constraint_config(config)

        assert errors == []

    def test_negative_weight_error(self):
        """Negative weights should produce error."""
        config = ConstraintConfigFull(
            weights=WeightsConfig(lambda_data=-1.0)
        )
        errors = validate_constraint_config(config)

        assert len(errors) > 0
        assert any("lambda_data" in e for e in errors)

    def test_invalid_solver_params(self):
        """Invalid solver params should produce errors."""
        config = ConstraintConfigFull(
            solver=SolverConfig(max_iterations=0)
        )
        errors = validate_constraint_config(config)

        assert any("max_iterations" in e for e in errors)

    def test_invalid_overlap_threshold(self):
        """Overlap threshold outside [0,1] should produce error."""
        config = ConstraintConfigFull(
            physics=PhysicsConfig(overlap_threshold=1.5)
        )
        errors = validate_constraint_config(config)

        assert any("overlap_threshold" in e for e in errors)


class TestGetDefaultConfig:
    """Tests for get_default_constraint_config function."""

    def test_returns_valid_config(self):
        """Should return a valid configuration."""
        config = get_default_constraint_config()

        assert isinstance(config, ConstraintConfigFull)
        errors = validate_constraint_config(config)
        assert errors == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
