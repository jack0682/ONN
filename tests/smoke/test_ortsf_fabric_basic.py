"""
Smoke test for ORTSF Fabric.

Verifies basic delay-robust control functionality.

Reference: spec/20_impl_plan.ir.yml IMPL_009
"""

import pytest
import numpy as np
import sys
import os
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from ortsf.fabric.ortsf_fabric import (
    ORTSFFabric,
    ORTSFConfig,
    create_default_ortsf_fabric,
    create_realtime_ortsf_fabric,
)
from onn.core.tensors import (
    ReasoningTrace,
    SensorObservation,
    ActuatorCommand,
    ControlMode,
    SemanticNode,
    JointState,
    BOUND_TENSOR_DIM,
)


@pytest.fixture
def mock_observation() -> SensorObservation:
    """Create a mock sensor observation."""
    return SensorObservation(
        timestamp_ns=time.time_ns(),
        frame_id="robot_base",
        rgb_images=[np.zeros((480, 640, 3), dtype=np.uint8)],
        depth_maps=[np.ones((480, 640), dtype=np.float32)],
        joint_state=JointState(
            position=np.zeros(6, dtype=np.float32),
            velocity=np.zeros(6, dtype=np.float32),
            effort=np.zeros(6, dtype=np.float32)
        )
    )


@pytest.fixture
def mock_trace() -> ReasoningTrace:
    """Create a mock reasoning trace."""
    current_time = time.time_ns()

    target_node = SemanticNode(
        node_id=1,
        bound_tensor=np.ones(BOUND_TENSOR_DIM, dtype=np.float32) * 0.5,
    )

    # Create cubic trajectory coefficients
    # 16 dims * 4 coefficients = 64 values
    coeffs = np.zeros((16, 4), dtype=np.float32)
    coeffs[:, 0] = 0.0  # Start position
    coeffs[:, 3] = 0.5  # End position (via cubic)

    return ReasoningTrace(
        timestamp_ns=current_time,
        target_state=[target_node],
        trajectory_coeffs=coeffs.flatten(),
        curvature=0.5,
        valid_until_ns=current_time + 500_000_000  # 500ms
    )


class TestORTSFSmoke:
    """Smoke tests for ORTSF Fabric."""

    def test_create_default_fabric(self):
        """Default fabric should be created successfully."""
        fabric = create_default_ortsf_fabric()
        assert fabric is not None
        assert fabric.config is not None

    def test_step_returns_command(self, mock_observation, mock_trace):
        """Step should return an ActuatorCommand."""
        fabric = create_default_ortsf_fabric()
        command = fabric.step(mock_trace, mock_observation)

        assert isinstance(command, ActuatorCommand)
        assert command.timestamp_ns > 0

    def test_command_has_valid_mode(self, mock_observation, mock_trace):
        """Command should have a valid control mode."""
        fabric = create_default_ortsf_fabric()
        command = fabric.step(mock_trace, mock_observation)

        assert isinstance(command.mode, ControlMode)

    def test_command_values_are_finite(self, mock_observation, mock_trace):
        """Command values should be finite."""
        fabric = create_default_ortsf_fabric()
        command = fabric.step(mock_trace, mock_observation)

        assert np.all(np.isfinite(command.command_values))

    def test_step_without_trace(self, mock_observation):
        """Step should work without a trace (returns safe command)."""
        fabric = create_default_ortsf_fabric()
        command = fabric.step(None, mock_observation)

        assert command is not None
        assert np.all(np.isfinite(command.command_values))

    def test_realtime_fabric(self, mock_observation, mock_trace):
        """Realtime fabric should produce valid commands."""
        fabric = create_realtime_ortsf_fabric(num_joints=6)
        command = fabric.step(mock_trace, mock_observation)

        assert command is not None
        assert len(command.command_values) == 6

    def test_multiple_steps(self, mock_observation, mock_trace):
        """Multiple steps should work without error."""
        fabric = create_default_ortsf_fabric()

        for _ in range(10):
            command = fabric.step(mock_trace, mock_observation)
            assert command is not None

    def test_delay_estimate(self, mock_observation, mock_trace):
        """Delay estimate should be reasonable."""
        fabric = create_default_ortsf_fabric()
        fabric.step(mock_trace, mock_observation)

        delay_ns = fabric.get_estimated_delay_ns()
        assert delay_ns >= 0
        assert delay_ns < 1_000_000_000  # Less than 1 second

    def test_reset(self, mock_observation, mock_trace):
        """Reset should clear controller state."""
        fabric = create_default_ortsf_fabric()
        fabric.step(mock_trace, mock_observation)

        fabric.reset()

        # After reset, there should be no trace
        assert fabric._current_trace is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
