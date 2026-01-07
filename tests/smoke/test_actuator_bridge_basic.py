"""
Smoke test for ActuatorBridge.

Verifies basic actuator command handling functionality.

Reference: spec/20_impl_plan.ir.yml IMPL_011
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from hal.actuator_bridge.actuator_bridge import (
    ActuatorBridge,
    ActuatorBridgeConfig,
    create_default_actuator_bridge,
    create_mock_actuator_bridge,
    create_safe_actuator_bridge,
)
from onn.core.tensors import ActuatorCommand, ControlMode


class TestActuatorBridgeSmoke:
    """Smoke tests for ActuatorBridge."""

    def test_create_default_bridge(self):
        """Default actuator bridge should be created successfully."""
        bridge = create_default_actuator_bridge()
        assert bridge is not None
        assert bridge.config.mock_mode is True

    def test_send_position_command(self):
        """Should successfully send a position command."""
        bridge = create_mock_actuator_bridge(num_joints=6)

        command = ActuatorCommand(
            timestamp_ns=1000,
            mode=ControlMode.POSITION,
            command_values=np.zeros(6, dtype=np.float32)
        )

        success = bridge.send(command)
        assert success is True

    def test_send_velocity_command(self):
        """Should successfully send a velocity command."""
        bridge = create_mock_actuator_bridge(num_joints=6)

        command = ActuatorCommand(
            timestamp_ns=1000,
            mode=ControlMode.VELOCITY,
            command_values=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        )

        success = bridge.send(command)
        assert success is True

    def test_clamping_position(self):
        """Position values should be clamped to limits."""
        bridge = create_safe_actuator_bridge(num_joints=6, conservative_limits=True)

        # Command with values outside limits
        command = ActuatorCommand(
            timestamp_ns=1000,
            mode=ControlMode.POSITION,
            command_values=np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)
        )

        bridge.send(command)
        last_cmd = bridge.get_last_command()

        # Values should be clamped to [-π/2, π/2]
        assert np.all(last_cmd.command_values <= np.pi / 2 + 1e-6)

    def test_rejects_nan_values(self):
        """Should reject commands with NaN values."""
        bridge = create_mock_actuator_bridge(num_joints=6)

        command = ActuatorCommand(
            timestamp_ns=1000,
            mode=ControlMode.POSITION,
            command_values=np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )

        with pytest.raises(ValueError, match="NaN"):
            bridge.send(command)

    def test_rejects_wrong_size(self):
        """Should reject commands with wrong number of values."""
        bridge = create_mock_actuator_bridge(num_joints=6)

        command = ActuatorCommand(
            timestamp_ns=1000,
            mode=ControlMode.POSITION,
            command_values=np.zeros(4, dtype=np.float32)  # Wrong size
        )

        with pytest.raises(ValueError, match="Expected 6"):
            bridge.send(command)

    def test_command_count_increments(self):
        """Command count should increment with each successful send."""
        bridge = create_mock_actuator_bridge(num_joints=6)
        initial_count = bridge.get_command_count()

        for i in range(3):
            command = ActuatorCommand(
                timestamp_ns=1000 + i,
                mode=ControlMode.POSITION,
                command_values=np.zeros(6, dtype=np.float32)
            )
            bridge.send(command)
            assert bridge.get_command_count() == initial_count + i + 1

    def test_outputs_within_limits(self):
        """All output commands should be within configured limits."""
        bridge = create_safe_actuator_bridge(num_joints=6)

        # Try various commands
        for val in [0.0, 1.0, 5.0, -5.0]:
            command = ActuatorCommand(
                timestamp_ns=1000,
                mode=ControlMode.POSITION,
                command_values=np.full(6, val, dtype=np.float32)
            )
            bridge.send(command)

            last = bridge.get_last_command()
            limits = bridge.get_limits()["position"]

            for i, v in enumerate(last.command_values):
                lo, hi = limits[i]
                assert lo <= v <= hi, f"Value {v} not in [{lo}, {hi}]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
