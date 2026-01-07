"""
Smoke test for SensorBridge.

Verifies basic sensor data acquisition functionality.

Reference: spec/20_impl_plan.ir.yml IMPL_010
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from hal.sensor_bridge.sensor_bridge import (
    SensorBridge,
    SensorBridgeConfig,
    create_default_sensor_bridge,
    create_mock_sensor_bridge,
)
from onn.core.tensors import SensorObservation, JointState


class TestSensorBridgeSmoke:
    """Smoke tests for SensorBridge."""

    def test_create_default_bridge(self):
        """Default sensor bridge should be created successfully."""
        bridge = create_default_sensor_bridge()
        assert bridge is not None
        assert bridge.config.mock_mode is True

    def test_acquire_returns_observation(self):
        """Acquire should return a valid SensorObservation."""
        bridge = create_mock_sensor_bridge()
        observation = bridge.acquire()

        assert isinstance(observation, SensorObservation)
        assert observation.timestamp_ns > 0
        assert observation.frame_id == "robot_base"

    def test_observation_has_images(self):
        """Observation should contain RGB images."""
        bridge = create_mock_sensor_bridge()
        observation = bridge.acquire()

        assert len(observation.rgb_images) > 0
        rgb = observation.rgb_images[0]
        assert rgb.shape[0] > 0  # Height
        assert rgb.shape[1] > 0  # Width
        assert rgb.shape[2] == 3  # RGB channels

    def test_observation_has_depth(self):
        """Observation should contain depth maps."""
        bridge = create_mock_sensor_bridge()
        observation = bridge.acquire()

        assert len(observation.depth_maps) > 0
        depth = observation.depth_maps[0]
        assert depth.shape[0] > 0
        assert depth.shape[1] > 0
        assert np.all(depth >= 0)  # Depth should be non-negative

    def test_observation_has_joint_state(self):
        """Observation should contain joint state."""
        bridge = create_mock_sensor_bridge(num_joints=6)
        observation = bridge.acquire()

        assert observation.joint_state is not None
        assert len(observation.joint_state.position) == 6
        assert len(observation.joint_state.velocity) == 6

    def test_timestamps_are_monotonic(self):
        """Timestamps should be monotonically increasing."""
        bridge = create_mock_sensor_bridge()

        observations = [bridge.acquire() for _ in range(5)]
        timestamps = [obs.timestamp_ns for obs in observations]

        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i-1], \
                f"Timestamp {i} is not monotonic"

    def test_acquisition_count_increments(self):
        """Acquisition count should increment with each acquire."""
        bridge = create_mock_sensor_bridge()
        initial_count = bridge.get_acquisition_count()

        for i in range(3):
            bridge.acquire()
            assert bridge.get_acquisition_count() == initial_count + i + 1

    def test_reset_clears_state(self):
        """Reset should clear bridge state."""
        bridge = create_mock_sensor_bridge()
        bridge.acquire()
        bridge.acquire()

        bridge.reset()

        assert bridge.get_acquisition_count() == 0
        assert bridge.get_last_observation() is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
