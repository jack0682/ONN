"""
SensorBridge - Hardware Abstraction Layer for Sensor Input.

Provides mock/simulated sensor data acquisition and time synchronization
for the CSA system.

Reference:
    - spec/20_impl_plan.ir.yml IMPL_010
    - spec/11_interfaces.ir.yml -> SensorObservation, JointState

Framework-agnostic: No ROS2, DDS, or external framework dependencies.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Callable
import logging
import time

from onn.core.tensors import (
    SensorObservation,
    JointState,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SensorBridgeConfig:
    """Configuration for the SensorBridge."""

    # Image dimensions
    image_width: int = 640
    image_height: int = 480
    num_cameras: int = 1

    # Joint configuration
    num_joints: int = 6

    # Frame ID
    frame_id: str = "robot_base"

    # Acquisition rate (Hz)
    acquisition_rate_hz: float = 30.0

    # Mock mode settings
    mock_mode: bool = True
    add_noise: bool = True
    noise_std: float = 0.01

    # Time synchronization tolerance (ns)
    sync_tolerance_ns: int = 10_000_000  # 10ms


# =============================================================================
# SensorBridge
# =============================================================================

class SensorBridge:
    """
    Hardware Abstraction Layer for sensor data acquisition.

    In mock mode, generates synthetic sensor data for testing.
    In production, this would interface with actual sensor drivers.

    Responsibilities:
    1. Acquire synchronized sensor data
    2. Apply time synchronization
    3. Validate data integrity

    Reference:
        - spec/20_impl_plan.ir.yml IMPL_010
    """

    def __init__(self, config: Optional[SensorBridgeConfig] = None):
        """
        Initialize the sensor bridge.

        Args:
            config: Bridge configuration. Uses defaults if None.
        """
        self.config = config or SensorBridgeConfig()

        # State
        self._last_observation: Optional[SensorObservation] = None
        self._last_timestamp_ns: int = 0
        self._acquisition_count: int = 0

        # Mock state for deterministic testing
        self._mock_joint_position = np.zeros(self.config.num_joints, dtype=np.float32)
        self._mock_phase: float = 0.0

    def acquire(self) -> SensorObservation:
        """
        Acquire sensor observation.

        Returns synchronized RGB-D images and joint state.

        Returns:
            SensorObservation with current sensor data

        Raises:
            RuntimeError: If acquisition fails (hardware mode only)
        """
        current_time_ns = time.time_ns()

        if self.config.mock_mode:
            observation = self._mock_observation(current_time_ns)
        else:
            observation = self._hardware_observation(current_time_ns)

        # Validate timestamp monotonicity
        if current_time_ns < self._last_timestamp_ns:
            logger.warning("Non-monotonic timestamp detected")

        self._last_timestamp_ns = current_time_ns
        self._last_observation = observation
        self._acquisition_count += 1

        logger.debug(f"Acquired observation #{self._acquisition_count}")
        return observation

    def _mock_observation(self, timestamp_ns: int) -> SensorObservation:
        """
        Generate mock sensor observation for testing.

        Creates synthetic RGB and depth images with simple patterns,
        and mock joint state with sinusoidal motion.

        Args:
            timestamp_ns: Timestamp for observation

        Returns:
            Mock SensorObservation
        """
        # Generate mock RGB image
        rgb_images = []
        for _ in range(self.config.num_cameras):
            rgb = self._generate_mock_rgb()
            rgb_images.append(rgb)

        # Generate mock depth map
        depth_maps = []
        for _ in range(self.config.num_cameras):
            depth = self._generate_mock_depth()
            depth_maps.append(depth)

        # Generate mock joint state
        joint_state = self._generate_mock_joint_state(timestamp_ns)

        return SensorObservation(
            timestamp_ns=timestamp_ns,
            frame_id=self.config.frame_id,
            rgb_images=rgb_images,
            depth_maps=depth_maps,
            joint_state=joint_state
        )

    def _generate_mock_rgb(self) -> np.ndarray:
        """Generate mock RGB image with gradient pattern."""
        h, w = self.config.image_height, self.config.image_width

        # Create gradient pattern
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # Red gradient (horizontal)
        rgb[:, :, 0] = np.linspace(0, 255, w).astype(np.uint8)

        # Green gradient (vertical)
        rgb[:, :, 1] = np.linspace(0, 255, h).reshape(-1, 1).astype(np.uint8)

        # Blue: phase-modulated pattern
        phase = self._mock_phase
        x = np.linspace(0, 4 * np.pi, w)
        pattern = ((np.sin(x + phase) + 1) * 127).astype(np.uint8)
        rgb[:, :, 2] = pattern

        # Add noise if configured
        if self.config.add_noise:
            noise = np.random.normal(0, 5, rgb.shape).astype(np.int16)
            rgb = np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return rgb

    def _generate_mock_depth(self) -> np.ndarray:
        """Generate mock depth map with spherical pattern."""
        h, w = self.config.image_height, self.config.image_width

        # Create depth pattern (distance from center)
        y, x = np.ogrid[:h, :w]
        cx, cy = w // 2, h // 2
        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        # Normalize to depth range (0.5 to 5.0 meters)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        depth = 0.5 + 4.5 * (distance / max_dist)

        # Add variation based on phase
        depth += 0.2 * np.sin(self._mock_phase)

        # Add noise if configured
        if self.config.add_noise:
            noise = np.random.normal(0, self.config.noise_std, depth.shape)
            depth = depth + noise

        return depth.astype(np.float32)

    def _generate_mock_joint_state(self, timestamp_ns: int) -> JointState:
        """Generate mock joint state with sinusoidal motion."""
        num_joints = self.config.num_joints

        # Update phase for time-varying motion
        dt = 1.0 / self.config.acquisition_rate_hz
        self._mock_phase += dt * 0.5  # Slow oscillation

        # Generate positions (sinusoidal)
        positions = np.zeros(num_joints, dtype=np.float32)
        for i in range(num_joints):
            freq = 0.2 + 0.1 * i  # Different frequency per joint
            positions[i] = 0.5 * np.sin(self._mock_phase * freq + i * 0.5)

        # Compute velocities (derivative)
        velocities = np.zeros(num_joints, dtype=np.float32)
        for i in range(num_joints):
            freq = 0.2 + 0.1 * i
            velocities[i] = 0.5 * freq * np.cos(self._mock_phase * freq + i * 0.5)

        # Efforts (zero for mock)
        efforts = np.zeros(num_joints, dtype=np.float32)

        # Add noise if configured
        if self.config.add_noise:
            positions += np.random.normal(0, self.config.noise_std, num_joints).astype(np.float32)
            velocities += np.random.normal(0, self.config.noise_std, num_joints).astype(np.float32)

        self._mock_joint_position = positions

        return JointState(
            position=positions,
            velocity=velocities,
            effort=efforts
        )

    def _hardware_observation(self, timestamp_ns: int) -> SensorObservation:
        """
        Acquire observation from actual hardware.

        This is a placeholder for hardware integration.
        In production, this would interface with camera drivers,
        depth sensors, and robot joint encoders.

        Args:
            timestamp_ns: Timestamp for observation

        Returns:
            SensorObservation from hardware
        """
        # TODO: Implement actual hardware acquisition
        raise NotImplementedError(
            "Hardware acquisition not implemented. "
            "Set mock_mode=True for testing."
        )

    def get_last_observation(self) -> Optional[SensorObservation]:
        """Get the most recent observation."""
        return self._last_observation

    def get_acquisition_count(self) -> int:
        """Get total number of acquisitions."""
        return self._acquisition_count

    def is_synchronized(self, tolerance_ns: Optional[int] = None) -> bool:
        """
        Check if sensors are synchronized within tolerance.

        Args:
            tolerance_ns: Synchronization tolerance in nanoseconds.
                         Uses config default if None.

        Returns:
            True if synchronized, False otherwise
        """
        if tolerance_ns is None:
            tolerance_ns = self.config.sync_tolerance_ns

        if self._last_observation is None:
            return False

        # Check timestamp is recent
        current_ns = time.time_ns()
        age_ns = current_ns - self._last_observation.timestamp_ns

        return age_ns < tolerance_ns

    def reset(self) -> None:
        """Reset bridge state."""
        self._last_observation = None
        self._last_timestamp_ns = 0
        self._acquisition_count = 0
        self._mock_joint_position = np.zeros(self.config.num_joints, dtype=np.float32)
        self._mock_phase = 0.0


# =============================================================================
# Factory Functions
# =============================================================================

def create_default_sensor_bridge() -> SensorBridge:
    """Create sensor bridge with default configuration."""
    return SensorBridge(SensorBridgeConfig())


def create_mock_sensor_bridge(
    num_joints: int = 6,
    image_size: tuple = (480, 640)
) -> SensorBridge:
    """
    Create mock sensor bridge for testing.

    Args:
        num_joints: Number of robot joints
        image_size: (height, width) of images

    Returns:
        SensorBridge configured for mock operation
    """
    config = SensorBridgeConfig(
        image_height=image_size[0],
        image_width=image_size[1],
        num_joints=num_joints,
        mock_mode=True,
        add_noise=True
    )
    return SensorBridge(config)
