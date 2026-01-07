"""
ActuatorBridge - Hardware Abstraction Layer for Actuator Output.

Provides clamping, validation, and forwarding of ActuatorCommands
to the robot hardware.

Reference:
    - spec/20_impl_plan.ir.yml IMPL_011
    - spec/11_interfaces.ir.yml -> ActuatorCommand

Framework-agnostic: No ROS2, DDS, or external framework dependencies.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Tuple
import logging
import time

from onn.core.tensors import (
    ActuatorCommand,
    ControlMode,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ActuatorBridgeConfig:
    """Configuration for the ActuatorBridge."""

    # Number of joints/actuators
    num_joints: int = 6

    # Position limits (per joint)
    position_limits: Optional[List[Tuple[float, float]]] = None

    # Velocity limits (per joint, absolute)
    velocity_limits: Optional[List[float]] = None

    # Torque/effort limits (per joint, absolute)
    torque_limits: Optional[List[float]] = None

    # Default limits if per-joint not specified
    default_position_limit: Tuple[float, float] = (-np.pi, np.pi)
    default_velocity_limit: float = 2.0  # rad/s
    default_torque_limit: float = 100.0  # Nm

    # Command rate limiting
    max_command_rate_hz: float = 1000.0  # Max 1kHz

    # Validation
    check_nan: bool = True
    check_limits: bool = True

    # Mock mode
    mock_mode: bool = True


# =============================================================================
# ActuatorBridge
# =============================================================================

class ActuatorBridge:
    """
    Hardware Abstraction Layer for actuator commands.

    Responsibilities:
    1. Validate incoming commands
    2. Apply safety clamping
    3. Forward to hardware (or mock)
    4. Track command history

    Reference:
        - spec/20_impl_plan.ir.yml IMPL_011
    """

    def __init__(self, config: Optional[ActuatorBridgeConfig] = None):
        """
        Initialize the actuator bridge.

        Args:
            config: Bridge configuration. Uses defaults if None.
        """
        self.config = config or ActuatorBridgeConfig()

        # Initialize limits
        self._position_limits = self._init_position_limits()
        self._velocity_limits = self._init_velocity_limits()
        self._torque_limits = self._init_torque_limits()

        # State
        self._last_command: Optional[ActuatorCommand] = None
        self._last_command_time_ns: int = 0
        self._command_count: int = 0
        self._clamped_count: int = 0

        # Callbacks for hardware integration
        self._send_callback: Optional[Callable[[ActuatorCommand], bool]] = None

    def _init_position_limits(self) -> List[Tuple[float, float]]:
        """Initialize position limits for each joint."""
        if self.config.position_limits:
            return self.config.position_limits

        return [self.config.default_position_limit] * self.config.num_joints

    def _init_velocity_limits(self) -> List[float]:
        """Initialize velocity limits for each joint."""
        if self.config.velocity_limits:
            return self.config.velocity_limits

        return [self.config.default_velocity_limit] * self.config.num_joints

    def _init_torque_limits(self) -> List[float]:
        """Initialize torque limits for each joint."""
        if self.config.torque_limits:
            return self.config.torque_limits

        return [self.config.default_torque_limit] * self.config.num_joints

    def send(self, command: ActuatorCommand) -> bool:
        """
        Send actuator command to hardware.

        Validates, clamps, and forwards the command.

        Args:
            command: ActuatorCommand to send

        Returns:
            True if command was sent successfully, False otherwise

        Raises:
            ValueError: If command is invalid and cannot be clamped
        """
        current_time_ns = time.time_ns()

        # Rate limiting
        if not self._check_rate_limit(current_time_ns):
            logger.warning("Command rate limit exceeded, dropping command")
            return False

        # Validate command
        validation_result = self._validate_command(command)
        if not validation_result[0]:
            logger.error(f"Command validation failed: {validation_result[1]}")
            raise ValueError(f"Invalid command: {validation_result[1]}")

        # Clamp command to limits
        clamped_command = self._clamp(command)

        # Track if clamping was applied
        if not np.allclose(clamped_command.command_values, command.command_values):
            self._clamped_count += 1
            logger.debug("Command values were clamped")

        # Send to hardware
        success = self._send_to_hardware(clamped_command)

        if success:
            self._last_command = clamped_command
            self._last_command_time_ns = current_time_ns
            self._command_count += 1

        return success

    def _validate_command(self, command: ActuatorCommand) -> Tuple[bool, str]:
        """
        Validate command structure and values.

        Args:
            command: Command to validate

        Returns:
            (is_valid, error_message)
        """
        # Check correct number of values
        if len(command.command_values) != self.config.num_joints:
            return False, f"Expected {self.config.num_joints} values, got {len(command.command_values)}"

        # Check for NaN/Inf
        if self.config.check_nan:
            if np.any(np.isnan(command.command_values)):
                return False, "Command contains NaN values"
            if np.any(np.isinf(command.command_values)):
                return False, "Command contains Inf values"

        # Check valid control mode
        if not isinstance(command.mode, ControlMode):
            return False, f"Invalid control mode: {command.mode}"

        return True, ""

    def _clamp(self, command: ActuatorCommand) -> ActuatorCommand:
        """
        Clamp command values to safe limits.

        Args:
            command: Command to clamp

        Returns:
            New ActuatorCommand with clamped values
        """
        values = command.command_values.copy()

        if command.mode == ControlMode.POSITION:
            for i in range(len(values)):
                lo, hi = self._position_limits[i]
                values[i] = np.clip(values[i], lo, hi)

        elif command.mode == ControlMode.VELOCITY:
            for i in range(len(values)):
                limit = self._velocity_limits[i]
                values[i] = np.clip(values[i], -limit, limit)

        elif command.mode == ControlMode.TORQUE:
            for i in range(len(values)):
                limit = self._torque_limits[i]
                values[i] = np.clip(values[i], -limit, limit)

        elif command.mode == ControlMode.IMPEDANCE:
            # Impedance mode: clamp position part (first half) and stiffness (second half)
            n = len(values) // 2
            for i in range(n):
                lo, hi = self._position_limits[i]
                values[i] = np.clip(values[i], lo, hi)
            # Stiffness should be non-negative
            for i in range(n, len(values)):
                values[i] = max(0.0, values[i])

        return ActuatorCommand(
            timestamp_ns=command.timestamp_ns,
            mode=command.mode,
            command_values=values
        )

    def _check_rate_limit(self, current_time_ns: int) -> bool:
        """Check if command rate is within limits."""
        if self._last_command_time_ns == 0:
            return True

        min_interval_ns = int(1e9 / self.config.max_command_rate_hz)
        elapsed_ns = current_time_ns - self._last_command_time_ns

        return elapsed_ns >= min_interval_ns

    def _send_to_hardware(self, command: ActuatorCommand) -> bool:
        """
        Send command to actual hardware.

        Args:
            command: Clamped command to send

        Returns:
            True if successful
        """
        if self.config.mock_mode:
            # Mock mode: just log and return success
            logger.debug(f"Mock send: mode={command.mode.value}, "
                        f"values={command.command_values[:3]}...")
            return True

        # Use callback if registered
        if self._send_callback:
            return self._send_callback(command)

        # TODO: Implement actual hardware interface
        raise NotImplementedError(
            "Hardware send not implemented. "
            "Set mock_mode=True or register a send callback."
        )

    def register_send_callback(
        self,
        callback: Callable[[ActuatorCommand], bool]
    ) -> None:
        """
        Register callback for sending commands to hardware.

        Args:
            callback: Function that takes ActuatorCommand and returns success
        """
        self._send_callback = callback

    def get_last_command(self) -> Optional[ActuatorCommand]:
        """Get the most recently sent command."""
        return self._last_command

    def get_command_count(self) -> int:
        """Get total number of commands sent."""
        return self._command_count

    def get_clamped_count(self) -> int:
        """Get number of commands that were clamped."""
        return self._clamped_count

    def get_limits(self) -> dict:
        """Get current limit configuration."""
        return {
            "position": self._position_limits,
            "velocity": self._velocity_limits,
            "torque": self._torque_limits,
        }

    def reset(self) -> None:
        """Reset bridge state."""
        self._last_command = None
        self._last_command_time_ns = 0
        self._command_count = 0
        self._clamped_count = 0


# =============================================================================
# Factory Functions
# =============================================================================

def create_default_actuator_bridge() -> ActuatorBridge:
    """Create actuator bridge with default configuration."""
    return ActuatorBridge(ActuatorBridgeConfig())


def create_mock_actuator_bridge(num_joints: int = 6) -> ActuatorBridge:
    """
    Create mock actuator bridge for testing.

    Args:
        num_joints: Number of robot joints

    Returns:
        ActuatorBridge configured for mock operation
    """
    config = ActuatorBridgeConfig(
        num_joints=num_joints,
        mock_mode=True,
        check_limits=True,
        max_command_rate_hz=1e9  # Effectively unlimited for testing
    )
    return ActuatorBridge(config)


def create_safe_actuator_bridge(
    num_joints: int = 6,
    conservative_limits: bool = True
) -> ActuatorBridge:
    """
    Create actuator bridge with conservative safety limits.

    Args:
        num_joints: Number of robot joints
        conservative_limits: If True, use very conservative limits

    Returns:
        ActuatorBridge with safety-focused configuration
    """
    if conservative_limits:
        pos_limit = (-np.pi / 2, np.pi / 2)  # ±90 degrees
        vel_limit = 0.5  # rad/s
        torque_limit = 20.0  # Nm
    else:
        pos_limit = (-np.pi, np.pi)
        vel_limit = 2.0
        torque_limit = 100.0

    config = ActuatorBridgeConfig(
        num_joints=num_joints,
        position_limits=[pos_limit] * num_joints,
        velocity_limits=[vel_limit] * num_joints,
        torque_limits=[torque_limit] * num_joints,
        mock_mode=True,
        check_limits=True,
        check_nan=True
    )
    return ActuatorBridge(config)
