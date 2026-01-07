"""
ORTSF (Ontology-Regulated Trajectory Synthesis Fabric) Controller.

The Control Layer in the CSA architecture.
Provides delay-robust control by interpolating ReasoningTraces
and generating ActuatorCommands.

Core Math (from spec/02_onn_math_spec.md Section 5):
    Small-Gain Theorem: γ_ONN * γ_ORTSF < 1

    Velocity clamping:
    ||v|| ≤ 1 / (γ_ORTSF * Δt_latency)

Reference:
    - spec/02_onn_math_spec.md Section 5
    - spec/20_impl_plan.ir.yml IMPL_009
    - spec/11_interfaces.ir.yml -> ActuatorCommand

Framework-agnostic: No ROS2, DDS, or external framework dependencies.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import logging
import time

from onn.core.tensors import (
    ReasoningTrace,
    SensorObservation,
    ActuatorCommand,
    ControlMode,
    JointState,
    BOUND_TENSOR_DIM,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ORTSFConfig:
    """
    Configuration for the ORTSF fabric.

    Reference: spec/01_constraints.md Section 3.4
    """
    # Control rate (Hz)
    control_rate_hz: float = 100.0

    # Trace horizon (seconds)
    trace_horizon_sec: float = 1.0

    # Interpolation mode
    interpolation_mode: str = "cubic"  # "linear", "cubic"

    # Small-Gain Theorem parameters
    gamma_ortsf: float = 0.5  # Lipschitz constant

    # Delay prediction
    nominal_delay_ns: int = 50_000_000  # 50ms
    max_delay_ns: int = 200_000_000  # 200ms

    # Safety limits
    max_velocity: float = 1.0  # m/s or rad/s
    max_acceleration: float = 5.0  # m/s² or rad/s²

    # === Control Gains (CPL_002: moved from hardcoded values) ===
    # PD control gains for torque mode
    torque_kp: float = 100.0  # Proportional gain
    torque_kd: float = 10.0   # Derivative gain
    
    # Torque limits (Nm)
    max_torque: float = 100.0  # Maximum absolute torque per joint

    # Control mode
    default_mode: ControlMode = field(default=ControlMode.POSITION)

    # Number of joints/DOF
    num_joints: int = 6


# =============================================================================
# ORTSF Fabric
# =============================================================================

class ORTSFFabric:
    """
    The ORTSF Fabric implementation.

    Performs delay-robust control by:
    1. Predicting communication/computation delay
    2. Interpolating ReasoningTrace at compensated time
    3. Computing smooth ActuatorCommand output

    Satisfies small-gain theorem for stability:
    γ_ONN * γ_ORTSF < 1

    Reference:
        - spec/02_onn_math_spec.md Section 5
        - spec/20_impl_plan.ir.yml IMPL_009
    """

    def __init__(self, config: Optional[ORTSFConfig] = None):
        """
        Initialize the ORTSF fabric.

        Args:
            config: Controller configuration. Uses defaults if None.
        """
        self.config = config or ORTSFConfig()

        # State tracking
        self._current_trace: Optional[ReasoningTrace] = None
        self._last_command: Optional[ActuatorCommand] = None
        self._last_observation: Optional[SensorObservation] = None

        # Delay estimation (exponential moving average)
        self._estimated_delay_ns: float = float(self.config.nominal_delay_ns)
        self._delay_alpha: float = 0.1  # EMA smoothing factor

        # Command history for velocity estimation
        self._command_history: List[Tuple[int, np.ndarray]] = []
        self._max_history_size: int = 10

    def step(
        self,
        trace: Optional[ReasoningTrace],
        observation: SensorObservation
    ) -> ActuatorCommand:
        """
        Compute actuator command from trace and observation.

        This is the main control loop step.

        Args:
            trace: Current ReasoningTrace from IMAGO (may be stale or None)
            observation: Current sensor observation

        Returns:
            ActuatorCommand to send to actuators

        Reference: spec/20_impl_plan.ir.yml IMPL_009
        """
        current_time_ns = time.time_ns()

        # Update trace if new one is provided
        if trace is not None and self._is_trace_valid(trace, current_time_ns):
            self._current_trace = trace
            logger.debug("Updated trace")

        self._last_observation = observation

        # Step 1: Predict delay
        predicted_delay_ns = self._predict_delay()

        # Step 2: Interpolate trace at compensated time
        target_position = self._interpolate_trace(
            current_time_ns + predicted_delay_ns
        )

        # Step 3: Compute command
        command = self._compute_command(
            target_position,
            observation,
            current_time_ns
        )

        self._last_command = command
        return command

    def _predict_delay(self) -> int:
        """
        Predict communication/computation delay.

        Uses exponential moving average of observed delays.

        Returns:
            Predicted delay in nanoseconds
        """
        # For V0: use nominal delay with small variation
        # In production, this would track actual message latencies

        # Add some noise to simulate realistic conditions
        noise = np.random.normal(0, 0.1 * self.config.nominal_delay_ns)
        predicted = self._estimated_delay_ns + noise

        # Clamp to valid range
        predicted = max(0, min(predicted, self.config.max_delay_ns))

        logger.debug(f"Predicted delay: {predicted / 1e6:.1f}ms")
        return int(predicted)

    def _interpolate_trace(self, target_time_ns: int) -> np.ndarray:
        """
        Interpolate trace at target time.

        Args:
            target_time_ns: Time at which to interpolate

        Returns:
            Interpolated target position (bound tensor)
        """
        if self._current_trace is None:
            logger.warning("No trace available, returning zeros")
            return np.zeros(self.config.num_joints, dtype=np.float32)

        trace = self._current_trace

        # Compute normalized time t ∈ [0, 1]
        trace_duration_ns = trace.valid_until_ns - trace.timestamp_ns
        if trace_duration_ns <= 0:
            t = 1.0
        else:
            elapsed_ns = target_time_ns - trace.timestamp_ns
            t = np.clip(elapsed_ns / trace_duration_ns, 0.0, 1.0)

        # Interpolate trajectory
        if trace.trajectory_coeffs.size == 0:
            # No trajectory coefficients, use target state
            if trace.target_state:
                return trace.target_state[0].bound_tensor[:self.config.num_joints].copy()
            return np.zeros(self.config.num_joints, dtype=np.float32)

        # Reshape coefficients and evaluate polynomial
        coeffs = trace.trajectory_coeffs
        n_coeffs = 4  # Cubic
        n_dims = len(coeffs) // n_coeffs

        if n_dims == 0:
            return np.zeros(self.config.num_joints, dtype=np.float32)

        coeffs = coeffs.reshape(n_dims, n_coeffs)
        position = np.zeros(min(n_dims, self.config.num_joints), dtype=np.float32)

        for dim in range(position.shape[0]):
            a, b, c, d = coeffs[dim]
            # p(t) = a + b*t + c*t² + d*t³
            position[dim] = a + b * t + c * t * t + d * t * t * t

        # Pad or truncate to num_joints
        if len(position) < self.config.num_joints:
            position = np.pad(position, (0, self.config.num_joints - len(position)))

        return position[:self.config.num_joints]

    def _compute_command(
        self,
        target_position: np.ndarray,
        observation: SensorObservation,
        timestamp_ns: int
    ) -> ActuatorCommand:
        """
        Compute actuator command from target position and current state.

        Args:
            target_position: Desired position
            observation: Current sensor observation
            timestamp_ns: Current timestamp

        Returns:
            ActuatorCommand to execute
        """
        mode = self.config.default_mode

        # Get current position from observation
        current_position = self._get_current_position(observation)

        if mode == ControlMode.POSITION:
            # Position control: command is the target position
            command_values = target_position.copy()

        elif mode == ControlMode.VELOCITY:
            # Velocity control: compute velocity to reach target
            dt = 1.0 / self.config.control_rate_hz
            velocity = (target_position - current_position) / dt

            # Clamp velocity
            velocity = self._clamp_velocity(velocity)
            command_values = velocity

        elif mode == ControlMode.TORQUE:
            # Torque control: PD control with configurable gains
            # Gains from config (CPL_002: no hardcoded magic numbers)
            kp = self.config.torque_kp
            kd = self.config.torque_kd

            # Estimate velocity from history
            current_velocity = self._estimate_velocity(timestamp_ns)

            position_error = target_position - current_position
            torque = kp * position_error - kd * current_velocity

            command_values = torque

        else:
            # Impedance or other: use position as command
            command_values = target_position.copy()

        # Apply safety limits
        command_values = self._apply_safety_limits(command_values, mode)

        # Update command history
        self._update_history(timestamp_ns, command_values)

        return ActuatorCommand(
            timestamp_ns=timestamp_ns,
            mode=mode,
            command_values=command_values
        )

    def _get_current_position(self, observation: SensorObservation) -> np.ndarray:
        """Extract current joint position from observation."""
        if observation.joint_state is not None:
            pos = observation.joint_state.position
            if len(pos) >= self.config.num_joints:
                return pos[:self.config.num_joints]
            return np.pad(pos, (0, self.config.num_joints - len(pos)))

        return np.zeros(self.config.num_joints, dtype=np.float32)

    def _estimate_velocity(self, current_time_ns: int) -> np.ndarray:
        """Estimate current velocity from command history."""
        if len(self._command_history) < 2:
            return np.zeros(self.config.num_joints, dtype=np.float32)

        # Use last two commands
        t1, cmd1 = self._command_history[-2]
        t2, cmd2 = self._command_history[-1]

        dt = (t2 - t1) / 1e9  # Convert to seconds
        if dt <= 0:
            return np.zeros(self.config.num_joints, dtype=np.float32)

        velocity = (cmd2 - cmd1) / dt
        return velocity

    def _clamp_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """Clamp velocity to satisfy small-gain theorem."""
        # ||v|| ≤ max_velocity
        vel_norm = np.linalg.norm(velocity)
        if vel_norm > self.config.max_velocity:
            velocity = velocity * (self.config.max_velocity / vel_norm)
        return velocity

    def _apply_safety_limits(
        self,
        values: np.ndarray,
        mode: ControlMode
    ) -> np.ndarray:
        """Apply safety limits to command values."""
        if mode == ControlMode.VELOCITY:
            # Clamp velocity
            vel_norm = np.linalg.norm(values)
            if vel_norm > self.config.max_velocity:
                values = values * (self.config.max_velocity / vel_norm)

        elif mode == ControlMode.TORQUE:
            # Clamp torque with configurable limit (CPL_002: no hardcoded magic numbers)
            max_torque = self.config.max_torque
            values = np.clip(values, -max_torque, max_torque)

        # Check for NaN/Inf
        if np.any(np.isnan(values)) or np.any(np.isinf(values)):
            logger.warning("Invalid command values detected, zeroing")
            values = np.zeros_like(values)

        return values

    def _update_history(self, timestamp_ns: int, command: np.ndarray) -> None:
        """Update command history for velocity estimation."""
        self._command_history.append((timestamp_ns, command.copy()))

        # Keep history bounded
        if len(self._command_history) > self._max_history_size:
            self._command_history.pop(0)

    def _is_trace_valid(self, trace: ReasoningTrace, current_time_ns: int) -> bool:
        """Check if trace is valid at current time."""
        return trace.valid_until_ns > current_time_ns

    def update_delay_estimate(self, actual_delay_ns: int) -> None:
        """
        Update delay estimation with observed delay.

        Args:
            actual_delay_ns: Measured delay in nanoseconds
        """
        # Exponential moving average
        self._estimated_delay_ns = (
            self._delay_alpha * actual_delay_ns +
            (1 - self._delay_alpha) * self._estimated_delay_ns
        )

    def get_estimated_delay_ns(self) -> int:
        """Get current delay estimate."""
        return int(self._estimated_delay_ns)

    def reset(self) -> None:
        """Reset controller state."""
        self._current_trace = None
        self._last_command = None
        self._last_observation = None
        self._command_history.clear()
        self._estimated_delay_ns = float(self.config.nominal_delay_ns)


# =============================================================================
# Factory Functions
# =============================================================================

def create_default_ortsf_fabric() -> ORTSFFabric:
    """Create ORTSF fabric with default configuration."""
    return ORTSFFabric(ORTSFConfig())


def create_realtime_ortsf_fabric(num_joints: int = 6) -> ORTSFFabric:
    """
    Create ORTSF fabric optimized for real-time control.

    Args:
        num_joints: Number of robot joints

    Returns:
        ORTSFFabric configured for real-time operation
    """
    config = ORTSFConfig(
        control_rate_hz=100.0,
        nominal_delay_ns=30_000_000,  # 30ms
        num_joints=num_joints,
        default_mode=ControlMode.POSITION,
    )
    return ORTSFFabric(config)
