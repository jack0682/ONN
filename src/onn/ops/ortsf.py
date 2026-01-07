"""
Ontological Real-Time Semantic Fabric (ORTSF).
Translates reasoning traces into actionable, delay-robust control commands.
"""

from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple

import time
from onn.core.tensors import ReasoningTrace, ActuatorCommand, ControlMode


class DeepDeltaPredictor(nn.Module):
    """
    Learned predictor for state transitions (Deep-Delta).
    Predicts the change in reasoning trace over a prediction horizon delta.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(
        self, current_trace: torch.Tensor, previous_trace: torch.Tensor, delta: float
    ) -> torch.Tensor:
        """
        Predict next trace state: R(t + delta) = R(t) + delta * MLP(R(t), R(t-1))
        """
        x = torch.cat([current_trace, previous_trace], dim=-1)
        delta_r = self.mlp(x)
        return current_trace + delta * delta_r

    def fit(self, history: List[torch.Tensor], delta: float, epochs: int = 5):
        """
        Self-supervised learning: train the MLP to predict R(t) from R(t-1) and R(t-2).
        Uses the collected reasoning trace as the ground truth.
        """
        if len(history) < 3:
            return

        self.train()
        for _ in range(epochs):
            total_loss = 0
            # Slide through history to create (R_prev, R_curr) -> R_next pairs
            for i in range(2, len(history)):
                r_prev2 = history[i - 2]
                r_prev1 = history[i - 1]
                r_target = history[i]

                self.optimizer.zero_grad()
                r_pred = self.forward(r_prev1, r_prev2, delta)
                loss = torch.mean((r_pred - r_target) ** 2)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        self.eval()


class ORTSFOperator:
    """
    The ORTSF composite operator: T_control o T_delay o T_predict.
    Implements Theorem 4 (Delay-Small Gain Stability) from the spec.
    """

    def __init__(
        self,
        predictor: DeepDeltaPredictor,
        delay_ms: float = 50.0,
        kp: float = 1.0,
        kd: float = 0.1,
    ):
        self.predictor = predictor
        self.delay = delay_ms / 1000.0
        self.kp = kp
        self.kd = kd
        self.history_buffer: List[torch.Tensor] = []
        self.last_predicted_error = None

    def predict(self, current_trace: torch.Tensor) -> torch.Tensor:
        """T_predict: Temporal extrapolation using Deep-Delta."""
        if not self.history_buffer:
            self.history_buffer.append(current_trace)
            return current_trace

        previous_trace = self.history_buffer[-1]
        predicted_trace = self.predictor(current_trace, previous_trace, self.delay)

        self.history_buffer.append(current_trace)
        if len(self.history_buffer) > 20:  # Larger buffer
            self.history_buffer.pop(0)

        return predicted_trace

    def compensate_delay(self, predicted_trace: torch.Tensor) -> torch.Tensor:
        """
        T_delay: Compensate for system latency.
        Abstracted Smith Predictor logic: Subtract estimated feedback error from predicted trace.
        """
        if self.last_predicted_error is None:
            return predicted_trace

        return predicted_trace - 0.1 * self.last_predicted_error  # Damped compensation

    def control_synthesis(self, compensated_trace: torch.Tensor) -> ActuatorCommand:
        """
        T_control: Map semantic trace to actuator commands using PD control.
        Assumes the trace represents a target state on the manifold.
        """
        # Target state extracted from trace (e.g., first 7 dims)
        target = compensated_trace[:7].detach().cpu().numpy()

        # Simple PD Control: u = Kp*e + Kd*de/dt
        # For MVP, we assume current state is 0 for error calculation
        current_state = np.zeros_like(target)
        error = target - current_state

        delta_error = error
        if self.last_predicted_error is not None:
            delta_error = error - self.last_predicted_error.cpu().numpy()[:7]

        self.last_predicted_error = compensated_trace.detach()

        u = self.kp * error + self.kd * delta_error

        return ActuatorCommand(
            timestamp_ns=time.time_ns(),
            command_values=u,
            mode=ControlMode.POSITION,
        )

    def transform(self, current_trace: torch.Tensor) -> ActuatorCommand:
        """Apply the full ORTSF transformation pipeline."""
        r_pred = self.predict(current_trace)
        r_comp = self.compensate_delay(r_pred)
        u_cmd = self.control_synthesis(r_comp)
        return u_cmd
