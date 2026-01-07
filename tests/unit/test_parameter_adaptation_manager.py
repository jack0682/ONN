import pytest
import numpy as np

from onn.ops.branching import ParameterAdaptationManager, AdaptationConfig


class TestParameterAdaptationManager:
    def setup_method(self):
        self.config = AdaptationConfig(
            min_history_len=10,
            steep_slope_threshold=-0.1,
            plateau_slope_threshold=-0.01,
            lr_increase_factor=1.2,
            lr_decrease_factor=0.9,
            min_gate_lr=0.001,
            max_gate_lr=0.5,
        )
        self.manager = ParameterAdaptationManager(self.config)
        self.meta_params = {"gate_lr": 0.1, "other_param": 1.0}

    def test_increase_lr_on_plateau(self):
        """LR should increase when residual slope is flat."""
        # Flat residual means high (close to 0) log-slope
        residual_history = np.linspace(0.1, 0.099, self.config.min_history_len).tolist()

        new_params, reason = self.manager.adapt(
            self.meta_params, residual_history, es_meta_params={"gate_lr": 0.1}
        )

        assert reason == "plateau"
        assert new_params["gate_lr"] > self.meta_params["gate_lr"]
        assert new_params["gate_lr"] == pytest.approx(0.11)
        assert new_params["other_param"] == self.meta_params["other_param"]

    def test_decrease_lr_on_steep_decline(self):
        """LR should decrease when residual is dropping quickly."""
        # Steeply falling residual means very negative log-slope
        residual_history = np.linspace(0.5, 0.1, self.config.min_history_len).tolist()

        new_params, reason = self.manager.adapt(self.meta_params, residual_history)

        assert reason == "improving"
        assert new_params["gate_lr"] < self.meta_params["gate_lr"]
        assert new_params["gate_lr"] == pytest.approx(
            0.1 * self.config.lr_decrease_factor
        )

    def test_no_change_on_normal_decline(self):
        """LR should not change for a moderate, healthy decline."""
        # A slope between the two thresholds
        # log_slope = (log(0.2) - log(0.3)) / 10 = (-1.6 - (-1.2)) / 10 = -0.04
        # This is between -0.1 and -0.01, so no change should happen.
        residual_history = np.linspace(0.3, 0.2, self.config.min_history_len).tolist()

        new_params, reason = self.manager.adapt(self.meta_params, residual_history)

        assert reason == "stable"
        assert new_params["gate_lr"] == self.meta_params["gate_lr"]

    def test_lr_clamping_max(self):
        """LR should not exceed max_gate_lr."""
        self.meta_params["gate_lr"] = 0.48
        # Provide es_meta_params that allows reaching the max
        # Upper bound: 0.5 * 1.1 = 0.55 > 0.5
        residual_history = np.linspace(0.1, 0.099, self.config.min_history_len).tolist()

        new_params, reason = self.manager.adapt(
            self.meta_params, residual_history, es_meta_params={"gate_lr": 0.5}
        )

        # 0.48 * 1.2 = 0.576, which should be clamped to 0.5 (max_gate_lr)
        assert new_params["gate_lr"] == self.config.max_gate_lr

    def test_trust_region_clamping(self):
        """Adaptation should be limited by the trust region percentage."""
        self.meta_params["gate_lr"] = 0.1
        # plateau triggers 1.2x increase: 0.1 * 1.2 = 0.12
        # trust region 10% limits it to: 0.1 * 1.1 = 0.11
        residual_history = np.linspace(0.1, 0.099, self.config.min_history_len).tolist()

        new_params, reason = self.manager.adapt(
            self.meta_params, residual_history, es_meta_params={"gate_lr": 0.1}
        )

        assert new_params["gate_lr"] == pytest.approx(0.11)

    def test_lr_clamping_min(self):
        """LR should not go below min_gate_lr."""
        self.meta_params["gate_lr"] = 0.0011
        # Steep residual to trigger decrease
        residual_history = np.linspace(0.5, 0.1, self.config.min_history_len).tolist()

        new_params, reason = self.manager.adapt(self.meta_params, residual_history)

        # 0.0011 * 0.9 = 0.00099, which should be clamped to 0.001
        assert new_params["gate_lr"] == self.config.min_gate_lr

    def test_insufficient_history(self):
        """Should not adapt if history is too short."""
        residual_history = [0.1, 0.05]  # len < min_history_len

        new_params, reason = self.manager.adapt(self.meta_params, residual_history)

        assert reason == "insufficient_history"
        assert new_params["gate_lr"] == self.meta_params["gate_lr"]
        assert new_params == self.meta_params
