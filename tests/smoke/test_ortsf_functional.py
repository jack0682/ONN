import torch
import numpy as np
from onn.ops.ortsf import ORTSFOperator, DeepDeltaPredictor


def test_ortsf_functional():
    input_dim = 64
    predictor = DeepDeltaPredictor(input_dim=input_dim)
    ortsf = ORTSFOperator(predictor=predictor, delay_ms=50.0)

    for t in range(5):
        current_trace = torch.randn(input_dim)
        cmd = ortsf.transform(current_trace)
        assert cmd is not None
        assert isinstance(cmd.command_values, np.ndarray)


def test_ortsf_zero_delay():
    input_dim = 64
    predictor = DeepDeltaPredictor(input_dim=input_dim)
    ortsf = ORTSFOperator(predictor=predictor, delay_ms=0.0)

    for t in range(5):
        current_trace = torch.randn(input_dim)
        cmd = ortsf.transform(current_trace)
        assert cmd is not None
        assert isinstance(cmd.command_values, np.ndarray)
        assert np.all(np.isfinite(cmd.command_values))


if __name__ == "__main__":
    test_ortsf_functional()
    test_ortsf_zero_delay()
