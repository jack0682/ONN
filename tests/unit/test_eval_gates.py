import pytest
from onn.core.tensors import EvalMetrics, GateReport
from onn.eval.gates import GateConfig, evaluate_gates, GateResult


def test_evaluate_gates_all_pass():
    metrics = EvalMetrics(
        violation_mean=0.01,
        violation_max=0.02,
        drift_mean=0.1,
        collapse_score=0.1,
        latency_mean=1.0,
        ricci_energy=10.0,
        contradiction_rate=0.0,
    )
    config = GateConfig()
    report = evaluate_gates(metrics, config)
    assert report.passed is True
    assert len(report.failed_gates) == 0


def test_evaluate_gates_violation_fails():
    metrics = EvalMetrics(violation_mean=0.1, violation_max=0.2)
    config = GateConfig(violation_mean=0.05, violation_max=0.1)
    report = evaluate_gates(metrics, config)
    assert report.passed is False
    assert "feasibility_mean" in report.failed_gates
    assert "feasibility_max" in report.failed_gates


def test_evaluate_gates_collapse_fails():
    metrics = EvalMetrics(collapse_score=0.005)
    config = GateConfig(collapse_min=0.01)
    report = evaluate_gates(metrics, config)
    assert report.passed is False
    assert "non_collapse" in report.failed_gates


def test_evaluate_gates_mixed_results():
    metrics = EvalMetrics(violation_mean=0.1, collapse_score=0.9)
    config = GateConfig(violation_mean=0.05, collapse_min=0.1)
    report = evaluate_gates(metrics, config)
    assert report.passed is False
    assert "feasibility_mean" in report.failed_gates
    assert "non_collapse" not in report.failed_gates
