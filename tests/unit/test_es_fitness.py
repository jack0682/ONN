import numpy as np
import pytest
from onn.es.fitness import FitnessConfig, EvalMetrics, compute_fitness


def test_fitness_config_defaults():
    config = FitnessConfig()
    assert config.w_violation == 10.0
    assert config.w_collapse == 1.0
    assert config.w_smoothness == -0.1


def test_compute_fitness_all_zero():
    metrics = EvalMetrics()
    fitness = compute_fitness(metrics)
    assert fitness == 0.0


def test_compute_fitness_with_violation():
    config = FitnessConfig(w_violation=10.0)
    metrics = EvalMetrics(violation_mean=0.5)
    fitness = compute_fitness(metrics, config)
    assert fitness == pytest.approx(-5.0)


def test_compute_fitness_with_collapse():
    config = FitnessConfig(w_collapse=2.0)
    metrics = EvalMetrics(collapse_score=0.8)
    fitness = compute_fitness(metrics, config)
    assert fitness == pytest.approx(1.6)


def test_compute_fitness_combined():
    config = FitnessConfig(w_violation=10.0, w_drift=1.0, w_collapse=1.0)
    metrics = EvalMetrics(violation_mean=0.1, drift_mean=0.2, collapse_score=0.9)
    fitness = compute_fitness(metrics, config)
    assert fitness == pytest.approx(-0.3)
