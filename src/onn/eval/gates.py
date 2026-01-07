"""Acceptance Gates for ONN Validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from onn.core import EvalMetrics, GateReport, GateResult

logger = logging.getLogger(__name__)


@dataclass
class GateConfig:
    """Acceptance gate thresholds."""

    violation_max: float = 0.1
    violation_mean: float = 0.05
    drift_max: float = 1.0
    collapse_min: float = 0.01
    latency_max: float = 5.0
    contradiction_max: float = 0.05
    ricci_max: float = 100.0


def evaluate_gates(
    metrics: EvalMetrics,
    config: Optional[GateConfig] = None,
) -> GateReport:
    """Check all gates and return pass/fail with details."""
    if config is None:
        config = GateConfig()

    results: List[GateResult] = []
    failed_gates: List[str] = []

    def check(gate_id, value, op, threshold, desc):
        passed = (op == "<=" and value <= threshold) or (
            op == ">=" and value >= threshold
        )
        results.append(GateResult(gate_id, passed, value, threshold, desc))
        if not passed:
            failed_gates.append(gate_id)

    check(
        "feasibility_mean",
        metrics.violation_mean,
        "<=",
        config.violation_mean,
        f"Mean violation ≤ {config.violation_mean}",
    )
    check(
        "feasibility_max",
        metrics.violation_max,
        "<=",
        config.violation_max,
        f"Max violation ≤ {config.violation_max}",
    )
    check(
        "fidelity",
        metrics.drift_mean,
        "<=",
        config.drift_max,
        f"Drift ≤ {config.drift_max}",
    )
    check(
        "non_collapse",
        metrics.collapse_score,
        ">=",
        config.collapse_min,
        f"Variance ≥ {config.collapse_min}",
    )
    check(
        "responsiveness",
        metrics.latency_mean,
        "<=",
        config.latency_max,
        f"Latency ≤ {config.latency_max}",
    )
    check(
        "structural",
        metrics.ricci_energy,
        "<=",
        config.ricci_max,
        f"Ricci ≤ {config.ricci_max}",
    )
    check(
        "consistency",
        metrics.contradiction_rate,
        "<=",
        config.contradiction_max,
        f"Contradiction ≤ {config.contradiction_max}",
    )

    overall_passed = not failed_gates
    return GateReport(
        passed=overall_passed,
        failed_gates=failed_gates,
        gate_results=results,
        metrics=metrics,
    )


def check_single_gate(
    gate_id: str,
    value: float,
    config: GateConfig,
) -> bool:
    """Check a single gate by ID."""
    thresholds = {
        "feasibility_mean": ("<=", config.violation_mean),
        "feasibility_max": ("<=", config.violation_max),
        "fidelity": ("<=", config.drift_max),
        "non_collapse": (">=", config.collapse_min),
        "responsiveness": ("<=", config.latency_max),
        "structural": ("<=", config.ricci_max),
        "consistency": ("<=", config.contradiction_max),
    }
    if gate_id not in thresholds:
        return True
    op, threshold = thresholds[gate_id]
    if op == "<=":
        return value <= threshold
    elif op == ">=":
        return value >= threshold
    return False


def create_strict_config() -> GateConfig:
    """Create a strict gate configuration for production."""
    return GateConfig(
        violation_max=0.05,
        violation_mean=0.01,
        drift_max=0.5,
        collapse_min=0.05,
        latency_max=3.0,
    )


def create_relaxed_config() -> GateConfig:
    """Create a relaxed gate configuration for development."""
    return GateConfig(
        violation_max=0.5,
        violation_mean=0.2,
        drift_max=5.0,
        collapse_min=0.001,
        latency_max=20.0,
        ricci_max=1000.0,
    )


def format_report(report: GateReport) -> str:
    """Format a gate report for display."""
    lines = [f"Gate Report: {'PASS' if report.passed else 'FAIL'}", "-" * 40]
    for gate in report.gate_results:
        status = "✓" if gate.passed else "✗"
        op = "≥" if "collapse" in gate.gate_id else "≤"
        lines.append(
            f"  {status} {gate.gate_id}: {gate.value:.4f} {op} {gate.threshold}"
        )
    if report.failed_gates:
        lines.append(f"\nFailed: {', '.join(report.failed_gates)}")
    return "\n".join(lines)
