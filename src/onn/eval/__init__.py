"""ONN Evaluation Package.

Provides metrics computation, acceptance gates, and reporting for ONN validation.

Reference:
    spec/20_impl_plan.ir.yml: IMPL_022
"""

from onn.eval.metrics import (
    compute_violation,
    compute_drift,
    compute_ricci_energy,
    compute_smoothness,
    compute_latency,
    compute_collapse_score,
    compute_all_metrics,
    metrics_to_summary,
)

from onn.eval.gates import (
    GateConfig,
    GateResult,
    GateReport,
    evaluate_gates,
    check_single_gate,
    create_strict_config,
    create_relaxed_config,
    format_report,
)

from onn.eval.reports import (
    ReportConfig,
    write_report,
    write_generation_report,
    write_gate_report,
    read_reports,
    summarize_reports,
    generate_experiment_summary,
    create_default_config,
)

__all__ = [
    # Metrics
    "compute_violation",
    "compute_drift",
    "compute_ricci_energy",
    "compute_smoothness",
    "compute_latency",
    "compute_collapse_score",
    "compute_all_metrics",
    "metrics_to_summary",
    # Gates
    "GateConfig",
    "GateResult",
    "GateReport",
    "evaluate_gates",
    "check_single_gate",
    "create_strict_config",
    "create_relaxed_config",
    "format_report",
    # Reports
    "ReportConfig",
    "write_report",
    "write_generation_report",
    "write_gate_report",
    "read_reports",
    "summarize_reports",
    "generate_experiment_summary",
    "create_default_config",
]
