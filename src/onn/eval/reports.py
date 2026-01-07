"""JSONL Logging and Summary Reporting.

This module provides utilities for logging evaluation results to JSONL
format and generating summary reports.

Reference:
    - spec/20_impl_plan.ir.yml: IMPL_022

Author: Claude (via IMPL_022)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class ReportConfig:
    """Report configuration."""
    output_dir: str = "reports"
    experiment_name: str = "onn_experiment"
    save_all_generations: bool = True
    save_best_only: bool = False


# ==============================================================================
# REPORT WRITING
# ==============================================================================

def write_report(
    data: Dict[str, Any],
    filepath: str,
    append: bool = True,
) -> None:
    """Write a single report entry to JSONL file.
    
    Args:
        data: Dictionary to write
        filepath: Path to JSONL file
        append: If True, append to file; otherwise overwrite
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp if not present
    if "timestamp" not in data:
        data["timestamp"] = datetime.now().isoformat()
    
    mode = "a" if append else "w"
    with open(path, mode) as f:
        f.write(json.dumps(data) + "\n")
    
    logger.debug(f"Wrote report to {filepath}")


def write_generation_report(
    generation: int,
    best_fitness: float,
    best_params: Dict[str, float],
    metrics: Dict[str, float],
    config: ReportConfig,
) -> None:
    """Write a generation report.
    
    Args:
        generation: Generation number
        best_fitness: Best fitness this generation
        best_params: Best parameter values
        metrics: Evaluation metrics
        config: Report configuration
    """
    filepath = Path(config.output_dir) / f"{config.experiment_name}_generations.jsonl"
    
    data = {
        "type": "generation",
        "generation": generation,
        "best_fitness": best_fitness,
        "best_params": best_params,
        "metrics": metrics,
    }
    
    write_report(data, str(filepath))


def write_gate_report(
    generation: int,
    passed: bool,
    failed_gates: List[str],
    metrics: Dict[str, float],
    config: ReportConfig,
) -> None:
    """Write a gate evaluation report.
    
    Args:
        generation: Generation number
        passed: Whether all gates passed
        failed_gates: List of failed gate IDs
        metrics: Evaluation metrics
        config: Report configuration
    """
    filepath = Path(config.output_dir) / f"{config.experiment_name}_gates.jsonl"
    
    data = {
        "type": "gate",
        "generation": generation,
        "passed": passed,
        "failed_gates": failed_gates,
        "metrics": metrics,
    }
    
    write_report(data, str(filepath))


# ==============================================================================
# REPORT READING
# ==============================================================================

def read_reports(filepath: str) -> List[Dict[str, Any]]:
    """Read all reports from a JSONL file.
    
    Args:
        filepath: Path to JSONL file
        
    Returns:
        List of report dictionaries
    """
    path = Path(filepath)
    if not path.exists():
        return []
    
    reports = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                reports.append(json.loads(line))
    
    return reports


def summarize_reports(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize a list of reports.
    
    Args:
        reports: List of report dictionaries
        
    Returns:
        Summary dictionary
    """
    if not reports:
        return {"error": "No reports to summarize"}
    
    # Filter generation reports
    gen_reports = [r for r in reports if r.get("type") == "generation"]
    gate_reports = [r for r in reports if r.get("type") == "gate"]
    
    summary = {
        "total_reports": len(reports),
        "generations": len(gen_reports),
        "gate_checks": len(gate_reports),
    }
    
    if gen_reports:
        fitnesses = [r["best_fitness"] for r in gen_reports]
        summary["fitness_initial"] = fitnesses[0]
        summary["fitness_final"] = fitnesses[-1]
        summary["fitness_best"] = max(fitnesses)
        summary["fitness_improvement"] = fitnesses[-1] - fitnesses[0]
    
    if gate_reports:
        passed = [r["passed"] for r in gate_reports]
        summary["gate_pass_rate"] = sum(passed) / len(passed)
        summary["gate_final_pass"] = passed[-1] if passed else False
    
    return summary


# ==============================================================================
# SUMMARY GENERATION
# ==============================================================================

def generate_experiment_summary(
    config: ReportConfig,
    include_all_generations: bool = False,
) -> str:
    """Generate a summary of an experiment.
    
    Args:
        config: Report configuration
        include_all_generations: If True, include all generation details
        
    Returns:
        Markdown-formatted summary
    """
    gen_file = Path(config.output_dir) / f"{config.experiment_name}_generations.jsonl"
    gate_file = Path(config.output_dir) / f"{config.experiment_name}_gates.jsonl"
    
    gen_reports = read_reports(str(gen_file))
    gate_reports = read_reports(str(gate_file))
    
    summary = summarize_reports(gen_reports + gate_reports)
    
    lines = [
        f"# Experiment Summary: {config.experiment_name}",
        "",
        f"- **Generations**: {summary.get('generations', 0)}",
        f"- **Gate Checks**: {summary.get('gate_checks', 0)}",
        "",
        "## Fitness",
        f"- Initial: {summary.get('fitness_initial', 'N/A'):.4f}" if isinstance(summary.get('fitness_initial'), (int, float)) else "- Initial: N/A",
        f"- Final: {summary.get('fitness_final', 'N/A'):.4f}" if isinstance(summary.get('fitness_final'), (int, float)) else "- Final: N/A", 
        f"- Best: {summary.get('fitness_best', 'N/A'):.4f}" if isinstance(summary.get('fitness_best'), (int, float)) else "- Best: N/A",
        "",
        "## Gates",
        f"- Pass Rate: {summary.get('gate_pass_rate', 0) * 100:.1f}%",
        f"- Final: {'PASS' if summary.get('gate_final_pass') else 'FAIL'}",
    ]
    
    if include_all_generations and gen_reports:
        lines.extend([
            "",
            "## Generation History",
            "",
            "| Gen | Fitness | Passed |",
            "|-----|---------|--------|",
        ])
        for i, gr in enumerate(gen_reports):
            gate_pass = gate_reports[i]["passed"] if i < len(gate_reports) else "?"
            lines.append(f"| {gr['generation']} | {gr['best_fitness']:.4f} | {gate_pass} |")
    
    return "\n".join(lines)


def create_default_config(experiment_name: str = "onn_exp") -> ReportConfig:
    """Create a default report configuration.
    
    Args:
        experiment_name: Name for the experiment
        
    Returns:
        ReportConfig instance
    """
    return ReportConfig(
        output_dir="reports",
        experiment_name=experiment_name,
    )
