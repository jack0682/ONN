"""Metric Computation for ONN Evaluation.

This module computes evaluation metrics for the ONN solver:
- Violation: Constraint satisfaction (Cx = τ)
- Drift: Distance from observations
- Ricci energy: Curvature regularization
- Smoothness: Temporal stability
- Latency: Event recovery time
- Collapse score: Embedding variance

Reference:
    - spec/11_interfaces.ir.yml: EvalMetrics
    - spec/20_impl_plan.ir.yml: IMPL_022

Author: Claude (via IMPL_022)
"""

from __future__ import annotations

import logging
from typing import List, Optional, TYPE_CHECKING

import torch
import numpy as np

if TYPE_CHECKING:
    from onn.core.graph import EdgeGraph
    from onn.core.cycles import CycleBasis

logger = logging.getLogger(__name__)


# ==============================================================================
# INDIVIDUAL METRICS
# ==============================================================================

def compute_violation(
    x: torch.Tensor,
    C: torch.Tensor,
    tau: torch.Tensor,
) -> float:
    """Compute constraint violation ||Cx - τ||_2.
    
    Args:
        x: Edge embeddings, shape (m, p)
        C: Cycle matrix, shape (q, m)
        tau: Target values, shape (q, p)
        
    Returns:
        Scalar violation (L2 norm)
    """
    if C.shape[0] == 0:
        return 0.0
    
    if x.dim() == 1:
        x = x.unsqueeze(1)
    if tau.dim() == 1:
        tau = tau.unsqueeze(1)
    
    residual = torch.mm(C, x) - tau
    return torch.norm(residual).item()


def compute_drift(
    x: torch.Tensor,
    x_obs: torch.Tensor,
) -> float:
    """Compute drift from observations ||x - x_obs||_2.
    
    Args:
        x: Stabilized embeddings, shape (m, p)
        x_obs: Observed embeddings, shape (m, p)
        
    Returns:
        Scalar drift (L2 norm)
    """
    return torch.norm(x - x_obs).item()


def compute_ricci_energy(
    x: torch.Tensor,
    edge_graph: "EdgeGraph",
) -> float:
    """Compute Ricci energy Σ_e Ric(e)².
    
    Args:
        x: Edge embeddings, shape (m, p)
        edge_graph: EdgeGraph instance
        
    Returns:
        Scalar Ricci energy
    """
    if edge_graph.num_edges == 0:
        return 0.0
    
    # Compute node degrees
    degrees = {}
    for ek in edge_graph.edge_keys:
        degrees[ek.source_id] = degrees.get(ek.source_id, 0) + 1
        degrees[ek.target_id] = degrees.get(ek.target_id, 0) + 1
    
    # Count triangles per edge
    adj = edge_graph._get_adjacency()
    
    ricci_sum = 0.0
    for ek in edge_graph.edge_keys:
        d_i = degrees.get(ek.source_id, 0)
        d_j = degrees.get(ek.target_id, 0)
        
        neighbors_i = set(adj.get(ek.source_id, []))
        neighbors_j = set(adj.get(ek.target_id, []))
        triangles = len(neighbors_i & neighbors_j)
        
        ric = 4 - d_i - d_j + 3 * triangles
        ricci_sum += ric ** 2
    
    return ricci_sum


def compute_smoothness(
    x: torch.Tensor,
    x_prev: torch.Tensor,
) -> float:
    """Compute smoothness ||x_t - x_{t-1}||_2.
    
    Args:
        x: Current embeddings, shape (m, p)
        x_prev: Previous embeddings, shape (m, p)
        
    Returns:
        Scalar smoothness (change magnitude)
    """
    return torch.norm(x - x_prev).item()


def compute_latency(
    violations: List[float],
    threshold: float,
    event_times: List[int],
) -> float:
    """Compute average recovery latency after events.
    
    Latency = number of steps until violation < threshold after event.
    
    Args:
        violations: Violation values at each time step
        threshold: Recovery threshold
        event_times: Time indices when events occurred
        
    Returns:
        Average latency (or 0.0 if no events)
    """
    if not event_times:
        return 0.0
    
    latencies = []
    for event_t in event_times:
        # Find recovery time
        for t in range(event_t + 1, len(violations)):
            if violations[t] < threshold:
                latencies.append(t - event_t)
                break
        else:
            # Never recovered
            latencies.append(len(violations) - event_t)
    
    return np.mean(latencies) if latencies else 0.0


def compute_collapse_score(x: torch.Tensor) -> float:
    """Compute collapse score (embedding variance).
    
    Higher score = less collapse = better.
    
    Args:
        x: Edge embeddings, shape (m, p)
        
    Returns:
        Mean variance across dimensions
    """
    if x.dim() == 1:
        x = x.unsqueeze(1)
    
    if x.shape[0] < 2:
        return 0.0
    
    # Variance per dimension, then mean
    var_per_dim = torch.var(x, dim=0)
    return var_per_dim.mean().item()


# ==============================================================================
# AGGREGATE METRICS
# ==============================================================================

def compute_all_metrics(
    x: torch.Tensor,
    x_obs: torch.Tensor,
    C: torch.Tensor,
    tau: torch.Tensor,
    edge_graph: "EdgeGraph",
    x_prev: Optional[torch.Tensor] = None,
) -> dict:
    """Compute all metrics for a single time step.
    
    Args:
        x: Stabilized embeddings
        x_obs: Observed embeddings
        C: Cycle matrix
        tau: Target values
        edge_graph: EdgeGraph instance
        x_prev: Previous embeddings (optional)
        
    Returns:
        Dictionary with all metric values
    """
    metrics = {
        "violation": compute_violation(x, C, tau),
        "drift": compute_drift(x, x_obs),
        "ricci_energy": compute_ricci_energy(x, edge_graph),
        "collapse_score": compute_collapse_score(x),
    }
    
    if x_prev is not None:
        metrics["smoothness"] = compute_smoothness(x, x_prev)
    else:
        metrics["smoothness"] = 0.0
    
    return metrics


def metrics_to_summary(metrics_list: List[dict]) -> dict:
    """Summarize metrics across time steps.
    
    Args:
        metrics_list: List of metric dictionaries
        
    Returns:
        Summary dictionary with mean/max values
    """
    if not metrics_list:
        return {}
    
    keys = metrics_list[0].keys()
    summary = {}
    
    for key in keys:
        values = [m[key] for m in metrics_list]
        summary[f"{key}_mean"] = np.mean(values)
        summary[f"{key}_max"] = np.max(values)
        summary[f"{key}_std"] = np.std(values)
    
    return summary
