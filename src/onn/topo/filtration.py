from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch


@dataclass
class BetaProfiles:
    tau_grid: np.ndarray
    beta0: np.ndarray
    beta1: np.ndarray


@dataclass
class TopoSummary:
    tau0_star: float
    tau1_star: float
    beta0_profile: np.ndarray
    beta1_profile: np.ndarray
    active_edge_profile: np.ndarray
    tau_grid: np.ndarray


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

    def num_components(self) -> int:
        roots = set(self.find(i) for i in range(len(self.parent)))
        return len(roots)


def compute_betti_numbers(
    num_nodes: int,
    edge_indices: np.ndarray,
    gates: np.ndarray,
    tau: float,
) -> Tuple[int, int]:
    """
    Compute β₀ (connected components) and β₁ (cycle rank) for graph G_τ.
    G_τ includes only edges where g_e >= τ.

    β₀ = number of connected components
    β₁ = |E_τ| - |V| + β₀  (cycle rank / cyclomatic number)
    """
    if num_nodes == 0:
        return 0, 0

    if len(edge_indices) == 0 or len(gates) == 0:
        return num_nodes, 0

    active_mask = gates >= tau
    active_edges = edge_indices[active_mask]
    num_active_edges = len(active_edges)

    if num_active_edges == 0:
        return num_nodes, 0

    uf = UnionFind(num_nodes)
    for src, tgt in active_edges:
        uf.union(int(src), int(tgt))

    beta0 = uf.num_components()
    beta1 = num_active_edges - num_nodes + beta0

    return beta0, max(0, beta1)


def compute_filtration_profiles(
    num_nodes: int,
    edge_indices: np.ndarray,
    gates: np.ndarray,
    tau_grid: Optional[np.ndarray] = None,
    num_tau_steps: int = 401,
) -> BetaProfiles:
    """
    Sweep τ from 1 → 0 and compute β₀(τ), β₁(τ).

    Higher resolution (default 401 steps) improves sensitivity to small gate changes.
    """
    if tau_grid is None:
        tau_grid = np.linspace(1.0, 0.0, num_tau_steps)

    beta0_arr = np.zeros(len(tau_grid), dtype=np.int32)
    beta1_arr = np.zeros(len(tau_grid), dtype=np.int32)

    for i, tau in enumerate(tau_grid):
        b0, b1 = compute_betti_numbers(num_nodes, edge_indices, gates, tau)
        beta0_arr[i] = b0
        beta1_arr[i] = b1

    return BetaProfiles(tau_grid=tau_grid, beta0=beta0_arr, beta1=beta1_arr)


def compute_tau_star(
    beta_profile: np.ndarray,
    tau_grid: np.ndarray,
) -> float:
    """
    Compute τ* = argmax_τ |Δβ(τ)| using smoothed finite differences
    for improved sensitivity across branches.
    """
    if len(beta_profile) < 2:
        return tau_grid[0] if len(tau_grid) > 0 else 0.5

    beta_f = beta_profile.astype(float)
    delta_raw = np.diff(beta_f)
    kernel = np.array([0.25, 0.5, 0.25])
    delta_smooth = np.convolve(delta_raw, kernel, mode="same")
    delta_beta = np.abs(delta_smooth)

    if np.max(delta_beta) == 0:
        return tau_grid[len(tau_grid) // 2]

    max_idx = int(np.argmax(delta_beta))
    return float(tau_grid[max_idx])


def compute_tau_star_v2(
    beta_profile: np.ndarray,
    tau_grid: np.ndarray,
    window_ratio: float = 0.1,
) -> float:
    """
    Compute τ* using the center of mass of the main peak of the |Δβ(τ)| profile.

    This method is robust to noise and minor secondary peaks by focusing only on
    the region around the most significant event in the filtration.

    Args:
        beta_profile: The Betti number profile, shape (n,).
        tau_grid: The filtration values (τ), shape (n,).
        window_ratio: The fraction of the total grid to use as a window
            around the main peak. e.g., 0.1 for a +/- 5% window.

    Returns:
        The computed τ* value.

    Method:
    1. Compute and smooth Δβ as in v1.
    2. Find the argmax of w(τ) = |Δβ_smooth(τ)| to locate the main peak.
    3. Define a window around this peak.
    4. Compute the center of mass (weighted average) on this windowed support.
    """
    if len(beta_profile) < 2:
        return tau_grid[0] if len(tau_grid) > 0 else 0.5

    beta_f = beta_profile.astype(float)
    delta_raw = np.diff(beta_f)
    kernel = np.array([0.25, 0.5, 0.25])
    delta_smooth = np.convolve(delta_raw, kernel, mode="same")
    delta_beta = np.abs(delta_smooth)

    total_delta = np.sum(delta_beta)
    if total_delta == 0:
        return tau_grid[len(tau_grid) // 2]

    # Find the main peak
    peak_idx = np.argmax(delta_beta)

    # Define a window around the peak
    grid_size = len(delta_beta)
    window_size = int(grid_size * window_ratio / 2)
    start_idx = max(0, peak_idx - window_size)
    end_idx = min(grid_size, peak_idx + window_size + 1)

    mask = np.zeros_like(delta_beta, dtype=bool)
    mask[start_idx:end_idx] = True

    filtered_deltas = delta_beta[mask]

    if np.sum(filtered_deltas) == 0:
        return tau_grid[peak_idx]  # Fallback to argmax if window is empty

    tau_midpoints = (tau_grid[:-1] + tau_grid[1:]) / 2.0
    filtered_midpoints = tau_midpoints[mask]

    return float(np.average(filtered_midpoints, weights=filtered_deltas))


def compute_topo_summary(
    num_nodes: int,
    edge_indices: np.ndarray,
    gates: np.ndarray,
    tau_grid: Optional[np.ndarray] = None,
) -> TopoSummary:
    """Compute full topological summary including τ*₀, τ*₁."""
    profiles = compute_filtration_profiles(num_nodes, edge_indices, gates, tau_grid)

    tau0_star = compute_tau_star(profiles.beta0, profiles.tau_grid)
    tau1_star = compute_tau_star(profiles.beta1, profiles.tau_grid)

    active_edge_profile = np.array([np.sum(gates >= tau) for tau in profiles.tau_grid])

    return TopoSummary(
        tau0_star=tau0_star,
        tau1_star=tau1_star,
        beta0_profile=profiles.beta0,
        beta1_profile=profiles.beta1,
        active_edge_profile=active_edge_profile,
        tau_grid=profiles.tau_grid,
    )


def gates_to_numpy(gates: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(gates, torch.Tensor):
        return gates.detach().cpu().numpy()
    return np.asarray(gates)


def edge_indices_to_numpy(edge_indices: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(edge_indices, torch.Tensor):
        return edge_indices.detach().cpu().numpy()
    return np.asarray(edge_indices)
