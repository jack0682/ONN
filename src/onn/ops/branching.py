from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import numpy as np
import torch
import copy


@dataclass
class StagnationSignal:
    stable: bool
    reasons: Dict[str, bool]


@dataclass
class StagnationConfig:
    window_size: int = 10
    eps_slope: float = 0.01
    eps_gate_std: float = 0.01
    eps_u_std: float = 0.05
    eps_tau_var: float = 0.01
    min_history_len: int = 5


class StagnationDetector:
    def __init__(self, config: Optional[StagnationConfig] = None):
        self.config = config or StagnationConfig()

    def detect(
        self,
        residual_history: List[float],
        gate_history: List[float],
        uncertainty_history: List[float],
        tau0_star_history: Optional[List[float]] = None,
        tau1_star_history: Optional[List[float]] = None,
    ) -> StagnationSignal:
        reasons = {
            "plateau": False,
            "saturation": False,
            "topo_lock": False,
        }

        if len(residual_history) < self.config.min_history_len:
            return StagnationSignal(stable=False, reasons=reasons)

        W = min(self.config.window_size, len(residual_history))

        plateau = self._check_residual_plateau(residual_history, W)
        saturation = self._check_mechanism_saturation(
            gate_history, uncertainty_history, W
        )
        topo_lock = self._check_topo_lock(tau0_star_history, tau1_star_history)

        reasons["plateau"] = plateau
        reasons["saturation"] = saturation
        reasons["topo_lock"] = topo_lock

        stable = plateau or saturation or topo_lock

        return StagnationSignal(stable=stable, reasons=reasons)

    def _check_residual_plateau(self, residual_history: List[float], W: int) -> bool:
        if len(residual_history) < W:
            return False

        recent = residual_history[-W:]
        r_start = max(recent[0], 1e-10)
        r_end = max(recent[-1], 1e-10)

        slope = (np.log(r_end) - np.log(r_start)) / W
        return slope > -self.config.eps_slope

    def _check_mechanism_saturation(
        self,
        gate_history: List[float],
        uncertainty_history: List[float],
        W: int,
    ) -> bool:
        if len(gate_history) < W or len(uncertainty_history) < W:
            return False

        gate_std = np.std(gate_history[-W:])
        u_std = np.std(uncertainty_history[-W:])

        return gate_std < self.config.eps_gate_std and u_std < self.config.eps_u_std

    def _check_topo_lock(
        self,
        tau0_star_history: Optional[List[float]],
        tau1_star_history: Optional[List[float]],
    ) -> bool:
        if tau0_star_history is None or tau1_star_history is None:
            return False
        if len(tau0_star_history) < 3 or len(tau1_star_history) < 3:
            return False

        tau0_var = (
            np.var(tau0_star_history[-5:])
            if len(tau0_star_history) >= 5
            else np.var(tau0_star_history)
        )
        tau1_var = (
            np.var(tau1_star_history[-5:])
            if len(tau1_star_history) >= 5
            else np.var(tau1_star_history)
        )

        return tau0_var < self.config.eps_tau_var and tau1_var < self.config.eps_tau_var


class BranchType(Enum):
    CONSERVATIVE = "conservative"
    EXPLORATION = "exploration"
    REWIRE = "rewire"


@dataclass
class BranchConfig:
    branch_type: BranchType
    seed: int

    gate_lr: float = 0.1
    gate_threshold: float = 0.5
    gate_min: float = 0.0
    gate_max: float = 1.0
    gate_reopen_evidence: float = 0.3

    uncertainty_lr: float = 0.05
    uncertainty_min: float = 0.01
    uncertainty_max: float = 10.0
    uncertainty_target_residual: float = 0.1

    min_active_edge_ratio: float = 0.3
    min_mean_gate: float = 0.2

    mutation_sigma: float = 0.1
    mutation_delta: float = 0.15
    apply_mutation: bool = False


def create_conservative_config(seed: int) -> BranchConfig:
    return BranchConfig(
        branch_type=BranchType.CONSERVATIVE,
        seed=seed,
        gate_lr=0.05,
        gate_threshold=0.3,
        gate_reopen_evidence=0.5,
        uncertainty_lr=0.02,
        uncertainty_max=3.0,
        min_active_edge_ratio=0.4,
        min_mean_gate=0.3,
        apply_mutation=False,
    )


def create_exploration_config(seed: int) -> BranchConfig:
    return BranchConfig(
        branch_type=BranchType.EXPLORATION,
        seed=seed,
        gate_lr=0.15,
        gate_threshold=0.5,
        mutation_sigma=0.2,
        mutation_delta=0.2,
        uncertainty_lr=0.1,
        uncertainty_max=15.0,
        apply_mutation=True,
    )


def create_rewire_config(seed: int) -> BranchConfig:
    return BranchConfig(
        branch_type=BranchType.REWIRE,
        seed=seed,
        gate_lr=0.1,
        gate_threshold=0.4,
        mutation_sigma=0.15,
        mutation_delta=0.1,
        apply_mutation=True,
    )


def boundary_gate_mutation(
    gates: np.ndarray,
    tau_star: float,
    delta: float,
    sigma: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Mutate gates near τ* boundary in logit space.
    Only edges with |g - τ*| < δ are mutated.
    """
    gates = gates.copy()
    near_boundary = np.abs(gates - tau_star) < delta

    if not np.any(near_boundary):
        return gates

    eps = 1e-6
    g_clipped = np.clip(gates[near_boundary], eps, 1.0 - eps)
    logit_g = np.log(g_clipped / (1.0 - g_clipped))

    noise = rng.normal(0, sigma, size=logit_g.shape)
    logit_g_mutated = logit_g + noise

    gates[near_boundary] = 1.0 / (1.0 + np.exp(-logit_g_mutated))
    gates = np.clip(gates, 0.0, 1.0)

    return gates


def meta_param_mutation(
    params: Dict[str, float],
    sigma_dict: Dict[str, float],
    rng: np.random.RandomState,
) -> Dict[str, float]:
    mutated = {}
    for key, val in params.items():
        sigma = sigma_dict.get(key, 0.1)
        mutated[key] = val + rng.normal(0, sigma * abs(val + 1e-6))
    return mutated


def _connected_components(
    num_nodes: int, edge_indices: np.ndarray, gates: np.ndarray, tau: float = 0.3
) -> List[set]:
    parent = list(range(num_nodes))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    active_edges = edge_indices[gates >= tau] if len(edge_indices) > 0 else []
    for src, tgt in active_edges:
        union(int(src), int(tgt))

    comp_map: Dict[int, List[int]] = {}
    for i in range(num_nodes):
        r = find(i)
        comp_map.setdefault(r, []).append(i)

    return [set(v) for v in comp_map.values()]


def topology_rewire_mutation(
    gates: np.ndarray,
    edge_indices: np.ndarray,
    num_nodes: int,
    beta0_current: int,
    beta1_current: int,
    rng: np.random.RandomState,
    target_beta0: Optional[int] = None,
    boost_cycles: bool = False,
    tau_connect: float = 0.3,
) -> np.ndarray:
    gates = gates.copy()

    if len(edge_indices) == 0 or num_nodes == 0:
        return gates

    components = _connected_components(num_nodes, edge_indices, gates, tau=tau_connect)

    if beta0_current > 1 and target_beta0 is not None and target_beta0 < beta0_current:
        comp_ids = list(range(len(components)))
        if len(comp_ids) >= 2:
            src_comp = components[0]
            tgt_comp = components[1]
            bridge_candidates = []
            for idx, (u, v) in enumerate(edge_indices):
                if gates[idx] < tau_connect:
                    if (u in src_comp and v in tgt_comp) or (
                        u in tgt_comp and v in src_comp
                    ):
                        bridge_candidates.append(idx)
            if bridge_candidates:
                chosen = rng.choice(bridge_candidates, size=1, replace=False)
                gates[chosen] = np.clip(gates[chosen] + 0.5, 0, 1)

    if boost_cycles and len(gates) > 0:
        cycle_candidates = []
        for idx, (u, v) in enumerate(edge_indices):
            if gates[idx] < 0.7:
                # if u and v already connected via another path, this edge can form/strengthen a cycle
                comps = _connected_components(
                    num_nodes,
                    edge_indices[np.arange(len(edge_indices)) != idx],
                    gates[np.arange(len(gates)) != idx],
                    tau=tau_connect,
                )
                same_component = any((u in c and v in c) for c in comps)
                if same_component:
                    cycle_candidates.append(idx)
        if cycle_candidates:
            chosen = rng.choice(
                cycle_candidates, size=min(2, len(cycle_candidates)), replace=False
            )
            gates[chosen] = np.clip(gates[chosen] + 0.3, 0, 1)

    return gates


@dataclass
class Branch:
    config: BranchConfig
    gates: np.ndarray
    meta_params: Dict[str, float]
    rng: np.random.RandomState


@dataclass
class BranchResult:
    branch: Branch
    final_gates: np.ndarray
    final_uncertainty: float
    active_edge_ratio: float
    mean_gate: float
    fitness: float
    converged: bool
    has_nan: bool
    tau0_star: float
    tau1_star: float
    beta0_final: int
    beta1_final: int


class BranchFactory:
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed

    def make_branches(
        self,
        parent_gates: np.ndarray,
        parent_meta_params: Dict[str, float],
        tau_star: float,
        edge_indices: Optional[np.ndarray] = None,
        num_nodes: int = 0,
        beta0_current: int = 1,
        beta1_current: int = 0,
        num_branches: int = 3,
    ) -> List[Branch]:
        branches = []

        configs = [
            create_conservative_config(self.base_seed),
            create_exploration_config(self.base_seed + 1),
            create_rewire_config(self.base_seed + 2),
        ]

        edge_indices = (
            np.asarray(edge_indices) if edge_indices is not None else np.array([])
        )

        for i, cfg in enumerate(configs[:num_branches]):
            rng = np.random.RandomState(cfg.seed)
            gates = parent_gates.copy()
            meta = parent_meta_params.copy()

            if cfg.apply_mutation:
                if cfg.branch_type == BranchType.EXPLORATION:
                    gates = boundary_gate_mutation(
                        gates, tau_star, cfg.mutation_delta, cfg.mutation_sigma, rng
                    )
                elif cfg.branch_type == BranchType.REWIRE:
                    gates = topology_rewire_mutation(
                        gates,
                        edge_indices=edge_indices,
                        num_nodes=num_nodes,
                        beta0_current=beta0_current,
                        beta1_current=beta1_current,
                        rng=rng,
                        boost_cycles=True,
                    )

            branches.append(Branch(config=cfg, gates=gates, meta_params=meta, rng=rng))

        return branches


@dataclass
class SurvivalConfig:
    min_active_edge_ratio: float = 0.2
    min_mean_gate: float = 0.1
    u_max_threshold: float = 10.0


from collections import deque


@dataclass
class DynamicBranchManagerConfig:
    spawn_threshold: float = 0.5  # Spawn if >50% of branches are stagnating
    stagnation_config: StagnationConfig = field(default_factory=StagnationConfig)
    max_branches: int = 10
    history_len: int = 20


class DynamicBranchManager:
    """
    Manages the lifecycle of branches, including dynamic spawning.
    """

    def __init__(
        self,
        initial_branches: List[Branch],
        config: Optional[DynamicBranchManagerConfig] = None,
        base_seed: int = 42,
    ):
        self.config = config or DynamicBranchManagerConfig()
        self.stagnation_detector = StagnationDetector(self.config.stagnation_config)
        self.branch_factory = BranchFactory(base_seed=base_seed)
        self.next_branch_id = 0

        self.branches: Dict[int, Branch] = {}
        self.history: Dict[int, Dict[str, deque]] = {}

        for branch in initial_branches:
            self.add_branch(branch)

    def add_branch(self, branch: Branch) -> int:
        """Adds a new branch to the manager and returns its ID."""
        if len(self.branches) >= self.config.max_branches:
            return -1

        branch_id = self.next_branch_id
        self.branches[branch_id] = branch
        self.history[branch_id] = {
            "fitness": deque(maxlen=self.config.history_len),
            "residual": deque(maxlen=self.config.history_len),
            "gate": deque(maxlen=self.config.history_len),
            "uncertainty": deque(maxlen=self.config.history_len),
            "tau0_star": deque(maxlen=self.config.history_len),
            "tau1_star": deque(maxlen=self.config.history_len),
        }
        self.next_branch_id += 1
        return branch_id

    def get_branch_by_config(
        self, config: BranchConfig
    ) -> Optional[Tuple[int, Branch]]:
        """Utility to find a branch by its config object."""
        for branch_id, branch in self.branches.items():
            if branch.config is config:
                return branch_id, branch
        return None

    def step(
        self,
        results: List[BranchResult],
        parent_gates: np.ndarray,
        parent_meta_params: Dict[str, float],
        parent_tau_star: float,
        edge_indices: np.ndarray,
        num_nodes: int,
    ) -> List[Branch]:
        """
        Process results, check for stagnation, and spawn new branches if needed.
        Returns a list of newly spawned branches.
        """
        self._update_histories(results)

        num_stagnated = 0
        for branch_id in self.branches:
            hist = self.history.get(branch_id)
            if not hist:
                continue

            signal = self.stagnation_detector.detect(
                residual_history=list(hist["residual"]),
                gate_history=list(hist["gate"]),
                uncertainty_history=list(hist["uncertainty"]),
                tau0_star_history=list(hist["tau0_star"]),
                tau1_star_history=list(hist["tau1_star"]),
            )
            if signal.stable:
                num_stagnated += 1

        spawned_branches = []
        stagnation_ratio = num_stagnated / len(self.branches) if self.branches else 0
        if stagnation_ratio >= self.config.spawn_threshold:
            parent_result = self._sample_parent_branch(results)
            if parent_result:
                new_branch = self._spawn_exploration_branch(
                    parent_branch=parent_result.branch,
                    parent_gates=parent_result.final_gates,
                    parent_meta_params=parent_result.branch.meta_params,
                    parent_tau_star=parent_result.tau0_star,
                    edge_indices=edge_indices,
                    num_nodes=num_nodes,
                )
                if new_branch:
                    branch_id = self.add_branch(new_branch)
                    if branch_id != -1:
                        spawned_branches.append(new_branch)

        return spawned_branches

    def _update_histories(self, results: List[BranchResult]):
        for r in results:
            branch_info = self.get_branch_by_config(r.branch.config)
            if branch_info is None:
                continue
            branch_id, _ = branch_info

            self.history[branch_id]["fitness"].append(r.fitness)
            self.history[branch_id]["tau0_star"].append(r.tau0_star)
            self.history[branch_id]["tau1_star"].append(r.tau1_star)
            self.history[branch_id]["gate"].append(r.mean_gate)
            self.history[branch_id]["uncertainty"].append(r.final_uncertainty)
            # Assuming 'residual' is part of fitness or needs to be passed in.
            # For now, using fitness as a proxy.
            self.history[branch_id]["residual"].append(1.0 / (r.fitness + 1e-6))

    def _sample_parent_branch(
        self, results: List[BranchResult], k: int = 3
    ) -> Optional[BranchResult]:
        """
        Selects a parent branch from the top-k survivors using fitness-proportional sampling.
        Returns the entire BranchResult of the chosen parent.
        """
        survivors = sorted(results, key=lambda r: r.fitness, reverse=True)
        if not survivors:
            return None

        top_k = survivors[:k]

        # Fitness normalization (softmax)
        fitness_values = np.array([r.fitness for r in top_k])
        # Shift to avoid overflow and handle negative fitness
        fitness_values = fitness_values - np.max(fitness_values)
        exp_fitness = np.exp(fitness_values)
        probabilities = exp_fitness / (np.sum(exp_fitness) + 1e-9)

        parent_result = np.random.choice(top_k, p=probabilities)

        return parent_result

    def _spawn_exploration_branch(
        self,
        parent_branch: Branch,
        parent_gates: np.ndarray,
        parent_meta_params: Dict[str, float],
        parent_tau_star: float,
        edge_indices: np.ndarray,
        num_nodes: int,
    ) -> Optional[Branch]:
        """Creates a single new exploration branch."""
        seed = self.branch_factory.base_seed + self.next_branch_id
        cfg = create_exploration_config(seed)

        rng = np.random.RandomState(cfg.seed)

        mutated_gates = boundary_gate_mutation(
            parent_gates, parent_tau_star, cfg.mutation_delta, cfg.mutation_sigma, rng
        )

        new_branch = Branch(
            config=cfg,
            gates=mutated_gates,
            meta_params=parent_meta_params.copy(),
            rng=rng,
        )
        return new_branch


class BranchSelector:
    def __init__(self, survival_config: Optional[SurvivalConfig] = None):
        self.survival_config = survival_config or SurvivalConfig()

    def select(
        self,
        results: List[BranchResult],
    ) -> Tuple[Optional[BranchResult], Optional[BranchResult]]:
        survivors = []

        for r in results:
            if self._passes_survival_filter(r):
                survivors.append(r)

        if not survivors:
            return None, None

        survivors.sort(key=lambda r: r.fitness, reverse=True)

        winner = survivors[0]
        backup = survivors[1] if len(survivors) > 1 else None

        return winner, backup

    def _passes_survival_filter(self, r: BranchResult) -> bool:
        if r.has_nan:
            return False

        if r.active_edge_ratio < self.survival_config.min_active_edge_ratio:
            return False

        if r.mean_gate < self.survival_config.min_mean_gate:
            return False

        if (
            r.final_uncertainty >= self.survival_config.u_max_threshold
            and not r.converged
        ):
            return False

        return True


@dataclass
class AdaptationConfig:
    min_history_len: int = 10
    steep_slope_threshold: float = -0.5
    plateau_slope_threshold: float = -0.05
    lr_increase_factor: float = 1.1
    lr_decrease_factor: float = 0.99
    min_gate_lr: float = 0.001
    max_gate_lr: float = 0.5
    trust_region_pct: float = 0.1  # Limit adaptation to ±10% of ES proposal


class ParameterAdaptationManager:
    """
    Manages self-tuning of meta-parameters like learning rates based on performance,
    while respecting a trust region around the ES-proposed values to avoid control conflict.
    """

    def __init__(self, config: Optional[AdaptationConfig] = None):
        self.config = config or AdaptationConfig()

    def adapt(
        self,
        meta_params: Dict[str, float],
        residual_history: List[float],
        es_meta_params: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, float], str]:
        """
        Adapts meta-parameters based on residual slope with a trust region around ES proposals.

        Returns:
            (adapted_meta_params, reason)
        """
        if len(residual_history) < self.config.min_history_len:
            return meta_params, "insufficient_history"

        es_meta_params = es_meta_params or meta_params

        W = self.config.min_history_len
        recent_residuals = residual_history[-W:]

        r_start = max(recent_residuals[0], 1e-10)
        r_end = max(recent_residuals[-1], 1e-10)

        log_slope = (np.log(r_end) - np.log(r_start)) / W

        new_meta_params = meta_params.copy()
        current_gate_lr = new_meta_params.get("gate_lr", 0.1)
        es_gate_lr = es_meta_params.get("gate_lr", current_gate_lr)

        reason = "stable"
        if log_slope < self.config.steep_slope_threshold:
            current_gate_lr *= self.config.lr_decrease_factor
            reason = "improving"
        elif log_slope > self.config.plateau_slope_threshold:
            current_gate_lr *= self.config.lr_increase_factor
            reason = "plateau"

        current_gate_lr = np.clip(
            current_gate_lr, self.config.min_gate_lr, self.config.max_gate_lr
        )

        # Trust-region clamp relative to ES proposal
        lower = es_gate_lr * (1 - self.config.trust_region_pct)
        upper = es_gate_lr * (1 + self.config.trust_region_pct)
        current_gate_lr = float(np.clip(current_gate_lr, lower, upper))

        new_meta_params["gate_lr"] = current_gate_lr
        return new_meta_params, reason
