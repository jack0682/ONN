"""
CSA (Correction-Stabilization-Action) Pipeline with ONN-ES Integration.

Integrates the full ONN pipeline:
    Sensor → SEGO → EdgeStabilizer → LOGOS → IMAGO → Action

With ES optimization loop:
    ES → W_lin → EdgeStabilizer → Fitness → ES

This module provides:
1. CSAPipeline: Full pipeline execution
2. CSAWithES: Pipeline + ES optimization loop
3. Episode generation for ES training

Reference:
    - spec/10_architecture.ir.yml: CSA pipeline
    - User roadmap: "ONN-ES CSA 파이프라인 연결"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Callable, TYPE_CHECKING

import torch
import numpy as np

from onn.core.tensors import (
    SensorObservation,
    RawSemanticGraph,
    StabilizedGraph,
    ReasoningTrace,
    MissionGoal,
)
from onn.ops.sego_anchor import SEGOGaugeAnchor, SEGOConfig, Detection
from onn.ops.logos_solver import LOGOSSolver, LOGOSConfig
from onn.ops.imago_planner import IMAGOPlanner, IMAGOConfig
from onn.ops.edge_stabilizer import (
    EdgeStabilizer,
    EdgeStabilizerConfig,
    EdgeStabilizerResult,
    stabilized_to_relation_embeddings,
)
from onn.es import (
    CMAESTrainer,
    ESConfig,
    Candidate,
    EpisodeStep,
    FitnessConfig,
    create_trainer_with_w_lin,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CSAConfig:
    """Configuration for the CSA pipeline."""

    # Component configs
    sego_config: Optional[SEGOConfig] = None
    edge_stabilizer_config: Optional[EdgeStabilizerConfig] = None
    logos_config: Optional[LOGOSConfig] = None
    imago_config: Optional[IMAGOConfig] = None

    # Pipeline options
    enable_edge_stabilizer: bool = True
    warm_start: bool = True


@dataclass
class CSAResult:
    """Result from CSA pipeline execution."""

    raw_graph: RawSemanticGraph
    stabilized_graph: StabilizedGraph
    reasoning_trace: Optional[ReasoningTrace]
    edge_stabilizer_result: Optional[EdgeStabilizerResult]

    # Metrics for ES
    logos_converged: bool = False
    logos_iterations: int = 0
    logos_energy: float = 0.0
    edge_violation: float = 0.0


# =============================================================================
# CSA Pipeline
# =============================================================================


class CSAPipeline:
    """Full CSA pipeline: SEGO → EdgeStabilizer → LOGOS → IMAGO.

    This pipeline integrates ONN-ES edge stabilization with the
    existing CSA components.

    Example:
        >>> pipeline = CSAPipeline(config)
        >>> result = pipeline.process(observation, goal)
        >>> # result.reasoning_trace contains the planned action
    """

    def __init__(self, config: Optional[CSAConfig] = None):
        """Initialize the CSA pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or CSAConfig()

        # Initialize components
        self.sego = SEGOGaugeAnchor(self.config.sego_config)
        self.edge_stabilizer = EdgeStabilizer(self.config.edge_stabilizer_config)
        self.logos = LOGOSSolver(self.config.logos_config)
        self.imago = IMAGOPlanner(self.config.imago_config)

        self._last_result: Optional[CSAResult] = None
        self._prev_edge_embeddings: Optional[torch.Tensor] = None

        logger.info("CSA Pipeline initialized with EdgeStabilizer")

    def process(
        self,
        observation: SensorObservation,
        goal: Optional[MissionGoal] = None,
        detections: Optional[List[Detection]] = None,
    ) -> CSAResult:
        """Process sensor observation through the full pipeline.

        Args:
            observation: Raw sensor data
            goal: Mission goal (required for IMAGO planning)
            detections: Optional pre-computed detections

        Returns:
            CSAResult with all intermediate and final outputs
        """
        # Step 1: SEGO - Perception
        raw_graph = self.sego.process(observation, detections)
        logger.debug(
            f"SEGO: {len(raw_graph.nodes)} nodes, {len(raw_graph.edge_candidates)} edges"
        )

        # Step 2: EdgeStabilizer - Edge embedding stabilization (ONN-ES)
        edge_stabilizer_result = None
        if self.config.enable_edge_stabilizer and raw_graph.edge_candidates:
            edge_stabilizer_result = self.edge_stabilizer.stabilize(
                raw_graph,
                x_prev=self._prev_edge_embeddings,
                warm_start=self.config.warm_start,
            )

            # Update edge relation_embeddings with stabilized values
            raw_graph = RawSemanticGraph(
                timestamp_ns=raw_graph.timestamp_ns,
                nodes=raw_graph.nodes,
                edge_candidates=stabilized_to_relation_embeddings(
                    edge_stabilizer_result,
                    raw_graph.edge_candidates,
                ),
            )

            # Cache for next frame
            self._prev_edge_embeddings = edge_stabilizer_result.stabilized_embeddings

        # Step 3: LOGOS - Constraint stabilization
        stabilized_graph = self.logos.solve(
            raw_graph, warm_start=self.config.warm_start
        )
        logger.debug(
            f"LOGOS: converged={stabilized_graph.is_valid}, "
            f"energy={stabilized_graph.global_energy:.4f}"
        )

        # Step 4: IMAGO - Planning (if goal provided and graph valid)
        reasoning_trace = None
        if goal is not None and stabilized_graph.is_valid:
            try:
                reasoning_trace = self.imago.plan(stabilized_graph, goal)
                logger.debug(
                    f"IMAGO: trace generated, curvature={reasoning_trace.curvature:.4f}"
                )
            except ValueError as e:
                logger.warning(f"IMAGO planning failed: {e}")

        # Build result
        result = CSAResult(
            raw_graph=raw_graph,
            stabilized_graph=stabilized_graph,
            reasoning_trace=reasoning_trace,
            edge_stabilizer_result=edge_stabilizer_result,
            logos_converged=stabilized_graph.is_valid,
            logos_iterations=stabilized_graph.iterations_used,
            logos_energy=stabilized_graph.global_energy,
            edge_violation=(
                edge_stabilizer_result.solver_result.breakdown.get("context", 0.0)
                if edge_stabilizer_result
                else 0.0
            ),
        )

        self._last_result = result
        return result

    def set_w_lin(self, w_lin: torch.Tensor) -> None:
        """Set W_lin in the EdgeStabilizer (for ES optimization)."""
        self.edge_stabilizer.set_w_lin(w_lin)

    def get_w_lin(self) -> Optional[torch.Tensor]:
        """Get current W_lin from EdgeStabilizer."""
        return self.edge_stabilizer.get_w_lin()

    def get_w_lin_shape(self) -> Tuple[int, int]:
        """Get W_lin shape for ES configuration."""
        return self.edge_stabilizer.get_w_lin_shape()

    def reset(self) -> None:
        """Reset all pipeline components."""
        self.sego.reset()
        self.edge_stabilizer.reset()
        self.logos.reset()
        self._prev_edge_embeddings = None
        self._last_result = None


# =============================================================================
# CSA with ES Optimization
# =============================================================================


@dataclass
class ESEpisode:
    """Episode data for ES fitness evaluation."""

    observations: List[SensorObservation]
    goals: List[Optional[MissionGoal]]
    detections: List[Optional[List[Detection]]]
    ground_truth: Optional[Dict] = None  # For supervised metrics


class CSAWithES:
    """CSA Pipeline with ES optimization loop.

    Optimizes W_lin in EdgeStabilizer to minimize pipeline loss.

    Example:
        >>> csa_es = CSAWithES(csa_config, es_config)
        >>> csa_es.train(episodes, num_generations=10)
        >>> # Pipeline now uses optimized W_lin
    """

    def __init__(
        self,
        csa_config: Optional[CSAConfig] = None,
        es_population_size: int = 16,
        es_seed: int = 42,
    ):
        """Initialize CSA with ES optimization.

        Args:
            csa_config: CSA pipeline configuration
            es_population_size: ES population size
            es_seed: Random seed for ES
        """
        self.pipeline = CSAPipeline(csa_config)
        self.fitness_config = FitnessConfig()

        # Create ES trainer with W_lin dimensions
        w_lin_shape = self.pipeline.get_w_lin_shape()
        self.trainer = create_trainer_with_w_lin(
            w_lin_shape=w_lin_shape,
            population_size=es_population_size,
            seed=es_seed,
        )

        self._generation = 0
        self._fitness_history: List[float] = []

        logger.info(
            f"CSAWithES initialized: W_lin shape={w_lin_shape}, "
            f"population={es_population_size}"
        )

    def evaluate_candidate(
        self,
        candidate: Candidate,
        episodes: List[ESEpisode],
    ) -> float:
        """Evaluate a candidate on multiple episodes.

        Args:
            candidate: ES candidate with hyperparams and W_lin
            episodes: List of episodes to evaluate on

        Returns:
            Fitness score (higher = better)
        """
        # Apply candidate's W_lin to pipeline
        if candidate.w_lin is not None:
            w_lin_tensor = torch.tensor(candidate.w_lin, dtype=torch.float32)
            self.pipeline.set_w_lin(w_lin_tensor)

        total_fitness = 0.0
        num_steps = 0

        for episode in episodes:
            self.pipeline.reset()

            for obs, goal, dets in zip(
                episode.observations,
                episode.goals,
                episode.detections,
            ):
                result = self.pipeline.process(obs, goal, dets)

                # Compute step fitness
                step_fitness = self._compute_step_fitness(result)
                total_fitness += step_fitness
                num_steps += 1

        return total_fitness / max(num_steps, 1)

    def _compute_step_fitness(self, result: CSAResult) -> float:
        """Compute fitness for a single pipeline step.

        Fitness = -violation - drift + collapse_score - latency

        Args:
            result: CSA pipeline result

        Returns:
            Step fitness (higher = better)
        """
        cfg = self.fitness_config

        # Edge violation (from EdgeStabilizer)
        violation = result.edge_violation

        # LOGOS energy as drift proxy
        drift = result.logos_energy

        # Convergence bonus
        converged_bonus = 1.0 if result.logos_converged else 0.0

        # Edge embedding variance (anti-collapse)
        if result.edge_stabilizer_result is not None:
            emb = result.edge_stabilizer_result.stabilized_embeddings
            collapse_score = torch.var(emb).item() if emb.numel() > 0 else 0.0
        else:
            collapse_score = 0.0

        fitness = (
            -cfg.w_violation * violation
            - cfg.w_drift * drift
            + cfg.w_collapse * collapse_score
            + converged_bonus
        )

        return fitness

    def train_step(self, episodes: List[ESEpisode]) -> Dict:
        """Run one ES generation.

        Args:
            episodes: Training episodes

        Returns:
            Training metrics
        """
        # Ask for candidates
        candidates = self.trainer.ask()

        # Evaluate each candidate
        fitnesses = []
        for candidate in candidates:
            fitness = self.evaluate_candidate(candidate, episodes)
            fitnesses.append(fitness)

        # Tell ES the results
        self.trainer.tell(candidates, fitnesses)

        self._generation += 1
        self._fitness_history.append(self.trainer.best_fitness)

        # Apply best W_lin to pipeline
        best_candidate, best_fitness = self.trainer.get_best()
        if best_candidate.w_lin is not None:
            w_lin_tensor = torch.tensor(best_candidate.w_lin, dtype=torch.float32)
            self.pipeline.set_w_lin(w_lin_tensor)

        return {
            "generation": self._generation,
            "best_fitness": best_fitness,
            "mean_fitness": np.mean(fitnesses),
            "std_fitness": np.std(fitnesses),
        }

    def train(
        self,
        episodes: List[ESEpisode],
        num_generations: int = 10,
        callback: Optional[Callable[[Dict], None]] = None,
    ) -> Dict:
        """Train ES for multiple generations.

        Args:
            episodes: Training episodes
            num_generations: Number of ES generations
            callback: Optional callback called after each generation

        Returns:
            Final training metrics
        """
        logger.info(f"Starting ES training: {num_generations} generations")

        for gen in range(num_generations):
            metrics = self.train_step(episodes)

            if callback:
                callback(metrics)

            logger.info(
                f"Generation {metrics['generation']}: "
                f"best={metrics['best_fitness']:.4f}, "
                f"mean={metrics['mean_fitness']:.4f}"
            )

        return self.trainer.get_report()

    def get_trained_w_lin(self) -> Optional[torch.Tensor]:
        """Get the optimized W_lin."""
        return self.pipeline.get_w_lin()

    def get_report(self) -> Dict:
        """Get training report."""
        return {
            "generations": self._generation,
            "fitness_history": self._fitness_history,
            **self.trainer.get_report(),
        }


# =============================================================================
# Episode Generation Utilities
# =============================================================================


def create_synthetic_episode(
    num_steps: int = 10,
    num_nodes: int = 5,
    image_shape: Tuple[int, int] = (480, 640),
    seed: Optional[int] = None,
) -> ESEpisode:
    """Create a synthetic episode for testing/training.

    Args:
        num_steps: Number of steps in the episode
        num_nodes: Number of objects per frame
        image_shape: Image dimensions (H, W)
        seed: Random seed

    Returns:
        Synthetic ESEpisode
    """
    rng = np.random.RandomState(seed)
    h, w = image_shape

    observations = []
    goals = []
    detections_list = []

    for t in range(num_steps):
        # Generate synthetic observation
        rgb = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
        depth = rng.rand(h, w).astype(np.float32) * 5.0

        obs = SensorObservation(
            timestamp_ns=t * 100_000_000,  # 100ms intervals
            frame_id="robot_base",
            rgb_images=[rgb],
            depth_maps=[depth],
        )
        observations.append(obs)

        # Generate synthetic detections
        dets = []
        for i in range(num_nodes):
            x = rng.randint(0, w - 100)
            y = rng.randint(0, h - 100)
            det_w = rng.randint(50, 150)
            det_h = rng.randint(50, 150)

            dets.append(
                Detection(
                    detection_id=i,
                    class_name=f"object_{i}",
                    confidence=rng.uniform(0.5, 1.0),
                    bbox=(x, y, det_w, det_h),
                    centroid_3d=np.array([x / w, y / h, rng.uniform(0.5, 3.0)]),
                )
            )
        detections_list.append(dets)

        # Goal for some steps
        if t % 3 == 0:
            goals.append(
                MissionGoal(
                    goal_id=f"goal_{t}",
                    verb="GRASP",
                    target_node_id=rng.randint(1, num_nodes + 1),
                )
            )
        else:
            goals.append(None)

    return ESEpisode(
        observations=observations,
        goals=goals,
        detections=detections_list,
    )


# =============================================================================
# Factory Functions
# =============================================================================


def create_default_csa_pipeline() -> CSAPipeline:
    """Create CSA pipeline with default configuration."""
    return CSAPipeline(CSAConfig())


def create_csa_with_es(
    population_size: int = 16,
    seed: int = 42,
) -> CSAWithES:
    """Create CSA pipeline with ES optimization.

    Args:
        population_size: ES population size
        seed: Random seed

    Returns:
        CSAWithES instance
    """
    return CSAWithES(
        csa_config=CSAConfig(),
        es_population_size=population_size,
        es_seed=seed,
    )
