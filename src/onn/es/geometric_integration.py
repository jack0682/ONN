"""Geometric Relation Integration with ONN-ES.

This module integrates the label-free relation understanding (Phase 1-3)
with the ONN-ES optimization framework.

Key Integration Points:
1. GeometricRelationEncoder → RelationEncoder (x_obs generation)
2. PredictiveRelationModel → Fitness (relation validation)
3. TemporalPredictiveModel → Episode evaluation (temporal consistency)
4. ContrastiveRelationLearner → Self-supervised refinement

Philosophy:
    "관계를 라벨 없이, 예측과 일관성으로 이해한다"

Reference:
    - Phase 1: GeometricRelationEncoder (SE(3) encoding)
    - Phase 2: PredictiveRelationModel (validation by prediction)
    - Phase 3: ContrastiveRelationLearner (self-supervised learning)

Author: Claude (ONN-ES Integration)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Any

import torch
import numpy as np

from onn.core.relation_geometry import (
    # Phase 1
    GeometricRelationEncoder,
    GeometricRelation,
    create_geometric_encoder,
    compose_relations,
    se3_distance,
    # Phase 2
    PredictiveRelationModel,
    create_predictive_model,
    TemporalPredictiveModel,
    create_temporal_model,
    RelationFitnessMetrics,
    compute_relation_fitness,
    # Phase 3
    ContrastiveRelationLearner,
    create_contrastive_learner,
    RelationClusterer,
    create_relation_clusterer,
)
from onn.core.tensors import SemanticNode, BOUND_TENSOR_DIM, FORM_TENSOR_DIM, INTENT_TENSOR_DIM

if TYPE_CHECKING:
    from onn.core.solver import ProjectionConsensusSolver, SolverResult
    from onn.core.graph import EdgeGraph
    from onn.es.ask_tell import Candidate

logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class GeometricFitnessConfig:
    """Extended fitness configuration with geometric metrics.

    Combines original ONN-ES metrics with geometric relation metrics.
    """
    # Original ONN-ES weights
    alpha_violation: float = 1.0
    alpha_drift: float = 0.5
    alpha_ricci: float = 0.1
    alpha_smooth: float = 0.1
    alpha_latency: float = 0.5
    alpha_collapse: float = 1.0

    # Geometric relation weights (NEW)
    alpha_prediction: float = 0.5      # Prediction accuracy reward
    alpha_bidirectional: float = 0.3   # Bidirectional consistency reward
    alpha_temporal: float = 0.3        # Temporal smoothness reward
    alpha_cycle: float = 0.5           # Cycle consistency reward
    alpha_contrastive: float = 0.2     # Contrastive loss penalty

    # Event handling
    recovery_threshold: float = 0.1


@dataclass
class GeometricEpisodeStep:
    """Extended episode step with geometric information.

    Contains both traditional ONN-ES data and geometric relation data.
    """
    # Original ONN-ES data
    x_obs: torch.Tensor
    edge_graph: "EdgeGraph"
    is_event: bool = False
    phi: Optional[torch.Tensor] = None

    # Geometric data (NEW)
    nodes: Optional[List[SemanticNode]] = None
    relations: Optional[Dict[Tuple[int, int], GeometricRelation]] = None
    timestamp: float = 0.0


@dataclass
class GeometricEvalMetrics:
    """Extended evaluation metrics with geometric relation quality.

    Combines original ONN-ES metrics with geometric metrics.
    """
    # Original ONN-ES metrics
    violation_mean: float = 0.0
    violation_max: float = 0.0
    drift_mean: float = 0.0
    ricci_energy: float = 0.0
    smoothness: float = 0.0
    latency_mean: float = 0.0
    collapse_score: float = 0.0

    # Geometric relation metrics (NEW)
    prediction_accuracy: float = 0.0
    bidirectional_consistency: float = 0.0
    temporal_smoothness: float = 0.0
    cycle_consistency: float = 0.0
    contrastive_loss: float = 0.0

    # Combined score
    geometric_score: float = 0.0


# ==============================================================================
# GEOMETRIC RELATION ENCODER (Extended)
# ==============================================================================

class GeometricRelationEncoderES:
    """Extended RelationEncoder with geometric SE(3) encoding.

    Combines:
    - Random projection (W_rp) for base structure
    - Learnable linear (W_lin) for ES adaptation
    - Geometric SE(3) encoding for label-free understanding

    Formula:
        x_obs = α * normalize(W_rp @ φ) + β * W_lin @ φ + γ * W_geo @ ξ

        where:
        - φ: Traditional pair features
        - ξ: SE(3) Lie algebra encoding (6D)
        - α, β, γ: Mixing weights
    """

    def __init__(
        self,
        input_dim: int = 64,
        output_dim: int = 32,
        geometric_dim: int = 6,
        use_geometric: bool = True,
        random_proj_seed: int = 42,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
    ):
        """Initialize the encoder.

        Args:
            input_dim: Traditional pair feature dimension
            output_dim: Output embedding dimension
            geometric_dim: Geometric encoding dimension (6 for SE(3))
            use_geometric: Whether to use geometric encoding
            random_proj_seed: Seed for random projection
            alpha: Weight for random projection
            beta: Weight for learnable linear
            gamma: Weight for geometric encoding
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.geometric_dim = geometric_dim
        self.use_geometric = use_geometric
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Random projection matrix (fixed)
        rng = np.random.RandomState(random_proj_seed)
        W_rp_np = rng.randn(output_dim, input_dim) / np.sqrt(input_dim)
        self.W_rp = torch.tensor(W_rp_np, dtype=torch.float32)

        # Learnable linear (ES-optimized)
        self.W_lin = torch.zeros(output_dim, input_dim)

        # Geometric projection (fixed but specialized)
        if use_geometric:
            W_geo_np = rng.randn(output_dim, geometric_dim) / np.sqrt(geometric_dim)
            self.W_geo = torch.tensor(W_geo_np, dtype=torch.float32)
        else:
            self.W_geo = None

        # Geometric encoder (Phase 1)
        self.geometric_encoder = create_geometric_encoder()

        logger.debug(
            f"GeometricRelationEncoderES: "
            f"input={input_dim}, output={output_dim}, geometric={use_geometric}"
        )

    def encode(
        self,
        phi: torch.Tensor,
        node_i: Optional[SemanticNode] = None,
        node_j: Optional[SemanticNode] = None,
    ) -> torch.Tensor:
        """Encode to edge embeddings with optional geometric enhancement.

        Args:
            phi: Pair features (m, input_dim) or (input_dim,)
            node_i: Source node (optional, for geometric encoding)
            node_j: Target node (optional, for geometric encoding)

        Returns:
            Edge embeddings x_obs
        """
        single = phi.dim() == 1
        if single:
            phi = phi.unsqueeze(0)

        device = phi.device

        # Component 1: Random projection
        x_rp = torch.mm(self.W_rp.to(device), phi.t()).t()
        x_rp = torch.nn.functional.normalize(x_rp, dim=1)

        # Component 2: Learnable linear
        x_lin = torch.mm(self.W_lin.to(device), phi.t()).t()

        # Component 3: Geometric encoding (if available)
        if self.use_geometric and node_i is not None and node_j is not None:
            relation = self.geometric_encoder.encode(node_i, node_j)
            xi = torch.tensor(relation.xi, dtype=torch.float32, device=device)
            if xi.dim() == 1:
                xi = xi.unsqueeze(0)
            x_geo = torch.mm(self.W_geo.to(device), xi.t()).t()
        else:
            x_geo = torch.zeros(phi.shape[0], self.output_dim, device=device)

        # Combine
        x_obs = self.alpha * x_rp + self.beta * x_lin + self.gamma * x_geo

        if single:
            x_obs = x_obs.squeeze(0)

        return x_obs

    def encode_batch_geometric(
        self,
        nodes: List[SemanticNode],
        edge_list: List[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, Dict[Tuple[int, int], GeometricRelation]]:
        """Encode all edges with full geometric information.

        Args:
            nodes: List of SemanticNode
            edge_list: List of (source_id, target_id) tuples

        Returns:
            (embeddings, relations_dict)
        """
        node_dict = {n.node_id: n for n in nodes}

        embeddings = []
        relations = {}

        for src_id, tgt_id in edge_list:
            node_i = node_dict.get(src_id)
            node_j = node_dict.get(tgt_id)

            if node_i is None or node_j is None:
                # Fallback: zero embedding
                embeddings.append(torch.zeros(self.output_dim))
                continue

            # Geometric relation
            relation = self.geometric_encoder.encode(node_i, node_j)
            relations[(src_id, tgt_id)] = relation

            # Create phi from form tensors (if available)
            phi_i = torch.tensor(node_i.form_tensor, dtype=torch.float32)
            phi_j = torch.tensor(node_j.form_tensor, dtype=torch.float32)
            phi = torch.cat([phi_i, phi_j])[:self.input_dim]  # Truncate/pad

            if len(phi) < self.input_dim:
                phi = torch.nn.functional.pad(phi, (0, self.input_dim - len(phi)))

            # Encode
            x = self.encode(phi, node_i, node_j)
            embeddings.append(x)

        return torch.stack(embeddings), relations

    def set_linear_weights(self, W_lin: torch.Tensor) -> None:
        """Set the learnable linear weights (from ES)."""
        if W_lin.shape != self.W_lin.shape:
            raise ValueError(f"Shape mismatch: {W_lin.shape} vs {self.W_lin.shape}")
        self.W_lin = W_lin.clone()

    def get_linear_weights(self) -> torch.Tensor:
        """Get the current linear weights."""
        return self.W_lin.clone()


# ==============================================================================
# GEOMETRIC EPISODE EVALUATION
# ==============================================================================

def evaluate_geometric_episode(
    solver: "ProjectionConsensusSolver",
    episode: List[GeometricEpisodeStep],
    config: Optional[GeometricFitnessConfig] = None,
    predictor: Optional[PredictiveRelationModel] = None,
    temporal_model: Optional[TemporalPredictiveModel] = None,
    contrastive_learner: Optional[ContrastiveRelationLearner] = None,
) -> GeometricEvalMetrics:
    """Evaluate episode with geometric relation metrics.

    This extends the standard ONN-ES evaluation with:
    - Relation prediction accuracy
    - Bidirectional consistency
    - Temporal smoothness
    - Cycle consistency
    - Contrastive loss

    Args:
        solver: ONN-ES solver
        episode: List of GeometricEpisodeStep
        config: Fitness configuration
        predictor: Predictive relation model (Phase 2)
        temporal_model: Temporal model (Phase 2)
        contrastive_learner: Contrastive learner (Phase 3)

    Returns:
        GeometricEvalMetrics
    """
    if config is None:
        config = GeometricFitnessConfig()

    if predictor is None:
        predictor = create_predictive_model()

    if temporal_model is None:
        temporal_model = create_temporal_model()
    else:
        temporal_model.reset()

    if contrastive_learner is None:
        contrastive_learner = create_contrastive_learner()

    geometric_encoder = create_geometric_encoder()

    if not episode:
        return GeometricEvalMetrics()

    # Standard ONN-ES metrics
    violations = []
    drifts = []
    ricci_energies = []
    smoothness_vals = []
    variance_vals = []
    latencies = []

    # Geometric metrics
    prediction_accuracies = []
    bidirectional_scores = []
    cycle_scores = []
    contrastive_losses = []

    x_prev = None
    last_event_step = -1
    prev_relations = None

    for t, step in enumerate(episode):
        # Run ONN-ES solver
        result = solver.solve(step.x_obs, step.edge_graph, x_prev=x_prev)
        x_hat = result.x

        # === Standard ONN-ES Metrics ===
        violation = result.breakdown.get("context", 0.0)
        violations.append(violation)

        drift = torch.norm(x_hat - step.x_obs).item()
        drifts.append(drift)

        ricci = result.breakdown.get("ricci", 0.0)
        ricci_energies.append(ricci)

        if x_prev is not None:
            smooth = torch.norm(x_hat - x_prev).item()
            smoothness_vals.append(smooth)

        var = torch.var(x_hat).item()
        variance_vals.append(var)

        if step.is_event:
            last_event_step = t
        elif last_event_step >= 0 and violation < config.recovery_threshold:
            latency = t - last_event_step
            latencies.append(latency)
            last_event_step = -1

        # === Geometric Relation Metrics ===
        if step.nodes is not None and len(step.nodes) >= 2:
            # Compute relation fitness (Phase 2)
            relation_metrics = compute_relation_fitness(
                geometric_encoder, predictor, step.nodes, temporal_model
            )

            prediction_accuracies.append(relation_metrics.prediction_accuracy)
            bidirectional_scores.append(relation_metrics.bidirectional_consistency)
            cycle_scores.append(relation_metrics.cycle_consistency)

            # Update temporal model
            if step.relations:
                for (src, tgt), rel in step.relations.items():
                    temporal_model.update(rel, step.timestamp)

            # Contrastive loss (Phase 3)
            if step.relations and prev_relations:
                curr_relations = list(step.relations.values())
                prev_rel_list = list(prev_relations.values())

                # Mine temporal pairs
                pairs = contrastive_learner.mine_temporal_pairs(
                    prev_rel_list, curr_relations
                )
                if pairs:
                    c_loss = contrastive_learner.compute_batch_loss(pairs)
                    contrastive_losses.append(c_loss)

            prev_relations = step.relations

        x_prev = x_hat

    # Aggregate metrics
    geometric_score = 0.0
    if prediction_accuracies:
        pred_acc = np.mean(prediction_accuracies)
        bidir = np.mean(bidirectional_scores)
        cycle = np.mean(cycle_scores)

        geometric_score = (
            config.alpha_prediction * pred_acc +
            config.alpha_bidirectional * bidir +
            config.alpha_cycle * cycle
        )

    if contrastive_losses:
        geometric_score -= config.alpha_contrastive * np.mean(contrastive_losses)

    return GeometricEvalMetrics(
        # Standard metrics
        violation_mean=np.mean(violations) if violations else 0.0,
        violation_max=np.max(violations) if violations else 0.0,
        drift_mean=np.mean(drifts) if drifts else 0.0,
        ricci_energy=np.mean(ricci_energies) if ricci_energies else 0.0,
        smoothness=np.mean(smoothness_vals) if smoothness_vals else 0.0,
        latency_mean=np.mean(latencies) if latencies else 0.0,
        collapse_score=np.mean(variance_vals) if variance_vals else 0.0,
        # Geometric metrics
        prediction_accuracy=np.mean(prediction_accuracies) if prediction_accuracies else 0.0,
        bidirectional_consistency=np.mean(bidirectional_scores) if bidirectional_scores else 0.0,
        temporal_smoothness=temporal_model.get_smoothness_score(),
        cycle_consistency=np.mean(cycle_scores) if cycle_scores else 0.0,
        contrastive_loss=np.mean(contrastive_losses) if contrastive_losses else 0.0,
        geometric_score=geometric_score,
    )


def compute_geometric_fitness(
    metrics: GeometricEvalMetrics,
    config: Optional[GeometricFitnessConfig] = None,
) -> float:
    """Compute total fitness from geometric metrics.

    Combines standard ONN-ES fitness with geometric relation quality.

    Args:
        metrics: GeometricEvalMetrics
        config: GeometricFitnessConfig

    Returns:
        Scalar fitness (higher = better)
    """
    if config is None:
        config = GeometricFitnessConfig()

    # Standard ONN-ES fitness
    standard_fitness = (
        - config.alpha_violation * metrics.violation_mean
        - config.alpha_drift * metrics.drift_mean
        + config.alpha_collapse * metrics.collapse_score
        - config.alpha_latency * metrics.latency_mean
        - config.alpha_ricci * metrics.ricci_energy
        - config.alpha_smooth * metrics.smoothness
    )

    # Geometric fitness (Phase 2 & 3)
    geometric_fitness = (
        + config.alpha_prediction * metrics.prediction_accuracy
        + config.alpha_bidirectional * metrics.bidirectional_consistency
        + config.alpha_temporal * metrics.temporal_smoothness
        + config.alpha_cycle * metrics.cycle_consistency
        - config.alpha_contrastive * metrics.contrastive_loss
    )

    return standard_fitness + geometric_fitness


# ==============================================================================
# CANDIDATE EVALUATION WITH GEOMETRIC RELATIONS
# ==============================================================================

def evaluate_geometric_candidate(
    candidate: "Candidate",
    episodes: List[List[GeometricEpisodeStep]],
    encoder: Optional[GeometricRelationEncoderES] = None,
    config: Optional[GeometricFitnessConfig] = None,
) -> float:
    """Evaluate candidate with geometric relation metrics.

    This is the main fitness function for CMA-ES with geometric understanding.

    Args:
        candidate: Candidate with hyperparams and W_lin
        episodes: List of geometric episodes
        encoder: GeometricRelationEncoderES
        config: GeometricFitnessConfig

    Returns:
        Scalar fitness
    """
    from onn.core.solver import create_solver_from_dict

    if config is None:
        config = GeometricFitnessConfig()

    # Set W_lin if provided
    if candidate.w_lin is not None and encoder is not None:
        w_lin_tensor = torch.tensor(candidate.w_lin, dtype=torch.float32)
        encoder.set_linear_weights(w_lin_tensor)

    # Create solver
    solver = create_solver_from_dict(candidate.hyperparams)

    # Create geometric models
    predictor = create_predictive_model()
    temporal_model = create_temporal_model()
    contrastive_learner = create_contrastive_learner()

    # Evaluate all episodes
    metrics_list = []
    for episode in episodes:
        metrics = evaluate_geometric_episode(
            solver, episode, config,
            predictor, temporal_model, contrastive_learner
        )
        metrics_list.append(metrics)

    # Aggregate
    if not metrics_list:
        return float('-inf')

    avg_metrics = GeometricEvalMetrics(
        violation_mean=np.mean([m.violation_mean for m in metrics_list]),
        violation_max=np.max([m.violation_max for m in metrics_list]),
        drift_mean=np.mean([m.drift_mean for m in metrics_list]),
        ricci_energy=np.mean([m.ricci_energy for m in metrics_list]),
        smoothness=np.mean([m.smoothness for m in metrics_list]),
        latency_mean=np.mean([m.latency_mean for m in metrics_list]),
        collapse_score=np.mean([m.collapse_score for m in metrics_list]),
        prediction_accuracy=np.mean([m.prediction_accuracy for m in metrics_list]),
        bidirectional_consistency=np.mean([m.bidirectional_consistency for m in metrics_list]),
        temporal_smoothness=np.mean([m.temporal_smoothness for m in metrics_list]),
        cycle_consistency=np.mean([m.cycle_consistency for m in metrics_list]),
        contrastive_loss=np.mean([m.contrastive_loss for m in metrics_list]),
    )

    return compute_geometric_fitness(avg_metrics, config)


# ==============================================================================
# SYNTHETIC GEOMETRIC EPISODE GENERATION
# ==============================================================================

def generate_geometric_episode(
    num_steps: int = 10,
    num_nodes: int = 5,
    embedding_dim: int = 32,
    event_prob: float = 0.1,
    noise_std: float = 0.1,
    motion_type: str = "static",  # "static", "linear", "circular"
    seed: Optional[int] = None,
) -> List[GeometricEpisodeStep]:
    """Generate synthetic episode with geometric relations.

    Creates a scene with nodes that have proper SE(3) frames,
    simulating realistic perception scenarios.

    Args:
        num_steps: Number of timesteps
        num_nodes: Number of objects in scene
        embedding_dim: Edge embedding dimension
        event_prob: Event probability per step
        noise_std: Observation noise
        motion_type: Type of object motion
        seed: Random seed

    Returns:
        List of GeometricEpisodeStep
    """
    from onn.core.graph import EdgeGraph

    rng = np.random.RandomState(seed)

    # Create initial node positions
    positions = []
    for i in range(num_nodes):
        if motion_type == "circular":
            angle = 2 * np.pi * i / num_nodes
            pos = np.array([np.cos(angle), np.sin(angle), 0.0])
        else:
            pos = rng.randn(3)
        positions.append(pos)

    # Create edges (fully connected for small graphs, sparse for large)
    edges = []
    if num_nodes <= 5:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edges.append((i, j))
                edges.append((j, i))  # Both directions
    else:
        # Sparse: each node connects to 2-3 neighbors
        for i in range(num_nodes):
            n_neighbors = min(3, num_nodes - 1)
            neighbors = rng.choice(
                [j for j in range(num_nodes) if j != i],
                n_neighbors, replace=False
            )
            for j in neighbors:
                if (i, j) not in edges:
                    edges.append((i, j))

    graph = EdgeGraph.from_edge_list(edges)

    # Geometric encoder
    geo_encoder = create_geometric_encoder()

    episode = []
    for t in range(num_steps):
        # Update positions based on motion type
        if motion_type == "linear":
            velocity = rng.randn(3) * 0.1
            positions = [p + velocity for p in positions]
        elif motion_type == "circular":
            angle_offset = t * 0.1
            positions = [
                np.array([
                    np.cos(2 * np.pi * i / num_nodes + angle_offset),
                    np.sin(2 * np.pi * i / num_nodes + angle_offset),
                    0.0
                ])
                for i in range(num_nodes)
            ]

        # Create nodes with current positions
        nodes = []
        for i, pos in enumerate(positions):
            bound = np.zeros(BOUND_TENSOR_DIM, dtype=np.float32)
            bound[0:3] = pos
            bound[3:6] = [0, 0, 1]  # Principal axis (Z-up)
            bound[6:9] = [0.1, 0.1, 0.1]  # Extents

            form = rng.randn(FORM_TENSOR_DIM).astype(np.float32)
            intent = rng.rand(INTENT_TENSOR_DIM).astype(np.float32)

            nodes.append(SemanticNode(
                node_id=i,
                bound_tensor=bound,
                form_tensor=form,
                intent_tensor=intent,
            ))

        # Encode geometric relations
        relations = {}
        for src, tgt in edges:
            rel = geo_encoder.encode(nodes[src], nodes[tgt])
            relations[(src, tgt)] = rel

        # Generate x_obs (with noise)
        x_obs_list = []
        for src, tgt in edges:
            rel = relations[(src, tgt)]
            # Use Lie algebra as embedding base
            xi = torch.tensor(rel.xi, dtype=torch.float32)
            # Project to embedding dim
            if embedding_dim > 6:
                padding = torch.randn(embedding_dim - 6) * 0.1
                emb = torch.cat([xi, padding])
            else:
                emb = xi[:embedding_dim]
            # Add noise
            emb = emb + torch.randn(embedding_dim) * noise_std
            x_obs_list.append(emb)

        x_obs = torch.stack(x_obs_list)

        # Random event
        is_event = rng.random() < event_prob
        if is_event:
            # Perturb positions
            idx = rng.randint(num_nodes)
            positions[idx] = positions[idx] + rng.randn(3) * 0.5

        episode.append(GeometricEpisodeStep(
            x_obs=x_obs,
            edge_graph=graph,
            is_event=is_event,
            nodes=nodes,
            relations=relations,
            timestamp=float(t) * 0.1,
        ))

    return episode


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

def create_geometric_encoder_es(
    input_dim: int = 64,
    output_dim: int = 32,
    use_geometric: bool = True,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
) -> GeometricRelationEncoderES:
    """Create geometric relation encoder for ES.

    Args:
        input_dim: Traditional pair feature dimension
        output_dim: Output embedding dimension
        use_geometric: Whether to use geometric encoding
        alpha: Weight for random projection
        beta: Weight for learnable linear (ES-optimized)
        gamma: Weight for geometric encoding
    """
    return GeometricRelationEncoderES(
        input_dim=input_dim,
        output_dim=output_dim,
        use_geometric=use_geometric,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )


def create_geometric_fitness_config(
    geometric_weight: float = 1.0,
) -> GeometricFitnessConfig:
    """Create geometric fitness configuration.

    Args:
        geometric_weight: Overall weight for geometric metrics (0-1)
    """
    return GeometricFitnessConfig(
        alpha_prediction=0.5 * geometric_weight,
        alpha_bidirectional=0.3 * geometric_weight,
        alpha_temporal=0.3 * geometric_weight,
        alpha_cycle=0.5 * geometric_weight,
        alpha_contrastive=0.2 * geometric_weight,
    )
