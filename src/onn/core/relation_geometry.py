"""
Geometric Relation Encoding - Label-Free Relation Understanding

Philosophy:
    "관계는 라벨이 아니라 변환 그 자체다"
    Relations are not discrete labels but continuous transformations.

    Instead of classifying "above", "below", "supports" - we encode
    the pure geometric structure: the SE(3) transformation between
    two coordinate frames.

Core Mathematics:
    SE(3) = Special Euclidean Group = {(R, t) | R ∈ SO(3), t ∈ ℝ³}

    Relation between entity A and B:
        T_rel = T_B ∘ T_A⁻¹

    Lie algebra representation:
        ξ = [ω, v]ᵀ ∈ ℝ⁶
        where ω = log(R) ∈ so(3) (axis-angle)
              v = t (translation)

Properties:
    - Continuous: Relations live in a smooth manifold
    - Equivariant: If scene transforms, relation transforms accordingly
    - Composable: T_AC = T_AB ∘ T_BC
    - Invertible: T_BA = T_AB⁻¹

Reference:
    - User Philosophy: "Resonance as Supreme Law"
    - Mathematical Foundation: Lie Groups and Robotics
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import logging

from onn.core.tensors import SemanticNode, BOUND_TENSOR_DIM

logger = logging.getLogger(__name__)


# =============================================================================
# Mathematical Primitives: SO(3) and SE(3)
# =============================================================================

def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Convert vector to skew-symmetric matrix.

    [v]_× = | 0   -v₃  v₂ |
            | v₃   0  -v₁ |
            |-v₂  v₁   0  |
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=np.float64)


def vee_so3(M: np.ndarray) -> np.ndarray:
    """Extract vector from skew-symmetric matrix (inverse of skew_symmetric)."""
    return np.array([M[2, 1], M[0, 2], M[1, 0]], dtype=np.float64)


def exp_so3(omega: np.ndarray) -> np.ndarray:
    """Exponential map: so(3) → SO(3) (Rodrigues formula).

    Given axis-angle ω, compute rotation matrix R = exp([ω]_×)

    R = I + sin(θ)/θ [ω]_× + (1-cos(θ))/θ² [ω]_×²

    where θ = ||ω||
    """
    theta = np.linalg.norm(omega)

    if theta < 1e-10:
        # Small angle approximation: R ≈ I + [ω]_×
        return np.eye(3) + skew_symmetric(omega)

    omega_hat = skew_symmetric(omega / theta)

    # Rodrigues formula
    R = (np.eye(3) +
         np.sin(theta) * omega_hat +
         (1 - np.cos(theta)) * (omega_hat @ omega_hat))

    return R


def log_so3(R: np.ndarray) -> np.ndarray:
    """Logarithm map: SO(3) → so(3).

    Given rotation matrix R, compute axis-angle ω = log(R)

    θ = arccos((tr(R) - 1) / 2)
    [ω]_× = θ/(2sin(θ)) (R - Rᵀ)
    """
    # Ensure R is a valid rotation matrix
    R = np.asarray(R, dtype=np.float64)

    # Compute rotation angle
    trace = np.trace(R)
    trace = np.clip(trace, -1.0, 3.0)  # Numerical safety

    cos_theta = (trace - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-10:
        # Near identity: ω ≈ vee(R - I)
        return vee_so3(R - np.eye(3))

    if np.abs(theta - np.pi) < 1e-6:
        # Near 180 degrees: need special handling
        # Find the column of (R + I) with largest norm
        M = R + np.eye(3)
        col_norms = np.linalg.norm(M, axis=0)
        k = np.argmax(col_norms)
        v = M[:, k]
        v = v / np.linalg.norm(v)
        return v * theta

    # General case
    omega_hat = theta / (2 * np.sin(theta)) * (R - R.T)
    return vee_so3(omega_hat)


def exp_se3(xi: np.ndarray) -> np.ndarray:
    """Exponential map: se(3) → SE(3).

    Given twist ξ = [ω, v], compute transformation T = exp(ξ̂)

    T = | R   Jv |
        | 0    1 |

    where R = exp([ω]_×)
          J = I + (1-cos(θ))/θ² [ω]_× + (θ-sin(θ))/θ³ [ω]_×²
    """
    omega = xi[:3]
    v = xi[3:6]

    theta = np.linalg.norm(omega)

    T = np.eye(4)

    if theta < 1e-10:
        # Pure translation
        T[:3, :3] = np.eye(3)
        T[:3, 3] = v
        return T

    # Rotation part
    R = exp_so3(omega)
    T[:3, :3] = R

    # Translation part with left Jacobian
    omega_hat = skew_symmetric(omega / theta)
    J = (np.eye(3) +
         (1 - np.cos(theta)) / theta * omega_hat +
         (theta - np.sin(theta)) / theta * (omega_hat @ omega_hat))

    T[:3, 3] = J @ v

    return T


def log_se3(T: np.ndarray) -> np.ndarray:
    """Logarithm map: SE(3) → se(3).

    Given transformation T, compute twist ξ = log(T)
    """
    R = T[:3, :3]
    t = T[:3, 3]

    omega = log_so3(R)
    theta = np.linalg.norm(omega)

    if theta < 1e-10:
        # Pure translation
        return np.concatenate([omega, t])

    # Inverse of left Jacobian
    omega_hat = skew_symmetric(omega / theta)

    # J_inv = I - 0.5[ω]_× + (1/θ² - (1+cos(θ))/(2θsin(θ))) [ω]_×²
    half_theta = theta / 2
    cot_half = np.cos(half_theta) / (np.sin(half_theta) + 1e-10)

    J_inv = (np.eye(3) -
             0.5 * omega_hat * theta +
             (1 - half_theta * cot_half) * (omega_hat @ omega_hat))

    v = J_inv @ t

    return np.concatenate([omega, v])


def invert_se3(T: np.ndarray) -> np.ndarray:
    """Invert SE(3) transformation.

    T⁻¹ = | Rᵀ  -Rᵀt |
          | 0     1  |
    """
    T_inv = np.eye(4)
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T_inv


def compose_se3(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """Compose two SE(3) transformations: T1 ∘ T2."""
    return T1 @ T2


# =============================================================================
# Frame Extraction from Semantic Nodes
# =============================================================================

def frame_from_bound_tensor(bound_tensor: np.ndarray) -> np.ndarray:
    """Extract SE(3) frame from bound tensor.

    Bound tensor layout (from sego_enhanced.py):
        [0:3]  - Position (centroid)
        [3:6]  - Principal axis (orientation)
        [6:9]  - Extents (L, W, H)
        [9]    - Radius
        [10:12]- Aspect ratios
        [12:15]- Surface normal
        [15]   - Convexity

    We construct a frame from position and principal axis:
        - Translation: position
        - Rotation: align Z-axis with principal axis
    """
    T = np.eye(4, dtype=np.float64)

    # Translation from centroid
    position = bound_tensor[0:3]
    T[:3, 3] = position

    # Rotation from principal axis
    principal_axis = bound_tensor[3:6]
    norm = np.linalg.norm(principal_axis)

    if norm > 1e-6:
        # Normalize
        z_axis = principal_axis / norm

        # Construct orthonormal basis using Gram-Schmidt
        # Find a vector not parallel to z_axis
        if abs(z_axis[0]) < 0.9:
            helper = np.array([1, 0, 0])
        else:
            helper = np.array([0, 1, 0])

        x_axis = np.cross(helper, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-10)

        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-10)

        # Rotation matrix columns are the basis vectors
        R = np.column_stack([x_axis, y_axis, z_axis])
        T[:3, :3] = R

    return T


def frame_from_node(node: SemanticNode) -> np.ndarray:
    """Extract SE(3) frame from SemanticNode."""
    return frame_from_bound_tensor(node.bound_tensor)


# =============================================================================
# Geometric Relation Encoder
# =============================================================================

@dataclass
class GeometricRelation:
    """A label-free geometric relation between two entities.

    Instead of discrete labels like "above" or "supports", this
    captures the continuous transformation structure.

    Attributes:
        xi: Lie algebra element [ω, v] ∈ ℝ⁶
        T_rel: The SE(3) transformation matrix (4x4)
        distance: Euclidean distance between centroids
        angle: Rotation angle in radians
    """
    xi: np.ndarray          # (6,) Lie algebra encoding
    T_rel: np.ndarray       # (4, 4) SE(3) transformation
    distance: float         # Euclidean distance
    angle: float            # Rotation angle

    # Source and target (for reference, not for classification)
    source_id: int
    target_id: int

    def __post_init__(self):
        """Compute derived quantities."""
        if self.distance is None:
            self.distance = np.linalg.norm(self.T_rel[:3, 3])
        if self.angle is None:
            self.angle = np.linalg.norm(self.xi[:3])

    def to_embedding(self, include_scale: bool = True) -> np.ndarray:
        """Convert to embedding vector for downstream use.

        Returns:
            If include_scale: [ω, v, distance, angle, ω_norm, v_norm] (12,)
            Else: [ω, v] (6,)
        """
        if not include_scale:
            return self.xi.astype(np.float32)

        omega = self.xi[:3]
        v = self.xi[3:6]

        omega_norm = np.linalg.norm(omega)
        v_norm = np.linalg.norm(v)

        # Normalize for scale invariance
        if omega_norm > 1e-6:
            omega_unit = omega / omega_norm
        else:
            omega_unit = omega

        if v_norm > 1e-6:
            v_unit = v / v_norm
        else:
            v_unit = v

        embedding = np.concatenate([
            omega_unit,           # (3,) rotation direction
            v_unit,               # (3,) translation direction
            [self.distance],      # (1,) distance
            [self.angle],         # (1,) rotation angle
            [omega_norm],         # (1,) rotation magnitude
            [v_norm],             # (1,) translation magnitude
        ])

        return embedding.astype(np.float32)

    def invert(self) -> 'GeometricRelation':
        """Get the inverse relation (B → A instead of A → B)."""
        T_inv = invert_se3(self.T_rel)
        xi_inv = log_se3(T_inv)

        return GeometricRelation(
            xi=xi_inv,
            T_rel=T_inv,
            distance=self.distance,  # Same distance
            angle=self.angle,        # Same angle magnitude
            source_id=self.target_id,
            target_id=self.source_id,
        )


class GeometricRelationEncoder:
    """Encodes relations as pure geometric transformations.

    Philosophy:
        "The word is seed of reality" - but we go beyond words.
        We encode the geometric structure that PRECEDES linguistic labels.

        "Above" is just a human label for certain configurations of
        the relative transformation T_rel where t_z > 0.

        We don't classify - we encode the full continuous structure.

    Example:
        >>> encoder = GeometricRelationEncoder()
        >>> relation = encoder.encode(node_cup, node_table)
        >>> # relation.xi contains the full SE(3) structure
        >>> # No labels needed - the geometry speaks for itself
    """

    def __init__(self, use_log_distance: bool = True):
        """Initialize encoder.

        Args:
            use_log_distance: Use log(1 + distance) for better scale handling
        """
        self.use_log_distance = use_log_distance

    def encode(
        self,
        node_i: SemanticNode,
        node_j: SemanticNode,
    ) -> GeometricRelation:
        """Encode the geometric relation from node_i to node_j.

        The relation T_rel expresses j's frame in i's local coordinates:
            T_rel = T_i⁻¹ @ T_j

        This is SE(3)-INVARIANT: if we transform both frames by T_world,
        the relation remains unchanged:
            T_rel_new = (T_world @ T_i)⁻¹ @ (T_world @ T_j)
                      = T_i⁻¹ @ T_world⁻¹ @ T_world @ T_j
                      = T_i⁻¹ @ T_j = T_rel

        Args:
            node_i: Source node (reference frame)
            node_j: Target node

        Returns:
            GeometricRelation containing the SE(3) structure
        """
        # Extract frames
        T_i = frame_from_node(node_i)
        T_j = frame_from_node(node_j)

        # Relative transformation: T_rel = T_i⁻¹ @ T_j (INVARIANT form)
        T_rel = compose_se3(invert_se3(T_i), T_j)

        # Convert to Lie algebra
        xi = log_se3(T_rel)

        # Compute metrics
        distance = np.linalg.norm(T_rel[:3, 3])
        angle = np.linalg.norm(xi[:3])

        if self.use_log_distance:
            distance = np.log1p(distance)

        return GeometricRelation(
            xi=xi,
            T_rel=T_rel,
            distance=distance,
            angle=angle,
            source_id=node_i.node_id,
            target_id=node_j.node_id,
        )

    def encode_symmetric(
        self,
        node_i: SemanticNode,
        node_j: SemanticNode,
    ) -> np.ndarray:
        """Encode a symmetric (order-invariant) relation.

        For some applications, we want R(A,B) = R(B,A).
        This is achieved by encoding both directions and combining.

        Returns:
            Symmetric embedding (24,)
        """
        rel_ij = self.encode(node_i, node_j)
        rel_ji = rel_ij.invert()

        emb_ij = rel_ij.to_embedding()
        emb_ji = rel_ji.to_embedding()

        # Symmetric combination: concatenate sorted by source_id
        if node_i.node_id < node_j.node_id:
            return np.concatenate([emb_ij, emb_ji])
        else:
            return np.concatenate([emb_ji, emb_ij])

    def encode_all_pairs(
        self,
        nodes: List[SemanticNode],
    ) -> Dict[Tuple[int, int], GeometricRelation]:
        """Encode all pairwise relations in a set of nodes.

        Returns:
            Dictionary mapping (source_id, target_id) to GeometricRelation
        """
        relations = {}

        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i != j:
                    rel = self.encode(node_i, node_j)
                    relations[(node_i.node_id, node_j.node_id)] = rel

        return relations

    def batch_encode(
        self,
        nodes: List[SemanticNode],
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Batch encode all relations as a tensor.

        Returns:
            embeddings: (num_pairs, embedding_dim) array
            pairs: List of (source_id, target_id) tuples
        """
        relations = self.encode_all_pairs(nodes)

        embeddings = []
        pairs = []

        for (src, tgt), rel in relations.items():
            embeddings.append(rel.to_embedding())
            pairs.append((src, tgt))

        if embeddings:
            return np.stack(embeddings), pairs
        else:
            return np.zeros((0, 12)), []


# =============================================================================
# Equivariance Verification
# =============================================================================

def verify_equivariance(
    encoder: GeometricRelationEncoder,
    node_i: SemanticNode,
    node_j: SemanticNode,
    T_world: np.ndarray,
    tol: float = 1e-5,
) -> bool:
    """Verify that encoding is SE(3)-equivariant.

    If we transform both nodes by T_world, the relative relation
    should remain unchanged (since T_rel = T_j @ T_i⁻¹).

    This is a fundamental property: the relation between two objects
    doesn't depend on where we observe them from.
    """
    # Original relation
    rel_orig = encoder.encode(node_i, node_j)

    # Transform both nodes
    T_i = frame_from_node(node_i)
    T_j = frame_from_node(node_j)

    T_i_new = compose_se3(T_world, T_i)
    T_j_new = compose_se3(T_world, T_j)

    # Create transformed nodes (simplified - just update position/orientation)
    node_i_new = _transform_node(node_i, T_i_new)
    node_j_new = _transform_node(node_j, T_j_new)

    # Relation after transformation
    rel_new = encoder.encode(node_i_new, node_j_new)

    # Check that T_rel is unchanged
    diff = np.linalg.norm(rel_orig.T_rel - rel_new.T_rel)

    if diff > tol:
        logger.warning(f"Equivariance violation: ||ΔT|| = {diff:.6f}")
        return False

    return True


def _transform_node(node: SemanticNode, T_new: np.ndarray) -> SemanticNode:
    """Create a new node with transformed frame.

    This properly transforms both position and orientation
    to maintain SE(3) equivariance.
    """
    new_bound = node.bound_tensor.copy()

    # Update position from the new frame's translation
    new_bound[0:3] = T_new[:3, 3]

    # Update principal axis: rotate the original axis by the new rotation
    # The principal axis is the Z-axis of the local frame
    new_bound[3:6] = T_new[:3, 2]  # Third column of rotation matrix

    # Also update surface normal if present (dims 12:15)
    if len(new_bound) >= 15:
        # Transform surface normal by rotation
        orig_normal = node.bound_tensor[12:15]
        if np.linalg.norm(orig_normal) > 1e-6:
            new_bound[12:15] = T_new[:3, :3] @ orig_normal

    return SemanticNode(
        node_id=node.node_id,
        bound_tensor=new_bound,
        form_tensor=node.form_tensor.copy(),
        intent_tensor=node.intent_tensor.copy(),
    )


# =============================================================================
# Relation Algebra: Composition and Structure
# =============================================================================

def compose_relations(
    rel_ab: GeometricRelation,
    rel_bc: GeometricRelation,
) -> GeometricRelation:
    """Compose two relations: R(A,C) = R(A,B) ∘ R(B,C).

    This captures the algebraic structure of relations.
    If A is below B, and B is below C, then A is below C.

    But we don't need labels - the composition is just:
        T_AC = T_BC @ T_AB
    """
    T_ac = compose_se3(rel_bc.T_rel, rel_ab.T_rel)
    xi_ac = log_se3(T_ac)

    return GeometricRelation(
        xi=xi_ac,
        T_rel=T_ac,
        distance=np.linalg.norm(T_ac[:3, 3]),
        angle=np.linalg.norm(xi_ac[:3]),
        source_id=rel_ab.source_id,
        target_id=rel_bc.target_id,
    )


def relation_distance(
    rel1: GeometricRelation,
    rel2: GeometricRelation,
) -> float:
    """Compute distance between two relations in SE(3).

    d(T₁, T₂) = ||log(T₁⁻¹ T₂)||

    This measures how "different" two relations are.
    """
    T_diff = compose_se3(invert_se3(rel1.T_rel), rel2.T_rel)
    xi_diff = log_se3(T_diff)

    return float(np.linalg.norm(xi_diff))


# =============================================================================
# Factory Functions
# =============================================================================

def create_geometric_encoder() -> GeometricRelationEncoder:
    """Create default geometric relation encoder."""
    return GeometricRelationEncoder(use_log_distance=True)


# =============================================================================
# PyTorch Integration
# =============================================================================

class GeometricRelationEncoderTorch:
    """PyTorch-compatible version for differentiable encoding.

    This allows gradient flow through the relation encoding,
    enabling end-to-end learning.
    """

    def __init__(self, use_log_distance: bool = True):
        self.use_log_distance = use_log_distance

    def encode_batch(
        self,
        positions_i: torch.Tensor,  # (B, 3)
        orientations_i: torch.Tensor,  # (B, 3) principal axis
        positions_j: torch.Tensor,  # (B, 3)
        orientations_j: torch.Tensor,  # (B, 3)
    ) -> torch.Tensor:
        """Batch encode relations in PyTorch.

        Simplified version using position and direction only.
        Full SE(3) would require quaternion or rotation matrix input.

        Returns:
            Relation embeddings (B, 9): [Δp_normalized, Δp_norm, cos_angle, ...]
        """
        # Relative position
        delta_p = positions_j - positions_i  # (B, 3)
        distance = torch.norm(delta_p, dim=-1, keepdim=True)  # (B, 1)

        if self.use_log_distance:
            distance = torch.log1p(distance)

        # Normalized direction
        delta_p_norm = delta_p / (torch.norm(delta_p, dim=-1, keepdim=True) + 1e-8)

        # Relative orientation (cosine similarity of principal axes)
        ori_i_norm = orientations_i / (torch.norm(orientations_i, dim=-1, keepdim=True) + 1e-8)
        ori_j_norm = orientations_j / (torch.norm(orientations_j, dim=-1, keepdim=True) + 1e-8)

        cos_angle = (ori_i_norm * ori_j_norm).sum(dim=-1, keepdim=True)  # (B, 1)

        # Cross product for rotation axis
        cross = torch.cross(ori_i_norm, ori_j_norm, dim=-1)  # (B, 3)

        # Assemble embedding
        embedding = torch.cat([
            delta_p_norm,  # (B, 3) direction
            distance,      # (B, 1) distance
            cos_angle,     # (B, 1) relative orientation angle
            cross,         # (B, 3) rotation axis
        ], dim=-1)

        return embedding  # (B, 9)


# =============================================================================
# Phase 2: Predictive Relation Model
# =============================================================================

def se3_distance(T1: np.ndarray, T2: np.ndarray) -> float:
    """Compute geodesic distance between two SE(3) transformations.

    d(T₁, T₂) = ||log(T₁⁻¹ T₂)||

    This is the proper metric on SE(3) manifold, combining:
    - Rotation distance (geodesic on SO(3))
    - Translation distance (Euclidean)

    The norm is computed in the Lie algebra se(3).
    """
    T_diff = compose_se3(invert_se3(T1), T2)
    xi_diff = log_se3(T_diff)

    # Weighted combination: rotation and translation have different units
    # We use equal weighting, but this could be application-specific
    omega_norm = np.linalg.norm(xi_diff[:3])  # Rotation component
    v_norm = np.linalg.norm(xi_diff[3:6])     # Translation component

    return float(np.sqrt(omega_norm**2 + v_norm**2))


def se3_distance_separate(
    T1: np.ndarray,
    T2: np.ndarray,
) -> Tuple[float, float]:
    """Compute separate rotation and translation distances.

    Returns:
        (rotation_distance, translation_distance)
    """
    T_diff = compose_se3(invert_se3(T1), T2)
    xi_diff = log_se3(T_diff)

    omega_norm = float(np.linalg.norm(xi_diff[:3]))
    v_norm = float(np.linalg.norm(xi_diff[3:6]))

    return omega_norm, v_norm


@dataclass
class PredictionResult:
    """Result of a state prediction.

    Captures both the prediction and the error metrics.
    """
    T_predicted: np.ndarray        # Predicted SE(3) transformation
    T_actual: np.ndarray           # Actual SE(3) transformation
    prediction_error: float        # Combined SE(3) distance
    rotation_error: float          # Rotation-only error (radians)
    translation_error: float       # Translation-only error (meters)
    relation_confidence: float     # 1 / (1 + error) - higher = more confident


class PredictiveRelationModel:
    """Validates relations through state prediction.

    Philosophy:
        "관계가 맞다면, 다음 상태를 예측할 수 있다"
        If a relation is correct, we can predict states.

        This is the key insight for LABEL-FREE understanding:
        We don't ask "what IS the relation?" (label)
        We ask "does this relation PREDICT correctly?" (function)

    Core Mechanism:
        Given: source frame T_i, relation T_rel
        Predict: target frame T_j_pred = T_i @ T_rel
        Validate: compare T_j_pred with T_j_actual

    This allows:
        1. Relation validation without labels
        2. Relation refinement through error minimization
        3. Temporal prediction (dynamics)

    Example:
        >>> model = PredictiveRelationModel()
        >>> relation = encoder.encode(node_i, node_j)
        >>> result = model.validate_relation(node_i, node_j, relation)
        >>> if result.relation_confidence > 0.9:
        ...     print("Relation is accurate!")
    """

    def __init__(
        self,
        rotation_weight: float = 1.0,
        translation_weight: float = 1.0,
        confidence_scale: float = 1.0,
    ):
        """Initialize predictive model.

        Args:
            rotation_weight: Weight for rotation error in combined metric
            translation_weight: Weight for translation error in combined metric
            confidence_scale: Scale factor for confidence computation
        """
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
        self.confidence_scale = confidence_scale

    def predict_target_frame(
        self,
        T_source: np.ndarray,
        relation: GeometricRelation,
    ) -> np.ndarray:
        """Predict target frame from source frame and relation.

        Given T_i and T_rel, compute:
            T_j_pred = T_i @ T_rel

        This follows from the definition:
            T_rel = T_i⁻¹ @ T_j
            => T_j = T_i @ T_rel

        Args:
            T_source: Source frame SE(3) (4x4)
            relation: Geometric relation from source to target

        Returns:
            Predicted target frame SE(3) (4x4)
        """
        return compose_se3(T_source, relation.T_rel)

    def predict_source_frame(
        self,
        T_target: np.ndarray,
        relation: GeometricRelation,
    ) -> np.ndarray:
        """Predict source frame from target frame and relation (inverse).

        Given T_j and T_rel, compute:
            T_i_pred = T_j @ T_rel⁻¹

        Args:
            T_target: Target frame SE(3) (4x4)
            relation: Geometric relation from source to target

        Returns:
            Predicted source frame SE(3) (4x4)
        """
        T_rel_inv = invert_se3(relation.T_rel)
        return compose_se3(T_target, T_rel_inv)

    def compute_prediction_error(
        self,
        T_predicted: np.ndarray,
        T_actual: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Compute prediction error in SE(3).

        Returns:
            (combined_error, rotation_error, translation_error)
        """
        rot_err, trans_err = se3_distance_separate(T_predicted, T_actual)

        combined = np.sqrt(
            self.rotation_weight * rot_err**2 +
            self.translation_weight * trans_err**2
        )

        return combined, rot_err, trans_err

    def validate_relation(
        self,
        node_i: SemanticNode,
        node_j: SemanticNode,
        relation: GeometricRelation,
    ) -> PredictionResult:
        """Validate a relation by checking prediction accuracy.

        The relation is considered valid if it can accurately predict
        one node's frame from the other.

        Args:
            node_i: Source node
            node_j: Target node
            relation: The relation to validate

        Returns:
            PredictionResult with error metrics and confidence
        """
        # Get actual frames
        T_i = frame_from_node(node_i)
        T_j_actual = frame_from_node(node_j)

        # Predict target frame
        T_j_predicted = self.predict_target_frame(T_i, relation)

        # Compute errors
        combined, rot_err, trans_err = self.compute_prediction_error(
            T_j_predicted, T_j_actual
        )

        # Compute confidence: 1 / (1 + scaled_error)
        confidence = 1.0 / (1.0 + self.confidence_scale * combined)

        return PredictionResult(
            T_predicted=T_j_predicted,
            T_actual=T_j_actual,
            prediction_error=combined,
            rotation_error=rot_err,
            translation_error=trans_err,
            relation_confidence=confidence,
        )

    def validate_bidirectional(
        self,
        node_i: SemanticNode,
        node_j: SemanticNode,
        relation: GeometricRelation,
    ) -> Tuple[PredictionResult, PredictionResult]:
        """Validate relation in both directions.

        A truly valid relation should predict accurately in BOTH directions:
        - i → j (using T_rel)
        - j → i (using T_rel⁻¹)

        Returns:
            Tuple of (forward_result, backward_result)
        """
        forward = self.validate_relation(node_i, node_j, relation)

        # Backward: predict i from j using inverted relation
        T_j = frame_from_node(node_j)
        T_i_actual = frame_from_node(node_i)
        T_i_predicted = self.predict_source_frame(T_j, relation)

        combined, rot_err, trans_err = self.compute_prediction_error(
            T_i_predicted, T_i_actual
        )
        confidence = 1.0 / (1.0 + self.confidence_scale * combined)

        backward = PredictionResult(
            T_predicted=T_i_predicted,
            T_actual=T_i_actual,
            prediction_error=combined,
            rotation_error=rot_err,
            translation_error=trans_err,
            relation_confidence=confidence,
        )

        return forward, backward


# =============================================================================
# Temporal Prediction: Relations Over Time
# =============================================================================

@dataclass
class TemporalRelationState:
    """State of a relation over time.

    Captures not just the current relation, but also its dynamics:
    how it changes over time.
    """
    relation: GeometricRelation        # Current relation
    velocity_xi: np.ndarray            # Time derivative in Lie algebra (6,)
    timestamp: float                   # Time of observation

    def predict_at_time(self, t: float) -> GeometricRelation:
        """Predict relation at future time t.

        Uses first-order approximation:
            T_rel(t) ≈ T_rel(t0) @ exp(dt * ξ_velocity)

        where dt = t - t0

        Args:
            t: Target time

        Returns:
            Predicted GeometricRelation at time t
        """
        dt = t - self.timestamp

        # Compute incremental transformation
        delta_xi = dt * self.velocity_xi
        delta_T = exp_se3(delta_xi)

        # Compose with current relation
        T_pred = compose_se3(self.relation.T_rel, delta_T)
        xi_pred = log_se3(T_pred)

        return GeometricRelation(
            xi=xi_pred,
            T_rel=T_pred,
            distance=np.linalg.norm(T_pred[:3, 3]),
            angle=np.linalg.norm(xi_pred[:3]),
            source_id=self.relation.source_id,
            target_id=self.relation.target_id,
        )


class TemporalPredictiveModel:
    """Predicts relation dynamics over time.

    Philosophy:
        "존재는 영원한 맥동" (Existence is eternal pulsation)

        Relations are not static - they evolve. A cup on a table
        has a static relation, but a falling cup has a dynamic one.

        By learning the DYNAMICS of relations, we can:
        1. Predict future states
        2. Detect anomalies (when prediction fails)
        3. Understand causality (what causes relation changes)

    The key insight: smooth temporal evolution indicates
    a stable, well-understood relation.
    """

    def __init__(
        self,
        smoothness_weight: float = 0.1,
        prediction_horizon: float = 0.1,  # seconds
    ):
        """Initialize temporal model.

        Args:
            smoothness_weight: Weight for temporal smoothness loss
            prediction_horizon: Default prediction horizon in seconds
        """
        self.smoothness_weight = smoothness_weight
        self.prediction_horizon = prediction_horizon
        self._history: List[TemporalRelationState] = []

    def update(
        self,
        relation: GeometricRelation,
        timestamp: float,
    ) -> Optional[TemporalRelationState]:
        """Update model with new observation.

        Estimates velocity from consecutive observations.

        Args:
            relation: New relation observation
            timestamp: Time of observation

        Returns:
            TemporalRelationState with estimated velocity (or None if first)
        """
        if not self._history:
            # First observation: zero velocity
            state = TemporalRelationState(
                relation=relation,
                velocity_xi=np.zeros(6),
                timestamp=timestamp,
            )
            self._history.append(state)
            return state

        # Estimate velocity from previous observation
        prev = self._history[-1]
        dt = timestamp - prev.timestamp

        if dt < 1e-6:
            # Too small time step
            return None

        # Velocity in Lie algebra: ξ_dot = log(T_prev⁻¹ T_curr) / dt
        T_diff = compose_se3(invert_se3(prev.relation.T_rel), relation.T_rel)
        xi_diff = log_se3(T_diff)
        velocity_xi = xi_diff / dt

        state = TemporalRelationState(
            relation=relation,
            velocity_xi=velocity_xi,
            timestamp=timestamp,
        )
        self._history.append(state)

        return state

    def predict(
        self,
        horizon: Optional[float] = None,
    ) -> Optional[GeometricRelation]:
        """Predict future relation.

        Args:
            horizon: Prediction horizon in seconds (uses default if None)

        Returns:
            Predicted relation at future time
        """
        if not self._history:
            return None

        current = self._history[-1]
        h = horizon if horizon is not None else self.prediction_horizon

        return current.predict_at_time(current.timestamp + h)

    def compute_temporal_loss(
        self,
        current: GeometricRelation,
        predicted: GeometricRelation,
    ) -> float:
        """Compute temporal prediction loss.

        How well did our previous prediction match the current observation?
        Low loss = stable, predictable relation
        High loss = chaotic or unexpected change

        Args:
            current: Current observed relation
            predicted: What we predicted for this time

        Returns:
            Temporal prediction loss
        """
        return se3_distance(current.T_rel, predicted.T_rel)

    def get_smoothness_score(self) -> float:
        """Compute temporal smoothness of the relation.

        Based on acceleration (second derivative):
        Low acceleration = smooth motion
        High acceleration = jerky/unpredictable

        Returns:
            Smoothness score (higher = smoother)
        """
        if len(self._history) < 3:
            return 1.0  # Not enough data

        # Compute acceleration (velocity change)
        velocities = [s.velocity_xi for s in self._history[-3:]]
        acc1 = velocities[-1] - velocities[-2]
        acc_norm = np.linalg.norm(acc1)

        # Convert to smoothness: 1 / (1 + acc)
        return 1.0 / (1.0 + acc_norm)

    def reset(self):
        """Clear history and reset model."""
        self._history = []


# =============================================================================
# Integration with ONN-ES Fitness
# =============================================================================

@dataclass
class RelationFitnessMetrics:
    """Metrics for evaluating relation quality.

    Used to integrate predictive validation with ONN-ES fitness.
    """
    prediction_accuracy: float     # 1 - normalized_error
    bidirectional_consistency: float  # min(forward, backward) confidence
    temporal_smoothness: float     # From TemporalPredictiveModel
    cycle_consistency: float       # For multi-relation cycles


def compute_relation_fitness(
    encoder: GeometricRelationEncoder,
    predictor: PredictiveRelationModel,
    nodes: List[SemanticNode],
    temporal_model: Optional[TemporalPredictiveModel] = None,
) -> RelationFitnessMetrics:
    """Compute overall fitness of relation encoding.

    This aggregates prediction accuracy across all pairs,
    providing a single fitness score for ES optimization.

    Args:
        encoder: Geometric relation encoder
        predictor: Predictive relation model
        nodes: List of nodes to evaluate
        temporal_model: Optional temporal model for smoothness

    Returns:
        RelationFitnessMetrics
    """
    if len(nodes) < 2:
        return RelationFitnessMetrics(
            prediction_accuracy=1.0,
            bidirectional_consistency=1.0,
            temporal_smoothness=1.0,
            cycle_consistency=1.0,
        )

    # Evaluate all pairs
    prediction_errors = []
    bidirectional_scores = []

    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i >= j:
                continue

            relation = encoder.encode(node_i, node_j)
            fwd, bwd = predictor.validate_bidirectional(node_i, node_j, relation)

            prediction_errors.append(fwd.prediction_error)
            bidirectional_scores.append(min(fwd.relation_confidence, bwd.relation_confidence))

    # Aggregate metrics
    avg_error = np.mean(prediction_errors) if prediction_errors else 0.0
    prediction_accuracy = 1.0 / (1.0 + avg_error)

    bidirectional_consistency = np.mean(bidirectional_scores) if bidirectional_scores else 1.0

    # Temporal smoothness
    temporal_smoothness = temporal_model.get_smoothness_score() if temporal_model else 1.0

    # Cycle consistency (for triangles)
    cycle_consistency = _compute_cycle_consistency(encoder, nodes)

    return RelationFitnessMetrics(
        prediction_accuracy=prediction_accuracy,
        bidirectional_consistency=bidirectional_consistency,
        temporal_smoothness=temporal_smoothness,
        cycle_consistency=cycle_consistency,
    )


def _compute_cycle_consistency(
    encoder: GeometricRelationEncoder,
    nodes: List[SemanticNode],
) -> float:
    """Compute cycle consistency for triangles.

    For any three nodes A, B, C:
        T_AC should equal T_AB @ T_BC

    This is a fundamental algebraic constraint on relations.
    """
    if len(nodes) < 3:
        return 1.0

    errors = []

    # Check all triangles
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            for k in range(j + 1, len(nodes)):
                node_a, node_b, node_c = nodes[i], nodes[j], nodes[k]

                # Direct relation A → C
                rel_ac = encoder.encode(node_a, node_c)

                # Composed relation: A → B → C
                rel_ab = encoder.encode(node_a, node_b)
                rel_bc = encoder.encode(node_b, node_c)
                rel_ac_composed = compose_relations(rel_ab, rel_bc)

                # Compare
                error = se3_distance(rel_ac.T_rel, rel_ac_composed.T_rel)
                errors.append(error)

    if not errors:
        return 1.0

    avg_error = np.mean(errors)
    return 1.0 / (1.0 + avg_error)


# =============================================================================
# Factory Functions
# =============================================================================

def create_predictive_model(
    rotation_weight: float = 1.0,
    translation_weight: float = 1.0,
) -> PredictiveRelationModel:
    """Create default predictive relation model."""
    return PredictiveRelationModel(
        rotation_weight=rotation_weight,
        translation_weight=translation_weight,
    )


def create_temporal_model(
    smoothness_weight: float = 0.1,
    prediction_horizon: float = 0.1,
) -> TemporalPredictiveModel:
    """Create temporal predictive model."""
    return TemporalPredictiveModel(
        smoothness_weight=smoothness_weight,
        prediction_horizon=prediction_horizon,
    )


# =============================================================================
# Phase 3: Contrastive Relation Learning
# =============================================================================

@dataclass
class ContrastivePair:
    """A pair for contrastive learning.

    Contains anchor, positive, and optionally multiple negatives.
    """
    anchor: GeometricRelation          # Reference relation
    positive: GeometricRelation        # Same-type relation (should be close)
    negatives: List[GeometricRelation] # Different-type relations (should be far)
    pair_type: str = "unknown"         # Type of positive pair


class ContrastiveRelationLearner:
    """Self-supervised learner for relation understanding.

    Philosophy:
        "라벨 없이 관계를 발견한다"

        We don't need humans to label "above", "below", "supports".
        Instead, we learn from NATURAL STRUCTURE:

        1. Temporal Consistency: Same object pair, consecutive frames
           → The relation should be similar (smooth motion)

        2. Inverse Symmetry: T_AB and T_BA⁻¹ are the same relation
           → They should be close in embedding space

        3. Cycle Closure: T_AC ≈ T_AB @ T_BC
           → Composed relations should match direct relations

        4. Spatial Similarity: Objects at similar relative positions
           → Should have similar relations

    This is TRUE label-free learning:
    - No human annotation needed
    - Relations discovered from geometric consistency
    - Clustering emerges naturally in SE(3) space

    Example:
        >>> learner = ContrastiveRelationLearner(temperature=0.1)
        >>> pairs = learner.mine_temporal_pairs(relations_t0, relations_t1)
        >>> loss = learner.compute_loss(pairs)
    """

    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 1.0,
        distance_type: str = "se3",  # "se3", "lie", "euclidean"
    ):
        """Initialize contrastive learner.

        Args:
            temperature: Temperature for softmax (lower = sharper)
            margin: Margin for triplet-style losses
            distance_type: How to measure relation distance
        """
        self.temperature = temperature
        self.margin = margin
        self.distance_type = distance_type

    def relation_distance(
        self,
        rel1: GeometricRelation,
        rel2: GeometricRelation,
    ) -> float:
        """Compute distance between two relations.

        Args:
            rel1: First relation
            rel2: Second relation

        Returns:
            Distance in chosen metric
        """
        if self.distance_type == "se3":
            return se3_distance(rel1.T_rel, rel2.T_rel)
        elif self.distance_type == "lie":
            # Distance in Lie algebra (linear approximation)
            return float(np.linalg.norm(rel1.xi - rel2.xi))
        elif self.distance_type == "euclidean":
            # Embedding distance
            emb1 = rel1.to_embedding()
            emb2 = rel2.to_embedding()
            return float(np.linalg.norm(emb1 - emb2))
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")

    def compute_infonce_loss(
        self,
        anchor: GeometricRelation,
        positive: GeometricRelation,
        negatives: List[GeometricRelation],
    ) -> float:
        """Compute InfoNCE contrastive loss.

        L = -log( exp(-d(a,p)/τ) / (exp(-d(a,p)/τ) + Σ exp(-d(a,n)/τ)) )

        This encourages:
        - Small distance between anchor and positive
        - Large distance between anchor and negatives

        Args:
            anchor: Anchor relation
            positive: Positive (similar) relation
            negatives: Negative (different) relations

        Returns:
            InfoNCE loss value
        """
        tau = self.temperature

        # Positive distance
        d_pos = self.relation_distance(anchor, positive)
        exp_pos = np.exp(-d_pos / tau)

        # Negative distances
        exp_negs = []
        for neg in negatives:
            d_neg = self.relation_distance(anchor, neg)
            exp_negs.append(np.exp(-d_neg / tau))

        # InfoNCE
        denominator = exp_pos + sum(exp_negs)
        loss = -np.log(exp_pos / (denominator + 1e-10))

        return float(loss)

    def compute_triplet_loss(
        self,
        anchor: GeometricRelation,
        positive: GeometricRelation,
        negative: GeometricRelation,
    ) -> float:
        """Compute triplet margin loss.

        L = max(0, d(a,p) - d(a,n) + margin)

        Args:
            anchor: Anchor relation
            positive: Positive (similar) relation
            negative: Negative (different) relation

        Returns:
            Triplet loss value
        """
        d_pos = self.relation_distance(anchor, positive)
        d_neg = self.relation_distance(anchor, negative)

        loss = max(0.0, d_pos - d_neg + self.margin)
        return loss

    def compute_batch_loss(
        self,
        pairs: List[ContrastivePair],
        loss_type: str = "infonce",
    ) -> float:
        """Compute loss over a batch of contrastive pairs.

        Args:
            pairs: List of ContrastivePair
            loss_type: "infonce" or "triplet"

        Returns:
            Average loss over batch
        """
        if not pairs:
            return 0.0

        losses = []
        for pair in pairs:
            if loss_type == "infonce":
                loss = self.compute_infonce_loss(
                    pair.anchor, pair.positive, pair.negatives
                )
            elif loss_type == "triplet":
                # Use first negative for triplet
                if pair.negatives:
                    loss = self.compute_triplet_loss(
                        pair.anchor, pair.positive, pair.negatives[0]
                    )
                else:
                    loss = 0.0
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

            losses.append(loss)

        return float(np.mean(losses))

    # =========================================================================
    # Positive Pair Mining (Self-Supervised)
    # =========================================================================

    def mine_temporal_pairs(
        self,
        relations_t0: List[GeometricRelation],
        relations_t1: List[GeometricRelation],
        max_distance: float = 0.5,
    ) -> List[ContrastivePair]:
        """Mine positive pairs from temporal consistency.

        Same object pair at consecutive times should have similar relations.
        This is the primary source of self-supervision.

        Args:
            relations_t0: Relations at time t
            relations_t1: Relations at time t+1
            max_distance: Maximum temporal change to consider "same"

        Returns:
            List of ContrastivePair with temporal positives
        """
        pairs = []

        # Match relations by (source_id, target_id)
        rel_dict_t0 = {(r.source_id, r.target_id): r for r in relations_t0}
        rel_dict_t1 = {(r.source_id, r.target_id): r for r in relations_t1}

        for key, rel_t0 in rel_dict_t0.items():
            if key in rel_dict_t1:
                rel_t1 = rel_dict_t1[key]

                # Check if temporal change is small enough
                dist = self.relation_distance(rel_t0, rel_t1)
                if dist < max_distance:
                    # Mine negatives: different object pairs
                    negatives = [
                        r for k, r in rel_dict_t0.items()
                        if k != key
                    ]

                    pairs.append(ContrastivePair(
                        anchor=rel_t0,
                        positive=rel_t1,
                        negatives=negatives[:10],  # Limit negatives
                        pair_type="temporal",
                    ))

        return pairs

    def mine_inverse_pairs(
        self,
        relations: List[GeometricRelation],
    ) -> List[ContrastivePair]:
        """Mine positive pairs from inverse symmetry.

        T_AB and inv(T_BA) represent the same geometric relationship.
        They should be close in embedding space.

        Args:
            relations: List of relations

        Returns:
            List of ContrastivePair with inverse positives
        """
        pairs = []

        # Group relations by object pair (undirected)
        pair_dict: Dict[Tuple[int, int], List[GeometricRelation]] = {}
        for rel in relations:
            # Canonical key (smaller id first)
            key = (min(rel.source_id, rel.target_id),
                   max(rel.source_id, rel.target_id))
            if key not in pair_dict:
                pair_dict[key] = []
            pair_dict[key].append(rel)

        # Find A→B and B→A pairs
        for key, rels in pair_dict.items():
            if len(rels) >= 2:
                # Should have both directions
                rel_ab = None
                rel_ba = None
                for r in rels:
                    if r.source_id == key[0]:
                        rel_ab = r
                    else:
                        rel_ba = r

                if rel_ab is not None and rel_ba is not None:
                    # T_AB should be close to inv(T_BA)
                    rel_ba_inv = rel_ba.invert()

                    # Negatives: relations to other objects
                    negatives = [
                        r for r in relations
                        if r.source_id not in key and r.target_id not in key
                    ]

                    pairs.append(ContrastivePair(
                        anchor=rel_ab,
                        positive=rel_ba_inv,
                        negatives=negatives[:10],
                        pair_type="inverse",
                    ))

        return pairs

    def mine_cycle_pairs(
        self,
        encoder: GeometricRelationEncoder,
        nodes: List[SemanticNode],
    ) -> List[ContrastivePair]:
        """Mine positive pairs from cycle closure.

        T_AC should equal T_AB @ T_BC (transitivity).
        Direct and composed relations should be close.

        Args:
            encoder: Geometric relation encoder
            nodes: List of nodes to form triangles

        Returns:
            List of ContrastivePair with cycle positives
        """
        pairs = []

        if len(nodes) < 3:
            return pairs

        # Encode all pairs
        relations = encoder.encode_all_pairs(nodes)

        # For each triangle A-B-C
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                for k in range(j + 1, len(nodes)):
                    a_id = nodes[i].node_id
                    b_id = nodes[j].node_id
                    c_id = nodes[k].node_id

                    # Get direct A→C
                    rel_ac = relations.get((a_id, c_id))
                    if rel_ac is None:
                        continue

                    # Get A→B and B→C
                    rel_ab = relations.get((a_id, b_id))
                    rel_bc = relations.get((b_id, c_id))
                    if rel_ab is None or rel_bc is None:
                        continue

                    # Compose A→B→C
                    rel_ac_composed = compose_relations(rel_ab, rel_bc)

                    # Negatives: other relations
                    negatives = [
                        r for (src, tgt), r in relations.items()
                        if src != a_id or tgt != c_id
                    ]

                    pairs.append(ContrastivePair(
                        anchor=rel_ac,
                        positive=rel_ac_composed,
                        negatives=negatives[:10],
                        pair_type="cycle",
                    ))

        return pairs

    def mine_similarity_pairs(
        self,
        relations: List[GeometricRelation],
        threshold: float = 0.3,
    ) -> List[ContrastivePair]:
        """Mine positive pairs from geometric similarity.

        Relations with similar SE(3) transformations are positives.
        This encourages clustering of similar relations.

        Args:
            relations: List of relations
            threshold: Distance threshold for "similar"

        Returns:
            List of ContrastivePair with similarity positives
        """
        pairs = []

        for i, rel_i in enumerate(relations):
            # Find similar relations
            positives = []
            negatives = []

            for j, rel_j in enumerate(relations):
                if i == j:
                    continue

                dist = self.relation_distance(rel_i, rel_j)
                if dist < threshold:
                    positives.append(rel_j)
                else:
                    negatives.append(rel_j)

            # Create pair if we found positives
            if positives:
                pairs.append(ContrastivePair(
                    anchor=rel_i,
                    positive=positives[0],  # Closest positive
                    negatives=negatives[:10],
                    pair_type="similarity",
                ))

        return pairs


# =============================================================================
# Relation Clustering
# =============================================================================

@dataclass
class RelationCluster:
    """A cluster of similar relations."""
    cluster_id: int
    centroid_xi: np.ndarray           # Cluster center in Lie algebra
    members: List[GeometricRelation]  # Relations in this cluster
    variance: float                    # Intra-cluster variance


class RelationClusterer:
    """Cluster relations in SE(3) space.

    Philosophy:
        "관계의 종류가 자연스럽게 드러난다"

        Without labels, similar relations will cluster together.
        "Above" relations will form one cluster,
        "beside" relations will form another.

        The clusters ARE the relation types - discovered, not assigned.

    Algorithm:
        1. Embed relations in Lie algebra (ℝ⁶)
        2. Cluster using k-means or hierarchical clustering
        3. Interpret clusters as relation types
    """

    def __init__(
        self,
        n_clusters: int = 5,
        algorithm: str = "kmeans",  # "kmeans", "hierarchical", "dbscan"
    ):
        """Initialize clusterer.

        Args:
            n_clusters: Number of clusters (for kmeans)
            algorithm: Clustering algorithm
        """
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.centroids: Optional[np.ndarray] = None
        self.clusters: List[RelationCluster] = []

    def fit(self, relations: List[GeometricRelation]) -> List[RelationCluster]:
        """Fit clusters to relations.

        Args:
            relations: List of relations to cluster

        Returns:
            List of RelationCluster
        """
        if len(relations) < self.n_clusters:
            # Not enough relations
            return [RelationCluster(
                cluster_id=0,
                centroid_xi=np.zeros(6),
                members=relations,
                variance=0.0,
            )]

        # Extract Lie algebra representations
        xi_matrix = np.array([rel.xi for rel in relations])  # (N, 6)

        if self.algorithm == "kmeans":
            labels, centroids = self._kmeans(xi_matrix)
        elif self.algorithm == "hierarchical":
            labels, centroids = self._hierarchical(xi_matrix)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        self.centroids = centroids

        # Build clusters
        self.clusters = []
        for c in range(self.n_clusters):
            members = [rel for rel, lbl in zip(relations, labels) if lbl == c]
            if members:
                member_xi = np.array([m.xi for m in members])
                variance = float(np.mean(np.var(member_xi, axis=0)))
            else:
                variance = 0.0

            self.clusters.append(RelationCluster(
                cluster_id=c,
                centroid_xi=centroids[c],
                members=members,
                variance=variance,
            ))

        return self.clusters

    def predict(self, relation: GeometricRelation) -> int:
        """Predict cluster for a new relation.

        Args:
            relation: Relation to classify

        Returns:
            Cluster ID
        """
        if self.centroids is None:
            raise ValueError("Clusterer not fitted")

        xi = relation.xi
        distances = [np.linalg.norm(xi - c) for c in self.centroids]
        return int(np.argmin(distances))

    def _kmeans(
        self,
        X: np.ndarray,
        max_iters: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple k-means implementation.

        Args:
            X: Data matrix (N, 6)
            max_iters: Maximum iterations

        Returns:
            (labels, centroids)
        """
        n_samples = X.shape[0]
        k = self.n_clusters

        # Initialize centroids randomly
        rng = np.random.RandomState(42)
        idx = rng.choice(n_samples, k, replace=False)
        centroids = X[idx].copy()

        for _ in range(max_iters):
            # Assign to nearest centroid
            distances = np.array([
                [np.linalg.norm(x - c) for c in centroids]
                for x in X
            ])
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([
                X[labels == c].mean(axis=0) if np.sum(labels == c) > 0 else centroids[c]
                for c in range(k)
            ])

            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        return labels, centroids

    def _hierarchical(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple hierarchical clustering.

        Uses agglomerative clustering with average linkage.
        """
        n_samples = X.shape[0]
        k = self.n_clusters

        # Start with each point as its own cluster
        labels = np.arange(n_samples)
        n_current = n_samples

        while n_current > k:
            # Find closest pair of clusters
            unique_labels = np.unique(labels)
            min_dist = float('inf')
            merge_pair = (0, 1)

            for i, l1 in enumerate(unique_labels):
                for l2 in unique_labels[i+1:]:
                    # Average linkage
                    pts1 = X[labels == l1]
                    pts2 = X[labels == l2]
                    dist = np.mean([
                        np.linalg.norm(p1 - p2)
                        for p1 in pts1 for p2 in pts2
                    ])
                    if dist < min_dist:
                        min_dist = dist
                        merge_pair = (l1, l2)

            # Merge
            labels[labels == merge_pair[1]] = merge_pair[0]
            n_current = len(np.unique(labels))

        # Relabel to 0, 1, ..., k-1
        unique = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique)}
        labels = np.array([label_map[l] for l in labels])

        # Compute centroids
        centroids = np.array([
            X[labels == c].mean(axis=0)
            for c in range(k)
        ])

        return labels, centroids

    def get_cluster_interpretation(
        self,
        cluster: RelationCluster,
    ) -> Dict[str, float]:
        """Interpret what a cluster represents.

        Analyzes the centroid to describe the relation type.

        Args:
            cluster: Cluster to interpret

        Returns:
            Dictionary with interpretable features
        """
        xi = cluster.centroid_xi
        omega = xi[:3]  # Rotation part
        v = xi[3:6]      # Translation part

        omega_norm = np.linalg.norm(omega)
        v_norm = np.linalg.norm(v)

        interpretation = {
            "rotation_magnitude": float(omega_norm),
            "translation_magnitude": float(v_norm),
            "primarily_rotational": omega_norm > v_norm,
            "primarily_translational": v_norm > omega_norm,
            "cluster_size": len(cluster.members),
            "cluster_variance": cluster.variance,
        }

        # Interpret direction
        if v_norm > 1e-3:
            v_normalized = v / v_norm
            interpretation["translation_x"] = float(v_normalized[0])
            interpretation["translation_y"] = float(v_normalized[1])
            interpretation["translation_z"] = float(v_normalized[2])

            # Human-readable interpretation
            if v_normalized[2] > 0.7:
                interpretation["likely_type"] = "above/on_top"
            elif v_normalized[2] < -0.7:
                interpretation["likely_type"] = "below/under"
            elif abs(v_normalized[0]) > 0.7 or abs(v_normalized[1]) > 0.7:
                interpretation["likely_type"] = "beside/next_to"
            else:
                interpretation["likely_type"] = "diagonal"

        return interpretation


# =============================================================================
# PyTorch Integration for End-to-End Learning
# =============================================================================

class ContrastiveRelationLoss(torch.nn.Module):
    """PyTorch module for contrastive relation loss.

    Enables end-to-end learning with gradient descent.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        distance_type: str = "euclidean",
    ):
        super().__init__()
        self.temperature = temperature
        self.distance_type = distance_type

    def forward(
        self,
        anchor: torch.Tensor,      # (B, D) embeddings
        positive: torch.Tensor,    # (B, D) embeddings
        negatives: torch.Tensor,   # (B, N, D) embeddings
    ) -> torch.Tensor:
        """Compute InfoNCE loss.

        Args:
            anchor: Anchor embeddings (B, D)
            positive: Positive embeddings (B, D)
            negatives: Negative embeddings (B, N, D)

        Returns:
            Scalar loss
        """
        tau = self.temperature

        # Distances
        if self.distance_type == "euclidean":
            d_pos = torch.norm(anchor - positive, dim=-1)  # (B,)
            d_neg = torch.norm(
                anchor.unsqueeze(1) - negatives,
                dim=-1
            )  # (B, N)
        elif self.distance_type == "cosine":
            # Cosine distance = 1 - cosine_similarity
            anchor_norm = torch.nn.functional.normalize(anchor, dim=-1)
            positive_norm = torch.nn.functional.normalize(positive, dim=-1)
            negatives_norm = torch.nn.functional.normalize(negatives, dim=-1)

            d_pos = 1 - (anchor_norm * positive_norm).sum(dim=-1)
            d_neg = 1 - torch.bmm(
                negatives_norm,
                anchor_norm.unsqueeze(-1)
            ).squeeze(-1)
        else:
            raise ValueError(f"Unknown distance: {self.distance_type}")

        # InfoNCE
        exp_pos = torch.exp(-d_pos / tau)  # (B,)
        exp_neg = torch.exp(-d_neg / tau).sum(dim=-1)  # (B,)

        loss = -torch.log(exp_pos / (exp_pos + exp_neg + 1e-10))

        return loss.mean()


class LearnableRelationEncoder(torch.nn.Module):
    """Learnable encoder that maps raw features to relation embeddings.

    Can be trained with contrastive loss to learn better representations.
    """

    def __init__(
        self,
        input_dim: int = 12,   # From geometric relation
        hidden_dim: int = 64,
        output_dim: int = 32,
    ):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode relation features.

        Args:
            x: Relation features (B, input_dim)

        Returns:
            Learned embeddings (B, output_dim)
        """
        return self.encoder(x)


# =============================================================================
# Factory Functions
# =============================================================================

def create_contrastive_learner(
    temperature: float = 0.1,
    margin: float = 1.0,
) -> ContrastiveRelationLearner:
    """Create contrastive relation learner."""
    return ContrastiveRelationLearner(
        temperature=temperature,
        margin=margin,
    )


def create_relation_clusterer(
    n_clusters: int = 5,
) -> RelationClusterer:
    """Create relation clusterer."""
    return RelationClusterer(n_clusters=n_clusters)
