"""
SEGO (Semantic Graph Ontology Mapper) Operator - GaugeAnchor

The Perception Operator in the ONN Trinity.
Projects raw sensor data onto the Semantic Manifold (Gauge Anchoring).

Core Math (from spec/00_high_level_plan.md):
    S_i = Encoder(z_i)
    E_ij filtered by geometric proximity

Framework-agnostic: No ROS2, DDS, or external framework dependencies.

Reference:
    - spec/10_architecture.ir.yml -> modules[sego_gauge_anchor]
    - spec/11_interfaces.ir.yml -> data_schemas[SensorObservation, RawSemanticGraph]
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple, Protocol
from abc import ABC, abstractmethod

from onn.core.tensors import (
    SemanticNode,
    SemanticEdge,
    SensorObservation,
    RawSemanticGraph,
    JointState,
    BOUND_TENSOR_DIM,
    FORM_TENSOR_DIM,
    INTENT_TENSOR_DIM,
)


# -----------------------------------------------------------------------------
# Encoder Protocol (for pluggable feature extractors)
# -----------------------------------------------------------------------------

class FeatureEncoder(Protocol):
    """
    Protocol for feature encoders.

    Implementations can use CNN, ViT, or any feature extraction method.
    This keeps the SEGO anchor framework-agnostic.
    """

    def encode_visual(self, image: np.ndarray) -> np.ndarray:
        """Encode RGB image to feature vector."""
        ...

    def encode_depth(self, depth: np.ndarray) -> np.ndarray:
        """Encode depth map to feature vector."""
        ...


# -----------------------------------------------------------------------------
# Default Encoders (simple implementations for skeleton)
# -----------------------------------------------------------------------------

class SimpleFeatureEncoder:
    """
    Simple feature encoder for development/testing.

    In production, this would be replaced with CNN/ViT-based encoders.
    """

    def __init__(
        self,
        form_dim: int = FORM_TENSOR_DIM,
        bound_dim: int = BOUND_TENSOR_DIM
    ):
        self.form_dim = form_dim
        self.bound_dim = bound_dim

    def encode_visual(self, image: np.ndarray) -> np.ndarray:
        """
        Simple visual encoding via spatial statistics.

        TODO: Replace with learned encoder (CNN/ViT) in production.
        """
        if image.size == 0:
            return np.zeros(self.form_dim, dtype=np.float32)

        # Flatten and compute statistics
        flat = image.flatten().astype(np.float32)

        # Simple statistics-based encoding
        features = []

        # Mean and std per channel (if color image)
        if image.ndim == 3:
            for c in range(min(image.shape[-1], 4)):
                channel = image[..., c].flatten()
                features.extend([
                    np.mean(channel),
                    np.std(channel),
                    np.percentile(channel, 25),
                    np.percentile(channel, 75),
                ])
        else:
            features.extend([
                np.mean(flat),
                np.std(flat),
                np.percentile(flat, 25),
                np.percentile(flat, 75),
            ])

        # Pad or truncate to form_dim
        features = np.array(features, dtype=np.float32)
        if len(features) < self.form_dim:
            features = np.pad(features, (0, self.form_dim - len(features)))
        else:
            features = features[:self.form_dim]

        # Normalize
        norm = np.linalg.norm(features)
        if norm > 1e-8:
            features = features / norm

        return features

    def encode_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Encode depth map to bound tensor.

        The bound tensor represents physical occupancy/collision boundaries.

        TODO: Replace with proper 3D representation in production.
        """
        if depth.size == 0:
            return np.zeros(self.bound_dim, dtype=np.float32)

        flat = depth.flatten().astype(np.float32)

        # Compute spatial statistics
        features = [
            np.mean(flat),
            np.std(flat),
            np.min(flat),
            np.max(flat),
            np.percentile(flat, 10),
            np.percentile(flat, 50),
            np.percentile(flat, 90),
        ]

        # Add histogram-based features
        hist, _ = np.histogram(flat, bins=self.bound_dim - len(features), density=True)
        features.extend(hist.tolist())

        features = np.array(features[:self.bound_dim], dtype=np.float32)

        # Normalize to unit sphere (for projection constraint)
        norm = np.linalg.norm(features)
        if norm > 1e-8:
            features = features / norm

        return features


# -----------------------------------------------------------------------------
# Detection Result (for object detection integration)
# -----------------------------------------------------------------------------

@dataclass
class Detection:
    """
    Detected object from perception pipeline.

    This is an intermediate representation before conversion to SemanticNode.
    """
    detection_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    mask: Optional[np.ndarray] = None
    centroid_3d: Optional[np.ndarray] = None  # (x, y, z) in camera frame


# -----------------------------------------------------------------------------
# SEGOGaugeAnchor
# -----------------------------------------------------------------------------

@dataclass
class SEGOConfig:
    """Configuration for SEGO GaugeAnchor."""
    proximity_threshold: float = 2.0  # meters, for edge candidate generation
    min_detection_confidence: float = 0.3
    relation_embedding_dim: int = 16
    max_nodes: int = 100  # Maximum nodes to track
    edge_decay_rate: float = 0.9  # For temporal smoothing
    
    # === CPL_005: Configurable intent encoding constants ===
    # Image dimension normalization (used in _compute_intent)
    image_width_norm: float = 640.0  # Typical image width for normalization
    image_height_norm: float = 480.0  # Typical image height for normalization
    area_normalization: float = 100000.0  # Area divisor for size-based encoding
    depth_normalization: float = 5.0  # Depth divisor for 3D-based encoding


class SEGOGaugeAnchor:
    """
    The SEGO Operator implementation (GaugeAnchor).

    Projects raw sensor observations onto the Semantic Manifold,
    creating anchored tensor states and proposing edge relationships.

    This implements the "Gauge Anchoring" process where raw data z_i
    is projected to semantic node states S_i.

    Reference:
        - spec/10_architecture.ir.yml -> modules[sego_gauge_anchor]
        - spec/00_high_level_plan.md -> Section 3.2 (SEGO)
    """

    def __init__(
        self,
        config: Optional[SEGOConfig] = None,
        encoder: Optional[FeatureEncoder] = None
    ):
        """
        Initialize the SEGO anchor.

        Args:
            config: SEGO configuration parameters
            encoder: Feature encoder for visual/depth data
        """
        self.config = config or SEGOConfig()
        self.encoder = encoder or SimpleFeatureEncoder()

        # Node tracking state
        self._next_node_id: int = 1
        self._node_registry: dict[int, SemanticNode] = {}
        self._detection_to_node: dict[int, int] = {}  # detection_id -> node_id

    def process(
        self,
        observation: SensorObservation,
        detections: Optional[List[Detection]] = None
    ) -> RawSemanticGraph:
        """
        Process sensor observation and produce raw semantic graph.

        This is the main entry point for the SEGO operator.

        Args:
            observation: Raw sensor data (RGB, depth, joint state)
            detections: Optional pre-computed object detections

        Returns:
            RawSemanticGraph with projected nodes and edge candidates

        Reference: spec/10_architecture.ir.yml -> modules[sego_gauge_anchor]
        """
        # If no detections provided, use simple grid-based segmentation
        if detections is None:
            detections = self._simple_detection(observation)

        # Project detections to semantic nodes
        nodes = self._project_to_nodes(observation, detections)

        # Generate edge candidates based on geometric proximity
        edges = self._generate_edge_candidates(nodes)

        return RawSemanticGraph(
            timestamp_ns=observation.timestamp_ns,
            nodes=nodes,
            edge_candidates=edges
        )

    def _simple_detection(self, observation: SensorObservation) -> List[Detection]:
        """
        Simple placeholder detection for when no detector is provided.

        In production, this would be replaced with actual object detection
        (YOLO, Mask R-CNN, etc.).

        TODO: Integrate with actual detection pipeline.
        """
        detections = []

        # Create a single "scene" detection from the full image
        if observation.rgb_images:
            for idx, img in enumerate(observation.rgb_images):
                h, w = img.shape[:2] if img.ndim >= 2 else (1, 1)
                detections.append(Detection(
                    detection_id=idx,
                    class_name="scene",
                    confidence=1.0,
                    bbox=(0, 0, w, h),
                    centroid_3d=None
                ))

        return detections

    def _project_to_nodes(
        self,
        observation: SensorObservation,
        detections: List[Detection]
    ) -> List[SemanticNode]:
        """
        Project detections to SemanticNode tensors.

        This is the core "Gauge Anchoring" operation:
            S_i = Encoder(z_i)
        """
        nodes = []

        for det in detections:
            if det.confidence < self.config.min_detection_confidence:
                continue

            # Get or create node ID for this detection
            if det.detection_id in self._detection_to_node:
                node_id = self._detection_to_node[det.detection_id]
            else:
                node_id = self._next_node_id
                self._next_node_id += 1
                self._detection_to_node[det.detection_id] = node_id

            # Extract image patch for this detection
            rgb_patch = self._extract_patch(observation.rgb_images, det.bbox)
            depth_patch = self._extract_patch(observation.depth_maps, det.bbox)

            # Encode to tensors
            form_tensor = self.encoder.encode_visual(rgb_patch)
            
            # CPL_009: Explicitly populate geometry for Physics Loss compliance
            # Spec 1.1.1: b_0:3=Pos, b_8:11=Extents
            bound_tensor = self.encoder.encode_depth(depth_patch)
            
            # 1. Position (Use 3D centroid if available, else normalized 2D center)
            if det.centroid_3d is not None and len(det.centroid_3d) >= 3:
                bound_tensor[0] = det.centroid_3d[0]
                bound_tensor[1] = det.centroid_3d[1]
                bound_tensor[2] = det.centroid_3d[2]
            else:
                # Fallback: Normalized 2D position (approx 0-1 range)
                cx = det.bbox[0] + det.bbox[2] / 2
                cy = det.bbox[1] + det.bbox[3] / 2
                bound_tensor[0] = cx / 640.0  # Normalized X
                bound_tensor[1] = cy / 480.0  # Normalized Y
                bound_tensor[2] = 0.0

            # 2. Extents/Radius (b_11 used as collision radius)
            # Normalize size (assuming 100px is unit size for safety)
            size_norm = max(det.bbox[2], det.bbox[3]) / 100.0
            bound_tensor[8] = det.bbox[2] / 100.0  # L
            bound_tensor[9] = det.bbox[3] / 100.0  # W
            bound_tensor[11] = size_norm * 0.5     # Radius (approx half max dim)
            intent_tensor = self._compute_intent(det)

            node = SemanticNode(
                node_id=node_id,
                bound_tensor=bound_tensor,
                form_tensor=form_tensor,
                intent_tensor=intent_tensor
            )

            nodes.append(node)
            self._node_registry[node_id] = node

        return nodes

    def _extract_patch(
        self,
        images: List[np.ndarray],
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Extract image patch from bounding box."""
        if not images:
            return np.array([])

        img = images[0]  # Use first image
        x, y, w, h = bbox

        # Ensure valid bounds
        x = max(0, x)
        y = max(0, y)

        if img.ndim >= 2:
            h_img, w_img = img.shape[:2]
            x2 = min(x + w, w_img)
            y2 = min(y + h, h_img)

            if x2 > x and y2 > y:
                return img[y:y2, x:x2]

        return img

    def _compute_intent(self, detection: Detection) -> np.ndarray:
        """
        Compute intent tensor from detection using CONTINUOUS encoding.

        The intent tensor encodes functional affordances and task relevance.
        This uses continuous feature-based encoding instead of discrete class lookup
        to satisfy spec/01_constraints.md (no discrete logic in critical path).

        Encoding strategy:
        - Dims 0-3: Confidence-scaled activation
        - Dims 4-7: Size-based affordance (graspable for small, supportive for large)
        - Dims 8-11: Position-based affordance (based on bbox location)
        - Dims 12-15: Reserved for learned embeddings

        Reference: spec/01_constraints.md Section 2.2 (No Discrete Logic in Critical Paths)
        """
        intent = np.zeros(INTENT_TENSOR_DIM, dtype=np.float32)

        # === CONTINUOUS ENCODING (no class_name lookup) ===

        # 1. Confidence-based activation (dims 0-3)
        conf = detection.confidence
        intent[0] = conf  # General activation
        intent[1] = conf * 0.5  # Secondary activation
        intent[2] = conf * 0.3  # Tertiary activation
        intent[3] = 1.0 - conf  # Uncertainty signal

        # 2. Size-based affordance (dims 4-7)
        # Small objects are more likely graspable, large objects more supportive
        x, y, w, h = detection.bbox
        area = w * h
        # CPL_005: Use configurable area normalization
        normalized_size = min(area / self.config.area_normalization, 1.0)

        intent[4] = 1.0 - normalized_size  # Graspability (inverse of size)
        intent[5] = normalized_size  # Supportiveness (proportional to size)
        intent[6] = min(w / max(h, 1), 1.0)  # Aspect ratio (horizontal)
        intent[7] = min(h / max(w, 1), 1.0)  # Aspect ratio (vertical)

        # 3. Position-based affordance (dims 8-11)
        # Objects at the bottom are often supportive, at center are manipulable
        center_x = x + w / 2
        center_y = y + h / 2

        # CPL_005: Use configurable image dimensions for normalization
        norm_x = center_x / self.config.image_width_norm
        norm_y = center_y / self.config.image_height_norm

        intent[8] = 1.0 - norm_y  # Higher objects (top of scene)
        intent[9] = norm_y  # Lower objects (bottom of scene)
        intent[10] = abs(0.5 - norm_x) * 2  # Distance from center (horizontal)
        intent[11] = abs(0.5 - norm_y) * 2  # Distance from center (vertical)

        # 4. 3D-based affordance if available (dims 12-15)
        if detection.centroid_3d is not None:
            depth = detection.centroid_3d[2] if len(detection.centroid_3d) > 2 else 0.0
            # CPL_005: Use configurable depth normalization
            depth_norm = self.config.depth_normalization
            intent[12] = min(depth / depth_norm, 1.0)  # Normalized depth
            intent[13] = 1.0 - min(depth / depth_norm, 1.0)  # Inverse depth (reachability)
        else:
            intent[12] = 0.5  # Unknown depth
            intent[13] = 0.5

        intent[14] = conf  # Redundant confidence for robustness
        intent[15] = 0.0  # Reserved

        # Clamp to [0, 1] (affordance probabilities)
        intent = np.clip(intent, 0.0, 1.0)

        return intent

    def _generate_edge_candidates(
        self,
        nodes: List[SemanticNode]
    ) -> List[SemanticEdge]:
        """
        Generate edge candidates based on geometric proximity.

        Edges are proposed for nodes that are spatially close.
        The LOGOS solver will later validate or prune these edges.
        """
        edges = []

        for i, ni in enumerate(nodes):
            for j, nj in enumerate(nodes):
                if i >= j:  # Avoid duplicates
                    continue

                # Compute proximity from bound tensors
                # (simplified: using tensor similarity as proxy for spatial proximity)
                proximity = self._compute_proximity(ni, nj)

                if proximity > self.config.proximity_threshold:
                    continue  # Too far apart

                # Create edge candidate
                relation_embedding = self._compute_relation_embedding(ni, nj)
                weight = 1.0 / (1.0 + proximity)  # Closer = stronger
                probability = min(1.0, weight)

                edge = SemanticEdge(
                    source_id=ni.node_id,
                    target_id=nj.node_id,
                    relation_embedding=relation_embedding,
                    weight=weight,
                    probability=probability
                )
                edges.append(edge)

        return edges

    def _compute_proximity(
        self,
        node_i: SemanticNode,
        node_j: SemanticNode
    ) -> float:
        """
        Compute spatial proximity between two nodes.

        Uses bound tensors as proxy for spatial location.
        Returns distance (lower = closer).
        """
        # Use L2 distance in bound tensor space
        diff = node_i.bound_tensor - node_j.bound_tensor
        return float(np.linalg.norm(diff))

    def _compute_relation_embedding(
        self,
        node_i: SemanticNode,
        node_j: SemanticNode
    ) -> np.ndarray:
        """
        Compute relation embedding between two nodes.

        The relation embedding encodes the nature of the relationship
        (support, containment, proximity, etc.).

        TODO: Learn relation embeddings from data.
        """
        dim = self.config.relation_embedding_dim

        # Simple relation encoding based on tensor differences
        bound_diff = node_i.bound_tensor[:dim // 2] - node_j.bound_tensor[:dim // 2]
        intent_diff = node_i.intent_tensor[:dim // 2] - node_j.intent_tensor[:dim // 2]

        relation = np.concatenate([bound_diff, intent_diff])

        # Pad or truncate
        if len(relation) < dim:
            relation = np.pad(relation, (0, dim - len(relation)))
        else:
            relation = relation[:dim]

        return relation.astype(np.float32)

    def reset(self) -> None:
        """Reset anchor state (clear node registry)."""
        self._next_node_id = 1
        self._node_registry.clear()
        self._detection_to_node.clear()

    def get_node(self, node_id: int) -> Optional[SemanticNode]:
        """Get a node from the registry by ID."""
        return self._node_registry.get(node_id)


# -----------------------------------------------------------------------------
# Factory function
# -----------------------------------------------------------------------------

def create_default_sego_anchor() -> SEGOGaugeAnchor:
    """Create SEGO anchor with default configuration."""
    return SEGOGaugeAnchor()
