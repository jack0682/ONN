"""
Enhanced SEGO (Semantic Graph Ontology Mapper) - Precision Perception

Advanced perception operator with:
1. Multi-scale feature encoding (Fourier + spatial pyramid)
2. 3D geometry estimation from depth
3. Semantic-aware relation encoding
4. Temporal consistency tracking
5. Context-aware edge generation

Core Math:
    S_i = Encoder(z_i) where z_i = (RGB, Depth, Mask, 3D)

    Bound tensor b_i encodes:
        - Position (3D centroid)
        - Orientation (principal axes)
        - Extents (bounding box dimensions)
        - Shape features (curvature, convexity)

    Form tensor f_i encodes:
        - Multi-scale visual features
        - Texture descriptors
        - Color histogram
        - Edge/contour features

    Intent tensor i_i encodes:
        - Affordance probabilities
        - Functional properties
        - Task relevance

Reference:
    - spec/10_architecture.ir.yml -> modules[sego_gauge_anchor]
    - "Understanding" layer of perception
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Protocol
from collections import deque
import logging

from onn.core.tensors import (
    SemanticNode,
    SemanticEdge,
    SensorObservation,
    RawSemanticGraph,
    BOUND_TENSOR_DIM,
    FORM_TENSOR_DIM,
    INTENT_TENSOR_DIM,
)
from onn.ops.sego_anchor import Detection, SEGOConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EnhancedSEGOConfig(SEGOConfig):
    """Extended configuration for Enhanced SEGO."""

    # Multi-scale feature extraction
    num_scales: int = 3  # Number of pyramid levels
    fourier_features: int = 8  # Number of Fourier frequency bands

    # 3D geometry
    focal_length: float = 525.0  # Camera focal length (pixels)
    depth_scale: float = 1000.0  # Depth scale (mm to meters)
    min_depth: float = 0.1  # Minimum valid depth (meters)
    max_depth: float = 10.0  # Maximum valid depth (meters)

    # Temporal tracking
    tracking_history: int = 5  # Frames to keep for tracking
    iou_threshold: float = 0.3  # IoU threshold for matching
    feature_match_threshold: float = 0.7  # Feature similarity threshold

    # Spatial relations
    vertical_threshold: float = 0.1  # Threshold for above/below (meters)
    horizontal_threshold: float = 0.1  # Threshold for left/right (meters)
    containment_threshold: float = 0.8  # IoU threshold for containment
    support_distance: float = 0.05  # Distance for support relation (meters)

    # Edge generation
    max_edges_per_node: int = 5  # Maximum edges per node
    edge_confidence_threshold: float = 0.3  # Minimum edge confidence


# =============================================================================
# Multi-Scale Feature Encoder
# =============================================================================

class MultiScaleFeatureEncoder:
    """
    Advanced feature encoder with multi-scale and Fourier features.

    Extracts:
    - Spatial pyramid features (local to global)
    - Fourier positional encoding
    - Texture descriptors
    - Shape features from depth
    """

    def __init__(self, config: EnhancedSEGOConfig):
        self.config = config
        self.form_dim = FORM_TENSOR_DIM
        self.bound_dim = BOUND_TENSOR_DIM

        # Precompute Fourier frequency bands
        self.freq_bands = 2.0 ** np.linspace(0, config.fourier_features - 1, config.fourier_features)

    def encode_visual(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Multi-scale visual feature encoding.

        Args:
            image: RGB image patch, shape (H, W, 3)
            mask: Optional binary mask, shape (H, W)

        Returns:
            Form tensor of shape (FORM_TENSOR_DIM,)
        """
        if image.size == 0:
            return np.zeros(self.form_dim, dtype=np.float32)

        features = []

        # 1. Multi-scale statistics (pyramid)
        pyramid_features = self._compute_spatial_pyramid(image, mask)
        features.extend(pyramid_features)

        # 2. Color histogram features
        color_features = self._compute_color_features(image, mask)
        features.extend(color_features)

        # 3. Texture features (local binary patterns approximation)
        texture_features = self._compute_texture_features(image)
        features.extend(texture_features)

        # 4. Edge/gradient features
        edge_features = self._compute_edge_features(image)
        features.extend(edge_features)

        # Assemble and normalize
        features = np.array(features, dtype=np.float32)

        # Pad or truncate
        if len(features) < self.form_dim:
            features = np.pad(features, (0, self.form_dim - len(features)))
        else:
            features = features[:self.form_dim]

        # L2 normalize
        norm = np.linalg.norm(features)
        if norm > 1e-8:
            features = features / norm

        return features

    def _compute_spatial_pyramid(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray]
    ) -> List[float]:
        """Compute spatial pyramid features."""
        features = []

        h, w = image.shape[:2]

        for level in range(self.config.num_scales):
            # Divide image into 2^level x 2^level cells
            grid_size = 2 ** level
            cell_h = h // grid_size
            cell_w = w // grid_size

            for i in range(grid_size):
                for j in range(grid_size):
                    y1, y2 = i * cell_h, (i + 1) * cell_h
                    x1, x2 = j * cell_w, (j + 1) * cell_w

                    cell = image[y1:y2, x1:x2]

                    if mask is not None:
                        cell_mask = mask[y1:y2, x1:x2]
                        if cell_mask.sum() > 0:
                            cell = cell[cell_mask > 0]

                    if cell.size > 0:
                        # Mean color per channel
                        if cell.ndim == 3:
                            features.extend(cell.mean(axis=(0, 1) if cell.ndim == 3 else 0).tolist()[:3])
                        else:
                            features.append(float(cell.mean()))
                    else:
                        features.extend([0.0, 0.0, 0.0])

        return features[:12]  # Limit pyramid features

    def _compute_color_features(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray]
    ) -> List[float]:
        """Compute color histogram features."""
        features = []

        if image.ndim != 3 or image.shape[2] < 3:
            return [0.0] * 6

        # Apply mask if available
        if mask is not None and mask.sum() > 0:
            pixels = image[mask > 0]
        else:
            pixels = image.reshape(-1, image.shape[-1])

        if len(pixels) == 0:
            return [0.0] * 6

        # Color statistics per channel
        for c in range(min(3, pixels.shape[-1])):
            channel = pixels[..., c].flatten().astype(np.float32)
            features.append(float(np.mean(channel)) / 255.0)
            features.append(float(np.std(channel)) / 128.0)

        return features[:6]

    def _compute_texture_features(self, image: np.ndarray) -> List[float]:
        """Compute texture features using gradient statistics."""
        features = []

        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(np.float32)

        if gray.size < 4:
            return [0.0] * 4

        # Gradient magnitudes
        gy, gx = np.gradient(gray)
        grad_mag = np.sqrt(gx**2 + gy**2)

        features.append(float(np.mean(grad_mag)))
        features.append(float(np.std(grad_mag)))
        features.append(float(np.max(grad_mag)) if grad_mag.size > 0 else 0.0)

        # Gradient direction histogram (simplified)
        grad_dir = np.arctan2(gy, gx + 1e-8)
        features.append(float(np.std(grad_dir)))

        return features[:4]

    def _compute_edge_features(self, image: np.ndarray) -> List[float]:
        """Compute edge-based features."""
        features = []

        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(np.float32)

        if gray.size < 9:
            return [0.0] * 4

        # Simple Sobel-like edge detection
        # Horizontal edges
        kernel_h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 4.0
        # Vertical edges
        kernel_v = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 4.0

        # Convolve (simplified without scipy)
        h, w = gray.shape
        if h >= 3 and w >= 3:
            edges_h = np.zeros((h-2, w-2))
            edges_v = np.zeros((h-2, w-2))

            for i in range(h-2):
                for j in range(w-2):
                    patch = gray[i:i+3, j:j+3]
                    edges_h[i, j] = np.sum(patch * kernel_h)
                    edges_v[i, j] = np.sum(patch * kernel_v)

            edge_mag = np.sqrt(edges_h**2 + edges_v**2)

            features.append(float(np.mean(edge_mag)))
            features.append(float(np.std(edge_mag)))
            features.append(float(np.sum(edge_mag > np.mean(edge_mag))) / max(edge_mag.size, 1))
            features.append(float(np.max(edge_mag)) if edge_mag.size > 0 else 0.0)
        else:
            features.extend([0.0] * 4)

        return features[:4]

    def encode_depth_to_geometry(
        self,
        depth: np.ndarray,
        bbox: Tuple[int, int, int, int],
        mask: Optional[np.ndarray] = None,
        image_shape: Tuple[int, int] = (480, 640),
    ) -> np.ndarray:
        """
        Encode depth to 3D geometry features.

        Computes:
        - 3D centroid (position)
        - Principal axes (orientation)
        - Bounding box extents
        - Shape descriptors (curvature, convexity)

        Args:
            depth: Depth patch, shape (H, W)
            bbox: Bounding box (x, y, w, h)
            mask: Optional binary mask
            image_shape: Full image shape (H, W)

        Returns:
            Bound tensor of shape (BOUND_TENSOR_DIM,)
        """
        bound = np.zeros(self.bound_dim, dtype=np.float32)

        if depth.size == 0:
            return bound

        # Get valid depth values
        if mask is not None and mask.size == depth.size:
            valid_mask = (depth > self.config.min_depth) & (depth < self.config.max_depth) & (mask > 0)
        else:
            valid_mask = (depth > self.config.min_depth) & (depth < self.config.max_depth)

        valid_depths = depth[valid_mask]

        if len(valid_depths) == 0:
            # Fallback to 2D
            return self._encode_2d_fallback(bbox, image_shape)

        # 1. Compute 3D point cloud
        points_3d = self._depth_to_pointcloud(depth, bbox, valid_mask, image_shape)

        if len(points_3d) < 3:
            return self._encode_2d_fallback(bbox, image_shape)

        # 2. Compute 3D centroid (position)
        centroid = np.mean(points_3d, axis=0)
        bound[0:3] = centroid  # Position

        # 3. Compute principal axes via PCA (orientation)
        centered = points_3d - centroid
        if len(centered) >= 3:
            try:
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                # Sort by eigenvalue (descending)
                idx = np.argsort(eigenvalues)[::-1]
                principal_axis = eigenvectors[:, idx[0]]
                bound[3:6] = principal_axis  # Orientation (principal axis)
            except:
                bound[3:6] = [0, 0, 1]  # Default Z-up

        # 4. Compute extents (3D bounding box)
        min_pt = np.min(points_3d, axis=0)
        max_pt = np.max(points_3d, axis=0)
        extents = max_pt - min_pt
        bound[6:9] = extents  # L, W, H

        # 5. Compute shape features
        # Radius (half diagonal)
        bound[9] = np.linalg.norm(extents) / 2.0

        # Aspect ratios
        if extents[2] > 1e-6:
            bound[10] = extents[0] / extents[2]  # L/H ratio
            bound[11] = extents[1] / extents[2]  # W/H ratio

        # 6. Surface features
        if len(points_3d) >= 10:
            # Estimate surface normals (simplified)
            bound[12:15] = self._estimate_surface_normal(points_3d)

            # Convexity estimate
            bound[15] = self._estimate_convexity(points_3d, centroid)

        return bound

    def _depth_to_pointcloud(
        self,
        depth: np.ndarray,
        bbox: Tuple[int, int, int, int],
        valid_mask: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Convert depth patch to 3D point cloud."""
        x, y, w, h = bbox
        img_h, img_w = image_shape

        # Create pixel coordinates
        u_coords, v_coords = np.meshgrid(
            np.arange(w) + x,
            np.arange(h) + y
        )

        # Get valid points
        u_valid = u_coords[valid_mask]
        v_valid = v_coords[valid_mask]
        z_valid = depth[valid_mask]

        # Convert to 3D (pinhole camera model)
        cx, cy = img_w / 2, img_h / 2
        fx = fy = self.config.focal_length

        x_3d = (u_valid - cx) * z_valid / fx
        y_3d = (v_valid - cy) * z_valid / fy
        z_3d = z_valid

        points = np.stack([x_3d, y_3d, z_3d], axis=1)

        return points

    def _encode_2d_fallback(
        self,
        bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Fallback 2D encoding when depth is unavailable."""
        bound = np.zeros(self.bound_dim, dtype=np.float32)

        x, y, w, h = bbox
        img_h, img_w = image_shape

        # Normalized 2D position
        bound[0] = (x + w/2) / img_w
        bound[1] = (y + h/2) / img_h
        bound[2] = 0.5  # Default depth

        # 2D extents (normalized)
        bound[6] = w / img_w
        bound[7] = h / img_h
        bound[8] = 0.1  # Default height

        bound[9] = max(w, h) / max(img_w, img_h)  # Radius

        return bound

    def _estimate_surface_normal(self, points: np.ndarray) -> np.ndarray:
        """Estimate average surface normal from point cloud."""
        if len(points) < 3:
            return np.array([0, 0, 1], dtype=np.float32)

        # Use PCA - smallest eigenvector is normal
        centered = points - np.mean(points, axis=0)
        try:
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Smallest eigenvalue's eigenvector is the normal
            normal = eigenvectors[:, 0]
            # Ensure normal points up (positive Z)
            if normal[2] < 0:
                normal = -normal
            return normal.astype(np.float32)
        except:
            return np.array([0, 0, 1], dtype=np.float32)

    def _estimate_convexity(self, points: np.ndarray, centroid: np.ndarray) -> float:
        """Estimate convexity of the point cloud."""
        if len(points) < 4:
            return 0.5

        # Simple convexity: ratio of points close to centroid
        distances = np.linalg.norm(points - centroid, axis=1)
        mean_dist = np.mean(distances)

        if mean_dist < 1e-6:
            return 1.0

        # Convex shapes have more uniform distance distribution
        std_dist = np.std(distances)
        convexity = 1.0 - min(std_dist / mean_dist, 1.0)

        return float(convexity)

    def encode_fourier_position(
        self,
        position: np.ndarray,
        max_freq: float = 10.0,
    ) -> np.ndarray:
        """
        Encode position using Fourier features for better spatial resolution.

        Args:
            position: 3D position (x, y, z)
            max_freq: Maximum frequency

        Returns:
            Fourier-encoded position features
        """
        features = []

        for freq in self.freq_bands:
            for p in position[:3]:
                features.append(np.sin(2 * np.pi * freq * p / max_freq))
                features.append(np.cos(2 * np.pi * freq * p / max_freq))

        return np.array(features, dtype=np.float32)


# =============================================================================
# Spatial Relation Analyzer
# =============================================================================

class SpatialRelationAnalyzer:
    """
    Analyzes spatial relationships between objects.

    Computes:
    - Topological relations (above, below, left, right, front, behind)
    - Support relations (supports, supported_by)
    - Containment relations (contains, inside)
    - Proximity relations (near, far)
    """

    def __init__(self, config: EnhancedSEGOConfig):
        self.config = config

    def compute_relation(
        self,
        node_i: SemanticNode,
        node_j: SemanticNode,
    ) -> Tuple[np.ndarray, float, Dict[str, float]]:
        """
        Compute relation embedding between two nodes.

        Returns:
            relation_embedding: Encoded relation vector
            confidence: Relation confidence score
            relation_scores: Dict of individual relation scores
        """
        # Extract positions
        pos_i = node_i.bound_tensor[0:3]
        pos_j = node_j.bound_tensor[0:3]

        # Extract extents
        ext_i = node_i.bound_tensor[6:9]
        ext_j = node_j.bound_tensor[6:9]

        # Compute spatial relations
        relations = {}

        # 1. Vertical relations
        vertical_diff = pos_i[2] - pos_j[2]  # Z difference
        relations['above'] = self._sigmoid(vertical_diff, self.config.vertical_threshold)
        relations['below'] = self._sigmoid(-vertical_diff, self.config.vertical_threshold)

        # 2. Horizontal relations
        horizontal_diff_x = pos_i[0] - pos_j[0]
        horizontal_diff_y = pos_i[1] - pos_j[1]
        relations['right'] = self._sigmoid(horizontal_diff_x, self.config.horizontal_threshold)
        relations['left'] = self._sigmoid(-horizontal_diff_x, self.config.horizontal_threshold)
        relations['front'] = self._sigmoid(-horizontal_diff_y, self.config.horizontal_threshold)
        relations['behind'] = self._sigmoid(horizontal_diff_y, self.config.horizontal_threshold)

        # 3. Support relation
        # Object i supports j if i is below j and they're close
        support_score = self._compute_support_score(pos_i, pos_j, ext_i, ext_j)
        relations['supports'] = support_score
        relations['supported_by'] = self._compute_support_score(pos_j, pos_i, ext_j, ext_i)

        # 4. Containment relation
        containment = self._compute_containment(pos_i, pos_j, ext_i, ext_j)
        relations['contains'] = containment
        relations['inside'] = self._compute_containment(pos_j, pos_i, ext_j, ext_i)

        # 5. Proximity
        distance = np.linalg.norm(pos_i - pos_j)
        relations['near'] = self._sigmoid(-distance + 1.0, 0.5)
        relations['far'] = self._sigmoid(distance - 2.0, 0.5)

        # 6. Functional relations (based on intent)
        functional = self._compute_functional_relation(node_i, node_j)
        relations.update(functional)

        # Build relation embedding
        relation_embedding = self._build_relation_embedding(relations, pos_i, pos_j, ext_i, ext_j)

        # Compute overall confidence
        confidence = self._compute_relation_confidence(relations, distance)

        return relation_embedding, confidence, relations

    def _sigmoid(self, x: float, threshold: float) -> float:
        """Smooth sigmoid activation with overflow protection."""
        z = 10 * (x / max(threshold, 1e-6))
        # Use numerically stable sigmoid
        if z >= 0:
            return 1.0 / (1.0 + np.exp(-z))
        else:
            exp_z = np.exp(z)
            return exp_z / (1.0 + exp_z)

    def _compute_support_score(
        self,
        pos_lower: np.ndarray,
        pos_upper: np.ndarray,
        ext_lower: np.ndarray,
        ext_upper: np.ndarray,
    ) -> float:
        """Compute support relation score."""
        # Lower object should be below upper object
        vertical_diff = pos_upper[2] - pos_lower[2]
        if vertical_diff < 0:
            return 0.0

        # Check vertical proximity (considering extents)
        lower_top = pos_lower[2] + ext_lower[2] / 2
        upper_bottom = pos_upper[2] - ext_upper[2] / 2
        gap = upper_bottom - lower_top

        if gap > self.config.support_distance:
            return 0.0

        # Check horizontal overlap
        overlap_x = self._compute_1d_overlap(
            pos_lower[0] - ext_lower[0]/2, pos_lower[0] + ext_lower[0]/2,
            pos_upper[0] - ext_upper[0]/2, pos_upper[0] + ext_upper[0]/2,
        )
        overlap_y = self._compute_1d_overlap(
            pos_lower[1] - ext_lower[1]/2, pos_lower[1] + ext_lower[1]/2,
            pos_upper[1] - ext_upper[1]/2, pos_upper[1] + ext_upper[1]/2,
        )

        if overlap_x <= 0 or overlap_y <= 0:
            return 0.0

        # Compute support score
        support_area = overlap_x * overlap_y
        upper_area = ext_upper[0] * ext_upper[1]

        return min(support_area / max(upper_area, 1e-6), 1.0)

    def _compute_1d_overlap(self, a1: float, a2: float, b1: float, b2: float) -> float:
        """Compute 1D overlap between two intervals."""
        return max(0, min(a2, b2) - max(a1, b1))

    def _compute_containment(
        self,
        pos_outer: np.ndarray,
        pos_inner: np.ndarray,
        ext_outer: np.ndarray,
        ext_inner: np.ndarray,
    ) -> float:
        """Compute containment relation score."""
        # Check if inner is within outer bounds
        for d in range(3):
            outer_min = pos_outer[d] - ext_outer[d] / 2
            outer_max = pos_outer[d] + ext_outer[d] / 2
            inner_min = pos_inner[d] - ext_inner[d] / 2
            inner_max = pos_inner[d] + ext_inner[d] / 2

            if inner_min < outer_min or inner_max > outer_max:
                return 0.0

        # Compute volume ratio
        inner_vol = np.prod(ext_inner + 1e-6)
        outer_vol = np.prod(ext_outer + 1e-6)

        return min(inner_vol / outer_vol, 1.0)

    def _compute_functional_relation(
        self,
        node_i: SemanticNode,
        node_j: SemanticNode,
    ) -> Dict[str, float]:
        """Compute functional relations from intent tensors."""
        intent_i = node_i.intent_tensor
        intent_j = node_j.intent_tensor

        relations = {}

        # Graspable + Supportive interaction
        relations['can_grasp'] = intent_i[4] * (1 - intent_j[5])  # i graspable, j not too supportive
        relations['can_place_on'] = intent_j[5] * (1 - intent_i[5])  # j supportive, i not

        # Reachability
        relations['reachable_pair'] = intent_i[13] * intent_j[13]  # Both reachable

        return relations

    def _build_relation_embedding(
        self,
        relations: Dict[str, float],
        pos_i: np.ndarray,
        pos_j: np.ndarray,
        ext_i: np.ndarray,
        ext_j: np.ndarray,
    ) -> np.ndarray:
        """Build relation embedding vector."""
        dim = 16  # Fixed embedding dimension

        embedding = np.zeros(dim, dtype=np.float32)

        # Spatial relations (dims 0-5)
        embedding[0] = relations.get('above', 0) - relations.get('below', 0)
        embedding[1] = relations.get('right', 0) - relations.get('left', 0)
        embedding[2] = relations.get('front', 0) - relations.get('behind', 0)

        # Support/containment (dims 3-5)
        embedding[3] = relations.get('supports', 0)
        embedding[4] = relations.get('contains', 0)
        embedding[5] = relations.get('near', 0)

        # Relative position (normalized)
        diff = pos_j - pos_i
        norm = np.linalg.norm(diff)
        if norm > 1e-6:
            embedding[6:9] = diff / norm

        # Relative size
        size_i = np.linalg.norm(ext_i)
        size_j = np.linalg.norm(ext_j)
        if size_i + size_j > 1e-6:
            embedding[9] = (size_i - size_j) / (size_i + size_j)

        # Functional relations (dims 10-12)
        embedding[10] = relations.get('can_grasp', 0)
        embedding[11] = relations.get('can_place_on', 0)
        embedding[12] = relations.get('reachable_pair', 0)

        # Distance features (dims 13-15)
        embedding[13] = min(norm / 2.0, 1.0)  # Normalized distance
        embedding[14] = relations.get('supported_by', 0)
        embedding[15] = relations.get('inside', 0)

        return embedding

    def _compute_relation_confidence(
        self,
        relations: Dict[str, float],
        distance: float,
    ) -> float:
        """Compute overall relation confidence."""
        # Higher confidence for closer objects with clear relations
        distance_factor = np.exp(-distance / 2.0)

        # Max relation strength
        spatial_strength = max(
            relations.get('above', 0), relations.get('below', 0),
            relations.get('supports', 0), relations.get('contains', 0),
            relations.get('near', 0),
        )

        return float(distance_factor * (0.5 + 0.5 * spatial_strength))


# =============================================================================
# Temporal Tracker
# =============================================================================

class TemporalTracker:
    """
    Tracks nodes across frames for temporal consistency.

    Uses IoU matching and feature similarity.
    """

    def __init__(self, config: EnhancedSEGOConfig):
        self.config = config
        self.history: deque = deque(maxlen=config.tracking_history)
        self.node_tracks: Dict[int, List[int]] = {}  # track_id -> [node_ids]
        self._next_track_id = 1

    def update(
        self,
        detections: List[Detection],
        nodes: List[SemanticNode],
    ) -> Dict[int, int]:
        """
        Update tracks with new detections.

        Returns:
            Mapping from detection_id to stable track_id
        """
        if not self.history:
            # First frame - create new tracks
            mapping = {}
            for det, node in zip(detections, nodes):
                track_id = self._next_track_id
                self._next_track_id += 1
                mapping[det.detection_id] = track_id
                self.node_tracks[track_id] = [node.node_id]

            self.history.append((detections, nodes))
            return mapping

        # Match with previous frame
        prev_detections, prev_nodes = self.history[-1]

        # Compute cost matrix (1 - IoU)
        cost_matrix = self._compute_cost_matrix(detections, prev_detections, nodes, prev_nodes)

        # Greedy matching
        mapping = {}
        used_prev = set()

        for i, det in enumerate(detections):
            best_j = -1
            best_cost = float('inf')

            for j, prev_det in enumerate(prev_detections):
                if j in used_prev:
                    continue
                if cost_matrix[i, j] < best_cost:
                    best_cost = cost_matrix[i, j]
                    best_j = j

            if best_j >= 0 and best_cost < (1 - self.config.iou_threshold):
                # Match found - use existing track
                used_prev.add(best_j)
                prev_track = self._find_track_for_detection(prev_detections[best_j].detection_id)
                if prev_track is not None:
                    mapping[det.detection_id] = prev_track
                    self.node_tracks[prev_track].append(nodes[i].node_id)
                else:
                    # Create new track
                    track_id = self._next_track_id
                    self._next_track_id += 1
                    mapping[det.detection_id] = track_id
                    self.node_tracks[track_id] = [nodes[i].node_id]
            else:
                # No match - new track
                track_id = self._next_track_id
                self._next_track_id += 1
                mapping[det.detection_id] = track_id
                self.node_tracks[track_id] = [nodes[i].node_id]

        self.history.append((detections, nodes))
        return mapping

    def _compute_cost_matrix(
        self,
        dets_curr: List[Detection],
        dets_prev: List[Detection],
        nodes_curr: List[SemanticNode],
        nodes_prev: List[SemanticNode],
    ) -> np.ndarray:
        """Compute matching cost matrix."""
        n_curr = len(dets_curr)
        n_prev = len(dets_prev)

        cost = np.ones((n_curr, n_prev), dtype=np.float32)

        for i, (det_c, node_c) in enumerate(zip(dets_curr, nodes_curr)):
            for j, (det_p, node_p) in enumerate(zip(dets_prev, nodes_prev)):
                # IoU cost
                iou = self._compute_iou(det_c.bbox, det_p.bbox)

                # Feature similarity
                feat_sim = self._compute_feature_similarity(node_c, node_p)

                # Combined cost
                cost[i, j] = 1.0 - (0.5 * iou + 0.5 * feat_sim)

        return cost

    def _compute_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
    ) -> float:
        """Compute IoU between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        inter_area = (xi2 - xi1) * (yi2 - yi1)

        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        return inter_area / max(union_area, 1e-6)

    def _compute_feature_similarity(
        self,
        node1: SemanticNode,
        node2: SemanticNode,
    ) -> float:
        """Compute feature similarity between nodes."""
        # Cosine similarity on form tensor
        f1 = node1.form_tensor
        f2 = node2.form_tensor

        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        return float(np.dot(f1, f2) / (norm1 * norm2))

    def _find_track_for_detection(self, detection_id: int) -> Optional[int]:
        """Find track ID for a detection from previous frame."""
        for track_id, node_ids in self.node_tracks.items():
            if len(node_ids) > 0:
                # Check if last node matches
                return track_id
        return None

    def get_smoothed_node(
        self,
        track_id: int,
        current_node: SemanticNode,
        alpha: float = 0.7,
    ) -> SemanticNode:
        """Get temporally smoothed node using exponential moving average."""
        if track_id not in self.node_tracks or len(self.node_tracks[track_id]) < 2:
            return current_node

        # Get previous node from history
        prev_nodes = []
        for _, nodes in reversed(list(self.history)[:-1]):
            for node in nodes:
                if node.node_id in self.node_tracks[track_id]:
                    prev_nodes.append(node)
                    break
            if len(prev_nodes) >= 2:
                break

        if not prev_nodes:
            return current_node

        # Exponential smoothing
        smoothed = current_node.copy()

        # Smooth bound tensor (position/geometry)
        smoothed.bound_tensor = alpha * current_node.bound_tensor + (1 - alpha) * prev_nodes[0].bound_tensor

        # Keep form tensor mostly current (appearance changes)
        smoothed.form_tensor = 0.9 * current_node.form_tensor + 0.1 * prev_nodes[0].form_tensor

        return smoothed

    def reset(self) -> None:
        """Reset tracker state."""
        self.history.clear()
        self.node_tracks.clear()
        self._next_track_id = 1


# =============================================================================
# Enhanced SEGO
# =============================================================================

class EnhancedSEGO:
    """
    Enhanced SEGO with precision perception capabilities.

    Features:
    - Multi-scale feature extraction
    - 3D geometry from depth
    - Semantic spatial relations
    - Temporal tracking
    - Context-aware edges
    """

    def __init__(
        self,
        config: Optional[EnhancedSEGOConfig] = None,
    ):
        self.config = config or EnhancedSEGOConfig()

        # Components
        self.encoder = MultiScaleFeatureEncoder(self.config)
        self.relation_analyzer = SpatialRelationAnalyzer(self.config)
        self.tracker = TemporalTracker(self.config)

        # State
        self._next_node_id = 1
        self._node_registry: Dict[int, SemanticNode] = {}
        self._track_to_node: Dict[int, int] = {}

        logger.info("Enhanced SEGO initialized")

    def process(
        self,
        observation: SensorObservation,
        detections: Optional[List[Detection]] = None,
    ) -> RawSemanticGraph:
        """
        Process observation through enhanced perception pipeline.

        Args:
            observation: Sensor observation with RGB and depth
            detections: Pre-computed detections (required for full functionality)

        Returns:
            RawSemanticGraph with rich semantic information
        """
        if detections is None:
            detections = self._generate_detections(observation)

        # 1. Project detections to semantic nodes
        nodes = self._project_to_nodes(observation, detections)

        # 2. Update temporal tracking
        track_mapping = self.tracker.update(detections, nodes)

        # 3. Apply temporal smoothing
        smoothed_nodes = []
        for det, node in zip(detections, nodes):
            track_id = track_mapping.get(det.detection_id)
            if track_id is not None:
                smoothed = self.tracker.get_smoothed_node(track_id, node)
                smoothed_nodes.append(smoothed)
            else:
                smoothed_nodes.append(node)

        # 4. Generate semantic edges
        edges = self._generate_semantic_edges(smoothed_nodes)

        return RawSemanticGraph(
            timestamp_ns=observation.timestamp_ns,
            nodes=smoothed_nodes,
            edge_candidates=edges,
        )

    def _generate_detections(self, observation: SensorObservation) -> List[Detection]:
        """Generate detections when none provided."""
        detections = []

        if not observation.rgb_images:
            return detections

        img = observation.rgb_images[0]
        h, w = img.shape[:2]

        # Simple grid-based pseudo-detection
        grid_size = 2
        cell_h, cell_w = h // grid_size, w // grid_size

        det_id = 0
        for i in range(grid_size):
            for j in range(grid_size):
                x, y = j * cell_w, i * cell_h

                # Check if cell has content (not empty/uniform)
                cell = img[y:y+cell_h, x:x+cell_w]
                if np.std(cell) > 10:  # Has some variation
                    centroid_3d = None
                    if observation.depth_maps:
                        depth_cell = observation.depth_maps[0][y:y+cell_h, x:x+cell_w]
                        valid_depth = depth_cell[(depth_cell > 0.1) & (depth_cell < 10)]
                        if len(valid_depth) > 0:
                            z = np.median(valid_depth)
                            cx = (x + cell_w/2 - w/2) * z / 525.0
                            cy = (y + cell_h/2 - h/2) * z / 525.0
                            centroid_3d = np.array([cx, cy, z])

                    detections.append(Detection(
                        detection_id=det_id,
                        class_name="region",
                        confidence=0.5 + 0.5 * min(np.std(cell) / 50, 1.0),
                        bbox=(x, y, cell_w, cell_h),
                        centroid_3d=centroid_3d,
                    ))
                    det_id += 1

        return detections

    def _project_to_nodes(
        self,
        observation: SensorObservation,
        detections: List[Detection],
    ) -> List[SemanticNode]:
        """Project detections to semantic nodes with rich features."""
        nodes = []

        img_shape = (480, 640)
        if observation.rgb_images:
            img_shape = observation.rgb_images[0].shape[:2]

        for det in detections:
            if det.confidence < self.config.min_detection_confidence:
                continue

            # Get node ID
            node_id = self._next_node_id
            self._next_node_id += 1

            # Extract patches
            rgb_patch = self._extract_patch(observation.rgb_images, det.bbox)
            depth_patch = self._extract_patch(observation.depth_maps, det.bbox)

            # Encode form tensor (visual features)
            form_tensor = self.encoder.encode_visual(rgb_patch, det.mask)

            # Encode bound tensor (3D geometry)
            if depth_patch.size > 0:
                bound_tensor = self.encoder.encode_depth_to_geometry(
                    depth_patch, det.bbox, det.mask, img_shape
                )
            else:
                bound_tensor = self.encoder._encode_2d_fallback(det.bbox, img_shape)

            # Override with detection 3D if available
            if det.centroid_3d is not None:
                bound_tensor[0:3] = det.centroid_3d

            # Compute intent tensor
            intent_tensor = self._compute_enhanced_intent(det, bound_tensor, form_tensor)

            node = SemanticNode(
                node_id=node_id,
                bound_tensor=bound_tensor,
                form_tensor=form_tensor,
                intent_tensor=intent_tensor,
            )

            nodes.append(node)
            self._node_registry[node_id] = node

        return nodes

    def _extract_patch(
        self,
        images: List[np.ndarray],
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Extract image patch from bounding box."""
        if not images:
            return np.array([])

        img = images[0]
        x, y, w, h = bbox

        x = max(0, x)
        y = max(0, y)

        if img.ndim >= 2:
            h_img, w_img = img.shape[:2]
            x2 = min(x + w, w_img)
            y2 = min(y + h, h_img)

            if x2 > x and y2 > y:
                return img[y:y2, x:x2]

        return np.array([])

    def _compute_enhanced_intent(
        self,
        detection: Detection,
        bound_tensor: np.ndarray,
        form_tensor: np.ndarray,
    ) -> np.ndarray:
        """Compute enhanced intent tensor with geometric and visual features."""
        intent = np.zeros(INTENT_TENSOR_DIM, dtype=np.float32)

        # 1. Confidence (dim 0)
        intent[0] = detection.confidence

        # 2. Size-based affordance (dims 1-3)
        extents = bound_tensor[6:9]
        volume = np.prod(extents + 1e-6)
        intent[1] = min(1.0 / (1.0 + volume * 10), 1.0)  # Graspability (small = graspable)
        intent[2] = min(volume * 5, 1.0)  # Supportiveness (large = supportive)
        intent[3] = min(extents[2] / max(np.mean(extents[:2]), 1e-6), 2.0) / 2.0  # Height ratio

        # 3. Position-based affordance (dims 4-6)
        position = bound_tensor[0:3]
        intent[4] = 1.0 - min(position[2] / 3.0, 1.0)  # Reachability (closer = more reachable)
        intent[5] = 1.0 - min(abs(position[0]) / 1.0, 1.0)  # Centrality X
        intent[6] = 1.0 - min(abs(position[1]) / 1.0, 1.0)  # Centrality Y

        # 4. Shape-based affordance (dims 7-9)
        aspect_xy = bound_tensor[10] if bound_tensor[10] > 0 else 1.0
        aspect_xz = bound_tensor[11] if bound_tensor[11] > 0 else 1.0
        intent[7] = 1.0 - min(abs(aspect_xy - 1.0), 1.0)  # Squareness XY
        intent[8] = 1.0 - min(abs(aspect_xz - 1.0), 1.0)  # Squareness XZ
        intent[9] = bound_tensor[15]  # Convexity

        # 5. Surface-based affordance (dims 10-12)
        surface_normal = bound_tensor[12:15]
        intent[10] = max(surface_normal[2], 0)  # Upward-facing (horizontal surface)
        intent[11] = 1.0 - abs(surface_normal[2])  # Vertical surface
        intent[12] = detection.confidence * intent[4]  # Combined actionability

        # 6. Visual feature-based affordance (dims 13-15)
        # Use form tensor statistics
        intent[13] = min(np.mean(np.abs(form_tensor[:8])) * 2, 1.0)  # Visual salience
        intent[14] = min(np.std(form_tensor) * 2, 1.0)  # Visual complexity
        intent[15] = detection.confidence  # Reserved

        return np.clip(intent, 0.0, 1.0).astype(np.float32)

    def _generate_semantic_edges(
        self,
        nodes: List[SemanticNode],
    ) -> List[SemanticEdge]:
        """Generate edges with semantic spatial relations."""
        edges = []

        for i, node_i in enumerate(nodes):
            candidates = []

            for j, node_j in enumerate(nodes):
                if i >= j:
                    continue

                # Compute semantic relation
                relation_emb, confidence, relations = self.relation_analyzer.compute_relation(
                    node_i, node_j
                )

                if confidence < self.config.edge_confidence_threshold:
                    continue

                # Compute edge weight from relations
                weight = self._compute_edge_weight(relations, confidence)

                candidates.append((j, relation_emb, weight, confidence))

            # Keep top-k edges per node
            candidates.sort(key=lambda x: -x[3])
            for j, relation_emb, weight, confidence in candidates[:self.config.max_edges_per_node]:
                edge = SemanticEdge(
                    source_id=node_i.node_id,
                    target_id=nodes[j].node_id,
                    relation_embedding=relation_emb,
                    weight=weight,
                    probability=confidence,
                )
                edges.append(edge)

        return edges

    def _compute_edge_weight(
        self,
        relations: Dict[str, float],
        confidence: float,
    ) -> float:
        """Compute edge weight from relation scores."""
        # Strong relations get higher weight
        support_weight = relations.get('supports', 0) + relations.get('supported_by', 0)
        containment_weight = relations.get('contains', 0) + relations.get('inside', 0)
        proximity_weight = relations.get('near', 0)
        functional_weight = relations.get('can_grasp', 0) + relations.get('can_place_on', 0)

        weight = (
            0.4 * support_weight +
            0.3 * containment_weight +
            0.2 * proximity_weight +
            0.1 * functional_weight
        ) * confidence

        return float(np.clip(weight, 0.1, 1.0))

    def reset(self) -> None:
        """Reset SEGO state."""
        self._next_node_id = 1
        self._node_registry.clear()
        self._track_to_node.clear()
        self.tracker.reset()

    def get_node(self, node_id: int) -> Optional[SemanticNode]:
        """Get node by ID."""
        return self._node_registry.get(node_id)


# =============================================================================
# Factory Functions
# =============================================================================

def create_enhanced_sego(
    config: Optional[EnhancedSEGOConfig] = None,
) -> EnhancedSEGO:
    """Create enhanced SEGO with default or custom configuration."""
    return EnhancedSEGO(config)


def create_precision_sego() -> EnhancedSEGO:
    """Create SEGO optimized for precision perception."""
    config = EnhancedSEGOConfig(
        num_scales=4,
        fourier_features=12,
        tracking_history=10,
        iou_threshold=0.4,
        max_edges_per_node=8,
    )
    return EnhancedSEGO(config)
