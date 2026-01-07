"""Integration tests for Enhanced SEGO (Precision Perception).

Tests the full enhanced perception pipeline:
- Multi-scale feature extraction
- 3D geometry estimation from depth
- Semantic spatial relations
- Temporal tracking
- Context-aware edge generation

Reference:
    - src/onn/ops/sego_enhanced.py
"""

import pytest
import numpy as np

from onn.core.tensors import (
    SensorObservation,
    SemanticNode,
    BOUND_TENSOR_DIM,
    FORM_TENSOR_DIM,
    INTENT_TENSOR_DIM,
)
from onn.ops import (
    EnhancedSEGO,
    EnhancedSEGOConfig,
    MultiScaleFeatureEncoder,
    SpatialRelationAnalyzer,
    TemporalTracker,
    create_enhanced_sego,
    create_precision_sego,
    Detection,
)


class TestMultiScaleFeatureEncoder:
    """Tests for MultiScaleFeatureEncoder."""

    def test_encoder_initialization(self):
        """Encoder should initialize with correct dimensions."""
        config = EnhancedSEGOConfig()
        encoder = MultiScaleFeatureEncoder(config)

        assert encoder.form_dim == FORM_TENSOR_DIM
        assert encoder.bound_dim == BOUND_TENSOR_DIM
        assert len(encoder.freq_bands) == config.fourier_features

    def test_visual_encoding(self):
        """Visual encoder should produce normalized form tensor."""
        config = EnhancedSEGOConfig()
        encoder = MultiScaleFeatureEncoder(config)

        # Create test image
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        form_tensor = encoder.encode_visual(image)

        assert form_tensor.shape == (FORM_TENSOR_DIM,)
        assert form_tensor.dtype == np.float32
        # Should be normalized
        norm = np.linalg.norm(form_tensor)
        assert 0.9 < norm < 1.1 or norm < 0.01  # Normalized or near-zero

    def test_visual_encoding_with_mask(self):
        """Visual encoder should respect binary mask."""
        config = EnhancedSEGOConfig()
        encoder = MultiScaleFeatureEncoder(config)

        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 1  # Center region

        form_with_mask = encoder.encode_visual(image, mask)
        form_without_mask = encoder.encode_visual(image)

        # Should be different with mask
        assert not np.allclose(form_with_mask, form_without_mask)

    def test_depth_to_geometry(self):
        """Depth encoder should produce valid bound tensor with 3D geometry."""
        config = EnhancedSEGOConfig()
        encoder = MultiScaleFeatureEncoder(config)

        # Create depth patch (1.5m average depth)
        depth = np.ones((100, 100), dtype=np.float32) * 1.5
        # Add some variation
        depth += np.random.randn(100, 100).astype(np.float32) * 0.1

        bbox = (100, 100, 100, 100)  # x, y, w, h
        image_shape = (480, 640)

        bound_tensor = encoder.encode_depth_to_geometry(depth, bbox, None, image_shape)

        assert bound_tensor.shape == (BOUND_TENSOR_DIM,)
        assert bound_tensor.dtype == np.float32

        # Position (centroid) should be around 1.5m depth
        assert 1.0 < bound_tensor[2] < 2.0  # Z coordinate

        # Extents should be non-zero
        assert bound_tensor[6] > 0  # L
        assert bound_tensor[7] > 0  # W
        assert bound_tensor[8] > 0  # H

    def test_2d_fallback(self):
        """Should fallback to 2D encoding when depth is invalid."""
        config = EnhancedSEGOConfig()
        encoder = MultiScaleFeatureEncoder(config)

        # Invalid depth (all zeros)
        depth = np.zeros((100, 100), dtype=np.float32)
        bbox = (200, 150, 80, 60)
        image_shape = (480, 640)

        bound_tensor = encoder.encode_depth_to_geometry(depth, bbox, None, image_shape)

        # Should use 2D fallback
        assert bound_tensor[0] > 0  # Normalized X
        assert bound_tensor[1] > 0  # Normalized Y
        assert bound_tensor[2] == 0.5  # Default depth

    def test_fourier_position_encoding(self):
        """Fourier encoding should produce frequency-based features."""
        config = EnhancedSEGOConfig(fourier_features=4)
        encoder = MultiScaleFeatureEncoder(config)

        position = np.array([0.5, 0.3, 1.0])

        fourier_features = encoder.encode_fourier_position(position)

        # 4 freq bands * 3 coords * 2 (sin+cos) = 24 features
        assert len(fourier_features) == 24
        # All should be in [-1, 1]
        assert np.all(np.abs(fourier_features) <= 1.0)


class TestSpatialRelationAnalyzer:
    """Tests for SpatialRelationAnalyzer."""

    def test_analyzer_initialization(self):
        """Analyzer should initialize correctly."""
        config = EnhancedSEGOConfig()
        analyzer = SpatialRelationAnalyzer(config)

        assert analyzer.config is not None

    def test_vertical_relations(self):
        """Should correctly identify above/below relations."""
        config = EnhancedSEGOConfig()
        analyzer = SpatialRelationAnalyzer(config)

        # Node A is above Node B
        node_above = create_test_node(position=[0, 0, 1.0])
        node_below = create_test_node(position=[0, 0, 0.2])

        _, _, relations = analyzer.compute_relation(node_above, node_below)

        assert relations['above'] > 0.5
        assert relations['below'] < 0.5

        # Reverse order
        _, _, relations_rev = analyzer.compute_relation(node_below, node_above)

        assert relations_rev['below'] > 0.5
        assert relations_rev['above'] < 0.5

    def test_support_relation(self):
        """Should correctly identify support relations."""
        config = EnhancedSEGOConfig()
        analyzer = SpatialRelationAnalyzer(config)

        # Table supports cup (table below, cup on top)
        table = create_test_node(position=[0, 0, 0.4], extents=[1.0, 0.6, 0.05])
        cup = create_test_node(position=[0, 0, 0.5], extents=[0.08, 0.08, 0.1])

        _, _, relations = analyzer.compute_relation(table, cup)

        # Table should support cup
        assert relations['supports'] > 0.3

    def test_containment_relation(self):
        """Should correctly identify containment relations."""
        config = EnhancedSEGOConfig()
        analyzer = SpatialRelationAnalyzer(config)

        # Box contains small object
        box = create_test_node(position=[0, 0, 0.3], extents=[0.3, 0.3, 0.2])
        small_obj = create_test_node(position=[0, 0, 0.3], extents=[0.05, 0.05, 0.05])

        _, _, relations = analyzer.compute_relation(box, small_obj)

        assert relations['contains'] > 0.0

    def test_proximity_relation(self):
        """Should correctly identify near/far relations."""
        config = EnhancedSEGOConfig()
        analyzer = SpatialRelationAnalyzer(config)

        # Close objects
        obj1 = create_test_node(position=[0, 0, 1.0])
        obj2 = create_test_node(position=[0.2, 0.1, 1.0])

        _, _, relations_close = analyzer.compute_relation(obj1, obj2)

        # Far objects
        obj3 = create_test_node(position=[3.0, 2.0, 1.0])
        _, _, relations_far = analyzer.compute_relation(obj1, obj3)

        assert relations_close['near'] > relations_far['near']
        assert relations_close['far'] < relations_far['far']

    def test_relation_embedding_shape(self):
        """Relation embedding should have correct shape."""
        config = EnhancedSEGOConfig()
        analyzer = SpatialRelationAnalyzer(config)

        node1 = create_test_node(position=[0, 0, 1.0])
        node2 = create_test_node(position=[0.5, 0.3, 1.2])

        relation_emb, confidence, _ = analyzer.compute_relation(node1, node2)

        assert relation_emb.shape == (16,)
        assert 0 <= confidence <= 1


class TestTemporalTracker:
    """Tests for TemporalTracker."""

    def test_tracker_initialization(self):
        """Tracker should initialize with empty state."""
        config = EnhancedSEGOConfig(tracking_history=5)
        tracker = TemporalTracker(config)

        assert len(tracker.history) == 0
        assert len(tracker.node_tracks) == 0

    def test_first_frame_tracking(self):
        """First frame should create new tracks."""
        config = EnhancedSEGOConfig()
        tracker = TemporalTracker(config)

        detections = [
            Detection(detection_id=0, class_name="obj_0", confidence=0.9, bbox=(100, 100, 50, 50)),
            Detection(detection_id=1, class_name="obj_1", confidence=0.8, bbox=(300, 200, 60, 60)),
        ]
        nodes = [create_test_node(node_id=i) for i in range(2)]

        mapping = tracker.update(detections, nodes)

        assert len(mapping) == 2
        assert 0 in mapping
        assert 1 in mapping
        # Track IDs should be unique
        assert mapping[0] != mapping[1]

    def test_consistent_tracking(self):
        """Same object in consecutive frames should get same track ID."""
        config = EnhancedSEGOConfig(iou_threshold=0.3)
        tracker = TemporalTracker(config)

        # Frame 1
        det1 = [Detection(detection_id=0, class_name="obj", confidence=0.9, bbox=(100, 100, 50, 50))]
        node1 = [create_test_node(node_id=0)]
        mapping1 = tracker.update(det1, node1)
        track_id_frame1 = mapping1[0]

        # Frame 2 - same object, slightly moved
        det2 = [Detection(detection_id=0, class_name="obj", confidence=0.9, bbox=(105, 102, 50, 50))]
        node2 = [create_test_node(node_id=1, form=node1[0].form_tensor)]  # Same appearance
        mapping2 = tracker.update(det2, node2)

        # Should maintain same track ID
        assert mapping2[0] == track_id_frame1

    def test_new_track_for_new_object(self):
        """New object should get new track ID."""
        config = EnhancedSEGOConfig(iou_threshold=0.3)
        tracker = TemporalTracker(config)

        # Frame 1 - one object
        det1 = [Detection(detection_id=0, class_name="obj", confidence=0.9, bbox=(100, 100, 50, 50))]
        node1 = [create_test_node(node_id=0)]
        mapping1 = tracker.update(det1, node1)

        # Frame 2 - completely new object (no overlap)
        det2 = [Detection(detection_id=0, class_name="obj", confidence=0.9, bbox=(500, 400, 50, 50))]
        node2 = [create_test_node(node_id=1)]
        mapping2 = tracker.update(det2, node2)

        # Should be a new track
        assert mapping2[0] != mapping1[0]

    def test_smoothed_node(self):
        """Temporal smoothing should blend consecutive frames when history is available."""
        config = EnhancedSEGOConfig()
        tracker = TemporalTracker(config)

        # Create consistent form tensor for "same object" across frames
        shared_form = np.random.randn(FORM_TENSOR_DIM).astype(np.float32)
        shared_form = shared_form / (np.linalg.norm(shared_form) + 1e-8)

        # Frame 1
        det1 = [Detection(detection_id=0, class_name="obj", confidence=0.9, bbox=(100, 100, 50, 50))]
        node1 = [create_test_node(node_id=0, position=[0, 0, 1.0], form=shared_form)]
        mapping1 = tracker.update(det1, node1)
        track_id = mapping1[0]

        # Frame 2 - moved but same appearance
        det2 = [Detection(detection_id=0, class_name="obj", confidence=0.9, bbox=(110, 105, 50, 50))]
        node2 = [create_test_node(node_id=1, position=[0.1, 0.05, 1.0], form=shared_form)]
        mapping2 = tracker.update(det2, node2)

        # Same object should get same track ID
        assert mapping2[0] == track_id

        # Get smoothed node - note: smoothing requires track history to be populated
        smoothed = tracker.get_smoothed_node(track_id, node2[0], alpha=0.5)

        # Smoothed node should exist and have valid tensors
        assert smoothed is not None
        assert smoothed.bound_tensor.shape == (BOUND_TENSOR_DIM,)

        # Track should have multiple node IDs recorded
        assert track_id in tracker.node_tracks
        assert len(tracker.node_tracks[track_id]) >= 2

    def test_tracker_reset(self):
        """Reset should clear all state."""
        config = EnhancedSEGOConfig()
        tracker = TemporalTracker(config)

        det = [Detection(detection_id=0, class_name="obj", confidence=0.9, bbox=(100, 100, 50, 50))]
        node = [create_test_node(node_id=0)]
        tracker.update(det, node)

        assert len(tracker.history) > 0

        tracker.reset()

        assert len(tracker.history) == 0
        assert len(tracker.node_tracks) == 0


class TestEnhancedSEGO:
    """Tests for full Enhanced SEGO pipeline."""

    def test_enhanced_sego_initialization(self):
        """EnhancedSEGO should initialize all components."""
        sego = create_enhanced_sego()

        assert sego.encoder is not None
        assert sego.relation_analyzer is not None
        assert sego.tracker is not None

    def test_process_with_detections(self):
        """Should process observation with detections."""
        sego = create_enhanced_sego()

        # Create observation
        rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        depth = np.ones((480, 640), dtype=np.float32) * 1.5
        obs = SensorObservation(
            timestamp_ns=0,
            frame_id="robot_base",
            rgb_images=[rgb],
            depth_maps=[depth],
        )

        detections = [
            Detection(detection_id=0, class_name="cup", confidence=0.9,
                      bbox=(100, 100, 50, 50), centroid_3d=np.array([0.1, 0.05, 1.5])),
            Detection(detection_id=1, class_name="table", confidence=0.95,
                      bbox=(50, 200, 300, 100), centroid_3d=np.array([0, 0.2, 1.2])),
        ]

        raw_graph = sego.process(obs, detections)

        # Should have nodes for each detection
        assert len(raw_graph.nodes) == 2

        # Each node should have proper tensors
        for node in raw_graph.nodes:
            assert node.bound_tensor.shape == (BOUND_TENSOR_DIM,)
            assert node.form_tensor.shape == (FORM_TENSOR_DIM,)
            assert node.intent_tensor.shape == (INTENT_TENSOR_DIM,)

        # Should have edges
        assert len(raw_graph.edge_candidates) > 0

    def test_process_without_detections(self):
        """Should generate pseudo-detections when none provided."""
        sego = create_enhanced_sego()

        # Create observation with some content variation
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[100:200, 100:200] = 255  # White square
        rgb[300:350, 400:450] = 128  # Gray square

        obs = SensorObservation(
            timestamp_ns=0,
            frame_id="robot_base",
            rgb_images=[rgb],
            depth_maps=[],
        )

        raw_graph = sego.process(obs, detections=None)

        # Should generate some nodes
        assert len(raw_graph.nodes) >= 1

    def test_temporal_consistency(self):
        """Should maintain temporal consistency across frames."""
        sego = create_enhanced_sego()

        # Frame 1
        rgb1 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        obs1 = SensorObservation(
            timestamp_ns=0,
            frame_id="robot_base",
            rgb_images=[rgb1],
            depth_maps=[],
        )
        det1 = [Detection(detection_id=0, class_name="obj", confidence=0.9, bbox=(100, 100, 50, 50))]
        graph1 = sego.process(obs1, det1)
        pos1 = graph1.nodes[0].bound_tensor[0:3].copy()

        # Frame 2 - slightly moved
        rgb2 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        obs2 = SensorObservation(
            timestamp_ns=100_000_000,
            frame_id="robot_base",
            rgb_images=[rgb2],
            depth_maps=[],
        )
        det2 = [Detection(detection_id=0, class_name="obj", confidence=0.9, bbox=(105, 103, 50, 50))]
        graph2 = sego.process(obs2, det2)

        # Position should be smoothed (not jumping)
        pos2_raw = np.array([(105 + 25) / 640, (103 + 25) / 480, 0.5])
        pos2_actual = graph2.nodes[0].bound_tensor[0:3]

        # With smoothing, actual position should be closer to frame 1
        # than raw frame 2 position
        dist_raw = np.linalg.norm(pos2_raw - pos1)
        dist_smoothed = np.linalg.norm(pos2_actual - pos1)

        # Smoothed should be closer to previous (within some tolerance)
        assert dist_smoothed <= dist_raw + 0.1

    def test_edge_generation_with_relations(self):
        """Edges should encode semantic relations."""
        config = EnhancedSEGOConfig(edge_confidence_threshold=0.1)
        sego = EnhancedSEGO(config)

        rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        depth = np.ones((480, 640), dtype=np.float32) * 1.5
        obs = SensorObservation(
            timestamp_ns=0,
            frame_id="robot_base",
            rgb_images=[rgb],
            depth_maps=[depth],
        )

        # Cup on table scenario
        detections = [
            Detection(detection_id=0, class_name="table", confidence=0.95,
                      bbox=(100, 250, 400, 100), centroid_3d=np.array([0, 0, 0.75])),
            Detection(detection_id=1, class_name="cup", confidence=0.9,
                      bbox=(250, 200, 50, 50), centroid_3d=np.array([0, -0.05, 0.85])),
        ]

        raw_graph = sego.process(obs, detections)

        # Should have edge between table and cup
        assert len(raw_graph.edge_candidates) >= 1

        edge = raw_graph.edge_candidates[0]

        # Edge should have relation embedding
        assert edge.relation_embedding is not None
        assert len(edge.relation_embedding) == 16

        # Edge should have weight and probability
        assert 0 < edge.weight <= 1
        assert 0 < edge.probability <= 1

    def test_precision_sego_factory(self):
        """Precision SEGO should have enhanced settings."""
        sego = create_precision_sego()

        assert sego.config.num_scales == 4
        assert sego.config.fourier_features == 12
        assert sego.config.tracking_history == 10
        assert sego.config.max_edges_per_node == 8

    def test_reset_clears_state(self):
        """Reset should clear all internal state."""
        sego = create_enhanced_sego()

        # Process some frames
        rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        obs = SensorObservation(
            timestamp_ns=0,
            frame_id="robot_base",
            rgb_images=[rgb],
            depth_maps=[],
        )
        det = [Detection(detection_id=0, class_name="obj", confidence=0.9, bbox=(100, 100, 50, 50))]
        sego.process(obs, det)

        # Should have state
        assert sego._next_node_id > 1

        # Reset
        sego.reset()

        # State should be cleared
        assert sego._next_node_id == 1
        assert len(sego._node_registry) == 0
        assert len(sego.tracker.history) == 0

    def test_intent_tensor_computation(self):
        """Intent tensor should encode affordances correctly."""
        sego = create_enhanced_sego()

        rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        depth = np.ones((480, 640), dtype=np.float32) * 1.0  # 1m depth
        obs = SensorObservation(
            timestamp_ns=0,
            frame_id="robot_base",
            rgb_images=[rgb],
            depth_maps=[depth],
        )

        # Small, close object (should be graspable)
        det_small = [Detection(
            detection_id=0, class_name="cup", confidence=0.95,
            bbox=(300, 200, 30, 40), centroid_3d=np.array([0, 0, 0.8])
        )]

        graph = sego.process(obs, det_small)
        node = graph.nodes[0]

        # Check intent tensor
        intent = node.intent_tensor

        # Confidence (dim 0) should match detection
        assert intent[0] == pytest.approx(0.95, abs=0.01)

        # All values should be in [0, 1]
        assert np.all(intent >= 0)
        assert np.all(intent <= 1)


class TestIntegrationWithCSA:
    """Tests for integration with CSA pipeline."""

    def test_enhanced_sego_output_compatible_with_csa(self):
        """Enhanced SEGO output should work with CSA pipeline."""
        from onn.ops import CSAPipeline, CSAConfig

        # This test verifies the output format is compatible
        sego = create_enhanced_sego()

        rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        depth = np.ones((480, 640), dtype=np.float32) * 1.5
        obs = SensorObservation(
            timestamp_ns=0,
            frame_id="robot_base",
            rgb_images=[rgb],
            depth_maps=[depth],
        )

        detections = [
            Detection(detection_id=i, class_name=f"obj_{i}", confidence=0.9,
                      bbox=(100+i*100, 100, 50, 50))
            for i in range(3)
        ]

        raw_graph = sego.process(obs, detections)

        # Output should have correct structure for CSA
        assert hasattr(raw_graph, 'nodes')
        assert hasattr(raw_graph, 'edge_candidates')
        assert hasattr(raw_graph, 'timestamp_ns')

        # Nodes should have all required tensors
        for node in raw_graph.nodes:
            assert hasattr(node, 'node_id')
            assert hasattr(node, 'bound_tensor')
            assert hasattr(node, 'form_tensor')
            assert hasattr(node, 'intent_tensor')

        # Edges should have required fields
        for edge in raw_graph.edge_candidates:
            assert hasattr(edge, 'source_id')
            assert hasattr(edge, 'target_id')
            assert hasattr(edge, 'relation_embedding')


# =============================================================================
# Helper Functions
# =============================================================================

def create_test_node(
    node_id: int = 0,
    position: list = None,
    extents: list = None,
    form: np.ndarray = None,
) -> SemanticNode:
    """Create a test SemanticNode with specified properties."""
    bound = np.zeros(BOUND_TENSOR_DIM, dtype=np.float32)

    if position:
        bound[0:3] = position
    else:
        bound[0:3] = [0, 0, 1.0]

    if extents:
        bound[6:9] = extents
    else:
        bound[6:9] = [0.1, 0.1, 0.1]

    if form is None:
        form = np.random.randn(FORM_TENSOR_DIM).astype(np.float32)
        form = form / (np.linalg.norm(form) + 1e-8)

    intent = np.random.rand(INTENT_TENSOR_DIM).astype(np.float32)

    return SemanticNode(
        node_id=node_id,
        bound_tensor=bound,
        form_tensor=form,
        intent_tensor=intent,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
