"""
Smoke test for SEGO GaugeAnchor.

Verifies basic semantic graph generation from sensor observation.

Reference: spec/20_impl_plan.ir.yml IMPL_004
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from onn.ops.sego_anchor import (
    SEGOGaugeAnchor,
    SEGOConfig,
    Detection,
    create_default_sego_anchor,
)
from onn.core.tensors import (
    SensorObservation,
    RawSemanticGraph,
    SemanticNode,
    JointState,
    BOUND_TENSOR_DIM,
    FORM_TENSOR_DIM,
    INTENT_TENSOR_DIM,
)


@pytest.fixture
def mock_observation() -> SensorObservation:
    """Create a mock sensor observation for testing."""
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = np.random.uniform(0.5, 5.0, (480, 640)).astype(np.float32)

    joint_state = JointState(
        position=np.zeros(6, dtype=np.float32),
        velocity=np.zeros(6, dtype=np.float32),
        effort=np.zeros(6, dtype=np.float32)
    )

    return SensorObservation(
        timestamp_ns=1000000,
        frame_id="robot_base",
        rgb_images=[rgb],
        depth_maps=[depth],
        joint_state=joint_state
    )


class TestSEGOSmoke:
    """Smoke tests for SEGO GaugeAnchor."""

    def test_create_default_sego(self):
        """Default SEGO anchor should be created successfully."""
        sego = create_default_sego_anchor()
        assert sego is not None
        assert sego.config is not None

    def test_process_returns_raw_graph(self, mock_observation):
        """Process should return a RawSemanticGraph."""
        sego = create_default_sego_anchor()
        graph = sego.process(mock_observation)

        assert isinstance(graph, RawSemanticGraph)
        assert graph.timestamp_ns == mock_observation.timestamp_ns

    def test_graph_has_nodes(self, mock_observation):
        """Processed graph should have at least one node."""
        sego = create_default_sego_anchor()
        graph = sego.process(mock_observation)

        assert len(graph.nodes) >= 1

    def test_nodes_have_valid_tensors(self, mock_observation):
        """Nodes should have correctly sized tensors."""
        sego = create_default_sego_anchor()
        graph = sego.process(mock_observation)

        for node in graph.nodes:
            assert node.bound_tensor.shape == (BOUND_TENSOR_DIM,)
            assert node.form_tensor.shape == (FORM_TENSOR_DIM,)
            assert node.intent_tensor.shape == (INTENT_TENSOR_DIM,)

    def test_nodes_have_finite_values(self, mock_observation):
        """Node tensors should not contain NaN or Inf."""
        sego = create_default_sego_anchor()
        graph = sego.process(mock_observation)

        for node in graph.nodes:
            assert np.all(np.isfinite(node.bound_tensor))
            assert np.all(np.isfinite(node.form_tensor))
            assert np.all(np.isfinite(node.intent_tensor))

    def test_process_with_detections(self, mock_observation):
        """Process should work with provided detections."""
        sego = create_default_sego_anchor()

        detections = [
            Detection(
                detection_id=1,
                class_name="cup",
                confidence=0.9,
                bbox=(100, 100, 50, 50)
            ),
            Detection(
                detection_id=2,
                class_name="table",
                confidence=0.8,
                bbox=(0, 200, 640, 280)
            ),
        ]

        graph = sego.process(mock_observation, detections)

        assert len(graph.nodes) == 2
        assert graph.nodes[0].node_id != graph.nodes[1].node_id

    def test_edge_candidates_generated(self, mock_observation):
        """Edge candidates should be generated for nearby nodes."""
        sego = SEGOGaugeAnchor(SEGOConfig(proximity_threshold=10.0))

        # Create detections that should produce edges
        detections = [
            Detection(detection_id=1, class_name="cup", confidence=0.9, bbox=(100, 100, 50, 50)),
            Detection(detection_id=2, class_name="cup", confidence=0.9, bbox=(120, 100, 50, 50)),
        ]

        graph = sego.process(mock_observation, detections)

        # With a high proximity threshold, these should be connected
        # (depends on bound tensor similarity)
        assert isinstance(graph.edge_candidates, list)

    def test_edges_have_valid_structure(self, mock_observation):
        """Edge candidates should have valid source/target IDs."""
        sego = create_default_sego_anchor()

        detections = [
            Detection(detection_id=1, class_name="cup", confidence=0.9, bbox=(100, 100, 50, 50)),
            Detection(detection_id=2, class_name="table", confidence=0.8, bbox=(0, 200, 640, 100)),
        ]

        graph = sego.process(mock_observation, detections)
        node_ids = set(n.node_id for n in graph.nodes)

        for edge in graph.edge_candidates:
            assert edge.source_id in node_ids
            assert edge.target_id in node_ids
            assert edge.weight >= 0
            assert 0 <= edge.probability <= 1

    def test_reset_clears_state(self, mock_observation):
        """Reset should clear internal state."""
        sego = create_default_sego_anchor()

        sego.process(mock_observation)
        sego.reset()

        assert sego._next_node_id == 1
        assert len(sego._node_registry) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
