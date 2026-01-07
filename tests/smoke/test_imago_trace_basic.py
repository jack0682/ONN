"""
Smoke test for IMAGO Planner.

Verifies basic reasoning trace generation functionality.

Reference: spec/20_impl_plan.ir.yml IMPL_008
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from onn.ops.imago_planner import (
    IMAGOPlanner,
    IMAGOConfig,
    create_default_imago_planner,
    interpolate_trace,
    validate_trace,
)
from onn.core.tensors import (
    SemanticNode,
    SemanticEdge,
    StabilizedGraph,
    ReasoningTrace,
    MissionGoal,
    BOUND_TENSOR_DIM,
    FORM_TENSOR_DIM,
    INTENT_TENSOR_DIM,
)


@pytest.fixture
def stabilized_graph() -> StabilizedGraph:
    """Create a stabilized graph for testing."""
    nodes = [
        SemanticNode(
            node_id=1,
            bound_tensor=np.random.randn(BOUND_TENSOR_DIM).astype(np.float32),
            form_tensor=np.random.randn(FORM_TENSOR_DIM).astype(np.float32),
            intent_tensor=np.random.rand(INTENT_TENSOR_DIM).astype(np.float32),
        ),
        SemanticNode(
            node_id=2,
            bound_tensor=np.random.randn(BOUND_TENSOR_DIM).astype(np.float32),
            form_tensor=np.random.randn(FORM_TENSOR_DIM).astype(np.float32),
            intent_tensor=np.random.rand(INTENT_TENSOR_DIM).astype(np.float32),
        ),
        SemanticNode(
            node_id=3,
            bound_tensor=np.random.randn(BOUND_TENSOR_DIM).astype(np.float32),
            form_tensor=np.random.randn(FORM_TENSOR_DIM).astype(np.float32),
            intent_tensor=np.random.rand(INTENT_TENSOR_DIM).astype(np.float32),
        ),
    ]

    edges = [
        SemanticEdge(
            source_id=1, target_id=2,
            relation_embedding=np.random.randn(16).astype(np.float32),
            weight=1.0, probability=0.9
        ),
        SemanticEdge(
            source_id=2, target_id=3,
            relation_embedding=np.random.randn(16).astype(np.float32),
            weight=0.8, probability=0.8
        ),
    ]

    return StabilizedGraph(
        timestamp_ns=1000,
        nodes=nodes,
        edges=edges,
        global_energy=0.5
    )


@pytest.fixture
def mission_goal() -> MissionGoal:
    """Create a mission goal for testing."""
    return MissionGoal(
        goal_id="test-goal-1",
        verb="GRASP",
        target_node_id=1,
        constraints={}
    )


class TestIMAGOSmoke:
    """Smoke tests for IMAGO Planner."""

    def test_create_default_planner(self):
        """Default planner should be created successfully."""
        planner = create_default_imago_planner()
        assert planner is not None
        assert planner.config is not None

    def test_plan_returns_reasoning_trace(self, stabilized_graph, mission_goal):
        """Plan should return a ReasoningTrace."""
        planner = create_default_imago_planner()
        trace = planner.plan(stabilized_graph, mission_goal)

        assert isinstance(trace, ReasoningTrace)
        assert trace.timestamp_ns > 0

    def test_trace_has_target_state(self, stabilized_graph, mission_goal):
        """Trace should have target state nodes."""
        planner = create_default_imago_planner()
        trace = planner.plan(stabilized_graph, mission_goal)

        assert len(trace.target_state) > 0
        for node in trace.target_state:
            assert isinstance(node, SemanticNode)

    def test_trace_has_valid_until(self, stabilized_graph, mission_goal):
        """Trace should have a valid_until timestamp."""
        planner = create_default_imago_planner()
        trace = planner.plan(stabilized_graph, mission_goal)

        assert trace.valid_until_ns > trace.timestamp_ns
        assert validate_trace(trace, trace.timestamp_ns)

    def test_trace_has_curvature(self, stabilized_graph, mission_goal):
        """Trace should have curvature value."""
        planner = create_default_imago_planner()
        trace = planner.plan(stabilized_graph, mission_goal)

        assert np.isfinite(trace.curvature)

    def test_trace_has_trajectory_coeffs(self, stabilized_graph, mission_goal):
        """Trace should have trajectory coefficients."""
        planner = create_default_imago_planner()
        trace = planner.plan(stabilized_graph, mission_goal)

        assert trace.trajectory_coeffs is not None
        # May be empty if no trajectory needed
        if trace.trajectory_coeffs.size > 0:
            assert np.all(np.isfinite(trace.trajectory_coeffs))

    def test_interpolate_trace(self, stabilized_graph, mission_goal):
        """Trace should be interpolatable."""
        planner = create_default_imago_planner()
        trace = planner.plan(stabilized_graph, mission_goal)

        if trace.trajectory_coeffs.size > 0:
            # Interpolate at various times
            for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
                result = interpolate_trace(trace, t)
                assert result is not None
                assert np.all(np.isfinite(result))

    def test_different_verbs(self, stabilized_graph):
        """Planner should handle different goal verbs."""
        planner = create_default_imago_planner()

        verbs = ["GRASP", "POUR", "PLACE", "MONITOR", "MOVE"]
        for verb in verbs:
            goal = MissionGoal(
                goal_id=f"test-{verb}",
                verb=verb,
                target_node_id=1,
                constraints={}
            )
            trace = planner.plan(stabilized_graph, goal)
            assert trace is not None

    def test_nonexistent_target_node(self, stabilized_graph):
        """Planner should handle nonexistent target nodes gracefully."""
        planner = create_default_imago_planner()

        goal = MissionGoal(
            goal_id="test-missing",
            verb="GRASP",
            target_node_id=999,  # Doesn't exist
            constraints={}
        )

        # Should not crash, return fallback trace
        trace = planner.plan(stabilized_graph, goal)
        assert trace is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
