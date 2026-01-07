"""
Smoke test for MissionControl.

Verifies basic mission goal publishing and graph monitoring.

Reference: spec/20_impl_plan.ir.yml IMPL_012
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from app.mission_control.mission_control import (
    MissionControl,
    MissionControlConfig,
    GraphSummary,
    create_default_mission_control,
)
from onn.core.tensors import (
    MissionGoal,
    StabilizedGraph,
    SemanticNode,
    SemanticEdge,
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
    ]

    edges = [
        SemanticEdge(
            source_id=1, target_id=2,
            relation_embedding=np.random.randn(16).astype(np.float32),
            weight=1.0, probability=0.9
        ),
    ]

    return StabilizedGraph(
        timestamp_ns=1000,
        nodes=nodes,
        edges=edges,
        global_energy=0.5
    )


class TestMissionControlSmoke:
    """Smoke tests for MissionControl."""

    def test_create_default_control(self):
        """Default mission control should be created successfully."""
        control = create_default_mission_control()
        assert control is not None
        assert control.config is not None

    def test_publish_goal(self):
        """Should successfully publish a mission goal."""
        control = create_default_mission_control()
        goal = control.publish_goal("GRASP", target_node_id=1)

        assert isinstance(goal, MissionGoal)
        assert goal.verb == "GRASP"
        assert goal.target_node_id == 1
        assert goal.goal_id is not None

    def test_current_goal(self):
        """Should track current goal."""
        control = create_default_mission_control()

        assert control.get_current_goal() is None

        control.publish_goal("POUR", target_node_id=2)
        goal = control.get_current_goal()

        assert goal is not None
        assert goal.verb == "POUR"

    def test_clear_goal(self):
        """Should clear current goal."""
        control = create_default_mission_control()
        control.publish_goal("GRASP", target_node_id=1)
        control.clear_goal()

        assert control.get_current_goal() is None

    def test_update_graph(self, stabilized_graph):
        """Should update with new graph."""
        control = create_default_mission_control()
        control.update_graph(stabilized_graph)

        latest = control.get_latest_graph()
        assert latest is not None
        assert len(latest.nodes) == 2

    def test_graph_summary(self, stabilized_graph):
        """Should compute graph summary."""
        control = create_default_mission_control()
        control.update_graph(stabilized_graph)

        summary = control.latest_graph_summary()
        assert isinstance(summary, GraphSummary)
        assert summary.num_nodes == 2
        assert summary.num_edges == 1
        assert np.isfinite(summary.global_energy)

    def test_goal_with_constraints(self):
        """Should publish goal with constraints."""
        control = create_default_mission_control()
        goal = control.publish_goal(
            "POUR",
            target_node_id=1,
            constraints={"keep_upright": True, "speed": "slow"}
        )

        assert goal.constraints["keep_upright"] is True
        assert goal.constraints["speed"] == "slow"

    def test_goal_history(self):
        """Should track goal history."""
        control = create_default_mission_control()

        control.publish_goal("GRASP", target_node_id=1)
        control.publish_goal("POUR", target_node_id=2)
        control.publish_goal("PLACE", target_node_id=3)

        history = control.get_goal_history()
        assert len(history) == 3
        assert history[0].verb == "GRASP"
        assert history[2].verb == "PLACE"

    def test_graph_history(self, stabilized_graph):
        """Should track graph history."""
        control = create_default_mission_control()

        for i in range(3):
            graph = StabilizedGraph(
                timestamp_ns=1000 + i * 100,
                nodes=stabilized_graph.nodes,
                edges=stabilized_graph.edges,
                global_energy=1.0 - i * 0.2
            )
            control.update_graph(graph)

        history = control.get_graph_history()
        assert len(history) == 3

    def test_find_node(self, stabilized_graph):
        """Should find node by ID."""
        control = create_default_mission_control()
        control.update_graph(stabilized_graph)

        node = control.find_node_in_graph(1)
        assert node is not None
        assert node.node_id == 1

        missing = control.find_node_in_graph(999)
        assert missing is None

    def test_callback_registration(self):
        """Should support callback registration."""
        control = create_default_mission_control()

        received_goals = []
        control.register_goal_callback(lambda g: received_goals.append(g))

        control.publish_goal("GRASP", target_node_id=1)

        assert len(received_goals) == 1
        assert received_goals[0].verb == "GRASP"

    def test_reset(self, stabilized_graph):
        """Should reset state."""
        control = create_default_mission_control()
        control.publish_goal("GRASP", target_node_id=1)
        control.update_graph(stabilized_graph)

        control.reset()

        assert control.get_current_goal() is None
        assert control.get_latest_graph() is None
        assert len(control.get_goal_history()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
