"""
MissionControl - Application Layer for Mission Management.

Publishes MissionGoal and provides graph monitoring capabilities.

Reference:
    - spec/20_impl_plan.ir.yml IMPL_012
    - spec/11_interfaces.ir.yml -> MissionGoal, StabilizedGraph

Framework-agnostic: No ROS2, DDS, or external framework dependencies.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any
import logging
import time
import uuid

from onn.core.tensors import (
    MissionGoal,
    StabilizedGraph,
    SemanticNode,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MissionControlConfig:
    """Configuration for MissionControl."""

    # Goal publishing
    default_goal_timeout_sec: float = 30.0  # Default goal timeout

    # Graph monitoring
    monitor_interval_sec: float = 1.0  # How often to update stats
    energy_threshold: float = 1.0  # Alert if energy > threshold
    max_history_size: int = 100  # Max graph history entries

    # Callbacks
    enable_callbacks: bool = True


# =============================================================================
# Graph Summary
# =============================================================================

@dataclass
class GraphSummary:
    """Summary statistics for a StabilizedGraph."""
    timestamp_ns: int
    num_nodes: int
    num_edges: int
    global_energy: float
    avg_node_degree: float
    max_node_degree: int
    is_connected: bool
    energy_trend: str  # "decreasing", "stable", "increasing"


# =============================================================================
# MissionControl
# =============================================================================

class MissionControl:
    """
    Application layer for mission goal management and monitoring.

    Responsibilities:
    1. Create and publish MissionGoal messages
    2. Monitor StabilizedGraph for anomalies
    3. Track mission progress
    4. Provide graph statistics

    Reference:
        - spec/20_impl_plan.ir.yml IMPL_012
    """

    def __init__(self, config: Optional[MissionControlConfig] = None):
        """
        Initialize MissionControl.

        Args:
            config: Controller configuration. Uses defaults if None.
        """
        self.config = config or MissionControlConfig()

        # State
        self._current_goal: Optional[MissionGoal] = None
        self._goal_history: List[MissionGoal] = []
        self._latest_graph: Optional[StabilizedGraph] = None
        self._graph_history: List[GraphSummary] = []

        # Callbacks
        self._goal_callbacks: List[Callable[[MissionGoal], None]] = []
        self._graph_callbacks: List[Callable[[StabilizedGraph], None]] = []
        self._alert_callbacks: List[Callable[[str, Any], None]] = []

    def publish_goal(
        self,
        verb: str,
        target_node_id: int,
        constraints: Optional[Dict[str, Any]] = None,
        goal_id: Optional[str] = None
    ) -> MissionGoal:
        """
        Create and publish a new mission goal.

        Args:
            verb: Action verb (e.g., 'GRASP', 'POUR', 'PLACE')
            target_node_id: ID of target node in semantic graph
            constraints: Optional task-specific constraints
            goal_id: Optional goal ID (generated if None)

        Returns:
            Published MissionGoal

        Example:
            >>> goal = mission_control.publish_goal("GRASP", target_node_id=42)
        """
        if goal_id is None:
            goal_id = str(uuid.uuid4())[:8]

        if constraints is None:
            constraints = {}

        goal = MissionGoal(
            goal_id=goal_id,
            verb=verb.upper(),
            target_node_id=target_node_id,
            constraints=constraints
        )

        # Store as current goal
        self._current_goal = goal
        self._goal_history.append(goal)

        # Trim history if needed
        if len(self._goal_history) > self.config.max_history_size:
            self._goal_history.pop(0)

        # Notify callbacks
        if self.config.enable_callbacks:
            for callback in self._goal_callbacks:
                try:
                    callback(goal)
                except Exception as e:
                    logger.error(f"Goal callback error: {e}")

        logger.info(f"Published goal: {verb} on node {target_node_id} (id={goal_id})")
        return goal

    def update_graph(self, graph: StabilizedGraph) -> None:
        """
        Update with latest stabilized graph.

        Computes summary statistics and checks for anomalies.

        Args:
            graph: Latest StabilizedGraph from LOGOS
        """
        self._latest_graph = graph

        # Compute summary
        summary = self._compute_summary(graph)
        self._graph_history.append(summary)

        # Trim history
        if len(self._graph_history) > self.config.max_history_size:
            self._graph_history.pop(0)

        # Check for anomalies
        self._check_anomalies(summary)

        # Notify callbacks
        if self.config.enable_callbacks:
            for callback in self._graph_callbacks:
                try:
                    callback(graph)
                except Exception as e:
                    logger.error(f"Graph callback error: {e}")

    def latest_graph_summary(self) -> Optional[GraphSummary]:
        """
        Get summary of the latest graph.

        Returns:
            GraphSummary or None if no graph received
        """
        if not self._graph_history:
            return None
        return self._graph_history[-1]

    def get_current_goal(self) -> Optional[MissionGoal]:
        """Get currently active mission goal."""
        return self._current_goal

    def clear_goal(self) -> None:
        """Clear the current mission goal."""
        if self._current_goal:
            logger.info(f"Cleared goal: {self._current_goal.goal_id}")
        self._current_goal = None

    def _compute_summary(self, graph: StabilizedGraph) -> GraphSummary:
        """Compute summary statistics for a graph."""
        num_nodes = len(graph.nodes)
        num_edges = len(graph.edges)

        # Compute degrees
        degrees = self._compute_degrees(graph)
        avg_degree = np.mean(list(degrees.values())) if degrees else 0.0
        max_degree = max(degrees.values()) if degrees else 0

        # Check connectivity (simple check)
        is_connected = self._check_connected(graph)

        # Compute energy trend
        energy_trend = self._compute_energy_trend(graph.global_energy)

        return GraphSummary(
            timestamp_ns=graph.timestamp_ns,
            num_nodes=num_nodes,
            num_edges=num_edges,
            global_energy=graph.global_energy,
            avg_node_degree=avg_degree,
            max_node_degree=max_degree,
            is_connected=is_connected,
            energy_trend=energy_trend
        )

    def _compute_degrees(self, graph: StabilizedGraph) -> Dict[int, int]:
        """Compute degree of each node."""
        degrees: Dict[int, int] = {node.node_id: 0 for node in graph.nodes}
        for edge in graph.edges:
            if edge.source_id in degrees:
                degrees[edge.source_id] += 1
            if edge.target_id in degrees:
                degrees[edge.target_id] += 1
        return degrees

    def _check_connected(self, graph: StabilizedGraph) -> bool:
        """Simple connectivity check (BFS from first node)."""
        if not graph.nodes:
            return True

        # Build adjacency
        adj: Dict[int, List[int]] = {n.node_id: [] for n in graph.nodes}
        for e in graph.edges:
            if e.source_id in adj and e.target_id in adj:
                adj[e.source_id].append(e.target_id)
                adj[e.target_id].append(e.source_id)

        # BFS
        visited = set()
        queue = [graph.nodes[0].node_id]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            queue.extend([n for n in adj.get(node, []) if n not in visited])

        return len(visited) == len(graph.nodes)

    def _compute_energy_trend(self, current_energy: float) -> str:
        """Compute energy trend from history."""
        if len(self._graph_history) < 2:
            return "stable"

        # Look at last few energies
        recent = [s.global_energy for s in self._graph_history[-5:]]
        recent.append(current_energy)

        if len(recent) < 2:
            return "stable"

        # Compute trend
        diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
        avg_diff = np.mean(diffs)

        if avg_diff < -0.01:
            return "decreasing"
        elif avg_diff > 0.01:
            return "increasing"
        else:
            return "stable"

    def _check_anomalies(self, summary: GraphSummary) -> None:
        """Check for anomalies and trigger alerts."""
        alerts = []

        # High energy
        if summary.global_energy > self.config.energy_threshold:
            alerts.append(("high_energy", summary.global_energy))

        # Disconnected graph
        if not summary.is_connected and summary.num_nodes > 1:
            alerts.append(("disconnected", summary.num_nodes))

        # Energy increasing
        if summary.energy_trend == "increasing":
            alerts.append(("energy_increasing", summary.global_energy))

        # Trigger alert callbacks
        for alert_type, value in alerts:
            logger.warning(f"Anomaly detected: {alert_type} = {value}")
            for callback in self._alert_callbacks:
                try:
                    callback(alert_type, value)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

    # =========================================================================
    # Callback Registration
    # =========================================================================

    def register_goal_callback(
        self,
        callback: Callable[[MissionGoal], None]
    ) -> None:
        """Register callback for goal publications."""
        self._goal_callbacks.append(callback)

    def register_graph_callback(
        self,
        callback: Callable[[StabilizedGraph], None]
    ) -> None:
        """Register callback for graph updates."""
        self._graph_callbacks.append(callback)

    def register_alert_callback(
        self,
        callback: Callable[[str, Any], None]
    ) -> None:
        """Register callback for anomaly alerts."""
        self._alert_callbacks.append(callback)

    # =========================================================================
    # Query Interface
    # =========================================================================

    def get_goal_history(self) -> List[MissionGoal]:
        """Get history of published goals."""
        return list(self._goal_history)

    def get_graph_history(self) -> List[GraphSummary]:
        """Get history of graph summaries."""
        return list(self._graph_history)

    def get_energy_history(self) -> List[float]:
        """Get history of global energy values."""
        return [s.global_energy for s in self._graph_history]

    def get_latest_graph(self) -> Optional[StabilizedGraph]:
        """Get the latest StabilizedGraph."""
        return self._latest_graph

    def find_node_in_graph(self, node_id: int) -> Optional[SemanticNode]:
        """Find a node in the latest graph by ID."""
        if self._latest_graph is None:
            return None
        for node in self._latest_graph.nodes:
            if node.node_id == node_id:
                return node
        return None

    def reset(self) -> None:
        """Reset mission control state."""
        self._current_goal = None
        self._goal_history.clear()
        self._latest_graph = None
        self._graph_history.clear()


# =============================================================================
# Factory Functions
# =============================================================================

def create_default_mission_control() -> MissionControl:
    """Create MissionControl with default configuration."""
    return MissionControl(MissionControlConfig())


def create_monitoring_mission_control(
    energy_threshold: float = 0.5,
    monitor_interval_sec: float = 0.5
) -> MissionControl:
    """
    Create MissionControl with sensitive monitoring.

    Args:
        energy_threshold: Threshold for energy alerts
        monitor_interval_sec: Monitoring interval

    Returns:
        MissionControl with monitoring-focused configuration
    """
    config = MissionControlConfig(
        energy_threshold=energy_threshold,
        monitor_interval_sec=monitor_interval_sec,
        enable_callbacks=True
    )
    return MissionControl(config)
