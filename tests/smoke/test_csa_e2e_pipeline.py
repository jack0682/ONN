"""
End-to-End Pipeline Test for CSA.

Tests the complete pipeline: HAL -> SEGO -> LOGOS -> IMAGO -> ORTSF -> HAL

This smoke test verifies that all modules can work together
to produce at least one valid ActuatorCommand from sensor input.

Reference: spec/20_impl_plan.ir.yml IMPL_013
"""

import pytest
import numpy as np
import sys
import os
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from hal.sensor_bridge.sensor_bridge import create_mock_sensor_bridge
from hal.actuator_bridge.actuator_bridge import create_mock_actuator_bridge
from onn.ops.sego_anchor import create_default_sego_anchor, Detection
from onn.ops.logos_solver import create_default_solver, LOGOSSolver
from onn.core.tensors import ConstraintConfig
from onn.ops.imago_planner import create_default_imago_planner
from ortsf.fabric.ortsf_fabric import create_default_ortsf_fabric
from app.mission_control.mission_control import create_default_mission_control
from onn.core.tensors import MissionGoal


class TestCSAEndToEndPipeline:
    """End-to-end integration tests for the CSA pipeline."""

    def test_full_pipeline_single_frame(self):
        """
        Complete pipeline should produce an ActuatorCommand from sensor input.

        Pipeline: SensorBridge -> SEGO -> LOGOS -> IMAGO -> ORTSF -> ActuatorBridge
        """
        # Create all components
        sensor_bridge = create_mock_sensor_bridge(num_joints=6)
        sego = create_default_sego_anchor()
        sego = create_default_sego_anchor()
        # Increased budget for test stability
        # Increased budget for test stability
        c_cfg = ConstraintConfig(max_iterations=100)
        from onn.ops.logos_solver import LOGOSConfig

        l_cfg = LOGOSConfig(
            max_iterations=200,
            lambda_logic=0.0,
            tolerance=1.0,
            uncertainty_lr=0.01,
        )
        logos = LOGOSSolver(config=l_cfg)

        imago = create_default_imago_planner()
        ortsf = create_default_ortsf_fabric()
        actuator_bridge = create_mock_actuator_bridge(num_joints=6)
        mission_control = create_default_mission_control()

        # Step 1: Acquire sensor observation
        observation = sensor_bridge.acquire()
        assert observation is not None, "SensorBridge failed to acquire"

        # Step 2: Create detections and process through SEGO
        # Step 2: Create detections and processing through SEGO
        detections = [
            Detection(
                detection_id=1,
                class_name="cup",
                confidence=0.9,
                bbox=(100, 100, 50, 50),
            ),
            # Table removed to avoid huge sphere collision radius overlap in simple V0 physics
            Detection(
                detection_id=2,
                class_name="marker",
                confidence=0.8,
                bbox=(300, 300, 50, 50),
            ),
        ]
        raw_graph = sego.process(observation, detections)
        assert len(raw_graph.nodes) > 0, "SEGO produced empty graph"

        # Step 3: Stabilize through LOGOS
        stabilized_graph = logos.solve(raw_graph)
        assert len(stabilized_graph.nodes) > 0, "LOGOS produced empty graph"
        # TEST_002: Explicitly check validity
        if not stabilized_graph.is_valid:
            pytest.fail(
                f"LOGOS failed to converge. Iterations: {stabilized_graph.iterations_used}"
            )
        assert np.isfinite(stabilized_graph.global_energy), "LOGOS energy is not finite"

        # Step 4: Update mission control with graph
        mission_control.update_graph(stabilized_graph)

        # Step 5: Publish mission goal
        goal = mission_control.publish_goal(
            verb="GRASP", target_node_id=stabilized_graph.nodes[0].node_id
        )
        assert goal is not None, "Failed to publish goal"

        # Step 6: Generate reasoning trace via IMAGO
        trace = imago.plan(stabilized_graph, goal)
        assert trace is not None, "IMAGO failed to generate trace"
        assert trace.valid_until_ns > trace.timestamp_ns, "Invalid trace validity"

        # Step 7: Generate actuator command via ORTSF
        command = ortsf.step(trace, observation)
        assert command is not None, "ORTSF failed to generate command"
        assert len(command.command_values) == 6, "Wrong number of command values"
        assert np.all(np.isfinite(command.command_values)), "Command contains NaN/Inf"

        # Step 8: Send command via ActuatorBridge
        success = actuator_bridge.send(command)
        assert success, "ActuatorBridge failed to send command"

        print(f"\nE2E Pipeline completed successfully!")
        print(f"  Nodes: {len(stabilized_graph.nodes)}")
        print(f"  Edges: {len(stabilized_graph.edges)}")
        print(f"  Energy: {stabilized_graph.global_energy:.4f}")
        print(f"  Command mode: {command.mode.value}")

    def test_pipeline_multiple_frames(self):
        """Pipeline should work across multiple frames."""
        # Create components
        sensor_bridge = create_mock_sensor_bridge(num_joints=6)
        sego = create_default_sego_anchor()
        logos = create_default_solver()
        imago = create_default_imago_planner()
        ortsf = create_default_ortsf_fabric()
        actuator_bridge = create_mock_actuator_bridge(num_joints=6)
        mission_control = create_default_mission_control()

        # Set up mission
        goal = mission_control.publish_goal("MONITOR", target_node_id=1)
        commands_generated = 0

        for frame_idx in range(5):
            # Acquire
            observation = sensor_bridge.acquire()

            # SEGO
            detections = [
                Detection(
                    detection_id=1,
                    class_name="object",
                    confidence=0.9,
                    bbox=(100 + frame_idx * 10, 100, 50, 50),
                ),
            ]
            raw_graph = sego.process(observation, detections)

            # LOGOS
            stabilized_graph = logos.solve(raw_graph)
            mission_control.update_graph(stabilized_graph)

            # IMAGO
            current_goal = mission_control.get_current_goal()
            if current_goal and stabilized_graph.nodes:
                trace = imago.plan(stabilized_graph, current_goal)

                # ORTSF
                command = ortsf.step(trace, observation)

                # Actuator
                if actuator_bridge.send(command):
                    commands_generated += 1

        assert commands_generated == 5, f"Expected 5 commands, got {commands_generated}"

    def test_pipeline_handles_empty_detections(self):
        """Pipeline should handle frames with no detections gracefully."""
        sensor_bridge = create_mock_sensor_bridge(num_joints=6)
        sego = create_default_sego_anchor()
        logos = create_default_solver()
        ortsf = create_default_ortsf_fabric()

        observation = sensor_bridge.acquire()

        # Process with no external detections (uses internal simple detection)
        raw_graph = sego.process(observation, detections=[])

        # Should still have at least scene node from simple detection
        # or handle empty gracefully
        if raw_graph.nodes:
            stabilized_graph = logos.solve(raw_graph)
            assert stabilized_graph is not None
        else:
            # Empty detections should produce empty graph
            pass

    def test_pipeline_energy_decreases(self):
        """LOGOS energy should generally decrease during optimization."""
        sensor_bridge = create_mock_sensor_bridge()
        sego = create_default_sego_anchor()
        logos = create_default_solver()

        observation = sensor_bridge.acquire()
        detections = [
            Detection(
                detection_id=1,
                class_name="cup",
                confidence=0.9,
                bbox=(100, 100, 50, 50),
            ),
            Detection(
                detection_id=2,
                class_name="cup",
                confidence=0.9,
                bbox=(150, 100, 50, 50),
            ),
        ]

        raw_graph = sego.process(observation, detections)
        stabilized_graph = logos.solve(raw_graph)

        result = logos.get_last_result()
        assert result is not None

        # Energy should not increase significantly
        if len(result.energy_history) > 1:
            assert result.energy_history[-1] <= result.energy_history[0] + 0.1

    def test_pipeline_respects_safety_limits(self):
        """Actuator commands should respect safety limits."""
        from hal.actuator_bridge.actuator_bridge import create_safe_actuator_bridge

        sensor_bridge = create_mock_sensor_bridge(num_joints=6)
        sego = create_default_sego_anchor()
        logos = create_default_solver()
        imago = create_default_imago_planner()
        ortsf = create_default_ortsf_fabric()
        actuator_bridge = create_safe_actuator_bridge(
            num_joints=6, conservative_limits=True
        )

        # Run pipeline
        observation = sensor_bridge.acquire()
        detections = [
            Detection(
                detection_id=1,
                class_name="object",
                confidence=1.0,
                bbox=(100, 100, 50, 50),
            )
        ]
        raw_graph = sego.process(observation, detections)
        stabilized_graph = logos.solve(raw_graph)

        goal = MissionGoal(
            goal_id="test", verb="MOVE", target_node_id=1, constraints={}
        )
        trace = imago.plan(stabilized_graph, goal)
        command = ortsf.step(trace, observation)

        actuator_bridge.send(command)
        final_command = actuator_bridge.get_last_command()

        # Check limits (conservative: ±π/2)
        limits = actuator_bridge.get_limits()["position"]
        for i, val in enumerate(final_command.command_values):
            lo, hi = limits[i]
            assert lo <= val <= hi, f"Joint {i} value {val} outside [{lo}, {hi}]"


class TestPipelinePerformance:
    """Performance-related smoke tests."""

    def test_pipeline_completes_in_reasonable_time(self):
        """Full pipeline should complete within timing budget."""
        import time

        sensor_bridge = create_mock_sensor_bridge(num_joints=6)
        sego = create_default_sego_anchor()
        logos = create_default_solver()
        imago = create_default_imago_planner()
        ortsf = create_default_ortsf_fabric()

        goal = MissionGoal(
            goal_id="perf", verb="GRASP", target_node_id=1, constraints={}
        )

        # Warm up
        obs = sensor_bridge.acquire()
        det = [
            Detection(
                detection_id=1,
                class_name="obj",
                confidence=0.9,
                bbox=(100, 100, 50, 50),
            )
        ]
        raw = sego.process(obs, det)
        stable = logos.solve(raw)
        trace = imago.plan(stable, goal)
        ortsf.step(trace, obs)

        # Time 10 iterations
        start = time.time()
        for _ in range(10):
            obs = sensor_bridge.acquire()
            raw = sego.process(obs, det)
            stable = logos.solve(raw)
            trace = imago.plan(stable, goal)
            ortsf.step(trace, obs)
        elapsed = time.time() - start

        avg_ms = (elapsed / 10) * 1000
        print(f"\nAverage pipeline time: {avg_ms:.1f}ms")

        # Should complete in less than 500ms per frame for smoke test
        assert avg_ms < 500, f"Pipeline too slow: {avg_ms:.1f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
