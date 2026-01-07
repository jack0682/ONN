"""Integration tests for CSA Pipeline with ONN-ES.

Tests the full integration:
    SEGO → EdgeStabilizer → LOGOS → IMAGO

With ES optimization:
    ES → W_lin → EdgeStabilizer → Fitness → ES

Reference:
    - User requirement: "CSA 파이프라인에 연결해줘"
"""

import pytest
import torch
import numpy as np

from onn.core.tensors import SensorObservation, MissionGoal
from onn.ops import (
    # EdgeStabilizer
    EdgeStabilizer,
    EdgeStabilizerConfig,
    create_edge_stabilizer_for_es,
    extract_edge_features,
    # CSA Pipeline
    CSAPipeline,
    CSAConfig,
    CSAWithES,
    ESEpisode,
    create_default_csa_pipeline,
    create_csa_with_es,
    create_synthetic_episode,
    # SEGO
    SEGOGaugeAnchor,
    Detection,
)
from onn.es import create_trainer_with_w_lin, Candidate


class TestEdgeStabilizer:
    """Tests for EdgeStabilizer module."""

    def test_edge_stabilizer_initialization(self):
        """EdgeStabilizer should initialize with correct dimensions."""
        config = EdgeStabilizerConfig(
            edge_feature_dim=64,
            edge_embedding_dim=32,
        )
        stabilizer = EdgeStabilizer(config)

        assert stabilizer.get_w_lin_shape() == (32, 64)
        assert stabilizer.get_w_lin() is not None
        assert stabilizer.get_w_lin().shape == (32, 64)

    def test_edge_stabilizer_with_sego_output(self):
        """EdgeStabilizer should process SEGO output."""
        # Create SEGO output
        sego = SEGOGaugeAnchor()

        # Synthetic observation
        rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        depth = np.random.rand(480, 640).astype(np.float32) * 5.0
        obs = SensorObservation(
            timestamp_ns=0,
            frame_id="robot_base",
            rgb_images=[rgb],
            depth_maps=[depth],
        )

        # Generate detections
        detections = [
            Detection(detection_id=i, class_name=f"obj_{i}",
                      confidence=0.9, bbox=(i*100, i*50, 80, 80),
                      centroid_3d=np.array([i*0.1, i*0.05, 1.0]))
            for i in range(4)
        ]

        # Process through SEGO
        raw_graph = sego.process(obs, detections)
        assert len(raw_graph.nodes) > 0
        assert len(raw_graph.edge_candidates) > 0

        # Process through EdgeStabilizer
        stabilizer = create_edge_stabilizer_for_es(
            edge_feature_dim=64,
            edge_embedding_dim=32,
        )
        result = stabilizer.stabilize(raw_graph)

        # Verify result
        assert result.stabilized_embeddings is not None
        assert result.stabilized_embeddings.shape[0] == len(raw_graph.edge_candidates)
        assert result.stabilized_embeddings.shape[1] == 32
        assert result.solver_result.converged or result.solver_result.iterations_used > 0

    def test_w_lin_affects_stabilization(self):
        """Different W_lin should produce different outputs."""
        sego = SEGOGaugeAnchor()

        # Create observation
        rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        obs = SensorObservation(timestamp_ns=0, frame_id="robot_base", rgb_images=[rgb], depth_maps=[])

        detections = [
            Detection(detection_id=i, class_name=f"obj_{i}",
                      confidence=0.9, bbox=(i*100, 50, 80, 80))
            for i in range(3)
        ]

        raw_graph = sego.process(obs, detections)

        # Stabilizer with default W_lin (zeros)
        stabilizer = create_edge_stabilizer_for_es()
        result1 = stabilizer.stabilize(raw_graph)
        emb1 = result1.stabilized_embeddings.clone()

        # Set different W_lin
        w_lin = torch.randn(32, 64) * 0.5
        stabilizer.set_w_lin(w_lin)
        result2 = stabilizer.stabilize(raw_graph)
        emb2 = result2.stabilized_embeddings

        # Should be different
        diff = torch.norm(emb1 - emb2).item()
        assert diff > 0.01, "W_lin should affect stabilization output"


class TestCSAPipeline:
    """Tests for CSA Pipeline."""

    def test_pipeline_full_execution(self):
        """Pipeline should execute full SEGO → EdgeStabilizer → LOGOS → IMAGO."""
        pipeline = create_default_csa_pipeline()

        # Create observation
        rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        depth = np.random.rand(480, 640).astype(np.float32) * 5.0
        obs = SensorObservation(
            timestamp_ns=0,
            frame_id="robot_base",
            rgb_images=[rgb],
            depth_maps=[depth],
        )

        detections = [
            Detection(detection_id=i, class_name=f"obj_{i}",
                      confidence=0.9, bbox=(i*100+50, 100, 80, 80),
                      centroid_3d=np.array([i*0.2, 0.2, 1.5]))
            for i in range(3)
        ]

        goal = MissionGoal(goal_id="test_goal", verb="GRASP", target_node_id=1)

        # Process
        result = pipeline.process(obs, goal, detections)

        # Verify all components ran
        assert result.raw_graph is not None
        assert result.stabilized_graph is not None
        assert result.edge_stabilizer_result is not None

        # LOGOS should have run
        assert result.logos_iterations > 0

    def test_pipeline_without_goal(self):
        """Pipeline should work without a goal (no IMAGO)."""
        pipeline = create_default_csa_pipeline()

        rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        obs = SensorObservation(timestamp_ns=0, frame_id="robot_base", rgb_images=[rgb], depth_maps=[])

        detections = [
            Detection(detection_id=0, class_name="obj", confidence=0.9, bbox=(100, 100, 80, 80))
        ]

        result = pipeline.process(obs, goal=None, detections=detections)

        assert result.raw_graph is not None
        assert result.stabilized_graph is not None
        assert result.reasoning_trace is None  # No goal = no trace

    def test_pipeline_set_w_lin(self):
        """Pipeline should allow setting W_lin."""
        pipeline = create_default_csa_pipeline()

        shape = pipeline.get_w_lin_shape()
        assert shape[0] > 0 and shape[1] > 0

        # Set new W_lin
        new_w_lin = torch.randn(shape)
        pipeline.set_w_lin(new_w_lin)

        # Verify it was set
        retrieved = pipeline.get_w_lin()
        assert torch.allclose(new_w_lin, retrieved)


class TestCSAWithES:
    """Tests for CSA + ES optimization."""

    def test_csa_with_es_initialization(self):
        """CSAWithES should initialize correctly."""
        csa_es = create_csa_with_es(population_size=8, seed=42)

        report = csa_es.get_report()
        assert report["generations"] == 0
        assert "optimize_w_lin" in report
        assert report["optimize_w_lin"] is True

    def test_synthetic_episode_generation(self):
        """Should generate valid synthetic episodes."""
        episode = create_synthetic_episode(num_steps=5, num_nodes=3, seed=42)

        assert len(episode.observations) == 5
        assert len(episode.goals) == 5
        assert len(episode.detections) == 5

        for obs in episode.observations:
            assert obs.rgb_images is not None
            assert len(obs.rgb_images) == 1

    def test_candidate_evaluation(self):
        """Should evaluate candidates on episodes."""
        csa_es = create_csa_with_es(population_size=4, seed=42)

        # Create episode
        episode = create_synthetic_episode(num_steps=3, num_nodes=3, seed=42)

        # Get a candidate
        candidates = csa_es.trainer.ask()
        candidate = candidates[0]

        # Evaluate
        fitness = csa_es.evaluate_candidate(candidate, [episode])

        # Should return a finite number
        assert np.isfinite(fitness)

    def test_training_step(self):
        """Training step should update ES and fitness."""
        csa_es = create_csa_with_es(population_size=4, seed=42)

        episode = create_synthetic_episode(num_steps=3, num_nodes=3, seed=42)

        # Run one training step
        metrics = csa_es.train_step([episode])

        assert metrics["generation"] == 1
        assert "best_fitness" in metrics
        assert "mean_fitness" in metrics
        assert np.isfinite(metrics["best_fitness"])

    def test_multi_generation_training(self):
        """Should train for multiple generations."""
        csa_es = create_csa_with_es(population_size=4, seed=42)

        episode = create_synthetic_episode(num_steps=3, num_nodes=3, seed=42)

        # Train for 3 generations
        final_report = csa_es.train([episode], num_generations=3)

        assert csa_es._generation == 3
        assert len(csa_es._fitness_history) == 3
        assert final_report["generation"] == 3

    def test_w_lin_optimization_improves(self):
        """W_lin optimization should improve or maintain fitness."""
        csa_es = create_csa_with_es(population_size=8, seed=42)

        episode = create_synthetic_episode(num_steps=5, num_nodes=4, seed=42)

        # Train for several generations
        csa_es.train([episode], num_generations=5)

        # Check fitness improved or stayed stable
        history = csa_es._fitness_history
        assert len(history) == 5

        # Best fitness should be at least as good as first
        # (CMA-ES tracks best, so this should always be true)
        assert history[-1] >= history[0] - 1e-6  # Small tolerance

        # Get trained W_lin
        trained_w_lin = csa_es.get_trained_w_lin()
        assert trained_w_lin is not None


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_full_es_csa_loop(self):
        """Full ES → W_lin → CSA → Fitness → ES loop."""
        # 1. Create CSA with ES
        csa_es = create_csa_with_es(population_size=4, seed=42)

        # 2. Generate training data
        episodes = [
            create_synthetic_episode(num_steps=3, num_nodes=3, seed=i)
            for i in range(2)
        ]

        # 3. Train
        fitness_log = []
        def callback(metrics):
            fitness_log.append(metrics["best_fitness"])

        csa_es.train(episodes, num_generations=3, callback=callback)

        # 4. Verify training ran
        assert len(fitness_log) == 3

        # 5. Use trained pipeline
        pipeline = csa_es.pipeline

        rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        obs = SensorObservation(timestamp_ns=0, frame_id="robot_base", rgb_images=[rgb], depth_maps=[])
        detections = [
            Detection(detection_id=0, class_name="obj", confidence=0.9, bbox=(100, 100, 80, 80)),
            Detection(detection_id=1, class_name="obj", confidence=0.9, bbox=(200, 100, 80, 80)),
        ]

        result = pipeline.process(obs, goal=None, detections=detections)

        # 6. Verify pipeline works with trained W_lin
        assert result.stabilized_graph is not None
        assert result.edge_stabilizer_result is not None

    def test_es_trainer_with_csa_dimensions(self):
        """ES trainer should work with CSA pipeline dimensions."""
        # Get dimensions from CSA
        pipeline = create_default_csa_pipeline()
        w_lin_shape = pipeline.get_w_lin_shape()

        # Create ES trainer with same dimensions
        trainer = create_trainer_with_w_lin(
            w_lin_shape=w_lin_shape,
            population_size=4,
            seed=42,
        )

        # Generate candidates
        candidates = trainer.ask()

        # Apply each candidate to pipeline and verify
        for candidate in candidates:
            assert candidate.w_lin is not None
            assert candidate.w_lin.shape == w_lin_shape

            w_lin_tensor = torch.tensor(candidate.w_lin, dtype=torch.float32)
            pipeline.set_w_lin(w_lin_tensor)

            # Verify it's set
            retrieved = pipeline.get_w_lin()
            assert torch.allclose(w_lin_tensor, retrieved)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
