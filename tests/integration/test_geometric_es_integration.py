"""Integration tests for Geometric Relation + ONN-ES.

Tests the complete integration of:
- Phase 1: GeometricRelationEncoder
- Phase 2: PredictiveRelationModel
- Phase 3: ContrastiveRelationLearner
- ONN-ES: CMA-ES optimization

Philosophy:
    "관계를 라벨 없이, 예측과 일관성으로 이해한다"
"""

import pytest
import numpy as np
import torch

from onn.es import (
    # Original ONN-ES
    CMAESTrainer,
    create_default_trainer,
    create_trainer_with_w_lin,
    Candidate,
    # Geometric Integration
    GeometricFitnessConfig,
    GeometricEpisodeStep,
    GeometricEvalMetrics,
    GeometricRelationEncoderES,
    evaluate_geometric_episode,
    compute_geometric_fitness,
    evaluate_geometric_candidate,
    generate_geometric_episode,
    create_geometric_encoder_es,
    create_geometric_fitness_config,
)

from onn.core.solver import create_default_pc_solver
from onn.core.relation_geometry import (
    create_geometric_encoder,
    create_predictive_model,
    create_temporal_model,
    create_contrastive_learner,
)


class TestGeometricRelationEncoderES:
    """Tests for GeometricRelationEncoderES."""

    def test_create_encoder(self):
        """Should create encoder with correct dimensions."""
        encoder = create_geometric_encoder_es(
            input_dim=64,
            output_dim=32,
            use_geometric=True,
        )

        assert encoder.input_dim == 64
        assert encoder.output_dim == 32
        assert encoder.W_rp.shape == (32, 64)
        assert encoder.W_lin.shape == (32, 64)
        assert encoder.W_geo.shape == (32, 6)

    def test_encode_without_geometry(self):
        """Should encode using only random projection + linear."""
        encoder = create_geometric_encoder_es(
            input_dim=16,
            output_dim=8,
            use_geometric=False,
        )

        phi = torch.randn(5, 16)
        x_obs = encoder.encode(phi)

        assert x_obs.shape == (5, 8)

    def test_encode_with_geometry(self):
        """Should include geometric encoding when nodes provided."""
        from onn.core.tensors import SemanticNode, BOUND_TENSOR_DIM, FORM_TENSOR_DIM, INTENT_TENSOR_DIM

        encoder = create_geometric_encoder_es(
            input_dim=FORM_TENSOR_DIM * 2,
            output_dim=32,
            use_geometric=True,
        )

        # Create test nodes
        bound_a = np.zeros(BOUND_TENSOR_DIM, dtype=np.float32)
        bound_a[0:3] = [0, 0, 0]
        bound_a[3:6] = [0, 0, 1]

        bound_b = np.zeros(BOUND_TENSOR_DIM, dtype=np.float32)
        bound_b[0:3] = [1, 0, 0]
        bound_b[3:6] = [0, 0, 1]

        node_a = SemanticNode(
            node_id=0,
            bound_tensor=bound_a,
            form_tensor=np.random.randn(FORM_TENSOR_DIM).astype(np.float32),
            intent_tensor=np.random.rand(INTENT_TENSOR_DIM).astype(np.float32),
        )
        node_b = SemanticNode(
            node_id=1,
            bound_tensor=bound_b,
            form_tensor=np.random.randn(FORM_TENSOR_DIM).astype(np.float32),
            intent_tensor=np.random.rand(INTENT_TENSOR_DIM).astype(np.float32),
        )

        phi = torch.randn(FORM_TENSOR_DIM * 2)
        x_obs = encoder.encode(phi, node_a, node_b)

        assert x_obs.shape == (32,)

    def test_set_linear_weights(self):
        """Should update W_lin from ES."""
        encoder = create_geometric_encoder_es(input_dim=16, output_dim=8)

        new_weights = torch.randn(8, 16)
        encoder.set_linear_weights(new_weights)

        assert torch.allclose(encoder.W_lin, new_weights)


class TestGeometricEpisodeGeneration:
    """Tests for synthetic geometric episode generation."""

    def test_generate_static_episode(self):
        """Should generate static scene episode."""
        episode = generate_geometric_episode(
            num_steps=5,
            num_nodes=4,
            embedding_dim=16,
            motion_type="static",
            seed=42,
        )

        assert len(episode) == 5
        for step in episode:
            assert isinstance(step, GeometricEpisodeStep)
            assert step.nodes is not None
            assert len(step.nodes) == 4
            assert step.relations is not None
            assert step.x_obs.shape[1] == 16

    def test_generate_circular_episode(self):
        """Should generate circular motion episode."""
        episode = generate_geometric_episode(
            num_steps=10,
            num_nodes=3,
            embedding_dim=8,
            motion_type="circular",
            seed=42,
        )

        assert len(episode) == 10

        # Check that positions change over time
        pos_t0 = episode[0].nodes[0].bound_tensor[:3]
        pos_t5 = episode[5].nodes[0].bound_tensor[:3]
        assert not np.allclose(pos_t0, pos_t5)

    def test_generate_with_events(self):
        """Should generate events with specified probability."""
        episode = generate_geometric_episode(
            num_steps=100,
            num_nodes=3,
            event_prob=0.3,
            seed=42,
        )

        event_count = sum(1 for step in episode if step.is_event)
        # Expect roughly 30 events (with variance)
        assert 10 < event_count < 60


class TestGeometricEpisodeEvaluation:
    """Tests for geometric episode evaluation."""

    @pytest.fixture
    def solver(self):
        return create_default_pc_solver()

    @pytest.fixture
    def config(self):
        return create_geometric_fitness_config(geometric_weight=1.0)

    def test_evaluate_episode_basic(self, solver, config):
        """Should evaluate episode and return metrics."""
        episode = generate_geometric_episode(
            num_steps=5,
            num_nodes=3,
            embedding_dim=32,
            seed=42,
        )

        metrics = evaluate_geometric_episode(solver, episode, config)

        assert isinstance(metrics, GeometricEvalMetrics)
        assert metrics.violation_mean >= 0
        assert 0 <= metrics.prediction_accuracy <= 1
        assert 0 <= metrics.bidirectional_consistency <= 1
        assert 0 <= metrics.cycle_consistency <= 1

    def test_evaluate_episode_with_models(self, solver, config):
        """Should use provided models for evaluation."""
        episode = generate_geometric_episode(
            num_steps=5,
            num_nodes=4,
            seed=42,
        )

        predictor = create_predictive_model()
        temporal = create_temporal_model()
        contrastive = create_contrastive_learner()

        metrics = evaluate_geometric_episode(
            solver, episode, config,
            predictor=predictor,
            temporal_model=temporal,
            contrastive_learner=contrastive,
        )

        assert isinstance(metrics, GeometricEvalMetrics)
        assert metrics.temporal_smoothness > 0


class TestGeometricFitness:
    """Tests for geometric fitness computation."""

    def test_compute_fitness_positive_metrics(self):
        """Good metrics should give positive fitness."""
        metrics = GeometricEvalMetrics(
            violation_mean=0.01,
            drift_mean=0.05,
            collapse_score=0.5,
            prediction_accuracy=0.95,
            bidirectional_consistency=0.9,
            cycle_consistency=0.9,
            contrastive_loss=0.1,
        )

        config = create_geometric_fitness_config()
        fitness = compute_geometric_fitness(metrics, config)

        # Good metrics should give decent fitness
        assert fitness > 0

    def test_compute_fitness_bad_metrics(self):
        """Bad metrics should give lower fitness."""
        good_metrics = GeometricEvalMetrics(
            violation_mean=0.01,
            prediction_accuracy=0.95,
            cycle_consistency=0.9,
        )

        bad_metrics = GeometricEvalMetrics(
            violation_mean=1.0,
            prediction_accuracy=0.3,
            cycle_consistency=0.3,
        )

        config = create_geometric_fitness_config()
        good_fitness = compute_geometric_fitness(good_metrics, config)
        bad_fitness = compute_geometric_fitness(bad_metrics, config)

        assert good_fitness > bad_fitness


class TestCMAESWithGeometric:
    """Tests for CMA-ES integration with geometric relations."""

    def test_trainer_with_geometric_evaluation(self):
        """CMA-ES should optimize with geometric fitness."""
        # Create trainer with W_lin optimization
        trainer = create_trainer_with_w_lin(
            w_lin_shape=(32, 64),
            population_size=4,
        )

        # Generate episodes
        episodes = [
            generate_geometric_episode(
                num_steps=3,
                num_nodes=3,
                embedding_dim=32,
                seed=i,
            )
            for i in range(2)
        ]

        # Create encoder
        encoder = create_geometric_encoder_es(
            input_dim=64,
            output_dim=32,
        )

        config = create_geometric_fitness_config()

        # One ES iteration
        candidates = trainer.ask()
        assert len(candidates) == 4

        fitnesses = []
        for candidate in candidates:
            fitness = evaluate_geometric_candidate(
                candidate, episodes, encoder, config
            )
            fitnesses.append(fitness)

        trainer.tell(candidates, fitnesses)

        # Should have updated state
        assert trainer.state.generation == 1

    def test_fitness_improves_over_generations(self):
        """Fitness should improve (or stay stable) over generations."""
        trainer = create_trainer_with_w_lin(
            w_lin_shape=(16, 32),
            population_size=6,
        )

        episodes = [
            generate_geometric_episode(
                num_steps=3,
                num_nodes=3,
                embedding_dim=16,
                seed=i,
            )
            for i in range(2)
        ]

        encoder = create_geometric_encoder_es(
            input_dim=32,
            output_dim=16,
        )
        config = create_geometric_fitness_config()

        best_fitnesses = []

        for gen in range(5):
            candidates = trainer.ask()
            fitnesses = [
                evaluate_geometric_candidate(c, episodes, encoder, config)
                for c in candidates
            ]
            trainer.tell(candidates, fitnesses)
            best_fitnesses.append(max(fitnesses))

        # Fitness should not drastically decrease
        # (may fluctuate due to stochasticity, but trend should be stable/up)
        assert best_fitnesses[-1] >= best_fitnesses[0] - 1.0  # Allow some variance


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline(self):
        """Test complete pipeline from perception to optimization."""
        from onn.core.tensors import SemanticNode, BOUND_TENSOR_DIM, FORM_TENSOR_DIM, INTENT_TENSOR_DIM

        # 1. Create scene with objects
        nodes = []
        for i in range(4):
            bound = np.zeros(BOUND_TENSOR_DIM, dtype=np.float32)
            bound[0:3] = [float(i), 0, 0]  # Position
            bound[3:6] = [0, 0, 1]  # Orientation

            nodes.append(SemanticNode(
                node_id=i,
                bound_tensor=bound,
                form_tensor=np.random.randn(FORM_TENSOR_DIM).astype(np.float32),
                intent_tensor=np.random.rand(INTENT_TENSOR_DIM).astype(np.float32),
            ))

        # 2. Encode geometric relations
        geo_encoder = create_geometric_encoder()
        relations = geo_encoder.encode_all_pairs(nodes)

        assert len(relations) == 12  # 4 * 3 directed pairs

        # 3. Validate with predictive model
        predictor = create_predictive_model()
        for (src, tgt), rel in relations.items():
            result = predictor.validate_relation(
                nodes[src], nodes[tgt], rel
            )
            assert result.relation_confidence > 0.9  # Should be accurate

        # 4. Generate episode and evaluate
        episode = generate_geometric_episode(
            num_steps=5,
            num_nodes=4,
            seed=42,
        )

        solver = create_default_pc_solver()
        config = create_geometric_fitness_config()
        metrics = evaluate_geometric_episode(solver, episode, config)

        assert metrics.prediction_accuracy > 0
        assert metrics.cycle_consistency > 0

        # 5. Run one ES optimization step
        trainer = create_trainer_with_w_lin(
            w_lin_shape=(16, 32),
            population_size=4,
        )

        encoder = create_geometric_encoder_es(input_dim=32, output_dim=16)

        candidates = trainer.ask()
        fitnesses = [
            evaluate_geometric_candidate(c, [episode], encoder, config)
            for c in candidates
        ]
        trainer.tell(candidates, fitnesses)

        assert trainer.state.generation == 1
        assert all(isinstance(f, float) for f in fitnesses)

    def test_contrastive_learning_integration(self):
        """Test contrastive learning with ES."""
        # Generate two related episodes (temporal sequence)
        episode1 = generate_geometric_episode(
            num_steps=3,
            num_nodes=3,
            motion_type="linear",
            seed=42,
        )

        # Extract relations for contrastive learning
        learner = create_contrastive_learner()

        all_relations = []
        for step in episode1:
            if step.relations:
                all_relations.extend(step.relations.values())

        # Mine similarity pairs
        pairs = learner.mine_similarity_pairs(all_relations, threshold=1.0)

        # Compute contrastive loss
        if pairs:
            loss = learner.compute_batch_loss(pairs)
            assert loss >= 0

    def test_clustering_with_geometric_relations(self):
        """Test relation clustering."""
        from onn.core.relation_geometry import create_relation_clusterer

        # Generate diverse relations
        episode = generate_geometric_episode(
            num_steps=1,
            num_nodes=6,  # More nodes = more relations
            seed=42,
        )

        relations = list(episode[0].relations.values())

        # Cluster relations
        clusterer = create_relation_clusterer(n_clusters=3)
        clusters = clusterer.fit(relations)

        assert len(clusters) == 3

        # Each cluster should have interpretation
        for cluster in clusters:
            if cluster.members:
                interp = clusterer.get_cluster_interpretation(cluster)
                assert "translation_magnitude" in interp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
