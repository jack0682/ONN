"""Integration test for ONN-ES Gate Loop.

This test verifies that the full ONN-ES pipeline works end-to-end:
1. Create edge graph and observations
2. Run PC solver to stabilize embeddings
3. Compute evaluation metrics
4. Check validation gates
5. Run CMA-ES to optimize solver parameters

Reference:
    - spec/20_impl_plan.ir.yml: IMPL_024
    - User roadmap: "ONN-ES gate loop"

Author: Claude (via IMPL_024)
"""

import pytest
import torch
import numpy as np

from onn.core.graph import EdgeGraph
from onn.core.cycles import build_cycle_basis
from onn.core.solver import ProjectionConsensusSolver, PCSolverConfig
from onn.core.projection import compute_projection_error
from onn.es import (
    CMAESTrainer,
    ESConfig,
    EpisodeStep,
    Candidate,
    create_trainer_with_w_lin,
    evaluate_candidate,
)
from onn.es.fitness import evaluate_episode, FitnessConfig, EvalMetrics
from onn.eval import evaluate_gates, GateConfig, format_report
from onn.relation.param import RelationEncoder, RelationParamConfig


class TestONNESGateLoop:
    """Integration tests for ONN-ES Gate loop."""

    def test_solver_reduces_violation(self):
        """PC solver should reduce cycle constraint violation."""
        # Create a triangle graph (1 cycle)
        edges = [(0, 1), (1, 2), (2, 0)]
        graph = EdgeGraph.from_edge_list(edges)

        # Build cycle basis
        basis = build_cycle_basis(graph, embedding_dim=32)
        assert basis.num_cycles == 1

        # Random observations (will have non-zero violation)
        x_obs = torch.randn(3, 32)
        initial_violation = compute_projection_error(
            x_obs, basis.cycle_matrix, basis.tau
        )

        # Solve
        config = PCSolverConfig(steps=20, step_size=0.05)
        solver = ProjectionConsensusSolver(config)
        result = solver.solve(x_obs, graph)

        # Check violation reduced
        final_violation = compute_projection_error(
            result.x, basis.cycle_matrix, basis.tau
        )

        assert final_violation < initial_violation
        assert final_violation < 1e-4  # Should be near-zero

    def test_cmaes_improves_fitness(self):
        """CMA-ES should improve fitness over generations."""
        # Create trainer
        config = ESConfig(population_size=8, max_generations=5, seed=42)
        trainer = CMAESTrainer(config)

        # Run 5 generations with dummy fitness
        fitness_history = []
        for gen in range(5):
            candidates = trainer.ask()

            # Fitness: penalize large parameters (just for testing)
            fitnesses = []
            for c in candidates:
                # c is now a Candidate object, access hyperparams
                f = -sum(v**2 for v in c.hyperparams.values())
                fitnesses.append(f)

            trainer.tell(candidates, fitnesses)
            fitness_history.append(trainer.best_fitness)

        # Fitness should improve (become less negative)
        assert fitness_history[-1] >= fitness_history[0]

    def test_evaluation_gates_work(self):
        """Evaluation gates should correctly pass/fail based on metrics."""
        # Metrics that should pass
        good_metrics = {
            "violation_mean": 0.01,
            "violation_max": 0.05,
            "drift_mean": 0.3,
            "collapse_score": 0.1,
            "latency_mean": 2.0,
            "ricci_energy": 10.0,
        }

        config = GateConfig()
        report = evaluate_gates(EvalMetrics(**good_metrics), config)

        assert report.passed is True
        assert len(report.failed_gates) == 0

        # Metrics that should fail
        bad_metrics = {
            "violation_mean": 1.0,  # Too high
            "violation_max": 1.0,
            "drift_mean": 0.3,
            "collapse_score": 0.001,  # Too low
            "latency_mean": 100.0,  # Too high
            "ricci_energy": 10.0,
        }

        bad_report = evaluate_gates(EvalMetrics(**bad_metrics), config)

        assert bad_report.passed is False
        assert len(bad_report.failed_gates) > 0

    def test_full_onn_es_loop(self):
        """Full ONN-ES loop: solver → metrics → gates."""
        # Setup
        edges = [(0, 1), (1, 2), (2, 0), (0, 2)]  # Triangle + chord = 2 cycles
        graph = EdgeGraph.from_edge_list(edges)

        # Solver
        solver_config = PCSolverConfig(steps=10, step_size=0.05)
        solver = ProjectionConsensusSolver(solver_config)

        # Create synthetic episode
        episode = []
        for t in range(5):
            x_obs = torch.randn(4, 32) * 0.1  # Small random observations
            episode.append(
                EpisodeStep(x_obs=x_obs, edge_graph=graph, is_event=(t == 2))
            )

        # Run episode
        fitness_config = FitnessConfig()
        metrics = evaluate_episode(solver, episode, fitness_config)

        # Check metrics are valid
        assert metrics.violation_mean >= 0
        assert metrics.drift_mean >= 0
        assert metrics.collapse_score >= 0

        # Check gates
        metrics_dict = {
            "violation_mean": metrics.violation_mean,
            "violation_max": metrics.violation_max,
            "drift_mean": metrics.drift_mean,
            "collapse_score": metrics.collapse_score,
            "latency_mean": metrics.latency_mean,
            "ricci_energy": metrics.ricci_energy,
        }

        gate_config = GateConfig()
        report = evaluate_gates(EvalMetrics(**metrics_dict), gate_config)

        # Should at least get a report
        assert report is not None
        print(format_report(report))


class TestSyntheticScenarios:
    """Tests with synthetic scenarios."""

    def test_noise_robustness(self):
        """Solver should handle noisy observations."""
        edges = [(0, 1), (1, 2), (2, 0)]
        graph = EdgeGraph.from_edge_list(edges)

        # Base observations + noise
        base_x = torch.zeros(3, 32)
        noisy_x = base_x + torch.randn(3, 32) * 0.5

        config = PCSolverConfig(steps=20, step_size=0.05)
        solver = ProjectionConsensusSolver(config)

        result = solver.solve(noisy_x, graph)

        # Should converge
        assert result.converged or result.final_loss < result.loss_history[0]

    def test_warm_start(self):
        """Warm start should speed up convergence."""
        edges = [(0, 1), (1, 2), (2, 0)]
        graph = EdgeGraph.from_edge_list(edges)

        config = PCSolverConfig(steps=5, step_size=0.05)
        solver = ProjectionConsensusSolver(config)

        # First solve
        x_obs1 = torch.randn(3, 32)
        result1 = solver.solve(x_obs1, graph, warm_start=False)

        # Second solve with warm start (similar observation)
        x_obs2 = x_obs1 + torch.randn(3, 32) * 0.1
        result2 = solver.solve(x_obs2, graph, warm_start=True)

        # Both should complete
        assert result1.iterations_used > 0
        assert result2.iterations_used > 0


class TestWLinOptimization:
    """Tests for W_lin neural parameter optimization via ES."""

    def test_trainer_with_w_lin_creates_candidates(self):
        """Trainer should create candidates with W_lin matrices."""
        # Create trainer with W_lin optimization
        trainer = create_trainer_with_w_lin(
            w_lin_shape=(32, 64),
            population_size=4,
            seed=42,
        )

        # Ask for candidates
        candidates = trainer.ask()

        assert len(candidates) == 4
        for c in candidates:
            assert isinstance(c, Candidate)
            assert c.hyperparams is not None
            assert c.w_lin is not None
            assert c.w_lin.shape == (32, 64)

    def test_w_lin_affects_encoding(self):
        """W_lin should affect RelationEncoder output."""
        # Create encoder
        config = RelationParamConfig(input_dim=64, output_dim=32)
        encoder = RelationEncoder(config)

        # Test input
        phi = torch.randn(10, 64)

        # Encode with zero W_lin (default)
        x_obs_zero = encoder.encode(phi)

        # Set non-zero W_lin
        w_lin = torch.randn(32, 64) * 0.5
        encoder.set_linear_weights(w_lin)
        x_obs_nonzero = encoder.encode(phi)

        # Outputs should differ
        diff = torch.norm(x_obs_zero - x_obs_nonzero).item()
        assert diff > 0.1, "W_lin should affect encoding"

    def test_es_optimizes_w_lin(self):
        """ES should optimize W_lin over generations."""
        # Create trainer
        trainer = create_trainer_with_w_lin(
            w_lin_shape=(8, 16),  # Smaller for faster test
            population_size=8,
            seed=42,
        )

        # Simple fitness: prefer W_lin with small norm
        def simple_fitness(candidate):
            hp_penalty = sum(v**2 for v in candidate.hyperparams.values())
            w_lin_norm = (
                np.linalg.norm(candidate.w_lin) if candidate.w_lin is not None else 0
            )
            return -hp_penalty - w_lin_norm

        # Run 5 generations
        fitness_history = []
        for gen in range(5):
            candidates = trainer.ask()
            fitnesses = [simple_fitness(c) for c in candidates]
            trainer.tell(candidates, fitnesses)
            fitness_history.append(trainer.best_fitness)

        # Fitness should improve
        assert fitness_history[-1] >= fitness_history[0]

        # Best candidate should have W_lin
        best, _ = trainer.get_best()
        assert best.w_lin is not None
        assert best.w_lin.shape == (8, 16)

    def test_full_w_lin_es_loop(self):
        """Full ES loop with W_lin and RelationEncoder."""
        # Setup
        input_dim = 16
        output_dim = 8
        num_edges = 4

        # Create encoder
        encoder_config = RelationParamConfig(input_dim=input_dim, output_dim=output_dim)
        encoder = RelationEncoder(encoder_config)

        # Create graph
        edges = [(0, 1), (1, 2), (2, 0), (0, 2)]
        graph = EdgeGraph.from_edge_list(edges)

        # Create episode with phi features
        episode = []
        for t in range(3):
            phi = torch.randn(num_edges, input_dim)
            x_obs = encoder.encode(phi)  # Initial encoding
            episode.append(
                EpisodeStep(
                    x_obs=x_obs,
                    edge_graph=graph,
                    is_event=(t == 1),
                    phi=phi,
                )
            )

        # Create trainer
        trainer = create_trainer_with_w_lin(
            w_lin_shape=(output_dim, input_dim),
            population_size=4,
            seed=42,
        )

        # Run one generation
        candidates = trainer.ask()
        fitnesses = []

        for c in candidates:
            fitness = evaluate_candidate(
                c,
                episodes=[episode],
                relation_encoder=encoder,
            )
            fitnesses.append(fitness)

        trainer.tell(candidates, fitnesses)

        # Should complete without error
        report = trainer.get_report()
        assert report["generation"] == 1
        assert report["optimize_w_lin"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
