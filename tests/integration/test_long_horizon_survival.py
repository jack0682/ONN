import pytest
import numpy as np
import torch

from onn.ops.logos_solver import LOGOSSolver, LOGOSConfig
from onn.core.tensors import (
    SemanticNode,
    SemanticEdge,
    RawSemanticGraph,
    BOUND_TENSOR_DIM,
    FORM_TENSOR_DIM,
    INTENT_TENSOR_DIM,
)


class TestLongHorizonSurvival:
    def test_gate_no_creeping_collapse_500_steps(self):
        np.random.seed(42)
        torch.manual_seed(42)

        num_nodes = 5
        nodes = []
        for i in range(num_nodes):
            bound = np.random.randn(BOUND_TENSOR_DIM).astype(np.float32)
            bound[11] = abs(bound[11]) + 0.5
            node = SemanticNode(
                node_id=i,
                bound_tensor=bound,
                form_tensor=np.random.randn(FORM_TENSOR_DIM).astype(np.float32),
                intent_tensor=np.random.rand(INTENT_TENSOR_DIM).astype(np.float32),
                uncertainty=0.5,
            )
            nodes.append(node)

        edges = []
        for i in range(num_nodes - 1):
            edge = SemanticEdge(
                source_id=i,
                target_id=i + 1,
                relation_embedding=np.random.randn(16).astype(np.float32),
                weight=1.0,
                probability=0.95,
                gate=0.8,
            )
            edges.append(edge)

        graph = RawSemanticGraph(timestamp_ns=0, nodes=nodes, edge_candidates=edges)

        config = LOGOSConfig(
            max_iterations=10,
            learning_rate=0.01,
            uncertainty_lr=0.01,
            gate_lr=0.05,
        )
        solver = LOGOSSolver(config)

        active_edge_ratios = []
        mean_gates = []
        mean_uncertainties = []
        residual_norms = []

        r_min = 0.2
        g_min = 0.15
        K = 20
        # Residual cap is permissive—goal is stability (not explosion), not convergence
        # Random synthetic data may have high but stable residuals
        residual_cap = 100.0

        for t in range(500):
            result_graph = solver.solve(graph)
            result = solver.get_last_result()

            if result and result.final_gates is not None:
                final_gates = result.final_gates
                if hasattr(final_gates, "detach"):
                    final_gates = final_gates.detach().cpu().numpy()
                final_uncs = result.final_uncertainties
                if hasattr(final_uncs, "detach"):
                    final_uncs = final_uncs.detach().cpu().numpy()

                active_ratio = float(np.mean(final_gates > 0.1))
                mean_gate = float(np.mean(final_gates))
                mean_unc = float(np.mean(final_uncs))
                res_norm = float(np.mean(result.residual_norm_history[-1:]))

                active_edge_ratios.append(active_ratio)
                mean_gates.append(mean_gate)
                mean_uncertainties.append(mean_unc)
                residual_norms.append(res_norm)

        active_edge_ratios = np.array(active_edge_ratios)
        mean_gates = np.array(mean_gates)
        mean_uncertainties = np.array(mean_uncertainties)
        residual_norms = np.array(residual_norms)

        below_r_min = active_edge_ratios < r_min
        below_g_min = mean_gates < g_min
        consecutive_r = np.sum(
            np.convolve(below_r_min.astype(int), np.ones(K), mode="valid") == K
        )
        consecutive_g = np.sum(
            np.convolve(below_g_min.astype(int), np.ones(K), mode="valid") == K
        )

        assert consecutive_r == 0, (
            f"Active edge ratio collapsed for {consecutive_r} windows of {K} steps"
        )
        assert consecutive_g == 0, (
            f"Mean gate collapsed for {consecutive_g} windows of {K} steps"
        )

        assert np.all(mean_uncertainties >= 0.001), "Uncertainty became negative"
        assert np.all(np.isfinite(mean_uncertainties)), "Uncertainty has NaN/Inf"

        residuals_after_warmup = residual_norms[50:]
        assert np.all(residuals_after_warmup < residual_cap), (
            f"Residual exceeded cap: {np.max(residuals_after_warmup)}"
        )

    def test_uncertainty_bounds_long_horizon(self):
        np.random.seed(123)
        torch.manual_seed(123)

        num_nodes = 3
        nodes = []
        for i in range(num_nodes):
            bound = np.random.randn(BOUND_TENSOR_DIM).astype(np.float32)
            bound[11] = 0.5
            node = SemanticNode(
                node_id=i,
                bound_tensor=bound,
                uncertainty=1.0,
            )
            nodes.append(node)

        edges = [
            SemanticEdge(
                source_id=0,
                target_id=1,
                relation_embedding=np.random.randn(16).astype(np.float32),
                gate=0.7,
            ),
            SemanticEdge(
                source_id=1,
                target_id=2,
                relation_embedding=np.random.randn(16).astype(np.float32),
                gate=0.7,
            ),
        ]

        graph = RawSemanticGraph(timestamp_ns=0, nodes=nodes, edge_candidates=edges)

        config = LOGOSConfig(
            max_iterations=5,
            uncertainty_min=0.01,
            uncertainty_max=5.0,
            uncertainty_lr=0.05,
        )
        solver = LOGOSSolver(config)

        for _ in range(100):
            result_graph = solver.solve(graph)
            result = solver.get_last_result()

            if result and result.final_uncertainties is not None:
                u_final = result.final_uncertainties
                if hasattr(u_final, "detach"):
                    u_final = u_final.detach().cpu().numpy()

                assert np.all(u_final >= config.uncertainty_min), (
                    f"Uncertainty below min: {np.min(u_final)}"
                )
                assert np.all(u_final <= config.uncertainty_max), (
                    f"Uncertainty above max: {np.max(u_final)}"
                )
                assert np.all(np.isfinite(u_final)), "Uncertainty has NaN/Inf"
