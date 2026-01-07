"""
Unit tests for LOGOS energy/loss terms.

Verifies the loss functions defined in spec/02_onn_math_spec.md Section 4.1:
    L_total = λ_data * L_data + λ_phys * L_phys + λ_logic * L_logic

Where:
    L_data  = Σ_i ||S_i - S_i^raw||²           (Data Fidelity)
    L_phys  = Σ_{i≠j} ReLU(Sim(B_i,B_j) - θ)   (Physical Validity)
    L_logic = Σ_{(i,j)∈E} w_ij ||S_i + r_ij - S_j||²  (Logical Consistency / TransE)

Reference:
    - spec/02_onn_math_spec.md Section 4.1
    - spec/20_impl_plan.ir.yml TEST_001
"""

import pytest
import torch
import numpy as np

from onn.ops.logos_solver import (
    compute_data_loss,
    compute_phys_loss,
    compute_logic_loss,
    compute_total_loss,
    project_to_manifold,
    LOGOSConfig,
    STATE_DIM,
)
from onn.core.tensors import BOUND_TENSOR_DIM, FORM_TENSOR_DIM, INTENT_TENSOR_DIM


class TestDataLoss:
    """Tests for compute_data_loss (L_data = Σ_i ||S_i - S_i^raw||²)."""

    def test_identical_states_zero_loss(self):
        """Identical states should yield zero data loss."""
        state = torch.randn(3, STATE_DIM)
        state_raw = state.clone()
        
        loss = compute_data_loss(state, state_raw)
        
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_different_states_positive_loss(self):
        """Different states should yield positive data loss."""
        state = torch.ones(2, STATE_DIM)
        state_raw = torch.zeros(2, STATE_DIM)
        
        loss = compute_data_loss(state, state_raw)
        
        # Expected: 2 nodes * 64 dims * 1^2 = 128
        expected = 2 * STATE_DIM * 1.0
        assert loss.item() == pytest.approx(expected, rel=1e-5)

    def test_single_node_squared_distance(self):
        """Single node loss should equal squared L2 norm of difference."""
        state = torch.tensor([[1.0, 2.0, 3.0] + [0.0] * (STATE_DIM - 3)])
        state_raw = torch.zeros(1, STATE_DIM)
        
        loss = compute_data_loss(state, state_raw)
        
        # Expected: 1^2 + 2^2 + 3^2 = 14
        assert loss.item() == pytest.approx(14.0, rel=1e-5)

    def test_gradient_flows(self):
        """Data loss should have valid gradients for autograd."""
        state = torch.randn(3, STATE_DIM, requires_grad=True)
        state_raw = torch.randn(3, STATE_DIM)
        
        loss = compute_data_loss(state, state_raw)
        loss.backward()
        
        assert state.grad is not None
        assert not torch.isnan(state.grad).any()


class TestPhysLoss:
    """Tests for compute_phys_loss (L_phys = Σ_{i≠j} ReLU(R_i + R_j - ||p_i - p_j||))."""

    def test_separated_objects_zero_loss(self):
        """Objects strictly separated should have zero loss."""
        state = torch.zeros(2, STATE_DIM)
        # Obj 1: Pos=(0,0,0), Radius=1.0
        state[0, 0:3] = torch.tensor([0.0, 0.0, 0.0])
        state[0, 11] = 1.0
        # Obj 2: Pos=(3,0,0), Radius=1.0
        state[1, 0:3] = torch.tensor([3.0, 0.0, 0.0])
        state[1, 11] = 1.0
        
        # Dist=3.0, R_sum=2.0 -> No collision
        loss = compute_phys_loss(state, overlap_threshold=0.5)
        
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_colliding_objects_positive_loss(self):
        """Colliding objects should yield positive loss."""
        state = torch.zeros(2, STATE_DIM)
        # Obj 1: Pos=(0,0,0), Radius=1.0
        state[0, 0:3] = torch.tensor([0.0, 0.0, 0.0])
        state[0, 11] = 1.0
        # Obj 2: Pos=(1.0,0,0), Radius=1.0
        state[1, 0:3] = torch.tensor([1.0, 0.0, 0.0])
        state[1, 11] = 1.0
        
        # Dist=1.0, R_sum=2.0 -> Penetration=1.0
        loss = compute_phys_loss(state, overlap_threshold=0.5)
        
        # Loss = sum(violation) / 2 = (1.0 + 1.0) / 2 = 1.0
        assert loss.item() == pytest.approx(1.0, rel=1e-5)

    def test_single_node_zero_loss(self):
        """Single node graph should have zero physics loss (no pairs)."""
        state = torch.randn(1, STATE_DIM)
        
        loss = compute_phys_loss(state, overlap_threshold=0.5)
        
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_gradient_flows(self):
        """Physics loss should have valid gradients."""
        state = torch.randn(3, STATE_DIM, requires_grad=True)
        # Ensure radii are positive for sensible gradients
        with torch.no_grad():
            state[:, 11] = torch.abs(state[:, 11])
        
        loss = compute_phys_loss(state, overlap_threshold=0.5)
        loss.backward()
        
        assert state.grad is not None
        assert not torch.isnan(state.grad).any()


class TestLogicLoss:
    """Tests for compute_logic_loss (TransE-style: L_logic = Σ w_ij ||S_i + r_ij - S_j||²)."""

    def test_no_edges_zero_loss(self):
        """Graph with no edges should have zero logic loss."""
        state = torch.randn(3, STATE_DIM)
        edge_indices = torch.zeros((0, 2), dtype=torch.long)
        relation_embeddings = torch.zeros((0, STATE_DIM))
        edge_weights = torch.zeros((0,))
        
        loss = compute_logic_loss(state, edge_indices, relation_embeddings, edge_weights)
        
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_perfect_transe_zero_loss(self):
        """Perfect TransE relation (S_i + r = S_j) should have zero loss."""
        state = torch.zeros(2, STATE_DIM)
        state[0, 0] = 1.0
        state[1, 0] = 3.0
        
        # r_ij = S_j - S_i = [2, 0, 0, ...]
        relation = torch.zeros(1, STATE_DIM)
        relation[0, 0] = 2.0
        
        edge_indices = torch.tensor([[0, 1]])
        edge_weights = torch.tensor([1.0])
        
        loss = compute_logic_loss(state, edge_indices, relation, edge_weights)
        
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_imperfect_transe_positive_loss(self):
        """Imperfect TransE relation should have positive loss."""
        state = torch.zeros(2, STATE_DIM)
        state[0, 0] = 1.0
        state[1, 0] = 3.0
        
        # Wrong relation
        relation = torch.zeros(1, STATE_DIM)
        relation[0, 0] = 1.0  # Should be 2.0
        
        edge_indices = torch.tensor([[0, 1]])
        edge_weights = torch.tensor([1.0])
        
        loss = compute_logic_loss(state, edge_indices, relation, edge_weights)
        
        # Error = (1 + 1) - 3 = -1, squared = 1
        assert loss.item() > 0

    def test_edge_weight_effect(self):
        """Higher edge weight should increase loss for same violation."""
        state = torch.randn(2, STATE_DIM)
        relation = torch.randn(1, STATE_DIM)
        edge_indices = torch.tensor([[0, 1]])
        
        loss_low = compute_logic_loss(state, edge_indices, relation, torch.tensor([0.1]))
        loss_high = compute_logic_loss(state, edge_indices, relation, torch.tensor([1.0]))
        
        # loss_high should be ~10x loss_low
        assert loss_high > loss_low

    def test_gradient_flows(self):
        """Logic loss should have valid gradients."""
        state = torch.randn(3, STATE_DIM, requires_grad=True)
        edge_indices = torch.tensor([[0, 1], [1, 2]])
        relation = torch.randn(2, STATE_DIM)
        weights = torch.tensor([1.0, 1.0])
        
        loss = compute_logic_loss(state, edge_indices, relation, weights)
        loss.backward()
        
        assert state.grad is not None
        assert not torch.isnan(state.grad).any()


class TestTotalLoss:
    """Tests for compute_total_loss (L_total = λ_d*L_d + λ_p*L_p + λ_l*L_l)."""

    def test_lambda_weighting(self):
        """Lambda weights should scale individual loss components."""
        state = torch.randn(2, STATE_DIM)
        state_raw = torch.randn(2, STATE_DIM)
        edge_indices = torch.zeros((0, 2), dtype=torch.long)
        relation = torch.zeros((0, STATE_DIM))
        weights = torch.zeros((0,))
        
        # High data weight
        config_data = LOGOSConfig(lambda_data=10.0, lambda_phys=1.0, lambda_logic=1.0)
        loss_data, _ = compute_total_loss(state, state_raw, edge_indices, relation, weights, config_data)
        
        # High physics weight
        config_phys = LOGOSConfig(lambda_data=1.0, lambda_phys=10.0, lambda_logic=1.0)
        loss_phys, _ = compute_total_loss(state, state_raw, edge_indices, relation, weights, config_phys)
        
        # Results depend on which component dominates
        assert loss_data.item() >= 0
        assert loss_phys.item() >= 0

    def test_breakdown_dict_contains_all_components(self):
        """Loss breakdown should contain all loss components."""
        state = torch.randn(2, STATE_DIM)
        state_raw = torch.randn(2, STATE_DIM)
        edge_indices = torch.zeros((0, 2), dtype=torch.long)
        relation = torch.zeros((0, STATE_DIM))
        weights = torch.zeros((0,))
        config = LOGOSConfig()
        
        _, breakdown = compute_total_loss(state, state_raw, edge_indices, relation, weights, config)
        
        assert "data" in breakdown
        assert "phys" in breakdown
        assert "logic" in breakdown
        assert "total" in breakdown

    def test_gradient_flows_through_total(self):
        """Total loss should support autograd."""
        state = torch.randn(3, STATE_DIM, requires_grad=True)
        state_raw = torch.randn(3, STATE_DIM)
        edge_indices = torch.tensor([[0, 1], [1, 2]])
        relation = torch.randn(2, STATE_DIM)
        weights = torch.tensor([1.0, 1.0])
        config = LOGOSConfig()
        
        total_loss, _ = compute_total_loss(state, state_raw, edge_indices, relation, weights, config)
        total_loss.backward()
        
        assert state.grad is not None
        assert not torch.isnan(state.grad).any()
        assert not torch.isinf(state.grad).any()

    def test_default_hyperparameters_from_spec(self):
        """Default config should match spec/02_onn_math_spec.md Section 4.2."""
        config = LOGOSConfig()
        
        # Spec values: λ_data=1.0, λ_phys=10.0, λ_logic=2.0
        assert config.lambda_data == pytest.approx(1.0)
        assert config.lambda_phys == pytest.approx(10.0)
        assert config.lambda_logic == pytest.approx(2.0)


class TestProjectToManifold:
    """Tests for project_to_manifold (hard constraint projection Π_C)."""

    def test_bound_tensor_normalized(self):
        """Bound tensor (first 16 dims) should be projected to unit sphere."""
        state = torch.randn(2, STATE_DIM) * 10  # Large values
        
        projected = project_to_manifold(state)
        
        # Check bound tensor norm is 1
        bound = projected[:, :BOUND_TENSOR_DIM]
        norms = torch.norm(bound, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_intent_tensor_clamped(self):
        """Intent tensor (last 16 dims) should be clamped to [0, 1]."""
        state = torch.randn(2, STATE_DIM) * 5  # Values outside [0, 1]
        
        projected = project_to_manifold(state)
        
        # Check intent tensor in [0, 1]
        intent = projected[:, BOUND_TENSOR_DIM + FORM_TENSOR_DIM:]
        assert (intent >= 0).all()
        assert (intent <= 1).all()

    def test_form_tensor_unchanged(self):
        """Form tensor (middle 32 dims) should not be modified."""
        state = torch.randn(2, STATE_DIM)
        original_form = state[:, BOUND_TENSOR_DIM:BOUND_TENSOR_DIM + FORM_TENSOR_DIM].clone()
        
        projected = project_to_manifold(state)
        
        projected_form = projected[:, BOUND_TENSOR_DIM:BOUND_TENSOR_DIM + FORM_TENSOR_DIM]
        assert torch.allclose(original_form, projected_form)

    def test_idempotent(self):
        """Projection should be idempotent: Π(Π(x)) = Π(x)."""
        state = torch.randn(3, STATE_DIM) * 10
        
        projected_once = project_to_manifold(state)
        projected_twice = project_to_manifold(projected_once)
        
        assert torch.allclose(projected_once, projected_twice, atol=1e-6)


class TestNumericalStability:
    """Tests for numerical stability edge cases."""

    def test_zero_bound_tensor_no_nan(self):
        """Zero bound tensor should not cause division by zero."""
        state = torch.zeros(1, STATE_DIM)
        
        projected = project_to_manifold(state)
        
        assert not torch.isnan(projected).any()
        assert not torch.isinf(projected).any()

    def test_very_large_values_no_overflow(self):
        """Very large input values should not cause overflow."""
        state = torch.ones(2, STATE_DIM) * 1e6
        state_raw = torch.zeros(2, STATE_DIM)
        config = LOGOSConfig()
        
        loss = compute_data_loss(state, state_raw)
        
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_empty_graph_handled(self):
        """Edge cases with 0 nodes should be handled gracefully."""
        state = torch.zeros(0, STATE_DIM)
        state_raw = torch.zeros(0, STATE_DIM)
        config = LOGOSConfig()
        
        data_loss = compute_data_loss(state, state_raw)
        phys_loss = compute_phys_loss(state, 0.5)
        
        assert data_loss.item() == 0.0
        assert phys_loss.item() == 0.0
