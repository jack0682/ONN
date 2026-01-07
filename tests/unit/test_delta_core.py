"""Tests for Deep Delta Learning (DDL) Core Module.

Tests the DeltaResidualBlock implementation based on the DDL paper:
"Deep Delta Learning" (Zhang et al., 2026)

Key tests:
1. Spectral Properties (Theorem 3.1, Eq 3.4):
   - det(A) = 1 - beta
   - beta=0 -> Identity (det=1)
   - beta=1 -> Projection (det=0)
   - beta=2 -> Reflection (det=-1)

2. Gradient Flow:
   - Multiple layers should have non-zero gradients
   - End-to-end differentiability

3. Update Formula (Eq 2.5 / 3.7):
   - h_next = h + beta * k * (v - k^T h)

Author: Claude
"""

import torch
import torch.nn as nn
import unittest
import numpy as np

from onn.modules.delta import (
    DeltaResidualBlock,
    DeltaResidualStack,
    DeltaLayerWrapper,
    DeltaConfig,
    compute_householder_matrix,
    verify_spectral_properties,
)


class TestDeltaResidualBlock(unittest.TestCase):
    """Test DeltaResidualBlock implementation."""

    def setUp(self):
        self.dim = 16
        self.batch_size = 4
        torch.manual_seed(42)

    def test_initialization(self):
        """Verify initialization follows DDL paper (beta near 0 for identity)."""
        block = DeltaResidualBlock(self.dim, beta_init_bias=-4.0)

        # Test initial beta value
        h = torch.randn(1, self.dim)
        beta = block.compute_beta(h)

        # beta = 2 * sigmoid(-4) ~ 2 * 0.018 ~ 0.036
        expected_beta = 2.0 * torch.sigmoid(torch.tensor(-4.0))
        self.assertTrue(
            torch.allclose(beta, expected_beta, atol=0.1),
            f"Initial beta should be near {expected_beta.item():.4f}, got {beta.mean().item():.4f}"
        )

    def test_k_normalization(self):
        """Verify k is L2 normalized (Appendix A, Eq A.1)."""
        block = DeltaResidualBlock(self.dim)
        h = torch.randn(self.batch_size, self.dim)

        k = block.compute_k(h)
        k_norm = torch.norm(k, dim=-1)

        self.assertTrue(
            torch.allclose(k_norm, torch.ones_like(k_norm), atol=1e-5),
            f"k should be unit normalized, got norms: {k_norm}"
        )

    def test_beta_range(self):
        """Verify beta is in [0, 2] (Eq 2.6)."""
        block = DeltaResidualBlock(self.dim)

        # Test with various inputs
        for _ in range(10):
            h = torch.randn(self.batch_size, self.dim) * 10  # Large values
            beta = block.compute_beta(h)

            self.assertTrue(
                (beta >= 0).all() and (beta <= 2).all(),
                f"Beta should be in [0, 2], got min={beta.min():.4f}, max={beta.max():.4f}"
            )

    def test_forward_shape(self):
        """Check output shape consistency."""
        block = DeltaResidualBlock(self.dim)
        h = torch.randn(self.batch_size, self.dim)

        h_next, beta = block(h)

        self.assertEqual(h_next.shape, h.shape)
        self.assertIsInstance(beta.item(), float)

    def test_determinant_beta_1(self):
        """Test det(A) = 0 when beta = 1 (Eq 3.4, projection case)."""
        block = DeltaResidualBlock(self.dim)
        h = torch.randn(self.batch_size, self.dim)

        # Force beta = 1
        with torch.no_grad():
            # sigmoid(0) = 0.5, so 2*sigmoid(0) = 1
            block.beta_proj.weight.fill_(0)
            block.beta_proj.bias.fill_(0)

        A = block.compute_delta_operator(h)
        det = torch.det(A)

        self.assertTrue(
            torch.allclose(det, torch.zeros_like(det), atol=1e-4),
            f"det(A) should be 0 when beta=1, got {det}"
        )

    def test_determinant_beta_2(self):
        """Test det(A) = -1 when beta = 2 (Eq 3.4, reflection case)."""
        block = DeltaResidualBlock(self.dim)
        h = torch.randn(self.batch_size, self.dim)

        # Force beta = 2
        with torch.no_grad():
            # Need sigmoid(x) = 1, so x -> inf
            # In practice, use large positive bias
            block.beta_proj.weight.fill_(0)
            block.beta_proj.bias.fill_(10)  # sigmoid(10) ~ 1

        A = block.compute_delta_operator(h)
        det = torch.det(A)

        self.assertTrue(
            torch.allclose(det, -torch.ones_like(det), atol=1e-2),
            f"det(A) should be -1 when beta=2, got {det}"
        )

    def test_determinant_beta_0(self):
        """Test det(A) = 1 when beta = 0 (identity case)."""
        block = DeltaResidualBlock(self.dim)
        h = torch.randn(self.batch_size, self.dim)

        # Force beta = 0
        with torch.no_grad():
            block.beta_proj.weight.fill_(0)
            block.beta_proj.bias.fill_(-10)  # sigmoid(-10) ~ 0

        A = block.compute_delta_operator(h)
        det = torch.det(A)

        self.assertTrue(
            torch.allclose(det, torch.ones_like(det), atol=1e-4),
            f"det(A) should be 1 when beta=0, got {det}"
        )

    def test_determinant_formula(self):
        """Test det(A) = 1 - beta for various beta values (Eq 3.4)."""
        block = DeltaResidualBlock(self.dim)
        h = torch.randn(self.batch_size, self.dim)

        beta = block.compute_beta(h)
        A = block.compute_delta_operator(h)
        det = torch.det(A)

        expected_det = 1.0 - beta.squeeze(-1)

        self.assertTrue(
            torch.allclose(det, expected_det, atol=1e-4),
            f"det(A) should equal 1-beta. det={det}, 1-beta={expected_det}"
        )

    def test_identity_at_beta_zero(self):
        """Verify identity mapping when beta -> 0 (Section 3.3)."""
        block = DeltaResidualBlock(self.dim, beta_init_bias=-10)
        h = torch.randn(self.batch_size, self.dim)

        h_next, beta = block(h)

        # With very small beta, h_next should be close to h
        self.assertTrue(
            torch.allclose(h_next, h, atol=0.1),
            f"h_next should be close to h when beta~0, max diff: {(h_next - h).abs().max():.4f}"
        )

    def test_gradient_flow(self):
        """Check gradients flow through the block."""
        block = DeltaResidualBlock(self.dim)
        h = torch.randn(self.batch_size, self.dim, requires_grad=True)

        h_next, _ = block(h)
        loss = h_next.sum()
        loss.backward()

        # Check gradients exist and are non-zero
        self.assertIsNotNone(h.grad)
        self.assertGreater(h.grad.abs().sum().item(), 0)

        self.assertIsNotNone(block.k_proj.weight.grad)
        self.assertGreater(block.k_proj.weight.grad.abs().sum().item(), 0)

        self.assertIsNotNone(block.beta_proj.weight.grad)
        self.assertIsNotNone(block.beta_proj.bias.grad)

    def test_spectral_verification(self):
        """Test the spectral verification utility."""
        block = DeltaResidualBlock(self.dim)
        h = torch.randn(self.batch_size, self.dim)

        beta = block.compute_beta(h)
        A = block.compute_delta_operator(h)

        results = verify_spectral_properties(A, beta)

        self.assertTrue(
            results['passed'],
            f"Spectral verification failed: {results['details']}"
        )


class TestDeltaResidualStack(unittest.TestCase):
    """Test DeltaResidualStack (multiple layers)."""

    def setUp(self):
        self.dim = 16
        self.num_blocks = 4
        self.batch_size = 4
        torch.manual_seed(42)

    def test_deep_gradient_flow(self):
        """Verify gradients flow through deep stack."""
        stack = DeltaResidualStack(
            dim=self.dim,
            num_blocks=self.num_blocks,
            beta_init_bias=-4.0,
        )

        h = torch.randn(self.batch_size, self.dim, requires_grad=True)
        h_out, _ = stack(h)
        loss = h_out.sum()
        loss.backward()

        # Check input gradients
        self.assertIsNotNone(h.grad)
        self.assertGreater(
            h.grad.abs().sum().item(), 0,
            "Gradient should flow through deep stack"
        )

        # Check each block has gradients
        for i, block in enumerate(stack.blocks):
            self.assertIsNotNone(
                block.k_proj.weight.grad,
                f"Block {i} k_proj should have gradient"
            )
            self.assertGreater(
                block.k_proj.weight.grad.abs().sum().item(), 0,
                f"Block {i} gradient should be non-zero"
            )

    def test_betas_tracking(self):
        """Verify beta values are tracked correctly."""
        stack = DeltaResidualStack(
            dim=self.dim,
            num_blocks=self.num_blocks,
        )

        h = torch.randn(self.batch_size, self.dim)
        _, betas = stack(h, return_all_betas=True)

        self.assertEqual(
            len(betas), self.num_blocks,
            f"Should return {self.num_blocks} beta values"
        )

        # All betas should be in valid range
        for beta in betas:
            self.assertTrue(0 <= beta <= 2, f"Beta {beta} out of range [0, 2]")

    def test_output_shape(self):
        """Check output shape is preserved through stack."""
        stack = DeltaResidualStack(dim=self.dim, num_blocks=self.num_blocks)
        h = torch.randn(self.batch_size, self.dim)

        h_out, _ = stack(h)
        self.assertEqual(h_out.shape, h.shape)


class TestDeltaLayerWrapper(unittest.TestCase):
    """Test backward compatibility wrapper."""

    def setUp(self):
        self.dim = 16
        self.batch_size = 4
        self.seq_len = 8
        torch.manual_seed(42)

    def test_interface_compatibility(self):
        """Verify wrapper has same interface as original DeltaLayer."""
        wrapper = DeltaLayerWrapper(self.dim)

        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        update = torch.randn(self.batch_size, self.seq_len, self.dim)

        out, beta = wrapper(x, update)

        self.assertEqual(out.shape, x.shape)
        self.assertIsInstance(beta, float)

    def test_gradient_flow_with_update(self):
        """Check gradients flow with external update."""
        wrapper = DeltaLayerWrapper(self.dim)

        x = torch.randn(self.batch_size, self.seq_len, self.dim, requires_grad=True)
        update = torch.randn(self.batch_size, self.seq_len, self.dim, requires_grad=True)

        out, _ = wrapper(x, update)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(update.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0)


class TestHouseholderMatrix(unittest.TestCase):
    """Test Householder matrix computation."""

    def setUp(self):
        self.dim = 8
        torch.manual_seed(42)

    def test_householder_orthogonal(self):
        """Verify H is orthogonal: H^T H = I."""
        k = torch.randn(self.dim)
        H = compute_householder_matrix(k)

        I = torch.eye(self.dim)
        HtH = H.T @ H

        self.assertTrue(
            torch.allclose(HtH, I, atol=1e-5),
            "Householder matrix should be orthogonal"
        )

    def test_householder_symmetric(self):
        """Verify H is symmetric: H = H^T."""
        k = torch.randn(self.dim)
        H = compute_householder_matrix(k)

        self.assertTrue(
            torch.allclose(H, H.T, atol=1e-6),
            "Householder matrix should be symmetric"
        )

    def test_householder_involutory(self):
        """Verify H is involutory: H^2 = I."""
        k = torch.randn(self.dim)
        H = compute_householder_matrix(k)

        H2 = H @ H
        I = torch.eye(self.dim)

        self.assertTrue(
            torch.allclose(H2, I, atol=1e-5),
            "Householder matrix should be involutory (H^2 = I)"
        )

    def test_householder_determinant(self):
        """Verify det(H) = -1 (reflection)."""
        k = torch.randn(self.dim)
        H = compute_householder_matrix(k)

        det = torch.det(H)

        self.assertTrue(
            torch.allclose(det, torch.tensor(-1.0), atol=1e-5),
            f"Householder det should be -1, got {det.item()}"
        )


class TestDeltaUpdateFormula(unittest.TestCase):
    """Test the Delta update formula (Eq 2.5 / 3.7)."""

    def setUp(self):
        self.dim = 8
        torch.manual_seed(42)

    def test_update_formula_manual(self):
        """Manually verify h_next = h + beta * k * (v - k^T h)."""
        block = DeltaResidualBlock(self.dim, d_v=1)
        h = torch.randn(1, self.dim)

        # Get components
        k = block.compute_k(h)
        beta = block.compute_beta(h)
        v = block.compute_v(h)

        # Manual computation
        k_dot_h = (k * h).sum(dim=-1, keepdim=True)
        delta = v - k_dot_h
        h_next_manual = h + beta * k * delta

        # Block computation
        h_next_block, _ = block(h)

        self.assertTrue(
            torch.allclose(h_next_block, h_next_manual, atol=1e-5),
            f"Block output should match manual formula\n"
            f"Block: {h_next_block}\nManual: {h_next_manual}"
        )

    def test_erasure_along_k(self):
        """When v=0 and beta=1, k component should be erased."""
        block = DeltaResidualBlock(self.dim)
        h = torch.randn(1, self.dim)

        # Force beta=1, v=0
        with torch.no_grad():
            block.beta_proj.weight.fill_(0)
            block.beta_proj.bias.fill_(0)  # 2*sigmoid(0)=1
            block.v_proj.weight.fill_(0)
            block.v_proj.bias.fill_(0)

        k = block.compute_k(h)
        h_next, _ = block(h)

        # h_next should be orthogonal to k (k component erased)
        dot = (h_next * k).sum()
        self.assertTrue(
            torch.allclose(dot, torch.tensor(0.0), atol=1e-4),
            f"h_next should be orthogonal to k, dot product: {dot.item()}"
        )

    def test_reflection_along_k(self):
        """When v=0 and beta=2, k component should be reflected."""
        block = DeltaResidualBlock(self.dim)
        h = torch.randn(1, self.dim)

        # Force beta=2, v=0
        with torch.no_grad():
            block.beta_proj.weight.fill_(0)
            block.beta_proj.bias.fill_(10)  # 2*sigmoid(10)~2
            block.v_proj.weight.fill_(0)
            block.v_proj.bias.fill_(0)

        k = block.compute_k(h)
        k_dot_h_before = (k * h).sum()

        h_next, _ = block(h)
        k_dot_h_after = (k * h_next).sum()

        # Reflection: k component should be negated
        self.assertTrue(
            torch.allclose(k_dot_h_before, -k_dot_h_after, atol=1e-2),
            f"k component should be reflected: before={k_dot_h_before.item():.4f}, "
            f"after={k_dot_h_after.item():.4f}"
        )


if __name__ == '__main__':
    unittest.main()
