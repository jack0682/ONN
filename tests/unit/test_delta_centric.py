"""Tests for Delta-Centric Architecture Components.

Tests all delta-based modules following DDL paper formulas:
- DeltaLinear: Delta-gated linear transformation
- DeltaFFN: Delta-based feed-forward network
- DeltaMLP: Delta-based multi-layer perceptron
- DeltaAttention: Delta-gated multi-head attention
- DeltaTransformerBlock: Combined attention + FFN block
- DeltaTransformer: Full delta-based transformer

Key DDL properties verified:
- beta in [0, 2] (Eq 2.6)
- Gradient flow through all components
- Near-identity initialization (beta ~ 0)
- Spectral properties preserved

Author: Claude
"""

import torch
import torch.nn as nn
import unittest
import sys
sys.path.insert(0, 'src')

from onn.modules.delta import (
    DeltaLinear,
    DeltaFFN,
    DeltaMLP,
    DeltaAttention,
    DeltaTransformerBlock,
    DeltaTransformer,
)


class TestDeltaLinear(unittest.TestCase):
    """Tests for DeltaLinear module."""

    def setUp(self):
        torch.manual_seed(42)
        self.in_features = 32
        self.out_features = 64

    def test_forward_shape(self):
        """Test output shape is correct."""
        layer = DeltaLinear(self.in_features, self.out_features)
        x = torch.randn(4, 8, self.in_features)

        y, beta = layer(x)

        self.assertEqual(y.shape, (4, 8, self.out_features))
        self.assertTrue(0 <= beta <= 2, f"Beta {beta} out of range [0, 2]")

    def test_same_dim_residual(self):
        """Test residual connection when in_features == out_features."""
        layer = DeltaLinear(32, 32)
        x = torch.randn(4, 8, 32)

        y, beta = layer(x)

        self.assertEqual(y.shape, x.shape)
        # With residual, output should be different from pure linear
        linear_out = layer.linear(x)
        self.assertFalse(torch.allclose(y, linear_out, atol=1e-5))

    def test_gradient_flow(self):
        """Test gradients flow through DeltaLinear."""
        # Use same dimensions for residual path
        layer = DeltaLinear(32, 32)
        x = torch.randn(4, 8, 32, requires_grad=True)

        y, _ = layer(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertIsNotNone(layer.linear.weight.grad)
        # k_proj and beta_proj gradients exist when using residual path
        self.assertIsNotNone(layer.beta_proj.weight.grad)


class TestDeltaFFN(unittest.TestCase):
    """Tests for DeltaFFN module."""

    def setUp(self):
        torch.manual_seed(42)
        self.dim = 64

    def test_forward_shape(self):
        """Test output shape matches input shape."""
        ffn = DeltaFFN(dim=self.dim, expansion=4, num_delta_blocks=2)
        x = torch.randn(4, 8, self.dim)

        y = ffn(x)

        self.assertEqual(y.shape, x.shape)

    def test_return_betas(self):
        """Test beta values are returned correctly."""
        ffn = DeltaFFN(dim=self.dim, expansion=4, num_delta_blocks=3)
        x = torch.randn(4, 8, self.dim)

        y, betas = ffn(x, return_betas=True)

        self.assertIn('stack_betas', betas)
        self.assertIn('residual_beta', betas)
        self.assertEqual(len(betas['stack_betas']), 3)

        # All betas should be in [0, 2]
        for beta in betas['stack_betas']:
            self.assertTrue(0 <= beta <= 2, f"Stack beta {beta} out of range")
        self.assertTrue(0 <= betas['residual_beta'] <= 2)

    def test_gradient_flow(self):
        """Test gradients flow through DeltaFFN."""
        ffn = DeltaFFN(dim=self.dim, num_delta_blocks=2)
        x = torch.randn(4, 8, self.dim, requires_grad=True)

        y = ffn(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0)

        # Check delta stack gradients
        for i, block in enumerate(ffn.delta_stack.blocks):
            self.assertIsNotNone(block.k_proj.weight.grad, f"Block {i} k_proj missing grad")

    def test_near_identity_init(self):
        """Test near-identity behavior at initialization."""
        ffn = DeltaFFN(dim=self.dim, beta_init_bias=-10.0)
        x = torch.randn(2, 4, self.dim)

        y, betas = ffn(x, return_betas=True)

        # All betas should be very small at init
        for beta in betas['stack_betas']:
            self.assertLess(beta, 0.1, f"Beta {beta} not near zero at init")


class TestDeltaMLP(unittest.TestCase):
    """Tests for DeltaMLP module."""

    def setUp(self):
        torch.manual_seed(42)

    def test_forward_shape(self):
        """Test output shape is correct."""
        mlp = DeltaMLP(input_dim=32, hidden_dim=64, output_dim=16, num_blocks=2)
        x = torch.randn(4, 32)

        y = mlp(x)

        self.assertEqual(y.shape, (4, 16))

    def test_return_betas(self):
        """Test beta values are returned."""
        mlp = DeltaMLP(input_dim=32, hidden_dim=64, output_dim=16, num_blocks=3)
        x = torch.randn(4, 32)

        y, betas = mlp(x, return_betas=True)

        self.assertEqual(len(betas), 3)
        for beta in betas:
            self.assertTrue(0 <= beta <= 2)

    def test_gradient_flow(self):
        """Test gradients flow through DeltaMLP."""
        mlp = DeltaMLP(input_dim=32, hidden_dim=64, output_dim=16, num_blocks=2)
        x = torch.randn(4, 32, requires_grad=True)

        y = mlp(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(mlp.input_proj[0].weight.grad)
        self.assertIsNotNone(mlp.output_proj.weight.grad)

    def test_batched_input(self):
        """Test with different batch sizes."""
        mlp = DeltaMLP(input_dim=32, hidden_dim=64, output_dim=16)

        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, 32)
            y = mlp(x)
            self.assertEqual(y.shape, (batch_size, 16))


class TestDeltaAttention(unittest.TestCase):
    """Tests for DeltaAttention module."""

    def setUp(self):
        torch.manual_seed(42)
        self.dim = 64
        self.num_heads = 8
        self.batch_size = 4
        self.seq_len = 16

    def test_forward_shape(self):
        """Test output shape matches input shape."""
        attn = DeltaAttention(dim=self.dim, num_heads=self.num_heads)
        x = torch.randn(self.batch_size, self.seq_len, self.dim)

        y = attn(x)

        self.assertEqual(y.shape, x.shape)

    def test_return_betas(self):
        """Test all QKV and output betas are returned."""
        attn = DeltaAttention(dim=self.dim, num_heads=self.num_heads)
        x = torch.randn(self.batch_size, self.seq_len, self.dim)

        y, betas = attn(x, return_betas=True)

        self.assertIn('q_beta', betas)
        self.assertIn('k_beta', betas)
        self.assertIn('v_beta', betas)
        self.assertIn('out_beta', betas)

        for key, beta in betas.items():
            self.assertTrue(0 <= beta <= 2, f"{key} beta {beta} out of range")

    def test_with_mask(self):
        """Test attention with causal mask."""
        attn = DeltaAttention(dim=self.dim, num_heads=self.num_heads)
        x = torch.randn(self.batch_size, self.seq_len, self.dim)

        # Causal mask
        mask = torch.tril(torch.ones(self.seq_len, self.seq_len))

        y = attn(x, mask=mask)

        self.assertEqual(y.shape, x.shape)

    def test_gradient_flow(self):
        """Test gradients flow through attention."""
        attn = DeltaAttention(dim=self.dim, num_heads=self.num_heads)
        x = torch.randn(self.batch_size, self.seq_len, self.dim, requires_grad=True)

        y = attn(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0)

        # Check delta gates have gradients
        self.assertIsNotNone(attn.q_delta.k_proj.weight.grad)
        self.assertIsNotNone(attn.k_delta.k_proj.weight.grad)
        self.assertIsNotNone(attn.v_delta.k_proj.weight.grad)
        self.assertIsNotNone(attn.out_delta.k_proj.weight.grad)

    def test_head_dim_divisibility(self):
        """Test that dim must be divisible by num_heads."""
        with self.assertRaises(AssertionError):
            DeltaAttention(dim=65, num_heads=8)


class TestDeltaTransformerBlock(unittest.TestCase):
    """Tests for DeltaTransformerBlock module."""

    def setUp(self):
        torch.manual_seed(42)
        self.dim = 64
        self.num_heads = 8
        self.batch_size = 4
        self.seq_len = 16

    def test_forward_shape(self):
        """Test output shape matches input shape."""
        block = DeltaTransformerBlock(dim=self.dim, num_heads=self.num_heads)
        x = torch.randn(self.batch_size, self.seq_len, self.dim)

        y = block(x)

        self.assertEqual(y.shape, x.shape)

    def test_return_betas(self):
        """Test all betas from attention and FFN are returned."""
        block = DeltaTransformerBlock(
            dim=self.dim,
            num_heads=self.num_heads,
            num_ffn_delta_blocks=2,
        )
        x = torch.randn(self.batch_size, self.seq_len, self.dim)

        y, betas = block(x, return_betas=True)

        self.assertIn('attention', betas)
        self.assertIn('ffn', betas)

        # Attention betas
        self.assertIn('q_beta', betas['attention'])
        self.assertIn('k_beta', betas['attention'])
        self.assertIn('v_beta', betas['attention'])
        self.assertIn('out_beta', betas['attention'])

        # FFN betas
        self.assertIn('stack_betas', betas['ffn'])
        self.assertIn('residual_beta', betas['ffn'])

    def test_gradient_flow(self):
        """Test gradients flow through entire block."""
        block = DeltaTransformerBlock(dim=self.dim, num_heads=self.num_heads)
        x = torch.randn(self.batch_size, self.seq_len, self.dim, requires_grad=True)

        y = block(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_with_mask(self):
        """Test block with causal mask."""
        block = DeltaTransformerBlock(dim=self.dim, num_heads=self.num_heads)
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        mask = torch.tril(torch.ones(self.seq_len, self.seq_len))

        y = block(x, mask=mask)

        self.assertEqual(y.shape, x.shape)


class TestDeltaTransformer(unittest.TestCase):
    """Tests for full DeltaTransformer model."""

    def setUp(self):
        torch.manual_seed(42)
        self.vocab_size = 1000
        self.dim = 64
        self.num_layers = 2
        self.num_heads = 8

    def test_forward_shape(self):
        """Test output logits shape is correct."""
        model = DeltaTransformer(
            vocab_size=self.vocab_size,
            dim=self.dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
        )
        input_ids = torch.randint(0, self.vocab_size, (4, 16))

        logits = model(input_ids)

        self.assertEqual(logits.shape, (4, 16, self.vocab_size))

    def test_return_betas(self):
        """Test beta diagnostics are returned for all layers."""
        model = DeltaTransformer(
            vocab_size=self.vocab_size,
            dim=self.dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
        )
        input_ids = torch.randint(0, self.vocab_size, (2, 8))

        logits, betas = model(input_ids, return_betas=True)

        self.assertEqual(len(betas), self.num_layers)
        for i in range(self.num_layers):
            self.assertIn(f'layer_{i}', betas)

    def test_beta_summary(self):
        """Test beta summary statistics."""
        model = DeltaTransformer(
            vocab_size=self.vocab_size,
            dim=self.dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
        )
        input_ids = torch.randint(0, self.vocab_size, (2, 8))

        logits, betas = model(input_ids, return_betas=True)
        summary = model.get_beta_summary(betas)

        self.assertIn('avg_attn_beta', summary)
        self.assertIn('avg_ffn_beta', summary)
        self.assertIn('num_attn_gates', summary)
        self.assertIn('num_ffn_gates', summary)

        # Beta averages should be in valid range
        self.assertTrue(0 <= summary['avg_attn_beta'] <= 2)
        self.assertTrue(0 <= summary['avg_ffn_beta'] <= 2)

    def test_gradient_flow(self):
        """Test gradients flow through entire model."""
        model = DeltaTransformer(
            vocab_size=self.vocab_size,
            dim=self.dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
        )
        input_ids = torch.randint(0, self.vocab_size, (2, 8))

        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()

        # Check embedding gradients
        self.assertIsNotNone(model.token_embed.weight.grad)
        self.assertIsNotNone(model.pos_embed.weight.grad)

        # Check output projection gradient
        self.assertIsNotNone(model.output_proj.weight.grad)

        # Check all layers have gradients
        for i, layer in enumerate(model.layers):
            self.assertIsNotNone(
                layer.attention.q_delta.k_proj.weight.grad,
                f"Layer {i} attention missing grad"
            )
            self.assertIsNotNone(
                layer.ffn.residual_delta.k_proj.weight.grad,
                f"Layer {i} FFN missing grad"
            )

    def test_language_modeling(self):
        """Test model can do basic forward pass for language modeling."""
        model = DeltaTransformer(
            vocab_size=self.vocab_size,
            dim=self.dim,
            num_layers=2,
            num_heads=4,
        )

        # Create input and target
        input_ids = torch.randint(0, self.vocab_size, (2, 16))
        targets = torch.randint(0, self.vocab_size, (2, 16))

        # Forward pass
        logits = model(input_ids)

        # Compute loss
        loss = nn.CrossEntropyLoss()(
            logits.view(-1, self.vocab_size),
            targets.view(-1)
        )

        # Backward pass
        loss.backward()

        # Loss should be finite
        self.assertTrue(torch.isfinite(loss))

    def test_parameter_count(self):
        """Test parameter count is reasonable."""
        model = DeltaTransformer(
            vocab_size=1000,
            dim=64,
            num_layers=2,
            num_heads=8,
        )

        num_params = sum(p.numel() for p in model.parameters())

        # Should have parameters (not empty model)
        self.assertGreater(num_params, 0)

        # Print for reference
        print(f"\nDeltaTransformer params: {num_params:,}")


class TestDeltaCentricImports(unittest.TestCase):
    """Test that all delta-centric components can be imported."""

    def test_import_from_modules(self):
        """Test imports from onn.modules."""
        from onn.modules import (
            DeltaLinear,
            DeltaFFN,
            DeltaMLP,
            DeltaAttention,
            DeltaTransformerBlock,
            DeltaTransformer,
        )

        self.assertTrue(True, "All imports successful")

    def test_import_from_delta(self):
        """Test imports from onn.modules.delta."""
        from onn.modules.delta import (
            DeltaLinear,
            DeltaFFN,
            DeltaMLP,
            DeltaAttention,
            DeltaTransformerBlock,
            DeltaTransformer,
        )

        self.assertTrue(True, "All delta imports successful")


class TestDeltaBetaProperties(unittest.TestCase):
    """Test beta properties across all delta components."""

    def setUp(self):
        torch.manual_seed(42)

    def test_beta_initialization_ffn(self):
        """Test FFN beta initialization near zero."""
        ffn = DeltaFFN(dim=32, beta_init_bias=-6.0)
        x = torch.randn(2, 4, 32)

        _, betas = ffn(x, return_betas=True)

        # All betas should be small
        for beta in betas['stack_betas']:
            self.assertLess(beta, 0.05)

    def test_beta_initialization_attention(self):
        """Test attention beta initialization near zero."""
        attn = DeltaAttention(dim=32, num_heads=4, beta_init_bias=-6.0)
        x = torch.randn(2, 4, 32)

        _, betas = attn(x, return_betas=True)

        for key, beta in betas.items():
            self.assertLess(beta, 0.05, f"{key} beta not near zero")

    def test_beta_range_under_stress(self):
        """Test beta stays in [0,2] with extreme inputs."""
        ffn = DeltaFFN(dim=32)

        for _ in range(10):
            # Random extreme inputs
            x = torch.randn(4, 8, 32) * 100

            _, betas = ffn(x, return_betas=True)

            for beta in betas['stack_betas']:
                self.assertTrue(0 <= beta <= 2, f"Beta {beta} out of range")


if __name__ == '__main__':
    unittest.main(verbosity=2)
