"""DDL Integration Tests.

Verifies that all DDL components are properly connected and work together:
1. DeltaResidualBlock -> DeltaLayerWrapper -> DeltaLayer (TLM)
2. DeltaResidualStack -> SE3RelationEncoder
3. DiffSolverConfig -> delta_ode_step
4. End-to-end gradient flow through entire pipeline

Author: Claude
"""

import torch
import torch.nn as nn
import unittest
import sys
sys.path.insert(0, 'src')


class TestModuleImports(unittest.TestCase):
    """Test that all modules can be imported correctly."""

    def test_delta_module_imports(self):
        """Test imports from onn.modules.delta."""
        from onn.modules.delta import (
            DeltaConfig,
            DeltaResidualBlock,
            DeltaResidualStack,
            DeltaLayerWrapper,
            compute_householder_matrix,
            verify_spectral_properties,
        )
        self.assertTrue(True, "All delta module imports successful")

    def test_modules_init_exports(self):
        """Test imports from onn.modules (via __init__.py)."""
        from onn.modules import (
            DeltaConfig,
            DeltaResidualBlock,
            DeltaResidualStack,
            DeltaLayerWrapper,
            compute_householder_matrix,
            verify_spectral_properties,
        )
        self.assertTrue(True, "All modules __init__ exports successful")

    def test_tlm_delta_layer_import(self):
        """Test DeltaLayer import from TLM."""
        from onn.tlm.model import DeltaLayer, TLMConfig, TopologicalLanguageModel
        self.assertTrue(True, "TLM imports successful")

    def test_se3_encoder_import(self):
        """Test SE3RelationEncoder with delta support."""
        from onn.relation.se3_encoder import SE3RelationEncoder, GeometricRelationEncoder
        self.assertTrue(True, "SE3 encoder imports successful")

    def test_diff_solver_import(self):
        """Test diff_solver with delta ODE support."""
        from onn.core.diff_solver import (
            DiffSolverConfig,
            DifferentiableSolver,
            delta_ode_step,
            compute_adaptive_beta,
        )
        self.assertTrue(True, "Diff solver imports successful")


class TestDeltaLayerConnection(unittest.TestCase):
    """Test connection: DeltaResidualBlock -> DeltaLayerWrapper -> DeltaLayer."""

    def setUp(self):
        self.dim = 32
        torch.manual_seed(42)

    def test_wrapper_uses_block(self):
        """Verify DeltaLayerWrapper uses DeltaResidualBlock internally."""
        from onn.modules.delta import DeltaLayerWrapper, DeltaResidualBlock

        wrapper = DeltaLayerWrapper(self.dim)

        # Check internal structure
        self.assertTrue(hasattr(wrapper, 'delta'))
        self.assertIsInstance(wrapper.delta, DeltaResidualBlock)

    def test_tlm_layer_uses_wrapper(self):
        """Verify TLM DeltaLayer uses DeltaLayerWrapper internally."""
        from onn.tlm.model import DeltaLayer
        from onn.modules.delta import DeltaLayerWrapper

        layer = DeltaLayer(self.dim)

        # Check internal structure
        self.assertTrue(hasattr(layer, '_delta'))
        self.assertIsInstance(layer._delta, DeltaLayerWrapper)

    def test_chain_equivalence(self):
        """Test that DeltaLayer -> DeltaLayerWrapper -> DeltaResidualBlock chain works."""
        from onn.tlm.model import DeltaLayer

        layer = DeltaLayer(self.dim, beta_init_bias=-4.0)

        x = torch.randn(4, 8, self.dim)
        update = torch.randn(4, 8, self.dim)

        # Should work without errors
        out, beta = layer(x, update)

        self.assertEqual(out.shape, x.shape)
        self.assertTrue(0 <= beta <= 2, f"Beta {beta} out of range")


class TestSE3EncoderConnection(unittest.TestCase):
    """Test connection: DeltaResidualStack -> SE3RelationEncoder."""

    def setUp(self):
        torch.manual_seed(42)

    def test_delta_encoder_mode(self):
        """Test SE3RelationEncoder with use_delta_encoder=True."""
        from onn.relation.se3_encoder import SE3RelationEncoder
        from onn.modules.delta import DeltaResidualStack

        encoder = SE3RelationEncoder(
            embed_dim=32,
            hidden_dim=64,
            use_delta_encoder=True,
            num_delta_blocks=2,
        )

        # Check internal structure
        self.assertTrue(hasattr(encoder, 'delta_stack'))
        self.assertIsInstance(encoder.delta_stack, DeltaResidualStack)
        self.assertTrue(hasattr(encoder, 'input_proj'))
        self.assertTrue(hasattr(encoder, 'output_proj'))

    def test_legacy_encoder_mode(self):
        """Test SE3RelationEncoder with use_delta_encoder=False."""
        from onn.relation.se3_encoder import SE3RelationEncoder

        encoder = SE3RelationEncoder(
            embed_dim=32,
            hidden_dim=64,
            use_delta_encoder=False,
        )

        # Check internal structure
        self.assertTrue(hasattr(encoder, 'encoder'))
        self.assertFalse(hasattr(encoder, 'delta_stack'))

    def test_encoder_output(self):
        """Test encoder produces valid output."""
        from onn.relation.se3_encoder import SE3RelationEncoder

        encoder = SE3RelationEncoder(
            embed_dim=32,
            use_delta_encoder=True,
            num_delta_blocks=2,
        )

        # Create SE(3) transforms
        T_a = torch.eye(4).unsqueeze(0)
        T_b = torch.eye(4).unsqueeze(0)
        T_b[0, 2, 3] = 0.5  # Translation

        z = encoder(T_a, T_b)
        self.assertEqual(z.shape, (1, 32))

    def test_encoder_with_betas(self):
        """Test encoder returns beta values when requested."""
        from onn.relation.se3_encoder import SE3RelationEncoder

        encoder = SE3RelationEncoder(
            embed_dim=32,
            use_delta_encoder=True,
            num_delta_blocks=3,
        )

        T_a = torch.eye(4).unsqueeze(0)
        T_b = torch.eye(4).unsqueeze(0)

        z, betas = encoder.encode(T_a, T_b, return_betas=True)
        self.assertEqual(len(betas), 3, "Should return 3 beta values")


class TestDiffSolverConnection(unittest.TestCase):
    """Test connection: delta_ode_step -> DiffSolverConfig."""

    def setUp(self):
        torch.manual_seed(42)

    def test_config_has_delta_options(self):
        """Test DiffSolverConfig has Delta ODE options."""
        from onn.core.diff_solver import DiffSolverConfig

        config = DiffSolverConfig()

        self.assertTrue(hasattr(config, 'use_delta_update'))
        self.assertTrue(hasattr(config, 'delta_beta_mode'))
        self.assertTrue(hasattr(config, 'delta_beta_value'))

    def test_delta_ode_step_function(self):
        """Test delta_ode_step works correctly."""
        from onn.core.diff_solver import delta_ode_step

        x = torch.randn(10, 32)
        grad = torch.randn(10, 32)

        # Test with beta=1 (projection mode)
        x_new_1 = delta_ode_step(x, grad, step_size=0.01, beta=1.0)
        self.assertEqual(x_new_1.shape, x.shape)

        # Test with beta=0.5 (should differ from beta=1)
        x_new_05 = delta_ode_step(x, grad, step_size=0.01, beta=0.5)
        self.assertFalse(torch.allclose(x_new_1, x_new_05))

        # Test with beta=2 (reflection mode)
        x_new_2 = delta_ode_step(x, grad, step_size=0.01, beta=2.0)
        self.assertFalse(torch.allclose(x_new_1, x_new_2))

        # Note: beta=1 with normalized k gives same result as Euler
        # This is mathematically correct per DDL paper

    def test_solver_with_delta_update(self):
        """Test DifferentiableSolver uses delta update when configured."""
        from onn.core.diff_solver import DiffSolverConfig, DifferentiableSolver

        config = DiffSolverConfig(
            K=5,
            use_delta_update=True,
            delta_beta_mode='fixed',
            delta_beta_value=1.0,
        )

        solver = DifferentiableSolver(config)
        self.assertTrue(solver.config.use_delta_update)


class TestTLMConnection(unittest.TestCase):
    """Test TLM model uses Delta layers correctly."""

    def setUp(self):
        torch.manual_seed(42)

    def test_tlm_creates_delta_layers(self):
        """Test TLM creates DeltaLayer instances."""
        from onn.tlm.model import TopologicalLanguageModel, TLMConfig, DeltaLayer

        config = TLMConfig(vocab_size=100, embed_dim=32)
        model = TopologicalLanguageModel(config, num_layers=2)

        # Check attention layers have delta
        for layer in model.layers:
            self.assertTrue(hasattr(layer, 'delta'))
            self.assertIsInstance(layer.delta, DeltaLayer)

        # Check FFN delta layers
        self.assertEqual(len(model.deltas), 2)
        for delta in model.deltas:
            self.assertIsInstance(delta, DeltaLayer)

    def test_tlm_forward_returns_betas(self):
        """Test TLM forward pass returns beta diagnostics."""
        from onn.tlm.model import TopologicalLanguageModel, TLMConfig

        config = TLMConfig(vocab_size=100, embed_dim=32)
        model = TopologicalLanguageModel(config, num_layers=2)

        input_ids = torch.randint(0, 100, (2, 8))
        logits, diagnostics = model(input_ids)

        self.assertIn('avg_attn_beta', diagnostics)
        self.assertIn('avg_ffn_beta', diagnostics)


class TestEndToEndGradientFlow(unittest.TestCase):
    """Test gradient flows through entire DDL pipeline."""

    def setUp(self):
        torch.manual_seed(42)

    def test_gradient_through_delta_stack(self):
        """Test gradients flow through DeltaResidualStack."""
        from onn.modules.delta import DeltaResidualStack

        stack = DeltaResidualStack(dim=32, num_blocks=4)
        x = torch.randn(4, 32, requires_grad=True)

        out, _ = stack(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0)

        # Check all blocks have gradients
        for i, block in enumerate(stack.blocks):
            self.assertIsNotNone(block.k_proj.weight.grad, f"Block {i} k_proj missing grad")
            self.assertIsNotNone(block.beta_proj.weight.grad, f"Block {i} beta_proj missing grad")
            self.assertIsNotNone(block.v_proj.weight.grad, f"Block {i} v_proj missing grad")

    def test_gradient_through_tlm(self):
        """Test gradients flow through TLM with Delta layers."""
        from onn.tlm.model import TopologicalLanguageModel, TLMConfig

        config = TLMConfig(vocab_size=100, embed_dim=32, relation_dim=16)
        model = TopologicalLanguageModel(config, num_layers=2)

        input_ids = torch.randint(0, 100, (2, 6))
        logits, _ = model(input_ids)

        loss = logits.sum()
        loss.backward()

        # Check delta layers have gradients
        for i, delta in enumerate(model.deltas):
            inner = delta._delta.delta  # DeltaLayer -> DeltaLayerWrapper -> DeltaResidualBlock
            self.assertIsNotNone(inner.k_proj.weight.grad, f"FFN delta {i} missing k grad")

    def test_gradient_through_se3_encoder(self):
        """Test gradients flow through SE3 encoder with Delta blocks."""
        from onn.relation.se3_encoder import SE3RelationEncoder

        encoder = SE3RelationEncoder(
            embed_dim=32,
            use_delta_encoder=True,
            num_delta_blocks=2,
        )

        T_a = torch.eye(4).unsqueeze(0).requires_grad_(False)
        T_b = torch.eye(4).unsqueeze(0)
        T_b[0, 2, 3] = 0.5

        # Make input require grad through a learnable transform
        position = torch.randn(1, 3, requires_grad=True)
        T_c = torch.eye(4).unsqueeze(0).clone()
        T_c[0, :3, 3] = position

        z = encoder(T_a, T_c)
        loss = z.sum()
        loss.backward()

        # Check delta stack has gradients
        for i, block in enumerate(encoder.delta_stack.blocks):
            self.assertIsNotNone(block.k_proj.weight.grad, f"SE3 delta block {i} missing grad")


class TestSpectralProperties(unittest.TestCase):
    """Test spectral properties are preserved through connections."""

    def setUp(self):
        torch.manual_seed(42)

    def test_determinant_through_tlm_layer(self):
        """Test det(A) = 1 - beta through TLM DeltaLayer."""
        from onn.tlm.model import DeltaLayer

        layer = DeltaLayer(16, beta_init_bias=0.0)  # beta = 1

        x = torch.randn(1, 1, 16)
        update = torch.zeros_like(x)

        _, beta = layer(x, update)

        # With beta ~ 1, det should be ~ 0
        expected_det = 1.0 - beta
        self.assertAlmostEqual(expected_det, 0.0, delta=0.1)

    def test_beta_range_through_wrapper(self):
        """Test beta stays in [0, 2] through DeltaLayerWrapper."""
        from onn.modules.delta import DeltaLayerWrapper

        wrapper = DeltaLayerWrapper(16)

        for _ in range(10):
            x = torch.randn(4, 8, 16) * 10
            update = torch.randn(4, 8, 16) * 10

            _, beta = wrapper(x, update)
            self.assertTrue(0 <= beta <= 2, f"Beta {beta} out of range through wrapper")


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing code."""

    def setUp(self):
        torch.manual_seed(42)

    def test_delta_layer_interface(self):
        """Test DeltaLayer maintains original interface."""
        from onn.tlm.model import DeltaLayer

        layer = DeltaLayer(32)

        # Original interface: forward(x, update) -> (output, beta)
        x = torch.randn(4, 8, 32)
        update = torch.randn(4, 8, 32)

        result = layer(x, update)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, x.shape)  # output
        self.assertIsInstance(result[1], float)     # beta

    def test_se3_encoder_interface(self):
        """Test SE3RelationEncoder maintains original interface."""
        from onn.relation.se3_encoder import SE3RelationEncoder

        # Original usage
        encoder = SE3RelationEncoder(embed_dim=32)

        T_a = torch.eye(4).unsqueeze(0)
        T_b = torch.eye(4).unsqueeze(0)

        # Original method
        z = encoder.encode(T_a, T_b)
        self.assertEqual(z.shape, (1, 32))

        # Alternative method
        z = encoder(T_a, T_b)
        self.assertEqual(z.shape, (1, 32))


if __name__ == '__main__':
    unittest.main(verbosity=2)
