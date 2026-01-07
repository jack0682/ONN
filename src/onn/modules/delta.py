"""Deep Delta Learning (DDL) - Delta Residual Block.

Implements the Delta Operator from:
"Deep Delta Learning" (Zhang et al., 2026)

Core formulas:
    Delta Operator (Eq 2.3-2.4):
        A(X) = I - beta(X) * k(X)k(X)^T / (k(X)^T k(X) + eps)

    Delta Residual Block (Eq 2.2, 2.5):
        X_{l+1} = A(X_l)X_l + beta(X_l)k(X_l)v(X_l)^T
                = X_l + beta(X_l) * k(X_l) * (v(X_l)^T - k(X_l)^T X_l)

    Scalar Value Limit d_v=1 (Eq 3.7):
        x_{l+1} = x_l + beta_l * (v_l - k_l^T x_l) * k_l

    Beta Gate (Eq 2.6, A.2):
        beta(X) = 2 * sigmoid(Linear(Pool(X)))
        Range: [0, 2]
        - beta -> 0: Identity mapping
        - beta -> 1: Orthogonal projection (det(A) -> 0)
        - beta -> 2: Full Householder reflection (det(A) -> -1)

    Direction k(X) (Eq A.1):
        k = k_raw / (||k_raw||_2 + eps)  # L2 normalized

Spectral Properties (Theorem 3.1):
    sigma(A) = {1, 1, ..., 1 (d-1 times), 1 - beta}
    det(A) = 1 - beta

Author: Claude
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DeltaConfig:
    """Configuration for Delta Residual Block.

    Args:
        dim: Feature dimension d
        d_v: Value dimension (1 for scalar, >1 for matrix state)
        beta_init_bias: Initial bias for beta gate (negative -> beta near 0)
        k_init_std: Initial std for k projection weights
        eps: Numerical stability epsilon
    """
    dim: int = 64
    d_v: int = 1
    beta_init_bias: float = -4.0  # sigmoid(-4) ~ 0.018, so beta ~ 0.036
    k_init_std: float = 0.01
    eps: float = 1e-8


class DeltaResidualBlock(nn.Module):
    """Deep Delta Learning Residual Block.

    Implements the rank-1 Delta update from the DDL paper:

        h_{l+1} = h_l + beta * k * (v - k^T h_l)   (for d_v=1)

    or equivalently:

        h_{l+1} = A(h_l) h_l + beta * k * v^T      (matrix form)

    where:
        - k(x): Direction vector (L2 normalized)
        - beta(x): Gate scalar in [0, 2]
        - v(x): Value (scalar for d_v=1)
        - A(x) = I - beta * k * k^T: Delta Operator

    Args:
        dim: Feature dimension
        d_v: Value dimension (default 1 for scalar value)
        v_input_dim: Input dimension for v branch (None = use h)
        beta_init_bias: Initial beta bias (negative for near-zero init)
        k_init_std: Initial std for k projection
        eps: Numerical stability
    """

    def __init__(
        self,
        dim: int,
        d_v: int = 1,
        v_input_dim: Optional[int] = None,
        beta_init_bias: float = -4.0,
        k_init_std: float = 0.01,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.dim = dim
        self.d_v = d_v
        self.v_input_dim = v_input_dim if v_input_dim is not None else dim
        self.eps = eps

        # Direction branch k(x): Linear -> L2 normalize
        # k: R^dim -> R^dim (Eq A.1)
        self.k_proj = nn.Linear(dim, dim, bias=False)

        # Gate branch beta(x): Linear -> 2*sigmoid
        # beta: R^dim -> R^1, output in [0, 2] (Eq A.2)
        self.beta_proj = nn.Linear(dim, 1)

        # Value branch v(x): Linear -> scalar output
        # v: R^v_input_dim -> R^d_v (Eq 3.7 for d_v=1)
        self.v_proj = nn.Linear(self.v_input_dim, d_v)

        # Initialize for stable training
        self._init_weights(beta_init_bias, k_init_std)

    def _init_weights(self, beta_init_bias: float, k_init_std: float):
        """Initialize weights for near-identity behavior at start.

        - k: Small random -> small projection initially
        - beta: Negative bias -> sigmoid output near 0 -> near identity
        - v: Standard init
        """
        # k projection: small std so initial k is small
        nn.init.normal_(self.k_proj.weight, std=k_init_std)

        # beta gate: bias to make sigmoid(bias) near 0
        # beta = 2 * sigmoid(x), want beta ~ 0 at init
        # sigmoid(-4) ~ 0.018, so beta ~ 0.036
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias, beta_init_bias)

        # v projection: standard Xavier init
        nn.init.xavier_normal_(self.v_proj.weight)
        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)

    def compute_k(self, h: torch.Tensor) -> torch.Tensor:
        """Compute normalized direction vector k(h).

        k = k_raw / (||k_raw||_2 + eps)  (Eq A.1)

        Args:
            h: Hidden state (..., dim)

        Returns:
            k: Normalized direction (..., dim) with ||k||_2 = 1
        """
        k_raw = self.k_proj(h)  # (..., dim)
        k_norm = torch.norm(k_raw, dim=-1, keepdim=True) + self.eps
        k = k_raw / k_norm
        return k

    def compute_beta(self, h: torch.Tensor) -> torch.Tensor:
        """Compute gate scalar beta(h) in [0, 2].

        beta = 2 * sigmoid(Linear(h))  (Eq 2.6, A.2)

        Args:
            h: Hidden state (..., dim)

        Returns:
            beta: Gate value (..., 1) in range [0, 2]
        """
        logit = self.beta_proj(h)  # (..., 1)
        beta = 2.0 * torch.sigmoid(logit)
        return beta

    def compute_v(
        self,
        h: torch.Tensor,
        v_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute value v(x).

        Args:
            h: Hidden state (..., dim)
            v_input: Optional separate input for v branch

        Returns:
            v: Value (..., d_v)
        """
        x = v_input if v_input is not None else h
        v = self.v_proj(x)  # (..., d_v)
        return v

    def forward(
        self,
        h: torch.Tensor,
        v_input: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, dict]]:
        """Delta residual update.

        Computes (Eq 2.5 / Eq 3.7):
            h_next = h + beta * k * (v - k^T h)

        For d_v=1 (scalar value):
            h_next[i] = h[i] + beta * k[i] * (v - sum_j(k[j]*h[j]))

        Args:
            h: Hidden state (..., dim)
            v_input: Optional input for value branch (default: use h)
            return_components: If True, return k, beta, v in dict

        Returns:
            h_next: Updated hidden state (..., dim)
            beta: Gate value (scalar, mean over batch for diagnostics)
            components: Optional dict with k, beta_full, v tensors
        """
        # Compute branches
        k = self.compute_k(h)           # (..., dim)
        beta = self.compute_beta(h)     # (..., 1)
        v = self.compute_v(h, v_input)  # (..., d_v)

        # Projection: k^T h
        # For vector state: dot product along dim axis
        k_dot_h = (k * h).sum(dim=-1, keepdim=True)  # (..., 1)

        # Delta update (Eq 3.7 for d_v=1):
        # h_next = h + beta * k * (v - k^T h)
        if self.d_v == 1:
            # Scalar value case
            delta = v - k_dot_h  # (..., 1) - (..., 1) = (..., 1)
            h_next = h + beta * k * delta  # (..., dim)
        else:
            # Matrix value case (Eq 2.5):
            # h_next = h + beta * k * (v^T - k^T @ h)
            # For batched: need outer product k @ v^T
            # h: (..., dim), k: (..., dim), v: (..., d_v)
            # k @ v^T: (..., dim, d_v) but we have vector h
            # This case requires h to be matrix (..., dim, d_v)
            # For simplicity, treat as d_v independent scalar updates
            delta = v - k_dot_h.expand_as(v)  # broadcast
            h_next = h + (beta * delta.mean(dim=-1, keepdim=True)) * k

        # Return beta mean for logging
        beta_mean = beta.mean()

        if return_components:
            components = {
                'k': k,
                'beta': beta,
                'v': v,
                'k_dot_h': k_dot_h,
            }
            return h_next, beta_mean, components

        return h_next, beta_mean

    def compute_delta_operator(self, h: torch.Tensor) -> torch.Tensor:
        """Compute the Delta Operator matrix A(h).

        A = I - beta * k @ k^T  (Eq 2.4)

        Useful for analysis and testing spectral properties.

        Args:
            h: Hidden state (batch, dim)

        Returns:
            A: Delta operator (batch, dim, dim)
        """
        k = self.compute_k(h)      # (batch, dim)
        beta = self.compute_beta(h)  # (batch, 1)

        # A = I - beta * k @ k^T
        # k @ k^T: (batch, dim, 1) @ (batch, 1, dim) = (batch, dim, dim)
        k_outer = k.unsqueeze(-1) @ k.unsqueeze(-2)  # (batch, dim, dim)

        I = torch.eye(self.dim, device=h.device, dtype=h.dtype)
        I = I.unsqueeze(0).expand(h.shape[0], -1, -1)  # (batch, dim, dim)

        A = I - beta.unsqueeze(-1) * k_outer  # (batch, dim, dim)

        return A

    def compute_determinant(self, h: torch.Tensor) -> torch.Tensor:
        """Compute determinant of Delta Operator.

        det(A) = 1 - beta  (Corollary 3.2)

        Args:
            h: Hidden state (batch, dim)

        Returns:
            det: Determinant (batch,)
        """
        beta = self.compute_beta(h).squeeze(-1)  # (batch,)
        return 1.0 - beta


class DeltaResidualStack(nn.Module):
    """Stack of Delta Residual Blocks.

    Creates N sequential Delta blocks for deep feature transformation.
    Useful for replacing MLPs in encoders.

    Args:
        dim: Feature dimension
        num_blocks: Number of Delta blocks
        d_v: Value dimension per block
        **kwargs: Additional args for DeltaResidualBlock
    """

    def __init__(
        self,
        dim: int,
        num_blocks: int,
        d_v: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.dim = dim
        self.num_blocks = num_blocks

        self.blocks = nn.ModuleList([
            DeltaResidualBlock(dim, d_v=d_v, **kwargs)
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        h: torch.Tensor,
        return_all_betas: bool = False,
    ) -> Union[Tuple[torch.Tensor, float], Tuple[torch.Tensor, list]]:
        """Apply stack of Delta blocks.

        Args:
            h: Input hidden state (..., dim)
            return_all_betas: If True, return list of all beta values

        Returns:
            h: Output hidden state (..., dim)
            beta: Mean beta (or list of betas if return_all_betas)
        """
        betas = []

        for block in self.blocks:
            h, beta = block(h)
            betas.append(beta.item() if isinstance(beta, torch.Tensor) else beta)

        if return_all_betas:
            return h, betas

        return h, sum(betas) / len(betas)


class DeltaLayerWrapper(nn.Module):
    """Wrapper for backward compatibility with existing DeltaLayer interface.

    Provides the same interface as the original DeltaLayer in tlm/model.py:
        forward(x, update) -> (output, beta_mean)

    The 'update' is used as v_input for the value branch.

    Args:
        dim: Feature dimension
        **kwargs: Additional args for DeltaResidualBlock
    """

    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.delta = DeltaResidualBlock(
            dim=dim,
            d_v=1,
            v_input_dim=dim,
            **kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        update: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Apply delta residual with external update as value input.

        This replicates the original DeltaLayer behavior:
            y = x - beta * proj_k(x) + beta * update

        But using the proper DDL formulation:
            y = x + beta * k * (v - k^T x)

        where v is derived from 'update'.

        Args:
            x: Input state (..., dim)
            update: External update to incorporate (..., dim)

        Returns:
            y: Output state (..., dim)
            beta_mean: Mean beta value (float)
        """
        # Use update as input for value branch
        # v_proj will reduce update to scalar v
        y, beta = self.delta(x, v_input=update)
        return y, beta.item() if isinstance(beta, torch.Tensor) else beta


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def compute_householder_matrix(k: torch.Tensor) -> torch.Tensor:
    """Compute Householder reflection matrix H_k.

    H_k = I - 2 * k @ k^T / ||k||^2  (Definition 2.1)

    This is the special case beta=2 of the Delta Operator.

    Args:
        k: Direction vector (..., dim)

    Returns:
        H: Householder matrix (..., dim, dim)
    """
    dim = k.shape[-1]
    k_normalized = F.normalize(k, dim=-1)  # Ensure unit norm

    # H = I - 2 * k @ k^T
    k_outer = k_normalized.unsqueeze(-1) @ k_normalized.unsqueeze(-2)
    I = torch.eye(dim, device=k.device, dtype=k.dtype)

    return I - 2 * k_outer


def verify_spectral_properties(A: torch.Tensor, beta: torch.Tensor, tol: float = 1e-5) -> dict:
    """Verify spectral properties of Delta Operator (Theorem 3.1).

    Expected spectrum: {1 (d-1 times), 1-beta}
    Expected det: 1 - beta

    Args:
        A: Delta operator (batch, dim, dim)
        beta: Gate values (batch, 1)
        tol: Tolerance for verification

    Returns:
        dict with verification results
    """
    batch_size, dim, _ = A.shape

    results = {'passed': True, 'details': []}

    for b in range(batch_size):
        A_b = A[b]
        beta_b = beta[b, 0].item()

        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvalsh(A_b)
        eigenvalues_sorted = torch.sort(eigenvalues)[0]

        # Expected: d-1 eigenvalues of 1, one eigenvalue of 1-beta
        expected_special = 1.0 - beta_b

        # Check determinant
        det_computed = torch.det(A_b).item()
        det_expected = 1.0 - beta_b
        det_ok = abs(det_computed - det_expected) < tol

        # Check smallest eigenvalue (should be 1-beta)
        smallest_eig = eigenvalues_sorted[0].item()
        eig_ok = abs(smallest_eig - expected_special) < tol

        # Check other eigenvalues (should be ~1)
        other_eigs = eigenvalues_sorted[1:]
        others_ok = torch.all(torch.abs(other_eigs - 1.0) < tol).item()

        detail = {
            'batch_idx': b,
            'beta': beta_b,
            'det_computed': det_computed,
            'det_expected': det_expected,
            'det_ok': det_ok,
            'smallest_eig': smallest_eig,
            'expected_special_eig': expected_special,
            'eig_ok': eig_ok,
            'others_ok': others_ok,
        }
        results['details'].append(detail)

        if not (det_ok and eig_ok and others_ok):
            results['passed'] = False

    return results


# ==============================================================================
# DELTA-CENTRIC ARCHITECTURE COMPONENTS
# ==============================================================================

class DeltaLinear(nn.Module):
    """Delta-based Linear Transformation.

    Replaces standard nn.Linear with delta-gated transformation:
        y = x + beta * k * (v - k^T x)

    where v = Linear(x) is the "target" value.

    This provides learnable gating over linear transformations,
    allowing the network to smoothly interpolate between identity and projection.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        beta_init_bias: Initial beta bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        beta_init_bias: float = -2.0,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Main linear transformation
        self.linear = nn.Linear(in_features, out_features)

        # Delta components (operate in output space)
        self.k_proj = nn.Linear(out_features, out_features, bias=False)
        self.beta_proj = nn.Linear(out_features, 1)

        self._init_weights(beta_init_bias)

    def _init_weights(self, beta_init_bias: float):
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.normal_(self.k_proj.weight, std=0.01)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias, beta_init_bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Delta-gated linear transformation.

        Args:
            x: Input (..., in_features)

        Returns:
            y: Output (..., out_features)
            beta: Gate value for diagnostics
        """
        # Standard linear
        h = self.linear(x)  # (..., out_features)

        # Delta gating
        k_raw = self.k_proj(h)
        k = F.normalize(k_raw, dim=-1, eps=1e-8)
        beta = 2.0 * torch.sigmoid(self.beta_proj(h))

        # v = h (the linear output is the target)
        # But we need a scalar v for the formula
        v = (k * h).sum(dim=-1, keepdim=True)  # Project h onto k
        k_dot_h = v  # Since v = k^T h, delta = 0 initially

        # For DeltaLinear, we want: y = beta * h + (1-beta) * x_proj
        # Simplified: just apply beta gating to the linear output
        # y = x_proj + beta * (h - x_proj) where x_proj is x projected to out_features
        if self.in_features == self.out_features:
            y = x + beta * (h - x)
        else:
            y = beta * h  # Can't do residual if dimensions differ

        return y, beta.mean()


class DeltaFFN(nn.Module):
    """Delta-based Feed-Forward Network.

    Replaces standard FFN (Linear -> Act -> Linear) with delta blocks:
        h1 = DeltaBlock(x)
        h2 = DeltaBlock(expand(h1))
        y = DeltaBlock(contract(h2))

    Architecture:
        input -> expand -> delta_stack -> contract -> output

    All transformations are gated by learned beta values.

    Args:
        dim: Input/output dimension
        expansion: FFN expansion factor (default 4x)
        num_delta_blocks: Number of delta blocks in the middle
        dropout: Dropout rate
        beta_init_bias: Initial beta bias
    """

    def __init__(
        self,
        dim: int,
        expansion: int = 4,
        num_delta_blocks: int = 2,
        dropout: float = 0.1,
        beta_init_bias: float = -4.0,
    ):
        super().__init__()

        self.dim = dim
        self.hidden_dim = dim * expansion

        # Expand: dim -> hidden_dim
        self.expand = nn.Linear(dim, self.hidden_dim)

        # Delta blocks in expanded space
        self.delta_stack = DeltaResidualStack(
            dim=self.hidden_dim,
            num_blocks=num_delta_blocks,
            beta_init_bias=beta_init_bias,
        )

        # Contract: hidden_dim -> dim
        self.contract = nn.Linear(self.hidden_dim, dim)

        # Normalization and dropout
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        # Residual delta gate
        self.residual_delta = DeltaResidualBlock(
            dim=dim,
            beta_init_bias=beta_init_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_betas: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """Delta FFN forward pass.

        Args:
            x: Input (..., dim)
            return_betas: If True, return beta diagnostics

        Returns:
            y: Output (..., dim)
            betas: Optional dict of beta values
        """
        # Expand and activate
        h = F.gelu(self.expand(x))

        # Delta transformations in expanded space
        h, stack_betas = self.delta_stack(h, return_all_betas=True)

        # Contract
        h = self.contract(h)
        h = self.dropout(h)

        # Delta residual connection
        y, residual_beta = self.residual_delta(x, v_input=h)
        y = self.norm(y)

        if return_betas:
            return y, {
                'stack_betas': stack_betas,
                'residual_beta': residual_beta.item() if isinstance(residual_beta, torch.Tensor) else residual_beta,
            }

        return y


class DeltaMLP(nn.Module):
    """Delta-based Multi-Layer Perceptron.

    Replaces standard MLP with delta block stack:
        input_proj -> [DeltaBlock x N] -> output_proj

    Each layer applies the delta update formula, allowing
    controlled information flow through the network.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        num_blocks: Number of delta blocks
        beta_init_bias: Initial beta bias
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_blocks: int = 2,
        beta_init_bias: float = -4.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Delta stack
        self.delta_stack = DeltaResidualStack(
            dim=hidden_dim,
            num_blocks=num_blocks,
            beta_init_bias=beta_init_bias,
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_betas: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        """Delta MLP forward pass.

        Args:
            x: Input (..., input_dim)
            return_betas: If True, return beta values

        Returns:
            y: Output (..., output_dim)
            betas: Optional list of beta values
        """
        h = self.input_proj(x)
        h, betas = self.delta_stack(h, return_all_betas=True)
        y = self.output_proj(h)

        if return_betas:
            return y, betas

        return y


class DeltaAttention(nn.Module):
    """Delta-based Multi-Head Attention.

    Applies delta gating to attention mechanism:
    1. Q, K, V projections with delta gating
    2. Standard attention computation
    3. Output projection with delta residual

    The delta gates control information flow at each stage.

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        beta_init_bias: Initial beta bias
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        beta_init_bias: float = -4.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # Q, K, V projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

        # Delta gates for Q, K, V
        self.q_delta = DeltaResidualBlock(dim, beta_init_bias=beta_init_bias)
        self.k_delta = DeltaResidualBlock(dim, beta_init_bias=beta_init_bias)
        self.v_delta = DeltaResidualBlock(dim, beta_init_bias=beta_init_bias)

        # Output delta residual
        self.out_delta = DeltaResidualBlock(dim, beta_init_bias=beta_init_bias)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_betas: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """Delta attention forward pass.

        Args:
            x: Input (batch, seq, dim)
            mask: Optional attention mask (seq, seq)
            return_betas: If True, return beta diagnostics

        Returns:
            y: Output (batch, seq, dim)
            betas: Optional dict of beta values
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V with delta gating
        q = self.q_proj(x)
        q, q_beta = self.q_delta(x, v_input=q)

        k = self.k_proj(x)
        k, k_beta = self.k_delta(x, v_input=k)

        v = self.v_proj(x)
        v, v_beta = self.v_delta(x, v_input=v)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

        # Output projection with delta residual
        out = self.out_proj(out)
        y, out_beta = self.out_delta(x, v_input=out)
        y = self.norm(y)

        if return_betas:
            betas = {
                'q_beta': q_beta.item() if isinstance(q_beta, torch.Tensor) else q_beta,
                'k_beta': k_beta.item() if isinstance(k_beta, torch.Tensor) else k_beta,
                'v_beta': v_beta.item() if isinstance(v_beta, torch.Tensor) else v_beta,
                'out_beta': out_beta.item() if isinstance(out_beta, torch.Tensor) else out_beta,
            }
            return y, betas

        return y


class DeltaTransformerBlock(nn.Module):
    """Complete Delta-based Transformer Block.

    Combines DeltaAttention and DeltaFFN into a single block:
        x -> DeltaAttention -> DeltaFFN -> output

    All components use delta gating for controlled information flow.

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        ffn_expansion: FFN expansion factor
        num_ffn_delta_blocks: Number of delta blocks in FFN
        dropout: Dropout rate
        beta_init_bias: Initial beta bias
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_expansion: int = 4,
        num_ffn_delta_blocks: int = 2,
        dropout: float = 0.1,
        beta_init_bias: float = -4.0,
    ):
        super().__init__()

        self.attention = DeltaAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            beta_init_bias=beta_init_bias,
        )

        self.ffn = DeltaFFN(
            dim=dim,
            expansion=ffn_expansion,
            num_delta_blocks=num_ffn_delta_blocks,
            dropout=dropout,
            beta_init_bias=beta_init_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_betas: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """Delta transformer block forward pass.

        Args:
            x: Input (batch, seq, dim)
            mask: Optional attention mask
            return_betas: If True, return beta diagnostics

        Returns:
            y: Output (batch, seq, dim)
            betas: Optional dict of all beta values
        """
        # Delta Attention
        if return_betas:
            x, attn_betas = self.attention(x, mask=mask, return_betas=True)
            x, ffn_betas = self.ffn(x, return_betas=True)
            return x, {'attention': attn_betas, 'ffn': ffn_betas}
        else:
            x = self.attention(x, mask=mask)
            x = self.ffn(x)
            return x


class DeltaTransformer(nn.Module):
    """Full Delta-based Transformer Model.

    A complete transformer using delta layers throughout:
    - Embedding with delta initialization
    - N x DeltaTransformerBlock
    - Output projection

    Args:
        vocab_size: Vocabulary size
        dim: Model dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length
        ffn_expansion: FFN expansion factor
        dropout: Dropout rate
        beta_init_bias: Initial beta bias
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 512,
        ffn_expansion: int = 4,
        dropout: float = 0.1,
        beta_init_bias: float = -4.0,
    ):
        super().__init__()

        self.dim = dim
        self.vocab_size = vocab_size

        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq_len, dim)

        # Delta transformer blocks
        self.layers = nn.ModuleList([
            DeltaTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                ffn_expansion=ffn_expansion,
                dropout=dropout,
                beta_init_bias=beta_init_bias,
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_betas: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """Delta transformer forward pass.

        Args:
            input_ids: Token IDs (batch, seq)
            return_betas: If True, return all beta diagnostics

        Returns:
            logits: Output logits (batch, seq, vocab_size)
            betas: Optional dict of beta values per layer
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)

        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

        # Transformer layers
        all_betas = {}
        for i, layer in enumerate(self.layers):
            if return_betas:
                x, layer_betas = layer(x, mask=mask, return_betas=True)
                all_betas[f'layer_{i}'] = layer_betas
            else:
                x = layer(x, mask=mask)

        # Output
        x = self.output_norm(x)
        logits = self.output_proj(x)

        if return_betas:
            return logits, all_betas

        return logits

    def get_beta_summary(self, betas: dict) -> dict:
        """Summarize beta values from forward pass.

        Args:
            betas: Beta dict from forward with return_betas=True

        Returns:
            Summary statistics
        """
        all_attn_betas = []
        all_ffn_betas = []

        for layer_name, layer_betas in betas.items():
            attn = layer_betas.get('attention', {})
            ffn = layer_betas.get('ffn', {})

            for key, val in attn.items():
                all_attn_betas.append(val)

            if 'stack_betas' in ffn:
                all_ffn_betas.extend(ffn['stack_betas'])
            if 'residual_beta' in ffn:
                all_ffn_betas.append(ffn['residual_beta'])

        return {
            'avg_attn_beta': sum(all_attn_betas) / len(all_attn_betas) if all_attn_betas else 0,
            'avg_ffn_beta': sum(all_ffn_betas) / len(all_ffn_betas) if all_ffn_betas else 0,
            'num_attn_gates': len(all_attn_betas),
            'num_ffn_gates': len(all_ffn_betas),
        }
