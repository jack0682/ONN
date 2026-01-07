"""DDL vs ResNet Benchmark.

Compares Deep Delta Learning (DDL) against standard ResNet-style residual connections.

Metrics:
1. Training convergence speed
2. Gradient flow (gradient magnitude across layers)
3. Final loss/accuracy
4. Parameter efficiency

Author: Claude
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple
import sys
sys.path.insert(0, 'src')

from onn.modules.delta import DeltaResidualBlock, DeltaResidualStack


# ==============================================================================
# BASELINE: Standard ResNet Block
# ==============================================================================

class ResNetBlock(nn.Module):
    """Standard ResNet residual block."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.gelu(self.fc1(x))
        out = self.fc2(out)
        return self.norm(residual + out)


class ResNetStack(nn.Module):
    """Stack of ResNet blocks."""

    def __init__(self, dim: int, num_blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResNetBlock(dim) for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


# ==============================================================================
# MODELS FOR COMPARISON
# ==============================================================================

class DDLModel(nn.Module):
    """Model using DDL (Deep Delta Learning) blocks."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_blocks: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.delta_stack = DeltaResidualStack(
            dim=hidden_dim,
            num_blocks=num_blocks,
            d_v=1,
            beta_init_bias=-4.0,
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        x = F.gelu(self.input_proj(x))
        x, betas = self.delta_stack(x, return_all_betas=True)
        x = self.output_proj(x)
        return x, betas


class ResNetModel(nn.Module):
    """Model using standard ResNet blocks."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_blocks: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.resnet_stack = ResNetStack(hidden_dim, num_blocks)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.input_proj(x))
        x = self.resnet_stack(x)
        x = self.output_proj(x)
        return x


# ==============================================================================
# BENCHMARK TASKS
# ==============================================================================

def generate_regression_data(n_samples: int, input_dim: int, output_dim: int, noise: float = 0.1):
    """Generate synthetic regression data."""
    X = torch.randn(n_samples, input_dim)
    # Non-linear target function
    W = torch.randn(input_dim, output_dim) * 0.5
    y = torch.tanh(X @ W) + noise * torch.randn(n_samples, output_dim)
    return X, y


def generate_classification_data(n_samples: int, input_dim: int, n_classes: int):
    """Generate synthetic classification data."""
    X = torch.randn(n_samples, input_dim)
    # Generate labels based on input regions
    labels = (X[:, 0] + X[:, 1] > 0).long()
    if n_classes > 2:
        labels = (X[:, :n_classes].argmax(dim=1)).long()
    return X, labels


# ==============================================================================
# BENCHMARK FUNCTIONS
# ==============================================================================

@dataclass
class BenchmarkResult:
    model_name: str
    task: str
    final_loss: float
    convergence_step: int  # Step where loss < threshold
    train_time: float
    gradient_norms: List[float]
    loss_history: List[float]
    beta_history: List[List[float]]  # For DDL only
    num_params: int


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """Compute gradient norms per layer."""
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.norm().item()
    return norms


def benchmark_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    task: str,  # 'regression' or 'classification'
    model_name: str,
    num_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    convergence_threshold: float = 0.1,
) -> BenchmarkResult:
    """Run benchmark on a single model."""

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    gradient_norms_history = []
    beta_history = []
    convergence_step = num_epochs

    n_samples = X_train.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_betas = []

        # Shuffle data
        perm = torch.randperm(n_samples)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)

            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            optimizer.zero_grad()

            # Forward pass
            if model_name == 'DDL':
                out, betas = model(X_batch)
                epoch_betas.extend(betas)
            else:
                out = model(X_batch)

            # Compute loss
            if task == 'regression':
                loss = F.mse_loss(out, y_batch)
            else:
                loss = F.cross_entropy(out, y_batch)

            # Backward pass
            loss.backward()

            # Record gradient norms (last batch of epoch)
            if i == n_batches - 1:
                grad_norms = compute_gradient_norms(model)
                gradient_norms_history.append(sum(grad_norms.values()) / len(grad_norms))

            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

        if epoch_betas:
            beta_history.append(epoch_betas)

        # Check convergence
        if avg_loss < convergence_threshold and convergence_step == num_epochs:
            convergence_step = epoch

    train_time = time.time() - start_time

    return BenchmarkResult(
        model_name=model_name,
        task=task,
        final_loss=loss_history[-1],
        convergence_step=convergence_step,
        train_time=train_time,
        gradient_norms=gradient_norms_history,
        loss_history=loss_history,
        beta_history=beta_history,
        num_params=count_parameters(model),
    )


def run_comparison(
    input_dim: int = 32,
    hidden_dim: int = 64,
    output_dim: int = 10,
    num_blocks: int = 4,
    n_samples: int = 1000,
    num_epochs: int = 100,
    task: str = 'regression',
):
    """Run full comparison between DDL and ResNet."""

    print(f"\n{'='*60}")
    print(f"DDL vs ResNet Benchmark")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"Input dim: {input_dim}, Hidden dim: {hidden_dim}, Output dim: {output_dim}")
    print(f"Num blocks: {num_blocks}, Samples: {n_samples}, Epochs: {num_epochs}")
    print(f"{'='*60}\n")

    # Generate data
    if task == 'regression':
        X, y = generate_regression_data(n_samples, input_dim, output_dim)
    else:
        X, y = generate_classification_data(n_samples, input_dim, output_dim)

    # Create models
    torch.manual_seed(42)
    ddl_model = DDLModel(input_dim, hidden_dim, output_dim, num_blocks)

    torch.manual_seed(42)
    resnet_model = ResNetModel(input_dim, hidden_dim, output_dim, num_blocks)

    # Run benchmarks
    print("Training DDL model...")
    ddl_result = benchmark_model(
        ddl_model, X, y, task, 'DDL', num_epochs=num_epochs
    )

    print("Training ResNet model...")
    resnet_result = benchmark_model(
        resnet_model, X, y, task, 'ResNet', num_epochs=num_epochs
    )

    return ddl_result, resnet_result


def print_results(ddl_result: BenchmarkResult, resnet_result: BenchmarkResult):
    """Print comparison results."""

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}\n")

    # Table header
    print(f"{'Metric':<30} {'DDL':>15} {'ResNet':>15} {'Winner':>10}")
    print("-" * 70)

    # Final loss
    winner = 'DDL' if ddl_result.final_loss < resnet_result.final_loss else 'ResNet'
    print(f"{'Final Loss':<30} {ddl_result.final_loss:>15.6f} {resnet_result.final_loss:>15.6f} {winner:>10}")

    # Convergence step
    winner = 'DDL' if ddl_result.convergence_step < resnet_result.convergence_step else 'ResNet'
    print(f"{'Convergence Step':<30} {ddl_result.convergence_step:>15d} {resnet_result.convergence_step:>15d} {winner:>10}")

    # Training time
    winner = 'DDL' if ddl_result.train_time < resnet_result.train_time else 'ResNet'
    print(f"{'Training Time (s)':<30} {ddl_result.train_time:>15.3f} {resnet_result.train_time:>15.3f} {winner:>10}")

    # Parameters
    winner = 'DDL' if ddl_result.num_params < resnet_result.num_params else 'ResNet'
    print(f"{'Num Parameters':<30} {ddl_result.num_params:>15d} {resnet_result.num_params:>15d} {winner:>10}")

    # Gradient flow (average gradient norm)
    ddl_avg_grad = np.mean(ddl_result.gradient_norms[-10:]) if ddl_result.gradient_norms else 0
    resnet_avg_grad = np.mean(resnet_result.gradient_norms[-10:]) if resnet_result.gradient_norms else 0
    # Higher gradient flow is better (no vanishing)
    winner = 'DDL' if ddl_avg_grad > resnet_avg_grad else 'ResNet'
    print(f"{'Avg Gradient Norm (last 10)':<30} {ddl_avg_grad:>15.6f} {resnet_avg_grad:>15.6f} {winner:>10}")

    # Loss improvement
    ddl_improvement = (ddl_result.loss_history[0] - ddl_result.loss_history[-1]) / ddl_result.loss_history[0] * 100
    resnet_improvement = (resnet_result.loss_history[0] - resnet_result.loss_history[-1]) / resnet_result.loss_history[0] * 100
    winner = 'DDL' if ddl_improvement > resnet_improvement else 'ResNet'
    print(f"{'Loss Improvement (%)':<30} {ddl_improvement:>15.2f} {resnet_improvement:>15.2f} {winner:>10}")

    print("-" * 70)

    # Beta analysis for DDL
    if ddl_result.beta_history:
        print(f"\nDDL Beta Analysis:")
        final_betas = ddl_result.beta_history[-1] if ddl_result.beta_history else []
        if final_betas:
            print(f"  Final beta values: {[f'{b:.4f}' for b in final_betas[:5]]}...")
            print(f"  Beta mean: {np.mean(final_betas):.4f}")
            print(f"  Beta std: {np.std(final_betas):.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    ddl_wins = 0
    resnet_wins = 0

    if ddl_result.final_loss < resnet_result.final_loss:
        ddl_wins += 1
    else:
        resnet_wins += 1

    if ddl_result.convergence_step < resnet_result.convergence_step:
        ddl_wins += 1
    else:
        resnet_wins += 1

    if ddl_avg_grad > resnet_avg_grad:
        ddl_wins += 1
    else:
        resnet_wins += 1

    print(f"DDL wins: {ddl_wins}/3 key metrics")
    print(f"ResNet wins: {resnet_wins}/3 key metrics")

    if ddl_wins > resnet_wins:
        print("\n>>> DDL outperforms ResNet in this benchmark! <<<")
    elif resnet_wins > ddl_wins:
        print("\n>>> ResNet outperforms DDL in this benchmark. <<<")
    else:
        print("\n>>> Results are mixed - performance is similar. <<<")


def run_depth_scaling_test(depths: List[int] = [2, 4, 8, 16]):
    """Test how models scale with depth."""

    print(f"\n{'='*60}")
    print("DEPTH SCALING TEST")
    print(f"{'='*60}\n")

    results = []

    for depth in depths:
        print(f"\n--- Testing depth={depth} ---")

        torch.manual_seed(42)
        X, y = generate_regression_data(500, 32, 10)

        # DDL
        torch.manual_seed(42)
        ddl_model = DDLModel(32, 64, 10, depth)
        ddl_result = benchmark_model(ddl_model, X, y, 'regression', 'DDL', num_epochs=50)

        # ResNet
        torch.manual_seed(42)
        resnet_model = ResNetModel(32, 64, 10, depth)
        resnet_result = benchmark_model(resnet_model, X, y, 'regression', 'ResNet', num_epochs=50)

        results.append({
            'depth': depth,
            'ddl_loss': ddl_result.final_loss,
            'resnet_loss': resnet_result.final_loss,
            'ddl_grad': np.mean(ddl_result.gradient_norms[-5:]) if ddl_result.gradient_norms else 0,
            'resnet_grad': np.mean(resnet_result.gradient_norms[-5:]) if resnet_result.gradient_norms else 0,
        })

    print(f"\n{'='*60}")
    print("DEPTH SCALING RESULTS")
    print(f"{'='*60}")
    print(f"{'Depth':>8} {'DDL Loss':>12} {'ResNet Loss':>12} {'DDL Grad':>12} {'ResNet Grad':>12}")
    print("-" * 60)

    for r in results:
        print(f"{r['depth']:>8d} {r['ddl_loss']:>12.6f} {r['resnet_loss']:>12.6f} "
              f"{r['ddl_grad']:>12.6f} {r['resnet_grad']:>12.6f}")

    # Analysis
    print("\nAnalysis:")
    ddl_better_count = sum(1 for r in results if r['ddl_loss'] < r['resnet_loss'])
    print(f"- DDL had lower loss in {ddl_better_count}/{len(results)} depth configurations")

    # Gradient degradation
    if len(results) > 1:
        ddl_grad_ratio = results[-1]['ddl_grad'] / (results[0]['ddl_grad'] + 1e-8)
        resnet_grad_ratio = results[-1]['resnet_grad'] / (results[0]['resnet_grad'] + 1e-8)
        print(f"- DDL gradient retention (deep/shallow): {ddl_grad_ratio:.4f}")
        print(f"- ResNet gradient retention (deep/shallow): {resnet_grad_ratio:.4f}")

        if ddl_grad_ratio > resnet_grad_ratio:
            print("- DDL maintains gradient flow better with depth!")
        else:
            print("- ResNet maintains gradient flow better with depth.")


if __name__ == "__main__":
    # Main comparison
    print("\n" + "="*70)
    print(" BENCHMARK 1: Regression Task ")
    print("="*70)
    ddl_result, resnet_result = run_comparison(
        input_dim=32,
        hidden_dim=64,
        output_dim=10,
        num_blocks=4,
        n_samples=1000,
        num_epochs=100,
        task='regression',
    )
    print_results(ddl_result, resnet_result)

    # Classification task
    print("\n" + "="*70)
    print(" BENCHMARK 2: Classification Task ")
    print("="*70)
    ddl_result, resnet_result = run_comparison(
        input_dim=32,
        hidden_dim=64,
        output_dim=5,
        num_blocks=4,
        n_samples=1000,
        num_epochs=100,
        task='classification',
    )
    print_results(ddl_result, resnet_result)

    # Depth scaling test
    print("\n" + "="*70)
    print(" BENCHMARK 3: Depth Scaling ")
    print("="*70)
    run_depth_scaling_test(depths=[2, 4, 8, 16])
