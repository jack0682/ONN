#!/usr/bin/env python3
"""TLM Van der Pol Oscillator Training Script.

Van der Pol 진동자 궤적 예측 - Limit Cycle 학습 테스트.

시스템:
    ẍ - μ(1 - x²)ẋ + x = 0
    
    또는 상태 공간 형태:
    ẋ = y
    ẏ = μ(1 - x²)y - x
    
    μ: 비선형 파라미터
    - μ = 0: 단순 조화 진동자
    - μ > 0: Limit Cycle 발생 (모든 궤적이 폐곡선으로 수렴)
    - μ ↑: 사이클이 더 "찌그러진" 릴랙세이션 진동

핵심 특성:
    - 전역 안정 Limit Cycle: 어디서 시작하든 같은 궤도로 수렴
    - ONN의 "사이클 제약"과 수학적으로 완벽하게 대응

실행:
    PYTHONPATH=./src python3 scripts/train_vanderpol.py
    PYTHONPATH=./src python3 scripts/train_vanderpol.py --epochs 2000 --embed-dim 512

Author: Claude
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from scipy.integrate import odeint

# matplotlib optional
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("  (matplotlib not found, visualization disabled)")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from onn.tlm import TopologicalLanguageModel, TLMConfig


# ==============================================================================
# Van der Pol 시뮬레이션
# ==============================================================================

def vanderpol_derivatives(state, t, mu):
    """Van der Pol 미분 방정식.
    
    ẋ = y
    ẏ = μ(1 - x²)y - x
    """
    x, y = state
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return [dxdt, dydt]


def generate_vanderpol_trajectory(mu: float, x0: float, y0: float, 
                                   seq_len: int, dt: float = 0.1) -> np.ndarray:
    """Van der Pol 궤적 생성.
    
    Args:
        mu: 비선형 파라미터
        x0, y0: 초기 조건
        seq_len: 시퀀스 길이
        dt: 시간 간격
        
    Returns:
        (seq_len, 2) 형태의 궤적 [x, y]
    """
    t = np.linspace(0, seq_len * dt, seq_len)
    trajectory = odeint(vanderpol_derivatives, [x0, y0], t, args=(mu,))
    return trajectory


def estimate_limit_cycle_period(mu: float) -> float:
    """Limit cycle 주기 추정 (근사).
    
    μ가 작을 때: T ≈ 2π
    μ가 클 때: T ≈ (3 - 2*ln(2))μ + 2π/μ (relaxation oscillation)
    """
    if mu < 0.1:
        return 2 * np.pi
    elif mu < 3:
        return 2 * np.pi * (1 + mu**2 / 16)  # 근사
    else:
        return (3 - 2 * np.log(2)) * mu + 2 * np.pi / mu


class VanderPolDataset:
    """Van der Pol 궤적 데이터셋."""
    
    def __init__(self, vocab_size: int, seq_len: int,
                 mu_range: tuple = (0.5, 5.0),
                 init_range: tuple = (-3.0, 3.0),
                 dt: float = 0.1,
                 scale: float = 5.0):
        """
        Args:
            vocab_size: 양자화 레벨 수
            seq_len: 시퀀스 길이
            mu_range: μ 파라미터 범위
            init_range: 초기값 범위
            dt: 시간 간격
            scale: 값 스케일링
        """
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.mu_range = mu_range
        self.init_range = init_range
        self.dt = dt
        self.scale = scale
        self.state_dim = 2  # x, y
    
    def _quantize(self, x: np.ndarray) -> np.ndarray:
        """연속값을 이산 토큰으로 변환."""
        normalized = (x / self.scale + 1) / 2  # [0, 1]
        tokens = np.clip(normalized * self.vocab_size, 0, self.vocab_size - 1)
        return tokens.astype(np.int64)
    
    def _dequantize(self, tokens: np.ndarray) -> np.ndarray:
        """토큰을 연속값으로 변환."""
        normalized = tokens / self.vocab_size
        x = (normalized * 2 - 1) * self.scale
        return x
    
    def _trajectory_to_tokens(self, traj: np.ndarray) -> np.ndarray:
        """(seq_len, 2) 궤적을 (seq_len * 2,) 토큰 시퀀스로 변환.
        
        인터리빙: [x0, y0, x1, y1, x2, y2, ...]
        """
        flat = traj.flatten()  # [x0, y0, x1, y1, ...]
        return self._quantize(flat)
    
    def _tokens_to_trajectory(self, tokens: np.ndarray) -> np.ndarray:
        """토큰 시퀀스를 궤적으로 복원."""
        flat = self._dequantize(tokens)
        return flat.reshape(-1, 2)
    
    def sample(self, batch: int, difficulty: float = 0.5) -> tuple:
        """샘플 생성.
        
        난이도:
        - 낮음: 작은 μ (원형에 가까운 사이클)
        - 높음: 큰 μ (찌그러진 릴랙세이션 사이클)
        """
        inputs = []
        targets = []
        params = []
        
        for _ in range(batch):
            # μ 샘플링 (난이도에 따라)
            if difficulty < 0.3:
                mu = np.random.uniform(0.5, 1.5)  # Easy: 약한 비선형
            elif difficulty < 0.7:
                mu = np.random.uniform(1.0, 3.0)  # Medium
            else:
                mu = np.random.uniform(*self.mu_range)  # Hard: 전체 범위
            
            # 초기값 (Limit cycle 밖 또는 안에서 시작)
            x0 = np.random.uniform(*self.init_range)
            y0 = np.random.uniform(*self.init_range)
            
            # 궤적 생성 (입력 + 타겟 = seq_len + 1 스텝)
            # 2D 상태를 인터리빙하므로 실제 필요한 시간 스텝은 seq_len // 2 + 1
            time_steps = self.seq_len // 2 + 1
            traj = generate_vanderpol_trajectory(mu, x0, y0, time_steps, self.dt)
            
            # 토큰화
            tokens = self._trajectory_to_tokens(traj)
            
            # 입력: [x0, y0, x1, y1, ...], 타겟: [y0, x1, y1, x2, ...]
            inputs.append(tokens[:self.seq_len])
            targets.append(tokens[1:self.seq_len + 1])
            params.append({'mu': mu, 'x0': x0, 'y0': y0})
        
        return (torch.tensor(np.array(inputs), dtype=torch.long),
                torch.tensor(np.array(targets), dtype=torch.long),
                params)
    
    def sample_specific(self, batch: int, mu: float) -> tuple:
        """특정 μ로 샘플 생성."""
        inputs = []
        targets = []
        
        for _ in range(batch):
            x0 = np.random.uniform(*self.init_range)
            y0 = np.random.uniform(*self.init_range)
            
            time_steps = self.seq_len // 2 + 1
            traj = generate_vanderpol_trajectory(mu, x0, y0, time_steps, self.dt)
            tokens = self._trajectory_to_tokens(traj)
            
            inputs.append(tokens[:self.seq_len])
            targets.append(tokens[1:self.seq_len + 1])
        
        return (torch.tensor(np.array(inputs), dtype=torch.long),
                torch.tensor(np.array(targets), dtype=torch.long))


# ==============================================================================
# 학습률 스케줄러
# ==============================================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * self.current_epoch / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        return lr


# ==============================================================================
# 학습
# ==============================================================================

class VanderPolTrainer:
    """Van der Pol 학습 관리자."""
    
    def __init__(self, config: dict):
        self.config = config
        
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print(f"  Using: MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"  Using: CUDA GPU")
        else:
            self.device = torch.device("cpu")
            print(f"  Using: CPU")
        
        # 모델
        self.model_config = TLMConfig(
            vocab_size=config["vocab_size"],
            embed_dim=config["embed_dim"],
            relation_dim=config["relation_dim"],
            context_window=config["context_window"],
            num_heads=config["num_heads"],
            pc_steps=config["pc_steps"],
            pc_alpha=config["pc_alpha"],
            dropout=config["dropout"],
        )
        
        self.model = TopologicalLanguageModel(
            self.model_config,
            num_layers=config["num_layers"]
        ).to(self.device)
        
        # 데이터셋
        self.dataset = VanderPolDataset(
            config["vocab_size"],
            config["seq_len"],
            dt=config["dt"]
        )
        
        # 옵티마이저
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
        
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=config["warmup_epochs"],
            total_epochs=config["epochs"],
        )
        
        self.dt = config["dt"]
        
        self.history = defaultdict(list)
        self.best_loss = float('inf')
    
    def train_epoch(self, epoch: int):
        self.model.train()
        
        difficulty = min(1.0, epoch / (self.config["epochs"] * 0.7))
        
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_topo = 0.0
        epoch_attn_beta = 0.0
        epoch_ffn_beta = 0.0
        steps = 0
        
        for _ in range(self.config["steps_per_epoch"]):
            input_ids, target_ids, _ = self.dataset.sample(
                self.config["batch_size"], difficulty
            )
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            logits, diag = self.model(input_ids)
            
            loss = F.cross_entropy(
                logits.view(-1, self.config["vocab_size"]),
                target_ids.view(-1)
            )
            topo_loss = diag['total_violation'] * self.config["topo_weight"]
            total_loss = loss + topo_loss
            
            pred = logits.argmax(dim=-1)
            acc = (pred == target_ids).float().mean().item()
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_topo += topo_loss
            epoch_attn_beta += diag.get("avg_attn_beta", 0.0)
            epoch_ffn_beta += diag.get("avg_ffn_beta", 0.0)
            steps += 1
        
        lr = self.scheduler.step()
        
        return {
            "loss": epoch_loss / steps,
            "acc": epoch_acc / steps,
            "topo": epoch_topo / steps,
            "attn_beta": epoch_attn_beta / steps,
            "ffn_beta": epoch_ffn_beta / steps,
            "lr": lr,
            "difficulty": difficulty,
        }
    
    @torch.no_grad()
    def evaluate_by_mu(self):
        """μ별 평가."""
        self.model.eval()
        
        results = {}
        for name, mu in [("mu=1.0", 1.0), ("mu=2.0", 2.0), ("mu=4.0", 4.0)]:
            input_ids, target_ids = self.dataset.sample_specific(32, mu)
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            logits, diag = self.model(input_ids)
            ce_loss = F.cross_entropy(
                logits.view(-1, self.config["vocab_size"]),
                target_ids.view(-1)
            )
            
            pred = logits.argmax(dim=-1)
            acc = (pred == target_ids).float().mean().item()
            
            # 연속값 MAE
            pred_vals = self.dataset._dequantize(pred.cpu().numpy())
            target_vals = self.dataset._dequantize(target_ids.cpu().numpy())
            mae = np.abs(pred_vals - target_vals).mean()
            
            results[name] = {"ce": ce_loss.item(), "acc": acc, "mae": mae}
        
        return results
    
    @torch.no_grad()
    def visualize_limit_cycle(self, save_path: str = "vanderpol_prediction.png"):
        """Limit cycle 시각화."""
        if not HAS_MATPLOTLIB:
            print("\n  (matplotlib not available, skipping visualization)")
            return
        
        self.model.eval()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for idx, mu in enumerate([1.0, 2.0, 4.0]):
            ax_phase = axes[0, idx]
            ax_time = axes[1, idx]
            
            # 실제 궤적 생성
            x0, y0 = 0.5, 0.5
            time_steps = 100
            traj = generate_vanderpol_trajectory(mu, x0, y0, time_steps, self.dt)
            
            # 모델 예측 (처음 몇 스텝만)
            tokens = self.dataset._trajectory_to_tokens(traj[:self.config["seq_len"] // 2 + 1])
            input_tokens = torch.tensor(tokens[:self.config["seq_len"]], dtype=torch.long).unsqueeze(0).to(self.device)
            
            logits, _ = self.model(input_tokens)
            pred_tokens = logits[0].argmax(dim=-1).cpu().numpy()
            
            # 예측 궤적 복원
            pred_traj = self.dataset._tokens_to_trajectory(pred_tokens)
            actual_traj = traj[1:len(pred_traj) + 1]
            
            # Phase plot
            ax_phase.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.5, label='Full trajectory')
            ax_phase.plot(actual_traj[:, 0], actual_traj[:, 1], 'g-', linewidth=2, label='Actual (segment)')
            ax_phase.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', linewidth=2, label='Predicted')
            ax_phase.set_xlabel('x')
            ax_phase.set_ylabel('y')
            ax_phase.set_title(f'Phase Space (μ={mu})')
            ax_phase.legend(fontsize=8)
            ax_phase.grid(True, alpha=0.3)
            ax_phase.set_aspect('equal')
            
            # Time series
            t = np.arange(len(actual_traj)) * self.dt
            ax_time.plot(t, actual_traj[:, 0], 'g-', linewidth=2, label='Actual x')
            ax_time.plot(t, pred_traj[:, 0], 'r--', linewidth=2, label='Predicted x')
            ax_time.set_xlabel('Time')
            ax_time.set_ylabel('x(t)')
            ax_time.set_title(f'Time Series (μ={mu})')
            ax_time.legend(fontsize=8)
            ax_time.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"\n  시각화 저장: {save_path}")
    
    def train(self):
        print("=" * 70)
        print("TLM Van der Pol Oscillator Training (Limit Cycle Learning)")
        print("=" * 70)
        print(f"  Epochs: {self.config['epochs']}, LR: {self.config['lr']}")
        print(f"  Model: {sum(p.numel() for p in self.model.parameters()):,} params")
        print(f"  Task: Limit Cycle 궤적 예측")
        print(f"  μ range: {self.config.get('mu_range', (0.5, 5.0))}")
        print("-" * 70)
        
        start_time = time.perf_counter()
        
        for epoch in range(1, self.config["epochs"] + 1):
            metrics = self.train_epoch(epoch)
            
            for k, v in metrics.items():
                self.history[k].append(v)
            
            if metrics["loss"] < self.best_loss:
                self.best_loss = metrics["loss"]
            
            if epoch % self.config["log_every"] == 0:
                elapsed = time.perf_counter() - start_time
                print(f"  Epoch {epoch:4d}/{self.config['epochs']}: "
                      f"Loss={metrics['loss']:.4f}, Acc={metrics['acc']*100:.1f}%, "
                      f"Topo={metrics['topo']:.6f}, Beta(A/F)={metrics['attn_beta']:.3f}/{metrics['ffn_beta']:.3f}, "
                      f"Time={elapsed:.1f}s")
        
        total_time = time.perf_counter() - start_time
        
        # μ별 평가
        print("\n" + "-" * 70)
        print("μ별 성능 평가 (Limit Cycle 강도):")
        eval_results = self.evaluate_by_mu()
        for name, res in eval_results.items():
            print(f"  {name:10s}: CE={res['ce']:.4f}, Acc={res['acc']*100:.1f}%, MAE={res['mae']:.4f}")
        
        # 시각화
        self.visualize_limit_cycle()
        
        # 요약
        print("\n" + "=" * 70)
        print("학습 요약")
        print("=" * 70)
        print(f"  총 시간: {total_time:.1f}s")
        print(f"  최종 Loss: {self.history['loss'][-1]:.4f}")
        print(f"  최종 Accuracy: {self.history['acc'][-1]*100:.1f}%")
        print(f"\n  Limit Cycle 학습 인사이트:")
        print(f"    → μ=1.0 (약한 비선형): MAE={eval_results['mu=1.0']['mae']:.4f}")
        print(f"    → μ=2.0 (중간): MAE={eval_results['mu=2.0']['mae']:.4f}")
        print(f"    → μ=4.0 (강한 릴랙세이션): MAE={eval_results['mu=4.0']['mae']:.4f}")
        
        return self.history


# ==============================================================================
# 메인
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="TLM Van der Pol Training")
    
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=32, help="시퀀스 길이 (x,y 인터리빙)")
    parser.add_argument("--dt", type=float, default=0.1, help="시간 간격")
    
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--pc-steps", type=int, default=5)
    
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--save", action="store_true")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = {
        "epochs": args.epochs if not args.fast else 200,
        "lr": args.lr,
        "warmup_epochs": args.warmup if not args.fast else 20,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "dt": args.dt,
        "weight_decay": 0.01,
        "topo_weight": 0.1,
        "steps_per_epoch": 10 if not args.fast else 5,
        "log_every": args.log_every if not args.fast else 40,
        
        "vocab_size": 500,
        "embed_dim": args.embed_dim,
        "relation_dim": 64,
        "context_window": max(64, args.seq_len),
        "num_heads": args.heads,
        "num_layers": args.layers,
        "pc_steps": args.pc_steps,
        "pc_alpha": 0.8,
        "dropout": 0.1,
    }
    
    trainer = VanderPolTrainer(config)
    history = trainer.train()
    
    if args.save:
        torch.save({
            'model_state_dict': trainer.model.state_dict(),
            'config': config,
            'history': dict(history),
        }, 'best_vanderpol.pt')
        print(f"\n  모델 저장: best_vanderpol.pt")


if __name__ == "__main__":
    main()
