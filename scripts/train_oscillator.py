#!/usr/bin/env python3
"""TLM Damped Oscillator Training Script.

감쇠 진동자 궤적 예측을 통한 Lyapunov 안정성 학습 테스트.

시스템:
    ẍ + 2ζωẋ + ω²x = 0
    
    ζ: 감쇠비 (damping ratio)
    ω: 고유진동수 (natural frequency)
    
    ζ < 1: 감쇠 진동 (underdamped)
    ζ = 1: 임계 감쇠 (critically damped)
    ζ > 1: 과감쇠 (overdamped)

학습 목표:
    입력: [x(0), x(Δt), x(2Δt), ...]
    출력: [x(Δt), x(2Δt), x(3Δt), ...]
    
    → 다음 상태 예측 = 동역학 학습

실행:
    PYTHONPATH=./src python3 scripts/train_oscillator.py
    PYTHONPATH=./src python3 scripts/train_oscillator.py --epochs 1000 --embed-dim 256

Author: Claude
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

# matplotlib is optional for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("  (matplotlib not found, visualization disabled)")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from onn.tlm import TopologicalLanguageModel, TLMConfig


# ==============================================================================
# 감쇠 진동자 시뮬레이션
# ==============================================================================

def damped_oscillator_analytical(zeta: float, omega: float, x0: float, v0: float, 
                                  t: np.ndarray) -> np.ndarray:
    """감쇠 진동자 해석해.
    
    ẍ + 2ζωẋ + ω²x = 0
    
    Args:
        zeta: 감쇠비
        omega: 고유진동수
        x0: 초기 위치
        v0: 초기 속도
        t: 시간 배열
        
    Returns:
        x(t) 배열
    """
    if zeta < 1:
        # Underdamped
        omega_d = omega * np.sqrt(1 - zeta**2)
        A = x0
        B = (v0 + zeta * omega * x0) / omega_d
        x = np.exp(-zeta * omega * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
    elif zeta == 1:
        # Critically damped
        A = x0
        B = v0 + omega * x0
        x = (A + B * t) * np.exp(-omega * t)
    else:
        # Overdamped
        r1 = omega * (-zeta + np.sqrt(zeta**2 - 1))
        r2 = omega * (-zeta - np.sqrt(zeta**2 - 1))
        A = (v0 - r2 * x0) / (r1 - r2)
        B = (r1 * x0 - v0) / (r1 - r2)
        x = A * np.exp(r1 * t) + B * np.exp(r2 * t)
    
    return x


def generate_trajectory(zeta: float, omega: float, x0: float, v0: float,
                        seq_len: int, dt: float = 0.1) -> np.ndarray:
    """궤적 생성."""
    t = np.arange(seq_len) * dt
    return damped_oscillator_analytical(zeta, omega, x0, v0, t)


class OscillatorDataset:
    """감쇠 진동자 데이터셋."""
    
    def __init__(self, vocab_size: int, seq_len: int, 
                 zeta_range: tuple = (0.1, 2.0),
                 omega_range: tuple = (0.5, 2.0),
                 x0_range: tuple = (-5.0, 5.0),
                 v0_range: tuple = (-3.0, 3.0),
                 dt: float = 0.1,
                 scale: float = 10.0):
        """
        Args:
            vocab_size: 양자화 레벨 수
            seq_len: 시퀀스 길이
            zeta_range: 감쇠비 범위
            omega_range: 고유진동수 범위
            x0_range: 초기 위치 범위
            v0_range: 초기 속도 범위
            dt: 시간 간격
            scale: 값 스케일링 (vocab_size에 맞추기)
        """
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.zeta_range = zeta_range
        self.omega_range = omega_range
        self.x0_range = x0_range
        self.v0_range = v0_range
        self.dt = dt
        self.scale = scale
    
    def _quantize(self, x: np.ndarray) -> np.ndarray:
        """연속값을 이산 토큰으로 변환."""
        # x를 [0, vocab_size-1]로 매핑
        # x 범위: 대략 [-scale, scale]
        normalized = (x / self.scale + 1) / 2  # [0, 1]
        tokens = np.clip(normalized * self.vocab_size, 0, self.vocab_size - 1)
        return tokens.astype(np.int64)
    
    def _dequantize(self, tokens: np.ndarray) -> np.ndarray:
        """토큰을 연속값으로 변환."""
        normalized = tokens / self.vocab_size  # [0, 1]
        x = (normalized * 2 - 1) * self.scale  # [-scale, scale]
        return x
    
    def sample(self, batch: int, difficulty: float = 0.5) -> tuple:
        """샘플 생성.
        
        Args:
            batch: 배치 크기
            difficulty: 난이도 (0=쉬움, 1=어려움)
                - 낮은 난이도: 작은 zeta (진동 명확)
                - 높은 난이도: 다양한 zeta
        
        Returns:
            input_ids: (batch, seq_len)
            target_ids: (batch, seq_len)
            params: 파라미터 정보
        """
        inputs = []
        targets = []
        params = []
        
        for _ in range(batch):
            # 파라미터 샘플링
            if difficulty < 0.3:
                # Easy: underdamped only
                zeta = np.random.uniform(0.1, 0.5)
            elif difficulty < 0.7:
                # Medium: under + critical
                zeta = np.random.uniform(0.3, 1.2)
            else:
                # Hard: 전체 범위
                zeta = np.random.uniform(*self.zeta_range)
            
            omega = np.random.uniform(*self.omega_range)
            x0 = np.random.uniform(*self.x0_range)
            v0 = np.random.uniform(*self.v0_range)
            
            # 궤적 생성
            traj = generate_trajectory(zeta, omega, x0, v0, self.seq_len + 1, self.dt)
            
            # 양자화
            tokens = self._quantize(traj)
            
            inputs.append(tokens[:self.seq_len])
            targets.append(tokens[1:self.seq_len + 1])
            params.append({'zeta': zeta, 'omega': omega, 'x0': x0, 'v0': v0})
        
        return (torch.tensor(np.array(inputs), dtype=torch.long),
                torch.tensor(np.array(targets), dtype=torch.long),
                params)
    
    def sample_specific(self, batch: int, zeta: float, omega: float = 1.0) -> tuple:
        """특정 파라미터로 샘플 생성."""
        inputs = []
        targets = []
        
        for _ in range(batch):
            x0 = np.random.uniform(*self.x0_range)
            v0 = np.random.uniform(*self.v0_range)
            
            traj = generate_trajectory(zeta, omega, x0, v0, self.seq_len + 1, self.dt)
            tokens = self._quantize(traj)
            
            inputs.append(tokens[:self.seq_len])
            targets.append(tokens[1:self.seq_len + 1])
        
        return (torch.tensor(np.array(inputs), dtype=torch.long),
                torch.tensor(np.array(targets), dtype=torch.long))


# ==============================================================================
# 학습률 스케줄러
# ==============================================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-6):
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
# 학습 및 평가
# ==============================================================================

class OscillatorTrainer:
    """감쇠 진동자 학습 관리자."""
    
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
        self.dataset = OscillatorDataset(
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
        
        self.history = defaultdict(list)
        self.best_loss = float('inf')
    
    def train_epoch(self, epoch: int):
        self.model.train()
        
        difficulty = min(1.0, epoch / (self.config["epochs"] * 0.7))
        
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_topo = 0.0
        epoch_acc = 0.0
        steps = 0
        
        for _ in range(self.config["steps_per_epoch"]):
            input_ids, target_ids, _ = self.dataset.sample(
                self.config["batch_size"], difficulty
            )
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            logits, diag = self.model(input_ids)
            
            ce_loss = F.cross_entropy(
                logits.view(-1, self.config["vocab_size"]), 
                target_ids.view(-1)
            )
            topo_loss = diag['total_violation'] * self.config["topo_weight"]
            loss = ce_loss + topo_loss
            
            pred = logits.argmax(dim=-1)
            acc = (pred == target_ids).float().mean().item()
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()
            epoch_ce += ce_loss.item()
            epoch_topo += topo_loss
            epoch_acc += acc
            steps += 1
        
        lr = self.scheduler.step()
        
        return {
            "loss": epoch_loss / steps,
            "ce": epoch_ce / steps,
            "topo": epoch_topo / steps,
            "acc": epoch_acc / steps,
            "lr": lr,
            "difficulty": difficulty,
        }
    
    @torch.no_grad()
    def evaluate_by_damping(self):
        """감쇠비별 평가."""
        self.model.eval()
        
        results = {}
        for name, zeta in [("under_0.3", 0.3), ("critical_1.0", 1.0), ("over_1.5", 1.5)]:
            input_ids, target_ids = self.dataset.sample_specific(32, zeta)
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            logits, diag = self.model(input_ids)
            ce_loss = F.cross_entropy(logits.view(-1, self.config["vocab_size"]), target_ids.view(-1))
            
            pred = logits.argmax(dim=-1)
            acc = (pred == target_ids).float().mean().item()
            
            # 연속값 오차 계산
            pred_vals = self.dataset._dequantize(pred.cpu().numpy())
            target_vals = self.dataset._dequantize(target_ids.cpu().numpy())
            mae = np.abs(pred_vals - target_vals).mean()
            
            results[name] = {"ce": ce_loss.item(), "acc": acc, "mae": mae}
        
        return results
    
    @torch.no_grad()
    def visualize_prediction(self, save_path: str = "oscillator_prediction.png"):
        """예측 시각화."""
        if not HAS_MATPLOTLIB:
            print("\n  (matplotlib not available, skipping visualization)")
            return
        
        self.model.eval()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        for idx, (zeta, title) in enumerate([
            (0.2, "Underdamped (ζ=0.2)"),
            (1.0, "Critical (ζ=1.0)"),
            (1.5, "Overdamped (ζ=1.5)")
        ]):
            ax_traj = axes[0, idx]
            ax_err = axes[1, idx]
            
            # 샘플 생성
            omega, x0, v0 = 1.0, 3.0, 0.0
            t = np.arange(self.config["seq_len"] + 1) * self.config["dt"]
            actual_traj = generate_trajectory(zeta, omega, x0, v0, self.config["seq_len"] + 1, self.config["dt"])
            
            # 토큰화
            tokens = self.dataset._quantize(actual_traj)
            input_tokens = torch.tensor(tokens[:self.config["seq_len"]], dtype=torch.long).unsqueeze(0).to(self.device)
            
            # 예측
            logits, _ = self.model(input_tokens)
            pred_tokens = logits[0].argmax(dim=-1).cpu().numpy()
            pred_traj = self.dataset._dequantize(pred_tokens)
            
            # 실제 다음 값
            actual_next = actual_traj[1:]
            
            # 궤적 플롯
            ax_traj.plot(t[:-1], actual_traj[:-1], 'b-', label='Actual', linewidth=2)
            ax_traj.plot(t[:-1], pred_traj, 'r--', label='Predicted', linewidth=2)
            ax_traj.set_xlabel('Time')
            ax_traj.set_ylabel('x(t)')
            ax_traj.set_title(title)
            ax_traj.legend()
            ax_traj.grid(True, alpha=0.3)
            
            # 오차 플롯
            error = np.abs(pred_traj - actual_next)
            ax_err.plot(t[:-1], error, 'g-', linewidth=2)
            ax_err.set_xlabel('Time')
            ax_err.set_ylabel('|Error|')
            ax_err.set_title(f'MAE: {error.mean():.4f}')
            ax_err.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"\n  시각화 저장: {save_path}")
    
    def train(self):
        print("=" * 70)
        print("TLM Damped Oscillator Training")
        print("=" * 70)
        print(f"  Epochs: {self.config['epochs']}, LR: {self.config['lr']}")
        print(f"  Model: {sum(p.numel() for p in self.model.parameters()):,} params")
        print(f"  Task: 감쇠 진동자 다음 상태 예측")
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
                      f"Loss={metrics['loss']:.4f}, CE={metrics['ce']:.4f}, "
                      f"Acc={metrics['acc']*100:.1f}%, Topo={metrics['topo']:.6f}, "
                      f"Time={elapsed:.1f}s")
        
        total_time = time.perf_counter() - start_time
        
        # 감쇠비별 평가
        print("\n" + "-" * 70)
        print("감쇠비별 성능 평가:")
        eval_results = self.evaluate_by_damping()
        for name, res in eval_results.items():
            print(f"  {name:15s}: CE={res['ce']:.4f}, Acc={res['acc']*100:.1f}%, MAE={res['mae']:.4f}")
        
        # 시각화
        self.visualize_prediction()
        
        # 요약
        print("\n" + "=" * 70)
        print("학습 요약")
        print("=" * 70)
        print(f"  총 시간: {total_time:.1f}s")
        print(f"  초기 Loss: {self.history['loss'][0]:.4f}")
        print(f"  최종 Loss: {self.history['loss'][-1]:.4f}")
        print(f"  최종 Accuracy: {self.history['acc'][-1]*100:.1f}%")
        print(f"\n  Lyapunov 관점: 모델이 '안정 수렴 궤적'을 학습했는가?")
        print(f"    → Underdamped MAE: {eval_results['under_0.3']['mae']:.4f}")
        print(f"    → Critical MAE: {eval_results['critical_1.0']['mae']:.4f}")
        print(f"    → Overdamped MAE: {eval_results['over_1.5']['mae']:.4f}")
        
        return self.history


# ==============================================================================
# 메인
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="TLM Oscillator Training")
    
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.1, help="시간 간격")
    
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--pc-steps", type=int, default=5)
    
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--save", action="store_true", help="Best 모델 저장")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = {
        "epochs": args.epochs if not args.fast else 100,
        "lr": args.lr,
        "warmup_epochs": args.warmup if not args.fast else 10,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "dt": args.dt,
        "weight_decay": 0.01,
        "topo_weight": 0.1,
        "steps_per_epoch": 10 if not args.fast else 5,
        "log_every": args.log_every if not args.fast else 20,
        "save_best": args.save,
        
        "vocab_size": 500,
        "embed_dim": args.embed_dim,
        "relation_dim": 32,
        "context_window": max(32, args.seq_len),  # Fix: Ensure context window >= seq_len
        "num_heads": args.heads,
        "num_layers": args.layers,
        "pc_steps": args.pc_steps,
        "pc_alpha": 0.8,
        "dropout": 0.1,
    }
    
    trainer = OscillatorTrainer(config)
    history = trainer.train()
    
    # 모델 저장
    if args.save:
        torch.save({
            'model_state_dict': trainer.model.state_dict(),
            'config': config,
            'history': dict(history),
        }, 'best_oscillator.pt')
        print(f"\n  모델 저장: best_oscillator.pt")


if __name__ == "__main__":
    main()
