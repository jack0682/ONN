#!/usr/bin/env python3
"""TLM Collatz Conjecture Training Script.

콜라츠 추측(3n+1 문제)을 사용한 TLM 수학적 검증.

콜라츠 규칙:
- 짝수: n → n/2
- 홀수: n → 3n+1
- 모든 수는 결국 4 → 2 → 1 사이클에 도달 (추측)

학습 목표:
1. 다음 값 예측: n → next(n)
2. 궤적 생성: n → sequence until 1
3. 사이클 도달 패턴 학습

실행:
    PYTHONPATH=./src python3 scripts/train_collatz.py --epochs 1000
    PYTHONPATH=./src python3 scripts/train_collatz.py --epochs 2000 --embed-dim 512

Author: Claude
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# PYTHONPATH 설정
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from onn.tlm import TopologicalLanguageModel, TLMConfig


# ==============================================================================
# 콜라츠 데이터 생성
# ==============================================================================

def collatz_next(n: int) -> int:
    """콜라츠 다음 값."""
    if n % 2 == 0:
        return n // 2
    else:
        return 3 * n + 1


def collatz_sequence(n: int, max_len: int = 100) -> list:
    """콜라츠 시퀀스 생성."""
    seq = [n]
    while n != 1 and len(seq) < max_len:
        n = collatz_next(n)
        seq.append(n)
    return seq


def collatz_steps(n: int, max_steps: int = 1000) -> int:
    """1에 도달하는 스텝 수."""
    steps = 0
    while n != 1 and steps < max_steps:
        n = collatz_next(n)
        steps += 1
    return steps


class CollatzDataset:
    """콜라츠 시퀀스 데이터셋."""
    
    def __init__(self, vocab_size: int, seq_len: int, max_start: int = None):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.max_start = max_start or vocab_size
        
        # 사전 계산된 시퀀스 캐시
        self._cache = {}
    
    def _get_sequence(self, start: int) -> list:
        """캐시된 시퀀스 반환."""
        if start not in self._cache:
            self._cache[start] = collatz_sequence(start, max_len=self.seq_len * 2)
        return self._cache[start]
    
    def sample_next_prediction(self, batch: int) -> tuple:
        """다음 값 예측용 샘플.
        
        Returns:
            input_ids: (batch, seq_len) - 콜라츠 시퀀스
            target_ids: (batch, seq_len) - 다음 값들
        """
        inputs = []
        targets = []
        
        for _ in range(batch):
            # 시작점 랜덤 선택 (2 이상)
            start = np.random.randint(2, self.max_start)
            seq = self._get_sequence(start)
            
            # 시퀀스가 충분히 길면 사용
            if len(seq) >= self.seq_len + 1:
                input_seq = seq[:self.seq_len]
                target_seq = seq[1:self.seq_len + 1]
            else:
                # 짧으면 패딩 (1로)
                input_seq = seq[:self.seq_len] + [1] * (self.seq_len - len(seq))
                target_seq = seq[1:self.seq_len + 1] + [1] * (self.seq_len - len(seq) + 1)
                target_seq = target_seq[:self.seq_len]
            
            # vocab_size로 클리핑 (큰 수는 모듈러)
            input_seq = [x % self.vocab_size for x in input_seq]
            target_seq = [x % self.vocab_size for x in target_seq]
            
            inputs.append(input_seq)
            targets.append(target_seq)
        
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
    
    def sample_trajectory(self, batch: int) -> tuple:
        """전체 궤적 샘플.
        
        Returns:
            input_ids: (batch, seq_len) - 콜라츠 궤적
            pattern: 시작값
        """
        inputs = []
        starts = []
        
        for _ in range(batch):
            start = np.random.randint(2, self.max_start)
            seq = self._get_sequence(start)
            
            if len(seq) >= self.seq_len:
                input_seq = seq[:self.seq_len]
            else:
                input_seq = seq + [1] * (self.seq_len - len(seq))
            
            input_seq = [x % self.vocab_size for x in input_seq]
            inputs.append(input_seq)
            starts.append(start)
        
        return torch.tensor(inputs, dtype=torch.long), starts
    
    def sample_mixed(self, batch: int, difficulty: float = 0.5) -> tuple:
        """난이도에 따른 혼합 샘플.
        
        낮은 난이도: 작은 시작값 (짧은 궤적)
        높은 난이도: 큰 시작값 (긴 궤적)
        """
        # 난이도에 따라 시작 범위 조절
        max_start = int(2 + (self.max_start - 2) * difficulty)
        max_start = max(3, max_start)
        
        inputs = []
        targets = []
        
        for _ in range(batch):
            start = np.random.randint(2, max_start)
            seq = self._get_sequence(start)
            
            if len(seq) >= self.seq_len + 1:
                input_seq = seq[:self.seq_len]
                target_seq = seq[1:self.seq_len + 1]
            else:
                input_seq = seq[:self.seq_len] + [1] * (self.seq_len - len(seq))
                target_seq = seq[1:self.seq_len + 1] + [1] * max(0, self.seq_len - len(seq) + 1)
                target_seq = target_seq[:self.seq_len]
            
            input_seq = [x % self.vocab_size for x in input_seq]
            target_seq = [x % self.vocab_size for x in target_seq]
            
            inputs.append(input_seq)
            targets.append(target_seq)
        
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


# ==============================================================================
# 학습률 스케줄러
# ==============================================================================

class WarmupCosineScheduler:
    """Warmup + Cosine Annealing 스케줄러."""
    
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
# 학습 루프
# ==============================================================================

class CollatzTrainer:
    """콜라츠 학습 관리자."""
    
    def __init__(self, config: dict):
        self.config = config
        
        # 디바이스 선택
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print(f"  Using: MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"  Using: CUDA GPU")
        else:
            self.device = torch.device("cpu")
            print(f"  Using: CPU")
        
        # 모델 생성
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
        self.dataset = CollatzDataset(
            config["vocab_size"], 
            config["seq_len"],
            max_start=config["max_start"]
        )
        
        # 옵티마이저
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
        
        # 스케줄러
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=config["warmup_epochs"],
            total_epochs=config["epochs"],
            min_lr=config["lr"] * 0.01,
        )
        
        # 로깅
        self.history = defaultdict(list)
        self.best_loss = float('inf')
    
    def train_epoch(self, epoch: int):
        """한 에폭 학습."""
        self.model.train()
        
        # Curriculum: 난이도 점진적 증가
        difficulty = min(1.0, epoch / (self.config["epochs"] * 0.8))
        
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_topo = 0.0
        epoch_acc = 0.0
        steps = 0
        
        steps_per_epoch = self.config.get("steps_per_epoch", 10)
        
        for _ in range(steps_per_epoch):
            input_ids, target_ids = self.dataset.sample_mixed(
                self.config["batch_size"], difficulty
            )
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Forward
            logits, diag = self.model(input_ids)
            
            # Loss
            ce_loss = F.cross_entropy(
                logits.view(-1, self.config["vocab_size"]), 
                target_ids.view(-1)
            )
            topo_loss = diag['total_violation'] * self.config["topo_weight"]
            loss = ce_loss + topo_loss
            
            # Accuracy
            pred = logits.argmax(dim=-1)
            acc = (pred == target_ids).float().mean().item()
            
            # Backward
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
    def evaluate(self):
        """평가: 다양한 시작값으로 테스트."""
        self.model.eval()
        
        results = {}
        
        # 난이도별 평가
        for diff_name, diff_val, max_s in [("easy", 0.2, 20), ("medium", 0.5, 100), ("hard", 0.8, 500)]:
            input_ids, target_ids = self.dataset.sample_mixed(32, diff_val)
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            logits, diag = self.model(input_ids)
            ce_loss = F.cross_entropy(logits.view(-1, self.config["vocab_size"]), target_ids.view(-1))
            
            pred = logits.argmax(dim=-1)
            acc = (pred == target_ids).float().mean().item()
            
            # 첫 번째 다음 값 정확도 (next-step prediction)
            next_acc = (pred[:, 0] == target_ids[:, 0]).float().mean().item()
            
            results[diff_name] = {
                "ce": ce_loss.item(), 
                "acc": acc, 
                "next_acc": next_acc,
                "viol": diag["total_violation"]
            }
        
        return results
    
    @torch.no_grad()
    def demo_prediction(self):
        """데모: 특정 수에서 시작하는 예측."""
        self.model.eval()
        
        test_starts = [7, 27, 97, 127]
        print("\n  데모 예측:")
        
        for start in test_starts:
            # 실제 시퀀스
            actual = collatz_sequence(start, max_len=8)
            actual_clipped = [x % self.config["vocab_size"] for x in actual[:8]]
            
            # 모델 예측
            input_seq = torch.tensor([[start % self.config["vocab_size"]]], device=self.device)
            
            generated = [start % self.config["vocab_size"]]
            for _ in range(7):
                logits, _ = self.model(input_seq)
                next_token = logits[0, -1].argmax().item()
                generated.append(next_token)
                input_seq = torch.tensor([generated[-self.config["context_window"]:]], device=self.device)
            
            # 정확도 계산
            correct = sum(1 for a, p in zip(actual_clipped, generated) if a == p)
            
            print(f"    n={start:3d}: 실제={actual_clipped[:6]}... 예측={generated[:6]}... ({correct}/8 일치)")
    
    def train(self):
        """전체 학습."""
        print("=" * 70)
        print("TLM Collatz Conjecture Training")
        print("=" * 70)
        print(f"  Epochs: {self.config['epochs']}, LR: {self.config['lr']}")
        print(f"  Warmup: {self.config['warmup_epochs']} epochs")
        print(f"  Max Start: {self.config['max_start']}")
        print(f"  Model: {sum(p.numel() for p in self.model.parameters()):,} params")
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
                      f"Diff={metrics['difficulty']:.2f}, Time={elapsed:.1f}s")
        
        total_time = time.perf_counter() - start_time
        
        # 최종 평가
        print("\n" + "-" * 70)
        print("최종 난이도별 평가:")
        eval_results = self.evaluate()
        for diff, res in eval_results.items():
            print(f"  {diff:8s}: CE={res['ce']:.4f}, Acc={res['acc']*100:.1f}%, Next={res['next_acc']*100:.1f}%")
        
        # 데모 예측
        self.demo_prediction()
        
        # 요약
        print("\n" + "=" * 70)
        print("학습 요약")
        print("=" * 70)
        print(f"  총 시간: {total_time:.1f}s")
        print(f"  초기 Loss: {self.history['loss'][0]:.4f}")
        print(f"  최종 Loss: {self.history['loss'][-1]:.4f}")
        print(f"  최고 Loss: {self.best_loss:.4f}")
        print(f"  최종 Accuracy: {self.history['acc'][-1]*100:.1f}%")
        
        return self.history


# ==============================================================================
# 메인
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="TLM Collatz Training")
    
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--max-start", type=int, default=1000, help="콜라츠 시작값 최대")
    
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--pc-steps", type=int, default=5)
    
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--fast", action="store_true")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = {
        "epochs": args.epochs if not args.fast else 100,
        "lr": args.lr,
        "warmup_epochs": args.warmup if not args.fast else 10,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "max_start": args.max_start,
        "weight_decay": 0.01,
        "topo_weight": 0.1,
        "steps_per_epoch": 10 if not args.fast else 5,
        "log_every": args.log_every if not args.fast else 20,
        
        "vocab_size": 2000,  # 콜라츠는 큰 수가 나올 수 있음
        "embed_dim": args.embed_dim,
        "relation_dim": 64,
        "context_window": max(32, args.seq_len),
        "num_heads": args.heads,
        "num_layers": args.layers,
        "pc_steps": args.pc_steps,
        "pc_alpha": 0.8,
        "dropout": 0.1,
    }
    
    trainer = CollatzTrainer(config)
    history = trainer.train()


if __name__ == "__main__":
    main()
