#!/usr/bin/env python3
"""TLM Advanced Training Script.

학습 효과를 높이기 위한 고급 학습 스크립트.

특징:
1. Warmup + Cosine Annealing 학습률 스케줄
2. 점진적 난이도 증가 (Curriculum Learning)
3. Early Stopping & Best Model 저장
4. 상세 로깅 및 체크포인트
5. 다양한 패턴 데이터

실행:
    python scripts/train_tlm.py

    # 긴 학습
    python scripts/train_tlm.py --epochs 2000 --lr 5e-4

    # 빠른 테스트
    python scripts/train_tlm.py --epochs 100 --fast

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
# 데이터 생성
# ==============================================================================

class PatternDataset:
    """다양한 패턴의 시퀀스 생성기."""
    
    def __init__(self, vocab_size: int, seq_len: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        
        self.patterns = {
            "repeat_2": self._repeat_2,       # ABABAB
            "repeat_3": self._repeat_3,       # ABCABC
            "arithmetic_inc": self._arith_inc, # 1,2,3,4
            "arithmetic_skip": self._arith_skip, # 1,3,5,7
            "copy": self._copy,               # ABCD -> ABCD
            "reverse": self._reverse,         # ABCD -> DCBA
            "palindrome": self._palindrome,   # ABCCBA
            "fibonacci_mod": self._fib,       # Fibonacci mod vocab
            "random": self._random,
        }
    
    def _repeat_2(self, batch):
        base = torch.randint(0, self.vocab_size, (batch, 2))
        repeats = self.seq_len // 2 + 1
        return base.repeat(1, repeats)[:, :self.seq_len]
    
    def _repeat_3(self, batch):
        base = torch.randint(0, self.vocab_size, (batch, 3))
        repeats = self.seq_len // 3 + 1
        return base.repeat(1, repeats)[:, :self.seq_len]
    
    def _arith_inc(self, batch):
        start = torch.randint(0, self.vocab_size - self.seq_len, (batch, 1))
        return (start + torch.arange(self.seq_len)) % self.vocab_size
    
    def _arith_skip(self, batch):
        start = torch.randint(0, self.vocab_size - self.seq_len * 2, (batch, 1))
        return (start + torch.arange(self.seq_len) * 2) % self.vocab_size
    
    def _copy(self, batch):
        half = self.seq_len // 2
        first = torch.randint(0, self.vocab_size, (batch, half))
        return torch.cat([first, first], dim=1)[:, :self.seq_len]
    
    def _reverse(self, batch):
        half = self.seq_len // 2
        first = torch.randint(0, self.vocab_size, (batch, half))
        return torch.cat([first, first.flip(1)], dim=1)[:, :self.seq_len]
    
    def _palindrome(self, batch):
        half = self.seq_len // 2
        first = torch.randint(0, self.vocab_size, (batch, half))
        if self.seq_len % 2 == 1:
            mid = torch.randint(0, self.vocab_size, (batch, 1))
            return torch.cat([first, mid, first.flip(1)], dim=1)[:, :self.seq_len]
        return torch.cat([first, first.flip(1)], dim=1)[:, :self.seq_len]
    
    def _fib(self, batch):
        result = torch.zeros(batch, self.seq_len, dtype=torch.long)
        result[:, 0] = torch.randint(1, 10, (batch,))
        result[:, 1] = torch.randint(1, 10, (batch,))
        for i in range(2, self.seq_len):
            result[:, i] = (result[:, i-1] + result[:, i-2]) % self.vocab_size
        return result
    
    def _random(self, batch):
        return torch.randint(0, self.vocab_size, (batch, self.seq_len))
    
    def sample(self, batch: int, pattern: str = None):
        """샘플 생성."""
        if pattern is None:
            pattern = np.random.choice(list(self.patterns.keys()))
        return self.patterns[pattern](batch).long(), pattern
    
    def sample_curriculum(self, batch: int, difficulty: float):
        """Curriculum Learning: 난이도에 따라 패턴 선택."""
        if difficulty < 0.3:
            # Easy: 단순 반복
            patterns = ["repeat_2", "repeat_3", "arithmetic_inc"]
        elif difficulty < 0.6:
            # Medium: 복사 및 변형
            patterns = ["arithmetic_inc", "arithmetic_skip", "copy"]
        elif difficulty < 0.9:
            # Hard: 역순 및 팰린드롬
            patterns = ["copy", "reverse", "palindrome", "fibonacci_mod"]
        else:
            # Expert: 모든 패턴
            patterns = list(self.patterns.keys())
        
        pattern = np.random.choice(patterns)
        return self.patterns[pattern](batch).long(), pattern


# ==============================================================================
# 학습률 스케줄러
# ==============================================================================

class WarmupCosineScheduler:
    """Warmup + Cosine Annealing 스케줄러."""
    
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, 
                 min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        
        return lr
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


# ==============================================================================
# 학습 루프
# ==============================================================================

class TLMTrainer:
    """TLM 학습 관리자."""
    
    def __init__(self, config: dict):
        self.config = config
        
        # 디바이스 선택: MPS (Apple Silicon) > CUDA > CPU
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
        self.dataset = PatternDataset(config["vocab_size"], config["seq_len"])
        
        # 옵티마이저
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            betas=(0.9, 0.98),
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
        self.patience_counter = 0
    
    def train_epoch(self, epoch: int):
        """한 에폭 학습."""
        self.model.train()
        
        # Curriculum: 난이도 점진적 증가
        difficulty = min(1.0, epoch / (self.config["epochs"] * 0.7))
        
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_topo = 0.0
        steps = 0
        
        # 에폭당 스텝 수
        steps_per_epoch = self.config.get("steps_per_epoch", 10)
        
        for _ in range(steps_per_epoch):
            input_ids, pattern = self.dataset.sample_curriculum(
                self.config["batch_size"], difficulty
            )
            input_ids = input_ids.to(self.device)
            
            # Target: next token prediction
            target_ids = torch.cat([
                input_ids[:, 1:], 
                torch.randint(0, self.config["vocab_size"], (input_ids.size(0), 1), device=self.device)
            ], dim=1)
            
            # Forward
            logits, diag = self.model(input_ids)
            
            # Loss
            ce_loss = F.cross_entropy(
                logits.view(-1, self.config["vocab_size"]), 
                target_ids.view(-1)
            )
            topo_loss = diag['total_violation'] * self.config["topo_weight"]
            loss = ce_loss + topo_loss
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()
            epoch_ce += ce_loss.item()
            epoch_topo += topo_loss
            steps += 1
        
        # 스케줄러 스텝
        lr = self.scheduler.step()
        
        # 평균
        return {
            "loss": epoch_loss / steps,
            "ce": epoch_ce / steps,
            "topo": epoch_topo / steps,
            "lr": lr,
            "difficulty": difficulty,
        }
    
    @torch.no_grad()
    def evaluate(self):
        """평가."""
        self.model.eval()
        
        results = {}
        for pattern in self.dataset.patterns.keys():
            input_ids, _ = self.dataset.sample(self.config["batch_size"], pattern)
            input_ids = input_ids.to(self.device)
            target_ids = torch.cat([
                input_ids[:, 1:], 
                torch.zeros(input_ids.size(0), 1, dtype=torch.long, device=self.device)
            ], dim=1)
            
            logits, diag = self.model(input_ids)
            ce_loss = F.cross_entropy(logits.view(-1, self.config["vocab_size"]), target_ids.view(-1))
            
            pred = logits.argmax(dim=-1)
            acc = (pred == target_ids).float().mean().item()
            
            results[pattern] = {"ce": ce_loss.item(), "acc": acc, "viol": diag["total_violation"]}
        
        return results
    
    def train(self):
        """전체 학습."""
        print("=" * 70)
        print("TLM Advanced Training")
        print("=" * 70)
        print(f"  Epochs: {self.config['epochs']}, LR: {self.config['lr']}")
        print(f"  Warmup: {self.config['warmup_epochs']} epochs")
        print(f"  Model: {sum(p.numel() for p in self.model.parameters()):,} params")
        print("-" * 70)
        
        start_time = time.perf_counter()
        
        for epoch in range(1, self.config["epochs"] + 1):
            metrics = self.train_epoch(epoch)
            
            # 로깅
            for k, v in metrics.items():
                self.history[k].append(v)
            
            # Early stopping 체크
            if metrics["loss"] < self.best_loss:
                self.best_loss = metrics["loss"]
                self.patience_counter = 0
                # Best 모델 저장
                if self.config.get("save_best", False):
                    torch.save(self.model.state_dict(), "best_tlm.pt")
            else:
                self.patience_counter += 1
            
            # 진행 출력
            if epoch % self.config["log_every"] == 0:
                elapsed = time.perf_counter() - start_time
                print(f"  Epoch {epoch:4d}/{self.config['epochs']}: "
                      f"Loss={metrics['loss']:.4f}, CE={metrics['ce']:.4f}, "
                      f"Topo={metrics['topo']:.6f}, LR={metrics['lr']:.6f}, "
                      f"Diff={metrics['difficulty']:.2f}, Time={elapsed:.1f}s")
            
            # Early stopping
            if self.patience_counter >= self.config.get("patience", 999999):
                print(f"\n  Early stopping at epoch {epoch}")
                break
        
        total_time = time.perf_counter() - start_time
        
        # 최종 평가
        print("\n" + "-" * 70)
        print("최종 패턴별 평가:")
        eval_results = self.evaluate()
        for pattern, res in sorted(eval_results.items(), key=lambda x: x[1]["ce"]):
            print(f"  {pattern:15s}: CE={res['ce']:.4f}, Acc={res['acc']*100:5.1f}%")
        
        # 요약
        print("\n" + "=" * 70)
        print("학습 요약")
        print("=" * 70)
        print(f"  총 시간: {total_time:.1f}s")
        print(f"  초기 Loss: {self.history['loss'][0]:.4f}")
        print(f"  최종 Loss: {self.history['loss'][-1]:.4f}")
        print(f"  최고 Loss: {self.best_loss:.4f}")
        print(f"  Loss 감소: {(self.history['loss'][0] - self.history['loss'][-1]) / self.history['loss'][0] * 100:.1f}%")
        
        return self.history


# ==============================================================================
# 메인
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="TLM Training Script")
    
    # 학습 설정
    parser.add_argument("--epochs", type=int, default=1000, help="총 에폭 수")
    parser.add_argument("--lr", type=float, default=5e-4, help="초기 학습률")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup 에폭")
    parser.add_argument("--batch-size", type=int, default=16, help="배치 크기")
    parser.add_argument("--seq-len", type=int, default=12, help="시퀀스 길이")
    
    # 모델 설정
    parser.add_argument("--embed-dim", type=int, default=64, help="임베딩 차원")
    parser.add_argument("--layers", type=int, default=3, help="레이어 수")
    parser.add_argument("--heads", type=int, default=4, help="어텐션 헤드 수")
    parser.add_argument("--pc-steps", type=int, default=5, help="PC Solver 스텝")
    
    # 기타
    parser.add_argument("--log-every", type=int, default=50, help="로깅 주기")
    parser.add_argument("--fast", action="store_true", help="빠른 테스트 모드")
    parser.add_argument("--save", action="store_true", help="Best 모델 저장")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = {
        # 학습
        "epochs": args.epochs if not args.fast else 100,
        "lr": args.lr,
        "warmup_epochs": args.warmup if not args.fast else 10,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "weight_decay": 0.01,
        "topo_weight": 0.1,
        "steps_per_epoch": 10 if not args.fast else 5,
        "log_every": args.log_every if not args.fast else 20,
        "patience": 200,
        "save_best": args.save,
        
        # 모델
        "vocab_size": 500,
        "embed_dim": args.embed_dim,
        "relation_dim": 32,
        "context_window": 16,
        "num_heads": args.heads,
        "num_layers": args.layers,
        "pc_steps": args.pc_steps,
        "pc_alpha": 0.8,
        "dropout": 0.1,
    }
    
    trainer = TLMTrainer(config)
    history = trainer.train()
    
    # 히스토리 저장
    if args.save:
        with open("training_history.json", "w") as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f)
        print("\n  모델과 히스토리가 저장되었습니다.")


if __name__ == "__main__":
    main()
