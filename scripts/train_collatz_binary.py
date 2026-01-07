#!/usr/bin/env python3
"""TLM Binary Collatz Training Script.

이진 표현 기반 콜라츠 학습 - 무한 큰 수 일반화.

핵심 아이디어:
- 숫자를 이진 표현으로 변환 (n=27 → "11011")
- vocab_size = 3 (0, 1, PAD)
- 콜라츠 규칙이 비트 연산으로 해석됨:
  - 짝수 (LSB=0): n/2 = 오른쪽 shift
  - 홀수 (LSB=1): 3n+1 = 복잡한 비트 연산

이점:
- 1조(10¹²) 이상의 수도 50비트로 표현 가능
- 학습된 규칙이 모든 크기의 수에 일반화

실행:
    PYTHONPATH=./src python3 scripts/train_collatz_binary.py
    PYTHONPATH=./src python3 scripts/train_collatz_binary.py --epochs 2000 --test-large

Author: Claude
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from onn.tlm import TopologicalLanguageModel, TLMConfig


# ==============================================================================
# 이진 콜라츠 데이터 생성
# ==============================================================================

# 토큰 정의
PAD_TOKEN = 0
ZERO_BIT = 1
ONE_BIT = 2

def collatz_next(n: int) -> int:
    """콜라츠 다음 값."""
    if n % 2 == 0:
        return n // 2
    else:
        return 3 * n + 1

def int_to_binary_tokens(n: int, max_bits: int = 64) -> list:
    """정수를 이진 토큰 시퀀스로 변환 (LSB first).
    
    예: 27 = 11011 → [1,1,0,1,1] (LSB first)
    토큰: 0=PAD, 1=ZERO, 2=ONE
    """
    if n == 0:
        return [ZERO_BIT]
    
    bits = []
    while n > 0 and len(bits) < max_bits:
        bits.append(ONE_BIT if (n & 1) else ZERO_BIT)
        n >>= 1
    
    return bits

def binary_tokens_to_int(tokens: list) -> int:
    """이진 토큰 시퀀스를 정수로 변환 (LSB first)."""
    result = 0
    for i, t in enumerate(tokens):
        if t == ONE_BIT:
            result |= (1 << i)
    return result

def pad_sequence(seq: list, length: int) -> list:
    """시퀀스를 지정된 길이로 패딩."""
    if len(seq) >= length:
        return seq[:length]
    return seq + [PAD_TOKEN] * (length - len(seq))


class BinaryCollatzDataset:
    """이진 표현 콜라츠 데이터셋."""
    
    def __init__(self, seq_len: int = 64, max_bits: int = 48):
        """
        Args:
            seq_len: 전체 시퀀스 길이 (현재 + 다음 상태)
            max_bits: 최대 비트 수 (48비트 = 최대 ~281조)
        """
        self.seq_len = seq_len
        self.max_bits = max_bits
        self.half_len = seq_len // 2  # 현재 상태와 다음 상태 각각의 길이
    
    def sample(self, batch: int, difficulty: float = 0.5) -> tuple:
        """샘플 생성.
        
        입력: 현재 수의 이진 표현
        출력: 다음 수의 이진 표현
        
        난이도:
        - 0.0: 작은 수 (1~1000)
        - 0.5: 중간 수 (1K~1M)
        - 1.0: 큰 수 (1M~1T)
        """
        inputs = []
        targets = []
        
        for _ in range(batch):
            # 난이도에 따른 범위
            if difficulty < 0.3:
                n = np.random.randint(2, 1000)
            elif difficulty < 0.6:
                n = np.random.randint(1000, 1_000_000)
            elif difficulty < 0.9:
                n = np.random.randint(1_000_000, 1_000_000_000)
            else:
                # 1조까지
                n = np.random.randint(1_000_000_000, 1_000_000_000_000)
            
            # 콜라츠 다음 값
            next_n = collatz_next(n)
            
            # 이진 변환
            current_bits = int_to_binary_tokens(n, self.max_bits)
            next_bits = int_to_binary_tokens(next_n, self.max_bits)
            
            # 패딩
            current_padded = pad_sequence(current_bits, self.half_len)
            next_padded = pad_sequence(next_bits, self.half_len)
            
            # 입력: 현재 상태, 출력: 다음 상태
            inputs.append(current_padded)
            targets.append(next_padded)
        
        return (torch.tensor(inputs, dtype=torch.long),
                torch.tensor(targets, dtype=torch.long))
    
    def sample_specific(self, n: int) -> tuple:
        """특정 수에 대한 샘플."""
        next_n = collatz_next(n)
        
        current_bits = pad_sequence(
            int_to_binary_tokens(n, self.max_bits), 
            self.half_len
        )
        next_bits = pad_sequence(
            int_to_binary_tokens(next_n, self.max_bits), 
            self.half_len
        )
        
        return (torch.tensor([current_bits], dtype=torch.long),
                torch.tensor([next_bits], dtype=torch.long),
                n, next_n)


# ==============================================================================
# 학습
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


class BinaryCollatzTrainer:
    """이진 콜라츠 학습."""
    
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
        
        # 모델 (vocab=3: PAD, 0, 1)
        self.model_config = TLMConfig(
            vocab_size=3,  # PAD, ZERO, ONE
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
        self.dataset = BinaryCollatzDataset(
            seq_len=config["seq_len"],
            max_bits=config["max_bits"]
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
        epoch_acc = 0.0
        steps = 0
        
        for _ in range(self.config["steps_per_epoch"]):
            input_ids, target_ids = self.dataset.sample(
                self.config["batch_size"], difficulty
            )
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            logits, diag = self.model(input_ids)
            
            # Loss: 각 비트 예측
            loss = F.cross_entropy(
                logits.view(-1, 3),  # vocab=3
                target_ids.view(-1)
            )
            
            pred = logits.argmax(dim=-1)
            acc = (pred == target_ids).float().mean().item()
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc
            steps += 1
        
        lr = self.scheduler.step()
        
        return {
            "loss": epoch_loss / steps,
            "acc": epoch_acc / steps,
            "lr": lr,
            "difficulty": difficulty,
        }
    
    @torch.no_grad()
    def evaluate_scales(self):
        """다양한 스케일에서 평가."""
        self.model.eval()
        
        scales = [
            ("1~1K", 2, 1000),
            ("1K~1M", 1000, 1_000_000),
            ("1M~1B", 1_000_000, 1_000_000_000),
            ("1B~1T", 1_000_000_000, 1_000_000_000_000),
        ]
        
        results = {}
        for name, low, high in scales:
            correct = 0
            total = 0
            
            for _ in range(32):
                n = np.random.randint(low, min(high, 10**12))  # 안전 범위
                input_ids, target_ids, _, actual_next = self.dataset.sample_specific(n)
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                logits, _ = self.model(input_ids)
                pred_tokens = logits[0].argmax(dim=-1).cpu().tolist()
                target_tokens = target_ids[0].tolist()
                
                # 비트별 정확도
                bit_correct = sum(1 for p, t in zip(pred_tokens, target_tokens) if p == t)
                correct += bit_correct
                total += len(pred_tokens)
            
            results[name] = correct / total
        
        return results
    
    @torch.no_grad()
    def demo_large_numbers(self):
        """큰 수 데모."""
        self.model.eval()
        
        test_numbers = [
            27,                        # 유명한 예
            9780657630,               # 100억 근처
            989345275647,             # 1조 근처
            9999999999999,            # 10조 근처 (13자리)
        ]
        
        print("\n  큰 수 예측 데모:")
        for n in test_numbers:
            actual_next = collatz_next(n)
            
            input_ids, target_ids, _, _ = self.dataset.sample_specific(n)
            input_ids = input_ids.to(self.device)
            
            logits, _ = self.model(input_ids)
            pred_tokens = logits[0].argmax(dim=-1).cpu().tolist()
            pred_n = binary_tokens_to_int(pred_tokens)
            
            match = "✓" if pred_n == actual_next else "✗"
            print(f"    n={n:>15,} → 실제={actual_next:>15,}, 예측={pred_n:>15,} {match}")
    
    def train(self):
        print("=" * 70)
        print("TLM Binary Collatz Training (Trillion Scale)")
        print("=" * 70)
        print(f"  Epochs: {self.config['epochs']}, LR: {self.config['lr']}")
        print(f"  Max Bits: {self.config['max_bits']} (최대 ~{2**self.config['max_bits']:,})")
        print(f"  Model: {sum(p.numel() for p in self.model.parameters()):,} params")
        print(f"  Vocab: 3 (PAD, ZERO, ONE)")
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
                      f"Diff={metrics['difficulty']:.2f}, Time={elapsed:.1f}s")
        
        total_time = time.perf_counter() - start_time
        
        # 스케일별 평가
        print("\n" + "-" * 70)
        print("스케일별 비트 정확도:")
        eval_results = self.evaluate_scales()
        for scale, acc in eval_results.items():
            status = "✓" if acc > 0.9 else "⚠" if acc > 0.7 else "✗"
            print(f"  {scale:>10}: {acc*100:.1f}% {status}")
        
        # 큰 수 데모
        self.demo_large_numbers()
        
        # 요약
        print("\n" + "=" * 70)
        print("학습 요약")
        print("=" * 70)
        print(f"  총 시간: {total_time:.1f}s")
        print(f"  최종 Loss: {self.history['loss'][-1]:.4f}")
        print(f"  최종 Accuracy: {self.history['acc'][-1]*100:.1f}%")
        
        return self.history


# ==============================================================================
# 메인
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="TLM Binary Collatz Training")
    
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=64, help="시퀀스 길이 (입력+출력)")
    parser.add_argument("--max-bits", type=int, default=48, help="최대 비트 수")
    
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--pc-steps", type=int, default=5)
    
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--test-large", action="store_true", help="큰 수 테스트 강화")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = {
        "epochs": args.epochs if not args.fast else 200,
        "lr": args.lr,
        "warmup_epochs": args.warmup if not args.fast else 20,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "max_bits": args.max_bits,
        "weight_decay": 0.01,
        "steps_per_epoch": 10 if not args.fast else 5,
        "log_every": args.log_every if not args.fast else 40,
        
        "embed_dim": args.embed_dim,
        "relation_dim": 64,
        "context_window": 48,
        "num_heads": args.heads,
        "num_layers": args.layers,
        "pc_steps": args.pc_steps,
        "pc_alpha": 0.8,
        "dropout": 0.1,
    }
    
    trainer = BinaryCollatzTrainer(config)
    history = trainer.train()
    
    if args.save:
        torch.save({
            'model_state_dict': trainer.model.state_dict(),
            'config': config,
            'history': dict(history),
        }, 'best_collatz_binary.pt')
        print(f"\n  모델 저장: best_collatz_binary.pt")


if __name__ == "__main__":
    main()
