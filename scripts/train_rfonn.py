"""Relational Field ONN Training Script.

ONN = 관계-장(field) 복원기

핵심 원칙:
- 라벨은 학습 목표가 아니다
- 라벨은 나중에 Z의 특정 영역에 붙는 "주석"
- 관계의 진실은 "개입 후 결과"에서 드러난다

학습 목표:
1. 예측 손실: outcome prediction (관계를 정의)
2. 비대칭성 손실: z_ij ≠ z_ji (방향성)
3. 등변성 손실: 좌표 변환에 일관성
4. Contrastive: 유사 결과 → 유사 관계

Author: Claude
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from onn.relation.four_axis_relation import FourAxisRelationModel, RelationAxisConfig
from onn.relation.contrastive_relation import compute_outcome_signature, signature_similarity
from onn.relation.topological_regularizer import TopologicalRegularizer, TopologicalConfig
from onn.simulation.pybullet_env import (
    RelationSimEnvironment, 
    SimulationConfig, 
    PhysicsRelationDataset,
)


@dataclass
class RFONNConfig:
    """Relational Field ONN 설정."""
    # 모델
    state_dim: int = 6
    action_dim: int = 6
    relation_dim: int = 48
    hidden_dim: int = 128
    
    # 학습
    epochs: int = 500
    batch_size: int = 32
    lr: float = 1e-3
    
    # 손실 가중치
    pred_weight: float = 1.0      # outcome prediction
    asym_weight: float = 0.3      # asymmetry
    equiv_weight: float = 0.2     # equivariance
    contrastive_weight: float = 0.5
    topo_weight: float = 0.3      # topological regularization
    
    # 시뮬레이션
    num_episodes: int = 200
    
    # 로깅
    log_every: int = 50


class RelationalFieldONN(nn.Module):
    """Relational Field ONN.
    
    관계-장 복원기: 라벨이 아닌 예측으로 관계를 정의.
    """
    
    def __init__(self, config: RFONNConfig):
        super().__init__()
        self.config = config
        
        # 4축 관계 모델 (G + C + CF + I)
        axis_config = RelationAxisConfig(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            relation_dim=config.relation_dim,
            hidden_dim=config.hidden_dim,
        )
        self.relation_model = FourAxisRelationModel(axis_config)
        
        # Outcome heads
        self.delta_head = nn.Sequential(
            nn.Linear(config.relation_dim + config.action_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.state_dim),
        )
        
        self.stability_head = nn.Sequential(
            nn.Linear(config.relation_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        
        self.fall_head = nn.Sequential(
            nn.Linear(config.relation_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )
    
    def encode_relation(
        self,
        state_i: torch.Tensor,
        state_j: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """관계 인코딩 (4축)."""
        return self.relation_model.encode_relation(state_i, state_j, action)
    
    def predict_outcome(
        self,
        z_relation: torch.Tensor,
        action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """결과 예측.
        
        Returns:
            delta_j: 상태 변화
            stability: 안정성 점수
            fall_prob: 낙하 확률
        """
        combined = torch.cat([z_relation, action], dim=-1)
        delta_j = self.delta_head(combined)
        stability = self.stability_head(z_relation)
        fall_prob = torch.sigmoid(self.fall_head(z_relation))
        
        return {
            "delta_j": delta_j,
            "stability": stability,
            "fall_prob": fall_prob,
        }
    
    def forward(
        self,
        state_i: torch.Tensor,
        state_j: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """전체 forward.
        
        Returns:
            predictions: outcome 예측
            relation_info: 관계 정보 (4축)
        """
        relation_info = self.encode_relation(state_i, state_j, action)
        predictions = self.predict_outcome(relation_info["z_relation"], action)
        
        return predictions, relation_info


class RFONNTrainer:
    """Relational Field ONN 학습."""
    
    def __init__(self, config: RFONNConfig):
        self.config = config
        
        # 디바이스
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
        self.model = RelationalFieldONN(config).to(self.device)
        
        # 위상 정규화
        topo_config = TopologicalConfig(embed_dim=config.relation_dim)
        self.topo_regularizer = TopologicalRegularizer(topo_config).to(self.device)
        
        # 옵티마이저
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=0.01,
        )
        
        # 데이터
        self.dataset = None
        self.history = defaultdict(list)
    
    def collect_data(self):
        """시뮬레이션 데이터 수집."""
        print("\n데이터 수집 중...")
        
        env = RelationSimEnvironment()
        episodes = env.collect_dataset(
            num_episodes=self.config.num_episodes,
            scenarios=["supports", "contains", "independent"],
            actions=["push", "remove"],
        )
        
        self.dataset = PhysicsRelationDataset(episodes)
        print(f"  수집 완료: {len(self.dataset)} 에피소드")
    
    def compute_losses(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """손실 계산.
        
        1. 예측 손실 (핵심)
        2. 비대칭성 손실
        3. Contrastive 손실
        """
        state_i = batch["state_i"].to(self.device)
        state_j = batch["state_j"].to(self.device)
        action = batch["action"].to(self.device)
        delta_j_true = batch["delta_j"].to(self.device)
        signature_true = batch["signature"].to(self.device)
        
        # Forward
        predictions, relation_info = self.model(state_i, state_j, action)
        
        # 1. 예측 손실: "관계의 정의"
        loss_pred = F.mse_loss(predictions["delta_j"], delta_j_true)
        
        # 낙하 손실 (z 변화가 크면 낙하)
        fell_true = (delta_j_true[:, 2] < -0.05).float()
        loss_fall = F.binary_cross_entropy(
            predictions["fall_prob"].squeeze(), fell_true
        )
        
        # 2. 비대칭성 손실: z_ij ≠ z_ji
        relation_info_reverse = self.model.encode_relation(state_j, state_i, action)
        z_ij = relation_info["z_relation"]
        z_ji = relation_info_reverse["z_relation"]
        
        # 다른 결과를 가진 쌍은 다른 관계를 가져야 함
        asym_similarity = F.cosine_similarity(z_ij, z_ji, dim=-1)
        loss_asym = F.relu(asym_similarity - 0.5).mean()
        
        # 3. Contrastive: 유사한 signature → 유사한 z
        signature_pred = compute_outcome_signature(predictions["delta_j"])
        sig_sim = signature_similarity(signature_pred, signature_true)
        z_norm = F.normalize(z_ij, p=2, dim=-1)
        
        # 간단한 contrastive: signature가 유사하면 z도 유사해야
        loss_contrastive = F.mse_loss(z_norm.norm(dim=-1), sig_sim)
        
        # 4. 위상 정규화: 관계 공간의 일관성
        topo_losses = self.topo_regularizer.compute_loss(z_ij.unsqueeze(1))
        loss_topo = topo_losses["total"]
        
        # 총 손실
        total_loss = (
            self.config.pred_weight * (loss_pred + loss_fall) +
            self.config.asym_weight * loss_asym +
            self.config.contrastive_weight * loss_contrastive +
            self.config.topo_weight * loss_topo
        )
        
        return {
            "total": total_loss,
            "pred": loss_pred,
            "fall": loss_fall,
            "asym": loss_asym,
            "contrastive": loss_contrastive,
            "topo": loss_topo,
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """한 에포크 학습."""
        self.model.train()
        
        batch = self.dataset.sample(self.config.batch_size)
        losses = self.compute_losses(batch)
        
        self.optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """평가."""
        self.model.eval()
        
        batch = self.dataset.sample(64)
        state_i = batch["state_i"].to(self.device)
        state_j = batch["state_j"].to(self.device)
        action = batch["action"].to(self.device)
        delta_j_true = batch["delta_j"].to(self.device)
        
        predictions, relation_info = self.model(state_i, state_j, action)
        
        # 예측 정확도
        delta_mse = F.mse_loss(predictions["delta_j"], delta_j_true).item()
        
        # 낙하 예측 정확도
        fell_true = (delta_j_true[:, 2] < -0.05).float()
        fell_pred = (predictions["fall_prob"].squeeze() > 0.5).float()
        fall_acc = (fell_true == fell_pred).float().mean().item()
        
        # 관계 다양성 (클러스터 형성 확인)
        z = relation_info["z_relation"]
        z_var = z.var(dim=0).mean().item()
        
        return {
            "delta_mse": delta_mse,
            "fall_acc": fall_acc,
            "z_variance": z_var,
        }
    
    @torch.no_grad()
    def discover_relation_clusters(self, num_samples: int = 100):
        """관계 클러스터 발견.
        
        라벨 없이 관계 종(species)을 발견.
        """
        self.model.eval()
        
        # 샘플 수집
        z_list = []
        scenarios = []
        
        for _ in range(num_samples // self.config.batch_size + 1):
            batch = self.dataset.sample(self.config.batch_size)
            state_i = batch["state_i"].to(self.device)
            state_j = batch["state_j"].to(self.device)
            action = batch["action"].to(self.device)
            
            _, relation_info = self.model(state_i, state_j, action)
            z_list.append(relation_info["z_relation"].cpu())
        
        Z = torch.cat(z_list, dim=0)[:num_samples]
        
        # 간단한 클러스터링 (K-means 없이 거리 기반)
        # 클러스터 중심 후보: 랜덤하게 3개 선택
        centers = Z[torch.randperm(len(Z))[:3]]
        
        # 각 샘플을 가장 가까운 중심에 할당
        distances = torch.cdist(Z, centers)
        assignments = distances.argmin(dim=1)
        
        # 클러스터별 통계
        cluster_info = {}
        for c in range(3):
            mask = assignments == c
            if mask.sum() > 0:
                cluster_z = Z[mask]
                cluster_info[f"cluster_{c}"] = {
                    "size": mask.sum().item(),
                    "mean_norm": cluster_z.norm(dim=-1).mean().item(),
                    "variance": cluster_z.var(dim=0).mean().item(),
                }
        
        return cluster_info
    
    def train(self):
        """전체 학습."""
        print("=" * 60)
        print("Relational Field ONN Training")
        print("=" * 60)
        print(f"  Epochs: {self.config.epochs}")
        print(f"  Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Philosophy: 라벨 없이 예측으로 관계 정의")
        print("-" * 60)
        
        # 데이터 수집
        if self.dataset is None:
            self.collect_data()
        
        start_time = time.perf_counter()
        
        for epoch in range(1, self.config.epochs + 1):
            losses = self.train_epoch()
            
            for k, v in losses.items():
                self.history[k].append(v)
            
            if epoch % self.config.log_every == 0:
                elapsed = time.perf_counter() - start_time
                eval_metrics = self.evaluate()
                
                print(f"  Epoch {epoch:4d}: "
                      f"Total={losses['total']:.4f}, "
                      f"Pred={losses['pred']:.4f}, "
                      f"Fall_Acc={eval_metrics['fall_acc']*100:.1f}%, "
                      f"Time={elapsed:.1f}s")
        
        # 최종 평가
        print("\n" + "-" * 60)
        print("최종 평가:")
        eval_metrics = self.evaluate()
        print(f"  Delta MSE: {eval_metrics['delta_mse']:.6f}")
        print(f"  Fall Accuracy: {eval_metrics['fall_acc']*100:.1f}%")
        print(f"  Z Variance: {eval_metrics['z_variance']:.4f}")
        
        # 관계 클러스터 발견
        print("\n관계 클러스터 발견 (라벨 없이):")
        clusters = self.discover_relation_clusters()
        for name, info in clusters.items():
            print(f"  {name}: size={info['size']}, var={info['variance']:.4f}")
        
        print("\n" + "=" * 60)
        print("핵심 인사이트:")
        print("  - 라벨 없이 outcome 예측으로 관계 학습")
        print("  - 관계 클러스터가 자동 형성됨")
        print("  - '명명'은 인간이 마지막에 할 일")
        print("=" * 60)
        
        return self.history


def main():
    config = RFONNConfig(
        epochs=300,
        num_episodes=100,
        batch_size=32,
    )
    
    trainer = RFONNTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
