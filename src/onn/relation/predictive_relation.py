"""Predictive Relation Model.

물리적 귀결 기반 관계 학습 - "관계 = 행동의 결과를 예측하는 것"

핵심 아이디어:
- 관계는 "무엇이 일어나는가"로 정의됨
- "A가 B를 supports"의 본질: A를 치우면 B가 떨어진다
- 라벨 없이 예측 정확도로 관계를 학습

학습 방식:
- 입력: (node_i, node_j, action)
- 출력: delta_node_j (예측된 변화)
- 손실: MSE(predicted_delta, actual_delta)

사용 예:
    model = PredictiveRelationModel(state_dim=6, embed_dim=32)
    
    # 현재 상태
    state_cup = torch.tensor([0, 0, 0.1, 0, 0, 0])  # 위치 + 속도
    state_table = torch.tensor([0, 0, 0, 0, 0, 0])
    
    # 행동: 테이블을 치움
    action = torch.tensor([0, 0, -1, 0, 0, 0])  # 테이블 아래로 이동
    
    # 예측: 컵이 어떻게 될까?
    delta_cup = model(state_cup, state_table, action)
    # 예상: 컵도 따라 떨어짐 (supports 관계)

Author: Claude
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class RelationPredictionConfig:
    """관계 예측 모델 설정."""
    state_dim: int = 6        # 상태 차원 (위치 3 + 속도 3)
    action_dim: int = 6       # 행동 차원
    embed_dim: int = 64       # 관계 임베딩 차원
    hidden_dim: int = 128     # 히든 레이어 차원
    num_layers: int = 3       # 레이어 수
    dropout: float = 0.1
    use_attention: bool = True  # 관계 기반 어텐션 사용


class RelationInferenceModule(nn.Module):
    """관계 추론 모듈.
    
    두 객체의 상태에서 관계 임베딩을 추론합니다.
    SE3 인코더와 달리, 상태 기반 (위치+속도)입니다.
    """
    
    def __init__(self, state_dim: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        
        self.relation_encoder = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
    
    def forward(self, state_i: torch.Tensor, state_j: torch.Tensor) -> torch.Tensor:
        """관계 임베딩 추론.
        
        Args:
            state_i: (..., state_dim) 객체 i의 상태
            state_j: (..., state_dim) 객체 j의 상태
            
        Returns:
            (..., embed_dim) 관계 임베딩
        """
        z_i = self.state_encoder(state_i)
        z_j = self.state_encoder(state_j)
        
        # 비대칭 관계: i → j
        z_pair = torch.cat([z_i, z_j], dim=-1)
        z_rel = self.relation_encoder(z_pair)
        
        return z_rel


class OutcomePredictionModule(nn.Module):
    """결과 예측 모듈.
    
    관계 임베딩과 행동이 주어졌을 때, 
    객체 j의 상태 변화를 예측합니다.
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        action_dim: int, 
        state_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.action_encoder = nn.Linear(action_dim, embed_dim)
        
        layers = []
        input_dim = embed_dim * 2  # relation + action
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
        layers.append(nn.Linear(hidden_dim, state_dim))
        
        self.predictor = nn.Sequential(*layers)
    
    def forward(
        self, 
        z_relation: torch.Tensor, 
        action: torch.Tensor,
    ) -> torch.Tensor:
        """결과 예측.
        
        Args:
            z_relation: (..., embed_dim) 관계 임베딩
            action: (..., action_dim) 객체 i에 가해지는 행동
            
        Returns:
            (..., state_dim) 객체 j의 예측 상태 변화
        """
        z_action = self.action_encoder(action)
        z_combined = torch.cat([z_relation, z_action], dim=-1)
        delta_j = self.predictor(z_combined)
        return delta_j


class PredictiveRelationModel(nn.Module):
    """예측 기반 관계 모델.
    
    "관계"를 명시적으로 라벨링하지 않고,
    "객체 i에 행동을 가했을 때 객체 j가 어떻게 변하는가?"를
    예측하는 것으로 관계를 학습합니다.
    
    학습 후, 관계 임베딩 공간에서:
    - "supports" 관계: 비슷한 영역에 클러스터
    - "contains" 관계: 다른 영역에 클러스터
    - 새로운 관계: 자동으로 발견
    """
    
    def __init__(self, config: RelationPredictionConfig = None):
        super().__init__()
        
        if config is None:
            config = RelationPredictionConfig()
        self.config = config
        
        self.relation_module = RelationInferenceModule(
            state_dim=config.state_dim,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
        )
        
        self.outcome_module = OutcomePredictionModule(
            embed_dim=config.embed_dim,
            action_dim=config.action_dim,
            state_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
        )
        
        self.dropout = nn.Dropout(config.dropout)
    
    def infer_relation(
        self, 
        state_i: torch.Tensor, 
        state_j: torch.Tensor,
    ) -> torch.Tensor:
        """관계 임베딩 추론.
        
        Args:
            state_i: (..., state_dim) 객체 i의 상태
            state_j: (..., state_dim) 객체 j의 상태
            
        Returns:
            (..., embed_dim) 관계 임베딩
        """
        return self.relation_module(state_i, state_j)
    
    def predict_outcome(
        self,
        z_relation: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """행동 결과 예측.
        
        Args:
            z_relation: (..., embed_dim) 관계 임베딩
            action: (..., action_dim) 행동
            
        Returns:
            (..., state_dim) 예측 상태 변화
        """
        z_rel = self.dropout(z_relation)
        return self.outcome_module(z_rel, action)
    
    def forward(
        self,
        state_i: torch.Tensor,
        state_j: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """전체 forward pass.
        
        Args:
            state_i: (..., state_dim) 객체 i의 상태
            state_j: (..., state_dim) 객체 j의 상태
            action: (..., action_dim) 객체 i에 가해지는 행동
            
        Returns:
            delta_j: (..., state_dim) 객체 j의 예측 상태 변화
            z_relation: (..., embed_dim) 관계 임베딩
        """
        z_relation = self.infer_relation(state_i, state_j)
        delta_j = self.predict_outcome(z_relation, action)
        return delta_j, z_relation


class RelationDataGenerator:
    """관계 학습용 데이터 생성기.
    
    다양한 물리적 관계 시나리오를 생성합니다:
    - supports: A가 B를 지탱 (A 이동 → B도 이동)
    - contains: A가 B를 포함 (A 이동 → B도 이동, 회전도)
    - independent: 무관 (A 이동 → B 그대로)
    """
    
    def __init__(self, state_dim: int = 6, action_dim: int = 6):
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def generate_supports(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Supports 관계 데이터 생성.
        
        시나리오: A가 B 위에 있고, A가 B를 지탱함
        행동: B를 움직임
        결과: A도 따라 움직임
        """
        # 상태: [x, y, z, vx, vy, vz]
        state_A = torch.randn(batch_size, self.state_dim) * 0.5
        state_B = state_A.clone()
        state_B[:, 2] -= 0.1  # B가 A 아래에 있음
        
        # 행동: B를 수평 이동
        action = torch.zeros(batch_size, self.action_dim)
        action[:, 0] = torch.randn(batch_size) * 0.1  # x 방향 이동
        action[:, 1] = torch.randn(batch_size) * 0.1  # y 방향 이동
        
        # 결과: A도 같이 이동 (supports 관계)
        delta_A = torch.zeros(batch_size, self.state_dim)
        delta_A[:, 0] = action[:, 0]  # 같은 x 변화
        delta_A[:, 1] = action[:, 1]  # 같은 y 변화
        
        return {
            "state_i": state_B,  # 지탱하는 객체
            "state_j": state_A,  # 지탱받는 객체
            "action": action,
            "delta_j": delta_A,
            "relation_type": "supports",
        }
    
    def generate_contains(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Contains 관계 데이터 생성.
        
        시나리오: A (컨테이너) 안에 B (내용물)
        행동: A를 기울임/이동
        결과: B도 따라 움직이지만 추가 변화 있음
        """
        state_A = torch.randn(batch_size, self.state_dim) * 0.3
        state_B = state_A.clone() + torch.randn(batch_size, self.state_dim) * 0.05
        
        # 행동: A를 기울임 (회전 + 이동)
        action = torch.zeros(batch_size, self.action_dim)
        action[:, 3] = torch.randn(batch_size) * 0.2  # 회전
        action[:, 0] = torch.randn(batch_size) * 0.1
        
        # 결과: B는 A를 따라가지만 약간 다름 (내부에서 움직임)
        delta_B = torch.zeros(batch_size, self.state_dim)
        delta_B[:, 0] = action[:, 0] + torch.randn(batch_size) * 0.02
        delta_B[:, 1] = torch.randn(batch_size) * 0.03  # 기울어지면서 굴러감
        
        return {
            "state_i": state_A,
            "state_j": state_B,
            "action": action,
            "delta_j": delta_B,
            "relation_type": "contains",
        }
    
    def generate_independent(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Independent 관계 데이터 생성.
        
        시나리오: A와 B가 무관
        행동: A를 움직임
        결과: B는 그대로
        """
        state_A = torch.randn(batch_size, self.state_dim) * 0.5
        state_B = torch.randn(batch_size, self.state_dim) * 0.5
        
        action = torch.randn(batch_size, self.action_dim) * 0.1
        
        delta_B = torch.zeros(batch_size, self.state_dim)  # 변화 없음
        
        return {
            "state_i": state_A,
            "state_j": state_B,
            "action": action,
            "delta_j": delta_B,
            "relation_type": "independent",
        }
    
    def generate_mixed_batch(
        self, 
        batch_size: int, 
        supports_ratio: float = 0.4,
        contains_ratio: float = 0.3,
    ) -> Dict[str, torch.Tensor]:
        """혼합 배치 생성."""
        n_supports = int(batch_size * supports_ratio)
        n_contains = int(batch_size * contains_ratio)
        n_independent = batch_size - n_supports - n_contains
        
        batches = []
        if n_supports > 0:
            batches.append(self.generate_supports(n_supports))
        if n_contains > 0:
            batches.append(self.generate_contains(n_contains))
        if n_independent > 0:
            batches.append(self.generate_independent(n_independent))
        
        # 합치기
        result = {
            "state_i": torch.cat([b["state_i"] for b in batches], dim=0),
            "state_j": torch.cat([b["state_j"] for b in batches], dim=0),
            "action": torch.cat([b["action"] for b in batches], dim=0),
            "delta_j": torch.cat([b["delta_j"] for b in batches], dim=0),
        }
        
        # 셔플
        perm = torch.randperm(batch_size)
        for k in result:
            result[k] = result[k][perm]
        
        return result


def train_predictive_relation(
    epochs: int = 500,
    batch_size: int = 64,
    lr: float = 1e-3,
    verbose: bool = True,
) -> PredictiveRelationModel:
    """Predictive Relation Model 학습.
    
    Returns:
        학습된 모델
    """
    config = RelationPredictionConfig()
    model = PredictiveRelationModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    generator = RelationDataGenerator()
    
    if verbose:
        print("=" * 60)
        print("Predictive Relation Model Training")
        print("=" * 60)
        print(f"  Epochs: {epochs}")
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
        print("-" * 60)
    
    losses = []
    for epoch in range(1, epochs + 1):
        batch = generator.generate_mixed_batch(batch_size)
        
        delta_pred, z_rel = model(
            batch["state_i"],
            batch["state_j"],
            batch["action"],
        )
        
        loss = F.mse_loss(delta_pred, batch["delta_j"])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if verbose and epoch % 100 == 0:
            print(f"  Epoch {epoch:4d}: Loss = {loss.item():.6f}")
    
    if verbose:
        print("-" * 60)
        print(f"  Final Loss: {losses[-1]:.6f}")
        print("=" * 60)
    
    return model


def demo_predictive_relation():
    """Predictive Relation Model 데모."""
    print("=" * 60)
    print("Predictive Relation Model Demo")
    print("=" * 60)
    
    # 1. 모델 학습
    model = train_predictive_relation(epochs=300, verbose=True)
    model.eval()
    
    # 2. 관계 임베딩 비교
    print("\n관계 임베딩 분석:")
    print("-" * 60)
    
    generator = RelationDataGenerator()
    
    # 각 관계 유형에서 샘플 추출
    with torch.no_grad():
        supports_data = generator.generate_supports(16)
        contains_data = generator.generate_contains(16)
        indep_data = generator.generate_independent(16)
        
        z_supports = model.infer_relation(supports_data["state_i"], supports_data["state_j"])
        z_contains = model.infer_relation(contains_data["state_i"], contains_data["state_j"])
        z_indep = model.infer_relation(indep_data["state_i"], indep_data["state_j"])
        
        # 클러스터 중심
        center_supports = z_supports.mean(dim=0)
        center_contains = z_contains.mean(dim=0)
        center_indep = z_indep.mean(dim=0)
        
        # 클러스터 간 거리
        dist_s_c = (center_supports - center_contains).norm()
        dist_s_i = (center_supports - center_indep).norm()
        dist_c_i = (center_contains - center_indep).norm()
        
        print(f"  클러스터 간 거리:")
        print(f"    supports ↔ contains: {dist_s_c:.4f}")
        print(f"    supports ↔ independent: {dist_s_i:.4f}")
        print(f"    contains ↔ independent: {dist_c_i:.4f}")
        
        # 예측 정확도
        delta_pred_s, _ = model(supports_data["state_i"], supports_data["state_j"], supports_data["action"])
        delta_pred_i, _ = model(indep_data["state_i"], indep_data["state_j"], indep_data["action"])
        
        mse_supports = F.mse_loss(delta_pred_s, supports_data["delta_j"])
        mse_indep = F.mse_loss(delta_pred_i, indep_data["delta_j"])
        
        print(f"\n  예측 MSE:")
        print(f"    supports: {mse_supports:.6f}")
        print(f"    independent: {mse_indep:.6f}")
    
    print("\n" + "=" * 60)
    print("핵심 인사이트:")
    print("  - 라벨 없이 예측 학습으로 관계를 구별")
    print("  - 유사한 물리적 귀결 → 유사한 관계 임베딩")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    demo_predictive_relation()
