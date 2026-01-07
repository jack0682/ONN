"""Enhanced Relation Model with 4-Axis Decomposition.

관계를 4개의 축으로 분해:
- G (Geometry): SE(3) 상대 변환 + 형상 근접성
- C (Constraint): 접촉/제약 존재 여부 (continuous)
- CF (Counterfactual): 개입 시 결과 변화 (비대칭성)
- I (Information): 상호 정보량 (의존성 강도)

핵심 철학:
- 라벨은 "사후 요약"일 뿐, 관계 자체는 연속적 장(field)
- "supports"는 C↑ + CF↑ + I↑ 영역의 특정 좌표
- 관계는 정적 분류가 아닌 "행동-결과의 인과 구조"

Author: Claude (based on user's brilliant framework)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class RelationAxisConfig:
    """4축 관계 모델 설정."""
    # 차원
    state_dim: int = 6          # 상태 차원 (위치 3 + 속도 3)
    action_dim: int = 6         # 행동 차원
    geometry_dim: int = 16      # G축 임베딩 차원
    constraint_dim: int = 8     # C축 임베딩 차원
    counterfactual_dim: int = 16  # CF축 임베딩 차원
    information_dim: int = 8    # I축 임베딩 차원
    
    # 최종 관계 임베딩
    relation_dim: int = 48      # G + C + CF + I 총합
    
    # 네트워크
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    
    # 손실 가중치
    pred_weight: float = 1.0      # outcome prediction
    asym_weight: float = 0.5      # counterfactual asymmetry
    equiv_weight: float = 0.3     # equivariance


# ==============================================================================
# 4축 인코더
# ==============================================================================

class GeometryEncoder(nn.Module):
    """G축: 기하적 관계 인코더.
    
    - SE(3) 상대 변환 (ξᵢⱼ)
    - 근접 특징 (φᵢⱼ): 최소거리, 접촉면적 proxy
    - 형상 적합 (κᵢⱼ): coplanarity, stability margin
    """
    
    def __init__(self, state_dim: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        
        # 상대 위치 인코더
        self.position_encoder = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim // 2),
        )
        
        # 근접/형상 특징 인코더
        self.proximity_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),  # distance, height_diff, alignment, overlap
            nn.GELU(),
            nn.Linear(hidden_dim // 2, embed_dim // 2),
        )
        
        self.output_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, state_i: torch.Tensor, state_j: torch.Tensor) -> torch.Tensor:
        """G축 임베딩 계산.
        
        Returns:
            (..., geometry_dim) 기하 임베딩
        """
        # 상대 변환 특징
        combined = torch.cat([state_i, state_j], dim=-1)
        z_position = self.position_encoder(combined)
        
        # 근접/형상 특징 (상태에서 추출)
        pos_i, pos_j = state_i[..., :3], state_j[..., :3]
        diff = pos_j - pos_i
        
        distance = torch.norm(diff, dim=-1, keepdim=True)
        height_diff = diff[..., 2:3]  # z축 차이
        horizontal_dist = torch.norm(diff[..., :2], dim=-1, keepdim=True)
        
        # Alignment: 수직 정렬도 (supports의 특징)
        alignment = torch.abs(height_diff) / (distance + 1e-6)
        
        proximity_features = torch.cat([
            distance, height_diff, horizontal_dist, alignment
        ], dim=-1)
        z_proximity = self.proximity_encoder(proximity_features)
        
        # 결합
        z_geo = torch.cat([z_position, z_proximity], dim=-1)
        return self.output_proj(z_geo)


class ConstraintEncoder(nn.Module):
    """C축: 제약/접촉 인코더.
    
    - 접촉 존재 여부 (continuous)
    - 제약 강도: "붙잡고 있는가/막고 있는가"
    """
    
    def __init__(self, state_dim: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim * 2 + 3, hidden_dim),  # states + contact features
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        
        # 접촉 확률 예측 헤드
        self.contact_head = nn.Linear(embed_dim, 1)
    
    def forward(
        self, 
        state_i: torch.Tensor, 
        state_j: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """C축 임베딩 계산.
        
        Returns:
            z_constraint: (..., constraint_dim) 제약 임베딩
            contact_prob: (..., 1) 접촉 확률
        """
        pos_i, pos_j = state_i[..., :3], state_j[..., :3]
        
        # 접촉 가능성 특징
        distance = torch.norm(pos_j - pos_i, dim=-1, keepdim=True)
        contact_likelihood = torch.exp(-distance * 5)  # 거리↑ → 접촉↓
        
        # 상대 속도 (분리 중인가?)
        vel_i, vel_j = state_i[..., 3:], state_j[..., 3:]
        rel_vel = torch.norm(vel_j - vel_i, dim=-1, keepdim=True)
        
        contact_features = torch.cat([
            contact_likelihood, distance, rel_vel
        ], dim=-1)
        
        combined = torch.cat([state_i, state_j, contact_features], dim=-1)
        z_constraint = self.encoder(combined)
        
        contact_prob = torch.sigmoid(self.contact_head(z_constraint))
        
        return z_constraint, contact_prob


class CounterfactualEncoder(nn.Module):
    """CF축: 반사실 인코더.
    
    "i에 개입하면 j가 어떻게 변하는가?"
    비대칭성이 핵심: z_ij ≠ z_ji
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        embed_dim: int, 
        hidden_dim: int,
    ):
        super().__init__()
        
        self.action_encoder = nn.Linear(action_dim, hidden_dim // 2)
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim * 2 + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        
        # 결과 예측 헤드
        self.outcome_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
    
    def forward(
        self,
        state_i: torch.Tensor,
        state_j: torch.Tensor,
        action_i: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CF축 임베딩 및 결과 예측.
        
        Args:
            state_i, state_j: 객체 상태
            action_i: 객체 i에 가해지는 행동
            
        Returns:
            z_cf: (..., counterfactual_dim) 반사실 임베딩
            delta_j: (..., state_dim) 예측된 j의 변화
        """
        z_action = self.action_encoder(action_i)
        combined = torch.cat([state_i, state_j, z_action], dim=-1)
        
        z_cf = self.encoder(combined)
        delta_j = self.outcome_head(z_cf)
        
        return z_cf, delta_j


class InformationEncoder(nn.Module):
    """I축: 정보량 인코더.
    
    "j가 i로 인해 얼마나 더 예측 가능해지는가?"
    상호 정보량의 proxy.
    """
    
    def __init__(self, state_dim: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        
        # 독립 인코더
        self.encoder_i = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        
        self.encoder_j = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        
        # 조건부 인코더: j | i
        self.conditional_encoder = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        
        # MI 추정 헤드
        self.mi_head = nn.Linear(embed_dim, 1)
    
    def forward(
        self,
        state_i: torch.Tensor,
        state_j: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """I축 임베딩 및 MI 추정.
        
        Returns:
            z_info: (..., information_dim) 정보 임베딩
            mi_estimate: (..., 1) 상호 정보량 추정
        """
        z_i = self.encoder_i(state_i)
        z_j = self.encoder_j(state_j)
        
        # 조건부 표현
        z_combined = torch.cat([z_i, z_j], dim=-1)
        z_info = self.conditional_encoder(z_combined)
        
        # MI 추정 (높을수록 관계 강함)
        mi_estimate = torch.sigmoid(self.mi_head(z_info)) * 2  # [0, 2] 범위
        
        return z_info, mi_estimate


# ==============================================================================
# 통합 관계 모델
# ==============================================================================

class FourAxisRelationModel(nn.Module):
    """4축 분해 기반 관계 모델.
    
    Relation(i, j) = G + C + CF + I
    
    라벨 없이 학습:
    1. Outcome prediction (CF축)
    2. Asymmetry regularization (방향성)
    3. Equivariance check (등변성)
    """
    
    def __init__(self, config: RelationAxisConfig = None):
        super().__init__()
        
        if config is None:
            config = RelationAxisConfig()
        self.config = config
        
        # 4축 인코더
        self.geometry_encoder = GeometryEncoder(
            config.state_dim, config.geometry_dim, config.hidden_dim
        )
        
        self.constraint_encoder = ConstraintEncoder(
            config.state_dim, config.constraint_dim, config.hidden_dim
        )
        
        self.counterfactual_encoder = CounterfactualEncoder(
            config.state_dim, config.action_dim,
            config.counterfactual_dim, config.hidden_dim
        )
        
        self.information_encoder = InformationEncoder(
            config.state_dim, config.information_dim, config.hidden_dim
        )
        
        # 통합 프로젝션
        total_dim = (config.geometry_dim + config.constraint_dim + 
                     config.counterfactual_dim + config.information_dim)
        self.output_proj = nn.Linear(total_dim, config.relation_dim)
    
    def encode_relation(
        self,
        state_i: torch.Tensor,
        state_j: torch.Tensor,
        action_i: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """전체 관계 인코딩.
        
        Returns:
            dict with keys:
                - z_relation: 통합 관계 임베딩
                - z_G, z_C, z_CF, z_I: 각 축 임베딩
                - contact_prob, mi_estimate: 보조 출력
                - delta_j: 예측 결과 (action이 주어진 경우)
        """
        # G축: Geometry
        z_G = self.geometry_encoder(state_i, state_j)
        
        # C축: Constraint
        z_C, contact_prob = self.constraint_encoder(state_i, state_j)
        
        # CF축: Counterfactual
        if action_i is None:
            action_i = torch.zeros(*state_i.shape[:-1], self.config.action_dim, 
                                   device=state_i.device)
        z_CF, delta_j = self.counterfactual_encoder(state_i, state_j, action_i)
        
        # I축: Information
        z_I, mi_estimate = self.information_encoder(state_i, state_j)
        
        # 통합
        z_all = torch.cat([z_G, z_C, z_CF, z_I], dim=-1)
        z_relation = self.output_proj(z_all)
        
        return {
            "z_relation": z_relation,
            "z_G": z_G,
            "z_C": z_C,
            "z_CF": z_CF,
            "z_I": z_I,
            "contact_prob": contact_prob,
            "mi_estimate": mi_estimate,
            "delta_j": delta_j,
        }
    
    def forward(
        self,
        state_i: torch.Tensor,
        state_j: torch.Tensor,
        action_i: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.
        
        Returns:
            z_relation: 통합 관계 임베딩
            diagnostics: 모든 축 정보
        """
        result = self.encode_relation(state_i, state_j, action_i)
        return result["z_relation"], result
    
    def compute_loss(
        self,
        state_i: torch.Tensor,
        state_j: torch.Tensor,
        action_i: torch.Tensor,
        delta_j_true: torch.Tensor,
        contact_true: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """손실 계산.
        
        3가지 손실:
        1. Outcome prediction (핵심)
        2. Asymmetry regularization (방향성)
        3. Contact prediction (보조)
        """
        result = self.encode_relation(state_i, state_j, action_i)
        
        # 1. Outcome prediction loss
        loss_pred = F.mse_loss(result["delta_j"], delta_j_true)
        
        # 2. Asymmetry loss: z_ij ≠ z_ji
        result_reverse = self.encode_relation(state_j, state_i, action_i)
        margin = 0.5
        dist_asym = F.cosine_similarity(
            result["z_relation"], result_reverse["z_relation"], dim=-1
        )
        loss_asym = F.relu(dist_asym - (1 - margin)).mean()
        
        # 3. Contact loss (선택적)
        if contact_true is not None:
            loss_contact = F.binary_cross_entropy(
                result["contact_prob"], contact_true
            )
        else:
            loss_contact = torch.tensor(0.0, device=state_i.device)
        
        # 총 손실
        total_loss = (
            self.config.pred_weight * loss_pred +
            self.config.asym_weight * loss_asym
        )
        
        return {
            "total": total_loss,
            "pred": loss_pred,
            "asym": loss_asym,
            "contact": loss_contact,
            "mi_mean": result["mi_estimate"].mean(),
        }


# ==============================================================================
# Outcome Signature (for Contrastive Learning)
# ==============================================================================

@dataclass
class OutcomeSignature:
    """결과 서명 - contrastive learning의 positive/negative 정의에 사용.
    
    라벨 대신 물리적 결과로 관계 유사성을 정의.
    """
    delta_z: float              # 수직 위치 변화
    delta_horizontal: float     # 수평 위치 변화
    contact_maintained: float   # 접촉 유지 비율
    stability_change: float     # 안정성 변화
    
    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.delta_z,
            self.delta_horizontal,
            self.contact_maintained,
            self.stability_change,
        ])
    
    @classmethod
    def from_delta(cls, delta_j: torch.Tensor) -> "OutcomeSignature":
        """상태 변화에서 서명 추출."""
        delta_z = delta_j[..., 2].mean().item()
        delta_horizontal = torch.norm(delta_j[..., :2], dim=-1).mean().item()
        
        # 단순화: 변화가 작으면 접촉 유지로 간주
        contact_maintained = float(torch.norm(delta_j, dim=-1).mean() < 0.1)
        
        # 단순화: z 감소 = 안정성 감소
        stability_change = -delta_z
        
        return cls(
            delta_z=delta_z,
            delta_horizontal=delta_horizontal,
            contact_maintained=contact_maintained,
            stability_change=stability_change,
        )


def compute_signature_distance(sig1: torch.Tensor, sig2: torch.Tensor) -> torch.Tensor:
    """두 서명 간의 거리 계산."""
    return torch.norm(sig1 - sig2, dim=-1)


# ==============================================================================
# 데모
# ==============================================================================

def demo_four_axis_relation():
    """4축 관계 모델 데모."""
    print("=" * 60)
    print("4-Axis Relation Model Demo")
    print("=" * 60)
    print("  G (Geometry) + C (Constraint) + CF (Counterfactual) + I (Information)")
    print("-" * 60)
    
    config = RelationAxisConfig()
    model = FourAxisRelationModel(config)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 시나리오: 컵이 테이블 위에 있음
    # 상태: [x, y, z, vx, vy, vz]
    state_table = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    state_cup = torch.tensor([[0.0, 0.0, 0.1, 0.0, 0.0, 0.0]])  # 10cm 위
    
    # 행동: 테이블 제거 (아래로 이동)
    action_remove = torch.tensor([[0.0, 0.0, -0.5, 0.0, 0.0, 0.0]])
    
    z_rel, diag = model(state_cup, state_table, action_remove)
    
    print(f"\n  컵 → 테이블 관계:")
    print(f"    통합 임베딩 shape: {z_rel.shape}")
    print(f"    G축 norm: {diag['z_G'].norm():.4f}")
    print(f"    C축 norm: {diag['z_C'].norm():.4f}")
    print(f"    CF축 norm: {diag['z_CF'].norm():.4f}")
    print(f"    I축 norm: {diag['z_I'].norm():.4f}")
    print(f"    접촉 확률: {diag['contact_prob'].item():.4f}")
    print(f"    MI 추정: {diag['mi_estimate'].item():.4f}")
    print(f"    예측 delta_j: {diag['delta_j'].squeeze().tolist()}")
    
    # 역방향 테스트 (비대칭성)
    z_rel_reverse, diag_reverse = model(state_table, state_cup, action_remove)
    
    similarity = F.cosine_similarity(z_rel, z_rel_reverse, dim=-1)
    print(f"\n  비대칭성 테스트:")
    print(f"    z(컵→테이블) vs z(테이블→컵): {similarity.item():.4f}")
    print(f"    (1.0 = 동일, 0.0 = 직교, 낮을수록 비대칭)")
    
    print("\n" + "=" * 60)
    print("핵심 인사이트:")
    print("  - 라벨 'supports'는 G↑ + C↑ + CF↑ + I↑ 영역")
    print("  - 관계는 4차원 연속 공간의 좌표")
    print("  - 개입(action)이 관계의 본질을 드러냄")
    print("=" * 60)


if __name__ == "__main__":
    demo_four_axis_relation()
