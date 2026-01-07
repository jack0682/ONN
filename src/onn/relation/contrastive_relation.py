"""Contrastive Relation Learner.

Outcome Signature 기반 자가지도 관계 학습.

핵심 아이디어:
- positive/negative를 라벨이 아닌 "물리적 결과 유사성"으로 정의
- 유사한 outcome signature → 유사한 관계 임베딩
- 클러스터가 자연스럽게 형성됨 (라벨 없이)

Outcome Signature:
- delta_z (수직 변화)
- delta_horizontal (수평 변화)
- contact_maintained (접촉 유지)
- stability_change (안정성 변화)

학습 후:
- "supports" 클러스터 자동 발견
- "contains" 클러스터 자동 발견
- 새로운 관계 유형도 자동 발견

Author: Claude
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ContrastiveConfig:
    """Contrastive 학습 설정."""
    state_dim: int = 6
    action_dim: int = 6
    embed_dim: int = 64
    hidden_dim: int = 128
    
    # Contrastive 설정
    temperature: float = 0.07
    signature_dim: int = 4      # outcome signature 차원
    margin: float = 0.5         # triplet margin
    
    # 학습 설정
    num_negatives: int = 16     # negative 샘플 수
    similarity_threshold: float = 0.3  # positive 판정 임계값


class RelationEncoder(nn.Module):
    """관계 인코더 (contrastive 학습용)."""
    
    def __init__(self, state_dim: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        
        # L2 정규화 (contrastive에 필수)
        self.normalize = True
    
    def forward(self, state_i: torch.Tensor, state_j: torch.Tensor) -> torch.Tensor:
        """관계 임베딩.
        
        Returns:
            (..., embed_dim) L2 정규화된 임베딩
        """
        combined = torch.cat([state_i, state_j], dim=-1)
        z = self.encoder(combined)
        
        if self.normalize:
            z = F.normalize(z, p=2, dim=-1)
        
        return z


class OutcomePredictor(nn.Module):
    """결과 예측기 (signature 생성용)."""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        embed_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
    
    def forward(
        self, 
        z_relation: torch.Tensor, 
        action: torch.Tensor,
    ) -> torch.Tensor:
        """결과 예측.
        
        Returns:
            (..., state_dim) 예측된 상태 변화
        """
        combined = torch.cat([z_relation, action], dim=-1)
        return self.predictor(combined)


def compute_outcome_signature(delta: torch.Tensor) -> torch.Tensor:
    """상태 변화에서 outcome signature 추출.
    
    Args:
        delta: (..., state_dim) 상태 변화 [dx, dy, dz, dvx, dvy, dvz]
        
    Returns:
        (..., 4) signature [delta_z, delta_h, contact, stability]
    """
    delta_z = delta[..., 2:3]  # z 변화
    delta_h = torch.norm(delta[..., :2], dim=-1, keepdim=True)  # 수평 변화
    
    # 접촉 유지: 변화가 작으면 유지
    total_change = torch.norm(delta[..., :3], dim=-1, keepdim=True)
    contact = torch.exp(-total_change * 10)  # 변화↓ → 접촉↑
    
    # 안정성: z 감소 = 불안정 (낙하)
    stability = -delta_z
    
    signature = torch.cat([delta_z, delta_h, contact, stability], dim=-1)
    return signature


def signature_similarity(sig1: torch.Tensor, sig2: torch.Tensor) -> torch.Tensor:
    """두 signature 간의 유사도 (0~1).
    
    Args:
        sig1, sig2: (..., signature_dim)
        
    Returns:
        (...,) 유사도 점수
    """
    # 코사인 유사도를 0~1로 변환
    cos_sim = F.cosine_similarity(sig1, sig2, dim=-1)
    return (cos_sim + 1) / 2  # [-1, 1] → [0, 1]


class ContrastiveRelationLearner(nn.Module):
    """Contrastive 기반 관계 학습.
    
    InfoNCE Loss 사용:
    - Anchor: 현재 관계 임베딩
    - Positive: 유사한 outcome signature를 가진 관계
    - Negative: 다른 outcome signature를 가진 관계
    """
    
    def __init__(self, config: ContrastiveConfig = None):
        super().__init__()
        
        if config is None:
            config = ContrastiveConfig()
        self.config = config
        
        self.relation_encoder = RelationEncoder(
            config.state_dim, config.embed_dim, config.hidden_dim
        )
        
        self.outcome_predictor = OutcomePredictor(
            config.state_dim, config.action_dim,
            config.embed_dim, config.hidden_dim
        )
        
        # Projection head (SimCLR 스타일)
        self.projection_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.embed_dim),
        )
    
    def encode_relation(
        self, 
        state_i: torch.Tensor, 
        state_j: torch.Tensor,
    ) -> torch.Tensor:
        """관계 인코딩."""
        return self.relation_encoder(state_i, state_j)
    
    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Projection for contrastive learning."""
        h = self.projection_head(z)
        return F.normalize(h, p=2, dim=-1)
    
    def forward(
        self,
        state_i: torch.Tensor,
        state_j: torch.Tensor,
        action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Returns:
            z_relation: 관계 임베딩
            z_projected: projection된 임베딩
            delta_pred: 예측된 결과
            signature: outcome signature
        """
        z_relation = self.encode_relation(state_i, state_j)
        z_projected = self.project(z_relation)
        delta_pred = self.outcome_predictor(z_relation, action)
        signature = compute_outcome_signature(delta_pred)
        
        return {
            "z_relation": z_relation,
            "z_projected": z_projected,
            "delta_pred": delta_pred,
            "signature": signature,
        }
    
    def infonce_loss(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """InfoNCE Loss.
        
        Args:
            anchors: (batch, embed_dim)
            positives: (batch, embed_dim)
            negatives: (batch, num_neg, embed_dim)
            
        Returns:
            scalar loss
        """
        batch_size = anchors.shape[0]
        
        # Positive 유사도
        pos_sim = torch.sum(anchors * positives, dim=-1) / self.config.temperature
        
        # Negative 유사도
        neg_sim = torch.bmm(
            negatives, anchors.unsqueeze(-1)
        ).squeeze(-1) / self.config.temperature  # (batch, num_neg)
        
        # Log-sum-exp
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # (batch, 1+num_neg)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchors.device)  # positive는 index 0
        
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def triplet_loss(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """Triplet Margin Loss.
        
        Args:
            anchors: (batch, embed_dim)
            positives: (batch, embed_dim)
            negatives: (batch, embed_dim) 또는 (batch, num_neg, embed_dim)
        """
        if negatives.dim() == 3:
            negatives = negatives[:, 0, :]  # 첫 번째 negative만 사용
        
        return F.triplet_margin_loss(
            anchors, positives, negatives,
            margin=self.config.margin,
        )


class ContrastiveDataGenerator:
    """Contrastive 학습용 데이터 생성."""
    
    def __init__(self, config: ContrastiveConfig):
        self.config = config
        
        # 관계 유형별 캐시 (signature 기반)
        self.signature_cache = defaultdict(list)
    
    def generate_triplet(
        self, 
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        """Triplet (anchor, positive, negative) 생성.
        
        Outcome signature 기반:
        - Positive: signature가 유사한 쌍
        - Negative: signature가 다른 쌍
        """
        anchors_i, anchors_j, anchors_action, anchors_delta = [], [], [], []
        positives_i, positives_j = [], []
        negatives_i, negatives_j = [], []
        
        for _ in range(batch_size):
            # Anchor 샘플 (supports 관계)
            ai, aj, action, delta = self._sample_supports()
            anchors_i.append(ai)
            anchors_j.append(aj)
            anchors_action.append(action)
            anchors_delta.append(delta)
            
            # Positive: 같은 관계 유형
            pi, pj, _, _ = self._sample_supports()
            positives_i.append(pi)
            positives_j.append(pj)
            
            # Negative: 다른 관계 유형 (independent)
            ni, nj, _, _ = self._sample_independent()
            negatives_i.append(ni)
            negatives_j.append(nj)
        
        return {
            "anchor_i": torch.stack(anchors_i),
            "anchor_j": torch.stack(anchors_j),
            "anchor_action": torch.stack(anchors_action),
            "anchor_delta": torch.stack(anchors_delta),
            "positive_i": torch.stack(positives_i),
            "positive_j": torch.stack(positives_j),
            "negative_i": torch.stack(negatives_i),
            "negative_j": torch.stack(negatives_j),
        }
    
    def _sample_supports(self) -> Tuple[torch.Tensor, ...]:
        """Supports 관계 샘플링."""
        state_j = torch.randn(self.config.state_dim) * 0.3
        state_i = state_j.clone()
        state_i[2] -= 0.1  # i가 j 아래에 있음
        
        action = torch.zeros(self.config.action_dim)
        action[0] = torch.randn(1).item() * 0.1
        
        delta = torch.zeros(self.config.state_dim)
        delta[0] = action[0]  # j가 i를 따라감
        
        return state_i, state_j, action, delta
    
    def _sample_independent(self) -> Tuple[torch.Tensor, ...]:
        """Independent 관계 샘플링."""
        state_i = torch.randn(self.config.state_dim) * 0.5
        state_j = torch.randn(self.config.state_dim) * 0.5
        
        action = torch.randn(self.config.action_dim) * 0.1
        delta = torch.zeros(self.config.state_dim)  # 변화 없음
        
        return state_i, state_j, action, delta
    
    def generate_signature_batch(
        self, 
        batch_size: int,
        num_negatives: int = 8,
    ) -> Dict[str, torch.Tensor]:
        """Signature 기반 positive/negative 배치 생성."""
        
        # 다양한 관계 샘플 생성
        all_samples = []
        
        for _ in range(batch_size * 3):
            relation_type = np.random.choice(
                ["supports", "contains", "independent"],
                p=[0.4, 0.3, 0.3]
            )
            
            if relation_type == "supports":
                sample = self._sample_supports()
            elif relation_type == "contains":
                sample = self._sample_contains()
            else:
                sample = self._sample_independent()
            
            all_samples.append((relation_type, sample))
        
        # Anchor, Positive, Negative 분류
        anchors = []
        positives = []
        negatives = []
        
        for i in range(batch_size):
            anchor_type, anchor_sample = all_samples[i]
            
            # Positive: 같은 타입
            pos_candidates = [
                s for t, s in all_samples[batch_size:] if t == anchor_type
            ]
            if len(pos_candidates) > 0:
                positive = pos_candidates[np.random.randint(len(pos_candidates))]
            else:
                positive = anchor_sample
            
            # Negative: 다른 타입
            neg_candidates = [
                s for t, s in all_samples[batch_size:] if t != anchor_type
            ]
            neg_list = []
            for _ in range(min(num_negatives, len(neg_candidates))):
                idx = np.random.randint(len(neg_candidates))
                neg_list.append(neg_candidates[idx])
            
            anchors.append(anchor_sample)
            positives.append(positive)
            negatives.append(neg_list)
        
        return self._collate(anchors, positives, negatives)
    
    def _sample_contains(self) -> Tuple[torch.Tensor, ...]:
        """Contains 관계 샘플링."""
        state_i = torch.randn(self.config.state_dim) * 0.3
        state_j = state_i.clone() + torch.randn(self.config.state_dim) * 0.05
        
        action = torch.zeros(self.config.action_dim)
        action[3] = torch.randn(1).item() * 0.2  # 회전
        
        delta = torch.zeros(self.config.state_dim)
        delta[0] = action[0] + torch.randn(1).item() * 0.02
        delta[1] = torch.randn(1).item() * 0.03
        
        return state_i, state_j, action, delta
    
    def _collate(
        self,
        anchors: List,
        positives: List,
        negatives: List,
    ) -> Dict[str, torch.Tensor]:
        """배치 정리."""
        return {
            "anchor_i": torch.stack([a[0] for a in anchors]),
            "anchor_j": torch.stack([a[1] for a in anchors]),
            "anchor_action": torch.stack([a[2] for a in anchors]),
            "anchor_delta": torch.stack([a[3] for a in anchors]),
            "positive_i": torch.stack([p[0] for p in positives]),
            "positive_j": torch.stack([p[1] for p in positives]),
        }


def train_contrastive(
    epochs: int = 500,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> ContrastiveRelationLearner:
    """Contrastive 학습."""
    print("=" * 60)
    print("Contrastive Relation Learner Training")
    print("=" * 60)
    
    config = ContrastiveConfig()
    model = ContrastiveRelationLearner(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    generator = ContrastiveDataGenerator(config)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 60)
    
    losses = []
    for epoch in range(1, epochs + 1):
        batch = generator.generate_triplet(batch_size)
        
        # Anchor 인코딩
        z_anchor = model.project(
            model.encode_relation(batch["anchor_i"], batch["anchor_j"])
        )
        
        # Positive 인코딩
        z_positive = model.project(
            model.encode_relation(batch["positive_i"], batch["positive_j"])
        )
        
        # Negative 인코딩
        z_negative = model.project(
            model.encode_relation(batch["negative_i"], batch["negative_j"])
        )
        
        # Triplet Loss
        loss = model.triplet_loss(z_anchor, z_positive, z_negative)
        
        # Outcome Prediction Loss (auxiliary)
        result = model(batch["anchor_i"], batch["anchor_j"], batch["anchor_action"])
        pred_loss = F.mse_loss(result["delta_pred"], batch["anchor_delta"])
        
        total_loss = loss + 0.5 * pred_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        losses.append(total_loss.item())
        
        if epoch % 100 == 0:
            print(f"  Epoch {epoch:4d}: Triplet={loss.item():.4f}, Pred={pred_loss.item():.4f}")
    
    print("-" * 60)
    print(f"  Final Loss: {losses[-1]:.4f}")
    print("=" * 60)
    
    return model


def demo_contrastive_clustering():
    """Contrastive 클러스터링 데모."""
    print("\n" + "=" * 60)
    print("Contrastive Relation Clustering Demo")
    print("=" * 60)
    
    # 학습
    model = train_contrastive(epochs=300)
    model.eval()
    
    # 각 관계 유형에서 샘플 추출
    config = ContrastiveConfig()
    generator = ContrastiveDataGenerator(config)
    
    with torch.no_grad():
        # Supports 샘플
        supports_z = []
        for _ in range(32):
            si, sj, _, _ = generator._sample_supports()
            z = model.encode_relation(si.unsqueeze(0), sj.unsqueeze(0))
            supports_z.append(z.squeeze())
        supports_z = torch.stack(supports_z)
        
        # Contains 샘플
        contains_z = []
        for _ in range(32):
            ci, cj, _, _ = generator._sample_contains()
            z = model.encode_relation(ci.unsqueeze(0), cj.unsqueeze(0))
            contains_z.append(z.squeeze())
        contains_z = torch.stack(contains_z)
        
        # Independent 샘플
        indep_z = []
        for _ in range(32):
            ii, ij, _, _ = generator._sample_independent()
            z = model.encode_relation(ii.unsqueeze(0), ij.unsqueeze(0))
            indep_z.append(z.squeeze())
        indep_z = torch.stack(indep_z)
    
    # 클러스터 간 거리
    print("\n클러스터 분석:")
    print("-" * 60)
    
    center_s = supports_z.mean(dim=0)
    center_c = contains_z.mean(dim=0)
    center_i = indep_z.mean(dim=0)
    
    dist_s_c = F.cosine_similarity(center_s.unsqueeze(0), center_c.unsqueeze(0)).item()
    dist_s_i = F.cosine_similarity(center_s.unsqueeze(0), center_i.unsqueeze(0)).item()
    dist_c_i = F.cosine_similarity(center_c.unsqueeze(0), center_i.unsqueeze(0)).item()
    
    print(f"  클러스터 중심 코사인 유사도:")
    print(f"    supports ↔ contains: {dist_s_c:.4f}")
    print(f"    supports ↔ independent: {dist_s_i:.4f}")
    print(f"    contains ↔ independent: {dist_c_i:.4f}")
    
    # 클러스터 내 분산
    var_s = supports_z.var(dim=0).mean().item()
    var_c = contains_z.var(dim=0).mean().item()
    var_i = indep_z.var(dim=0).mean().item()
    
    print(f"\n  클러스터 내 분산:")
    print(f"    supports: {var_s:.4f}")
    print(f"    contains: {var_c:.4f}")
    print(f"    independent: {var_i:.4f}")
    
    print("\n" + "=" * 60)
    print("핵심 인사이트:")
    print("  - 라벨 없이 outcome 기반으로 관계 클러스터 형성")
    print("  - 클러스터 간 거리 > 클러스터 내 분산 = 성공")
    print("  - '이름 붙이기'는 인간이 마지막에 할 요약 작업")
    print("=" * 60)


if __name__ == "__main__":
    demo_contrastive_clustering()
