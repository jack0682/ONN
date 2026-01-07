"""Topological Regularization for Relational Fields.

관계-장의 위상적 일관성을 보장하는 정규화 항.

핵심 아이디어:
- 관계는 "무작위 벡터장"이 아니라 "정합적인 형태"를 가져야 함
- Forman-Ricci 곡률: 모순 관계(bottleneck) 탐지
- 호몰로지: 관계 패턴의 안정된 위상적 특징 유지

용어:
- 양수 곡률: 밀집된 영역 (클러스터)
- 음수 곡률: 병목/다리 (경계)
- 사이클 일관성: A→B→C→A 경로가 닫힘

사용 예:
    topo_reg = TopologicalRegularizer(embed_dim=48)
    loss_topo = topo_reg.compute_loss(z_relations, adjacency)

Author: Claude
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TopologicalConfig:
    """위상 정규화 설정."""
    embed_dim: int = 48
    
    # 곡률 설정
    curvature_weight: float = 0.1
    target_curvature: float = 0.0  # 균형 상태 목표
    
    # 사이클 설정
    cycle_weight: float = 0.2
    max_cycle_length: int = 4
    
    # 호몰로지 설정
    homology_weight: float = 0.1
    persistence_threshold: float = 0.1


class RelationalCurvature(nn.Module):
    """관계 공간에서의 이산 곡률 계산.
    
    Forman-Ricci 곡률의 연속 버전:
    - 관계 임베딩 간의 "곡률"을 계산
    - 모순 관계(불일치)를 탐지
    """
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 곡률 추정 네트워크
        self.curvature_net = nn.Sequential(
            nn.Linear(embed_dim * 3, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
    
    def compute_edge_curvature(
        self,
        z_ij: torch.Tensor,
        z_neighbors_i: torch.Tensor,
        z_neighbors_j: torch.Tensor,
    ) -> torch.Tensor:
        """엣지 (i, j)의 곡률 계산.
        
        Args:
            z_ij: (..., embed_dim) 엣지 i→j의 관계 임베딩
            z_neighbors_i: (num_neighbors, embed_dim) i의 다른 이웃들과의 관계
            z_neighbors_j: (num_neighbors, embed_dim) j의 다른 이웃들과의 관계
            
        Returns:
            스칼라 곡률 값
        """
        # 이웃 관계의 평균
        if z_neighbors_i.numel() > 0:
            z_avg_i = z_neighbors_i.mean(dim=0)
        else:
            z_avg_i = torch.zeros(self.embed_dim, device=z_ij.device)
        
        if z_neighbors_j.numel() > 0:
            z_avg_j = z_neighbors_j.mean(dim=0)
        else:
            z_avg_j = torch.zeros(self.embed_dim, device=z_ij.device)
        
        # 곡률 추정
        combined = torch.cat([z_ij, z_avg_i, z_avg_j], dim=-1)
        curvature = self.curvature_net(combined)
        
        return curvature
    
    def batch_curvature(
        self,
        z_relations: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """배치 곡률 계산.
        
        Args:
            z_relations: (batch, num_edges, embed_dim) 관계 임베딩
            adjacency: (batch, num_nodes, num_nodes) 인접 행렬
            
        Returns:
            (batch, num_edges) 엣지별 곡률
        """
        batch_size, num_edges, embed_dim = z_relations.shape
        
        # 간단한 곡률 근사: 이웃 관계와의 차이
        # 진정한 Forman-Ricci는 그래프 구조가 필요
        
        # 엣지 간 유사도 기반 곡률
        z_norm = F.normalize(z_relations, p=2, dim=-1)
        
        # 각 엣지가 다른 엣지들과 얼마나 일관적인가
        similarity = torch.bmm(z_norm, z_norm.transpose(-1, -2))  # (batch, num_edges, num_edges)
        
        # 평균 유사도 = 곡률 proxy (높으면 밀집 = 양수 곡률)
        curvature = similarity.mean(dim=-1) - 0.5  # 중심화
        
        return curvature


class CycleConsistency(nn.Module):
    """사이클 일관성 정규화.
    
    A → B → C → A 경로가 닫혀야 함:
    z_AB + z_BC + z_CA ≈ 0 (또는 identity)
    """
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 사이클 합성 연산자
        self.compose = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Tanh(),
        )
    
    def compose_relations(
        self,
        z_ab: torch.Tensor,
        z_bc: torch.Tensor,
    ) -> torch.Tensor:
        """두 관계 합성: z_ab ∘ z_bc = z_ac.
        
        Args:
            z_ab: A→B 관계
            z_bc: B→C 관계
            
        Returns:
            z_ac: A→C 추론된 관계
        """
        combined = torch.cat([z_ab, z_bc], dim=-1)
        return self.compose(combined)
    
    def cycle_loss(
        self,
        z_ab: torch.Tensor,
        z_bc: torch.Tensor,
        z_ca: torch.Tensor,
    ) -> torch.Tensor:
        """3-사이클 손실.
        
        z_ab ∘ z_bc ∘ z_ca ≈ identity (zero for additive)
        """
        # A → B → C
        z_ac_inferred = self.compose_relations(z_ab, z_bc)
        
        # A → C → A should be ~identity
        z_aa_inferred = self.compose_relations(z_ac_inferred, z_ca)
        
        # Identity = zero vector (additive 모델)
        loss = z_aa_inferred.pow(2).mean()
        
        return loss
    
    def batch_cycle_loss(
        self,
        z_relations: torch.Tensor,
        num_cycles: int = 10,
    ) -> torch.Tensor:
        """랜덤 사이클로 배치 손실 계산.
        
        Args:
            z_relations: (batch, num_edges, embed_dim)
            num_cycles: 샘플링할 사이클 수
            
        Returns:
            평균 사이클 손실
        """
        batch_size, num_edges, _ = z_relations.shape
        
        if num_edges < 3:
            return torch.tensor(0.0, device=z_relations.device)
        
        total_loss = 0.0
        
        for _ in range(num_cycles):
            # 랜덤 3-사이클 선택
            indices = torch.randperm(num_edges, device=z_relations.device)[:3]
            
            z_ab = z_relations[:, indices[0], :]
            z_bc = z_relations[:, indices[1], :]
            z_ca = z_relations[:, indices[2], :]
            
            total_loss = total_loss + self.cycle_loss(z_ab, z_bc, z_ca)
        
        return total_loss / num_cycles


class PersistenceRegularizer(nn.Module):
    """지속 호몰로지 기반 정규화.
    
    관계 공간의 "위상적 특징"이 노이즈에 안정적이어야 함:
    - 짧은 수명의 특징 = 노이즈
    - 긴 수명의 특징 = 진짜 구조
    """
    
    def __init__(self, embed_dim: int, threshold: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.threshold = threshold
    
    def compute_persistence(
        self,
        z_relations: torch.Tensor,
    ) -> torch.Tensor:
        """간단한 persistence proxy 계산.
        
        진짜 지속 호몰로지는 계산이 비싸므로,
        관계 임베딩의 안정성으로 근사.
        """
        # 관계 크기 분포
        norms = z_relations.norm(dim=-1)
        
        # 큰 관계는 "오래 지속"하는 특징
        # 작은 관계는 노이즈
        
        # threshold 이하는 제거 (노이즈)
        persistent_mask = norms > self.threshold
        
        # Persistence = 큰 관계의 비율
        persistence = persistent_mask.float().mean()
        
        return persistence
    
    def loss(
        self,
        z_relations: torch.Tensor,
        target_persistence: float = 0.7,
    ) -> torch.Tensor:
        """Persistence 손실.
        
        너무 많은 노이즈 관계가 있으면 벌점.
        """
        persistence = self.compute_persistence(z_relations)
        
        # Target 이상 유지
        loss = F.relu(target_persistence - persistence)
        
        return loss


class TopologicalRegularizer(nn.Module):
    """통합 위상 정규화.
    
    1. 곡률 정규화: 모순 관계 탐지
    2. 사이클 일관성: 경로 닫힘
    3. Persistence: 노이즈 억제
    """
    
    def __init__(self, config: TopologicalConfig = None):
        super().__init__()
        
        if config is None:
            config = TopologicalConfig()
        self.config = config
        
        self.curvature = RelationalCurvature(config.embed_dim)
        self.cycle = CycleConsistency(config.embed_dim)
        self.persistence = PersistenceRegularizer(
            config.embed_dim, config.persistence_threshold
        )
    
    def compute_loss(
        self,
        z_relations: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """전체 위상 손실 계산.
        
        Args:
            z_relations: (batch, num_edges, embed_dim) 또는 (batch, embed_dim)
            adjacency: (optional) 인접 행렬
            
        Returns:
            손실 딕셔너리
        """
        # 차원 조정
        if z_relations.dim() == 2:
            z_relations = z_relations.unsqueeze(1)  # (batch, 1, embed_dim)
        
        # 1. 곡률 손실
        curvatures = self.curvature.batch_curvature(
            z_relations, 
            adjacency if adjacency is not None else torch.zeros(1)
        )
        # 목표: 균형 곡률 (0에 가까움)
        loss_curvature = (curvatures - self.config.target_curvature).pow(2).mean()
        
        # 2. 사이클 손실
        loss_cycle = self.cycle.batch_cycle_loss(z_relations)
        
        # 3. Persistence 손실
        loss_persistence = self.persistence.loss(z_relations)
        
        # 총 손실
        total_loss = (
            self.config.curvature_weight * loss_curvature +
            self.config.cycle_weight * loss_cycle +
            self.config.homology_weight * loss_persistence
        )
        
        return {
            "total": total_loss,
            "curvature": loss_curvature,
            "cycle": loss_cycle,
            "persistence": loss_persistence,
        }


def demo_topological_regularizer():
    """위상 정규화 데모."""
    print("=" * 60)
    print("Topological Regularizer Demo")
    print("=" * 60)
    
    config = TopologicalConfig(embed_dim=48)
    regularizer = TopologicalRegularizer(config)
    
    print(f"  Parameters: {sum(p.numel() for p in regularizer.parameters()):,}")
    
    # 테스트 데이터
    batch_size = 8
    num_edges = 10
    
    # 1. 일관된 관계 (낮은 손실 기대)
    z_consistent = torch.randn(batch_size, num_edges, config.embed_dim) * 0.3
    losses_consistent = regularizer.compute_loss(z_consistent)
    
    print(f"\n  일관된 관계:")
    print(f"    Total: {losses_consistent['total'].item():.4f}")
    print(f"    Curvature: {losses_consistent['curvature'].item():.4f}")
    print(f"    Cycle: {losses_consistent['cycle'].item():.4f}")
    
    # 2. 불일치 관계 (높은 손실 기대)
    z_inconsistent = torch.randn(batch_size, num_edges, config.embed_dim) * 2.0
    losses_inconsistent = regularizer.compute_loss(z_inconsistent)
    
    print(f"\n  불일치 관계:")
    print(f"    Total: {losses_inconsistent['total'].item():.4f}")
    print(f"    Curvature: {losses_inconsistent['curvature'].item():.4f}")
    print(f"    Cycle: {losses_inconsistent['cycle'].item():.4f}")
    
    print("\n" + "=" * 60)
    print("핵심 인사이트:")
    print("  - 곡률: 관계 공간의 '형태' 유지")
    print("  - 사이클: A→B→C→A 경로 일관성")
    print("  - Persistence: 노이즈 vs 진짜 구조 구별")
    print("=" * 60)


if __name__ == "__main__":
    demo_topological_regularizer()
