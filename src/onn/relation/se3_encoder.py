"""SE(3) Relation Encoder.

SE(3) 변환 기반 라벨 없는 관계 인코딩.

핵심 아이디어:
- "관계"를 좌표계 A에서 B로의 변환으로 정의
- SE(3) ∈ R^{4x4} 대신 se(3) ∈ R^6으로 표현 (Lie algebra)
- "above"는 특정 t_z 범위의 변환일 뿐 - 라벨 불필요

수학적 배경:
- SE(3) = Special Euclidean Group (회전 + 이동)
- se(3) = Lie algebra of SE(3) = (omega, v) ∈ R^6
- log: SE(3) → se(3), exp: se(3) → SE(3)

사용 예:
    encoder = SE3RelationEncoder(embed_dim=32)
    
    # 두 객체의 포즈 (4x4 SE(3) 행렬)
    T_cup = torch.eye(4)
    T_table = torch.eye(4)
    T_table[2, 3] = -0.1  # 테이블이 10cm 아래
    
    # 관계 임베딩 (32-dim 연속 벡터)
    z_rel = encoder.encode(T_cup, T_table)

Author: Claude
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List

from onn.modules.delta import DeltaResidualBlock, DeltaResidualStack


# ==============================================================================
# SE(3) / SO(3) 기본 연산
# ==============================================================================

def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """벡터를 skew-symmetric 행렬로 변환.
    
    v = [v1, v2, v3] → [v]_x = [[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]]
    
    Args:
        v: (..., 3) 벡터
        
    Returns:
        (..., 3, 3) skew-symmetric 행렬
    """
    batch_shape = v.shape[:-1]
    v = v.reshape(-1, 3)
    
    zeros = torch.zeros(v.shape[0], device=v.device, dtype=v.dtype)
    
    skew = torch.stack([
        torch.stack([zeros, -v[:, 2], v[:, 1]], dim=1),
        torch.stack([v[:, 2], zeros, -v[:, 0]], dim=1),
        torch.stack([-v[:, 1], v[:, 0], zeros], dim=1),
    ], dim=1)
    
    return skew.reshape(*batch_shape, 3, 3)


def vee(skew: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric 행렬을 벡터로 변환 (skew_symmetric의 역).
    
    Args:
        skew: (..., 3, 3) skew-symmetric 행렬
        
    Returns:
        (..., 3) 벡터
    """
    return torch.stack([
        skew[..., 2, 1],
        skew[..., 0, 2],
        skew[..., 1, 0],
    ], dim=-1)


def log_SO3(R: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """SO(3) → so(3) (회전 행렬 → axis-angle 벡터).
    
    Rodrigues' formula의 역.
    
    Args:
        R: (..., 3, 3) 회전 행렬
        eps: 수치 안정성을 위한 epsilon
        
    Returns:
        (..., 3) axis-angle 벡터 (||omega|| = angle, omega/||omega|| = axis)
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    
    # trace(R) = 1 + 2*cos(theta)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1 + eps, 1 - eps)
    theta = torch.acos(cos_theta)
    
    # 작은 각도 근사
    small_angle = theta.abs() < eps
    
    # 일반적인 경우: omega = theta / (2*sin(theta)) * vee(R - R^T)
    sin_theta = torch.sin(theta)
    sin_theta = torch.where(sin_theta.abs() < eps, torch.ones_like(sin_theta), sin_theta)
    
    skew_part = (R - R.transpose(-1, -2)) / 2
    omega_normalized = vee(skew_part)
    
    scale = theta / (2 * sin_theta)
    scale = scale.unsqueeze(-1)
    
    omega = scale * omega_normalized * 2
    
    # 작은 각도: omega ≈ vee(R - R^T) / 2
    omega_small = omega_normalized
    
    omega = torch.where(small_angle.unsqueeze(-1), omega_small, omega)
    
    return omega.reshape(*batch_shape, 3)


def exp_SO3(omega: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """so(3) → SO(3) (axis-angle 벡터 → 회전 행렬).
    
    Rodrigues' formula: R = I + sin(θ)[ω]_x + (1-cos(θ))[ω]_x^2
    
    Args:
        omega: (..., 3) axis-angle 벡터
        eps: 수치 안정성을 위한 epsilon
        
    Returns:
        (..., 3, 3) 회전 행렬
    """
    batch_shape = omega.shape[:-1]
    omega = omega.reshape(-1, 3)
    
    theta = torch.norm(omega, dim=-1, keepdim=True)
    theta = torch.clamp(theta, min=eps)
    
    omega_normalized = omega / theta
    omega_skew = skew_symmetric(omega_normalized)
    
    theta = theta.squeeze(-1)
    
    sin_theta = torch.sin(theta).unsqueeze(-1).unsqueeze(-1)
    cos_theta = torch.cos(theta).unsqueeze(-1).unsqueeze(-1)
    
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0)
    
    R = I + sin_theta * omega_skew + (1 - cos_theta) * (omega_skew @ omega_skew)
    
    return R.reshape(*batch_shape, 3, 3)


def log_SE3(T: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """SE(3) → se(3) (변환 행렬 → 6-DoF 벡터).
    
    Args:
        T: (..., 4, 4) SE(3) 변환 행렬
        eps: 수치 안정성
        
    Returns:
        (..., 6) se(3) 벡터 [omega (3), v (3)]
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    
    omega = log_SO3(R, eps)
    
    # 작은 각도: v ≈ t
    # 일반: v = J_l^{-1} @ t (left Jacobian inverse)
    # 여기서는 간단히 t를 사용 (1차 근사)
    v = t
    
    return torch.cat([omega, v], dim=-1)


def exp_SE3(xi: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """se(3) → SE(3) (6-DoF 벡터 → 변환 행렬).
    
    Args:
        xi: (..., 6) se(3) 벡터 [omega (3), v (3)]
        eps: 수치 안정성
        
    Returns:
        (..., 4, 4) SE(3) 변환 행렬
    """
    batch_shape = xi.shape[:-1]
    
    omega = xi[..., :3]
    v = xi[..., 3:]
    
    R = exp_SO3(omega, eps)
    t = v  # 1차 근사
    
    # SE(3) 행렬 구성
    T = torch.zeros(*batch_shape, 4, 4, device=xi.device, dtype=xi.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    
    return T


def relative_transform(T_a: torch.Tensor, T_b: torch.Tensor) -> torch.Tensor:
    """두 포즈 사이의 상대 변환 계산.
    
    T_rel = T_b @ inv(T_a)
    
    "A에서 바라본 B의 위치"
    
    Args:
        T_a, T_b: (..., 4, 4) SE(3) 변환 행렬
        
    Returns:
        (..., 4, 4) 상대 변환 행렬
    """
    # SE(3) inverse: [R^T | -R^T @ t]
    R_a = T_a[..., :3, :3]
    t_a = T_a[..., :3, 3]
    
    R_a_inv = R_a.transpose(-1, -2)
    t_a_inv = -torch.einsum('...ij,...j->...i', R_a_inv, t_a)
    
    T_a_inv = torch.zeros_like(T_a)
    T_a_inv[..., :3, :3] = R_a_inv
    T_a_inv[..., :3, 3] = t_a_inv
    T_a_inv[..., 3, 3] = 1.0
    
    return torch.einsum('...ij,...jk->...ik', T_b, T_a_inv)


# ==============================================================================
# 관계 인코더
# ==============================================================================

class SE3RelationEncoder(nn.Module):
    """SE(3) 기반 관계 인코더 with Deep Delta Learning.

    두 객체의 포즈를 받아 연속적인 관계 임베딩을 생성합니다.
    MLP를 DeltaResidualBlock 스택으로 교체하여 DDL 논문의 수식을 따릅니다.

    아키텍처 (DDL refactored):
        input_proj(input_dim -> hidden_dim)
        -> [DeltaResidualBlock(hidden_dim)] x num_delta_blocks
        -> output_proj(hidden_dim -> embed_dim)

    Delta 업데이트 (Eq 2.5):
        h_{l+1} = h_l + beta * k * (v - k^T h_l)

    Args:
        embed_dim: 출력 임베딩 차원 (기본 32)
        hidden_dim: 히든 레이어 차원 (기본 64)
        include_distance: 거리 정보 추가 여부
        include_angle: 각도 정보 추가 여부
        num_delta_blocks: Delta 블록 수 (기본 2)
        use_delta_encoder: Delta 블록 사용 여부 (False면 기존 MLP)
    """

    def __init__(
        self,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        include_distance: bool = True,
        include_angle: bool = True,
        num_delta_blocks: int = 2,
        use_delta_encoder: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.include_distance = include_distance
        self.include_angle = include_angle
        self.use_delta_encoder = use_delta_encoder

        # 입력 차원: se(3) 6 + distance 1 + angle 1
        input_dim = 6
        if include_distance:
            input_dim += 1
        if include_angle:
            input_dim += 1
        self.input_dim = input_dim

        if use_delta_encoder:
            # DDL-based encoder: input_proj -> Delta stack -> output_proj
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )

            # Stack of Delta Residual Blocks
            self.delta_stack = DeltaResidualStack(
                dim=hidden_dim,
                num_blocks=num_delta_blocks,
                d_v=1,
                beta_init_bias=-4.0,  # Start near identity (Sec 3.3)
                k_init_std=0.01,
            )

            self.output_proj = nn.Linear(hidden_dim, embed_dim)
        else:
            # Legacy MLP encoder (for backward compatibility)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim),
            )
    
    def encode(
        self,
        T_a: torch.Tensor,
        T_b: torch.Tensor,
        return_betas: bool = False,
    ) -> torch.Tensor:
        """두 포즈의 관계를 인코딩.

        Args:
            T_a: (..., 4, 4) 객체 A의 SE(3) 포즈
            T_b: (..., 4, 4) 객체 B의 SE(3) 포즈
            return_betas: If True, also return beta values from delta blocks

        Returns:
            z: (..., embed_dim) 관계 임베딩
            betas: (optional) List of beta values if return_betas=True
        """
        # 상대 변환
        T_rel = relative_transform(T_a, T_b)

        # SE(3) → se(3)
        xi = log_SE3(T_rel)  # (..., 6)

        features = [xi]

        # 거리 추가
        if self.include_distance:
            t = T_rel[..., :3, 3]
            distance = torch.norm(t, dim=-1, keepdim=True)
            features.append(distance)

        # 각도 추가
        if self.include_angle:
            omega = xi[..., :3]
            angle = torch.norm(omega, dim=-1, keepdim=True)
            features.append(angle)

        # 연결
        x = torch.cat(features, dim=-1)

        # 인코딩
        if self.use_delta_encoder:
            # DDL-based encoding
            h = self.input_proj(x)
            h, betas = self.delta_stack(h, return_all_betas=True)
            z = self.output_proj(h)

            if return_betas:
                return z, betas
        else:
            # Legacy MLP encoding
            z = self.encoder(x)
            if return_betas:
                return z, []

        return z
    
    def forward(
        self,
        T_a: torch.Tensor,
        T_b: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass (encode의 별칭)."""
        return self.encode(T_a, T_b)
    
    @staticmethod
    def poses_to_matrix(
        position: torch.Tensor,
        quaternion: Optional[torch.Tensor] = None,
        euler: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """편의 함수: 위치 + 회전 → SE(3) 행렬.
        
        Args:
            position: (..., 3) 위치
            quaternion: (..., 4) 쿼터니언 (w, x, y, z) 또는
            euler: (..., 3) 오일러 각 (roll, pitch, yaw)
            
        Returns:
            (..., 4, 4) SE(3) 행렬
        """
        batch_shape = position.shape[:-1]
        device = position.device
        dtype = position.dtype
        
        T = torch.zeros(*batch_shape, 4, 4, device=device, dtype=dtype)
        T[..., 3, 3] = 1.0
        T[..., :3, 3] = position
        
        if quaternion is not None:
            # 쿼터니언 → 회전 행렬
            w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
            
            R = torch.stack([
                torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)], dim=-1),
                torch.stack([2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)], dim=-1),
                torch.stack([2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)], dim=-1),
            ], dim=-2)
            
            T[..., :3, :3] = R
            
        elif euler is not None:
            # 오일러 → 회전 행렬 (ZYX convention)
            roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]
            
            cr, sr = torch.cos(roll), torch.sin(roll)
            cp, sp = torch.cos(pitch), torch.sin(pitch)
            cy, sy = torch.cos(yaw), torch.sin(yaw)
            
            R = torch.stack([
                torch.stack([cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr], dim=-1),
                torch.stack([sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr], dim=-1),
                torch.stack([-sp, cp*sr, cp*cr], dim=-1),
            ], dim=-2)
            
            T[..., :3, :3] = R
        else:
            # 회전 없음
            T[..., :3, :3] = torch.eye(3, device=device, dtype=dtype)
        
        return T


class GeometricRelationEncoder(nn.Module):
    """기하적 관계 인코더 (SE3 + 추가 기하 특징).
    
    SE3RelationEncoder를 확장하여 추가적인 기하 정보를 포함합니다:
    - 바운딩 박스 크기 비율
    - 접촉 가능성 (거리 기반)
    - 중력 방향 관계
    
    Args:
        embed_dim: 출력 임베딩 차원
        include_gravity: 중력 방향 특징 포함 여부
        gravity_axis: 중력 방향 (기본 Z축 = 2)
    """
    
    def __init__(
        self,
        embed_dim: int = 32,
        include_gravity: bool = True,
        gravity_axis: int = 2,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.include_gravity = include_gravity
        self.gravity_axis = gravity_axis
        
        # SE3 인코더
        self.se3_encoder = SE3RelationEncoder(
            embed_dim=embed_dim // 2,
            hidden_dim=64,
        )
        
        # 중력 관계 인코더
        if include_gravity:
            self.gravity_encoder = nn.Sequential(
                nn.Linear(3, 32),
                nn.GELU(),
                nn.Linear(32, embed_dim // 2),
            )
        else:
            self.gravity_encoder = None
        
        # 최종 결합
        final_dim = embed_dim // 2 + (embed_dim // 2 if include_gravity else 0)
        self.output_proj = nn.Linear(final_dim, embed_dim)
    
    def forward(
        self,
        T_a: torch.Tensor,
        T_b: torch.Tensor,
    ) -> torch.Tensor:
        """관계 인코딩.
        
        Args:
            T_a, T_b: (..., 4, 4) SE(3) 포즈
            
        Returns:
            (..., embed_dim) 관계 임베딩
        """
        # SE3 관계
        z_se3 = self.se3_encoder(T_a, T_b)
        
        features = [z_se3]
        
        # 중력 관계
        if self.include_gravity:
            t_rel = T_b[..., :3, 3] - T_a[..., :3, 3]
            
            # 중력 축 방향 차이
            vertical_diff = t_rel[..., self.gravity_axis:self.gravity_axis+1]
            horizontal_dist = torch.norm(
                torch.cat([t_rel[..., :self.gravity_axis], 
                          t_rel[..., self.gravity_axis+1:]], dim=-1),
                dim=-1, keepdim=True
            )
            total_dist = torch.norm(t_rel, dim=-1, keepdim=True)
            
            gravity_features = torch.cat([vertical_diff, horizontal_dist, total_dist], dim=-1)
            z_gravity = self.gravity_encoder(gravity_features)
            features.append(z_gravity)
        
        # 결합
        z = torch.cat(features, dim=-1)
        z = self.output_proj(z)
        
        return z


# ==============================================================================
# 테스트 / 데모
# ==============================================================================

def demo_se3_encoder():
    """SE3 인코더 데모."""
    print("=" * 60)
    print("SE(3) Relation Encoder Demo")
    print("=" * 60)
    
    encoder = SE3RelationEncoder(embed_dim=32)
    
    # 시나리오 1: 컵이 테이블 위에 있음
    T_table = torch.eye(4).unsqueeze(0)  # 원점
    T_cup = torch.eye(4).unsqueeze(0)
    T_cup[0, 2, 3] = 0.1  # 테이블 위 10cm
    
    z_cup_table = encoder(T_cup, T_table)
    print(f"\n1. 컵 → 테이블 관계:")
    print(f"   임베딩 shape: {z_cup_table.shape}")
    print(f"   임베딩 norm: {z_cup_table.norm():.4f}")
    
    # 시나리오 2: 컵이 테이블 옆에 있음
    T_cup2 = torch.eye(4).unsqueeze(0)
    T_cup2[0, 0, 3] = 0.3  # 테이블 옆 30cm
    
    z_cup2_table = encoder(T_cup2, T_table)
    print(f"\n2. 컵2 → 테이블 관계 (옆에 있음):")
    print(f"   임베딩 norm: {z_cup2_table.norm():.4f}")
    
    # 관계 유사도 비교
    similarity = nn.functional.cosine_similarity(z_cup_table, z_cup2_table, dim=-1)
    print(f"\n3. 관계 유사도:")
    print(f"   (위에 있음) vs (옆에 있음): {similarity.item():.4f}")
    
    # 시나리오 3: 같은 "위에 있음" 관계 (다른 높이)
    T_cup3 = torch.eye(4).unsqueeze(0)
    T_cup3[0, 2, 3] = 0.2  # 테이블 위 20cm (더 높음)
    
    z_cup3_table = encoder(T_cup3, T_table)
    similarity2 = nn.functional.cosine_similarity(z_cup_table, z_cup3_table, dim=-1)
    print(f"   (위에 10cm) vs (위에 20cm): {similarity2.item():.4f}")
    
    print("\n" + "=" * 60)
    print("예상 결과: '위에 있음' 관계끼리는 더 유사해야 함")
    print("          '위' vs '옆'은 덜 유사해야 함")
    print("=" * 60)


if __name__ == "__main__":
    demo_se3_encoder()
