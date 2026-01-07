"""PyBullet Simulation for Relation Learning.

실제 물리 시뮬레이션을 통한 관계 데이터 생성.

시나리오:
1. Stacking (supports): 블록 A가 B 위에 있음
2. Container (contains): 컵 안에 구슬이 있음
3. Independent: 무관한 두 객체

각 시나리오에서:
- 행동(action) 적용 (push, remove, lift)
- 결과(outcome) 관측
- 관계 라벨 없이 데이터만 수집

사용 예:
    env = RelationSimEnvironment()
    data = env.collect_episode("supports", action="remove_support")
    # data = {"state_before", "state_after", "action", "delta", "outcome_signature"}

Requirements:
    pip install pybullet

Author: Claude
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import torch
import time

# PyBullet optional (시스템에 없을 수 있음)
try:
    import pybullet as p
    import pybullet_data
    HAS_PYBULLET = True
except ImportError:
    HAS_PYBULLET = False
    print("  (PyBullet not found - using mock simulation)")


@dataclass
class SimulationConfig:
    """시뮬레이션 설정."""
    gui: bool = False           # GUI 모드 (디버깅용)
    time_step: float = 1/240    # 물리 타임스텝
    gravity: float = -9.81      # 중력
    
    # 행동 설정
    push_force: float = 5.0
    lift_height: float = 0.2
    remove_distance: float = 0.5
    
    # 관측 설정
    settle_steps: int = 120     # 안정화 대기 스텝
    episode_steps: int = 240    # 에피소드 길이


@dataclass  
class ObjectState:
    """객체 상태."""
    position: np.ndarray        # [x, y, z]
    orientation: np.ndarray     # quaternion [x, y, z, w]
    velocity: np.ndarray        # [vx, vy, vz]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    
    def to_tensor(self) -> torch.Tensor:
        """상태를 텐서로 변환 (6-DoF: pos + vel)."""
        return torch.tensor(
            np.concatenate([self.position, self.velocity]),
            dtype=torch.float32
        )
    
    @classmethod
    def from_pybullet(cls, body_id: int, physics_client: int = 0) -> "ObjectState":
        """PyBullet에서 상태 읽기."""
        pos, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=physics_client)
        vel, ang_vel = p.getBaseVelocity(body_id, physicsClientId=physics_client)
        return cls(
            position=np.array(pos),
            orientation=np.array(orn),
            velocity=np.array(vel),
            angular_velocity=np.array(ang_vel),
        )


@dataclass
class EpisodeData:
    """에피소드 데이터."""
    scenario: str               # "supports", "contains", "independent"
    action_type: str            # "push", "remove", "lift"
    
    state_i_before: ObjectState
    state_j_before: ObjectState
    state_i_after: ObjectState
    state_j_after: ObjectState
    
    action: np.ndarray          # 행동 벡터
    delta_j: np.ndarray         # j의 상태 변화
    
    # 결과 서명
    fell: bool                  # j가 떨어졌는가?
    contact_maintained: bool    # 접촉이 유지되었는가?
    stability_change: float     # 안정성 변화량
    
    def to_tensors(self) -> Dict[str, torch.Tensor]:
        """학습용 텐서로 변환."""
        return {
            "state_i": self.state_i_before.to_tensor(),
            "state_j": self.state_j_before.to_tensor(),
            "action": torch.tensor(self.action, dtype=torch.float32),
            "delta_j": torch.tensor(self.delta_j, dtype=torch.float32),
            "outcome_signature": torch.tensor([
                self.delta_j[2],  # z 변화
                np.linalg.norm(self.delta_j[:2]),  # 수평 변화
                float(self.contact_maintained),
                self.stability_change,
            ], dtype=torch.float32),
        }


class RelationSimEnvironment:
    """관계 학습용 PyBullet 시뮬레이션 환경."""
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.physics_client = None
        self.objects = {}
        
        if not HAS_PYBULLET:
            print("  Warning: PyBullet not available, using mock data")
    
    def connect(self):
        """PyBullet 연결."""
        if not HAS_PYBULLET:
            return
        
        if self.config.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.config.gravity)
        p.setTimeStep(self.config.time_step)
        
        # 바닥
        self.plane_id = p.loadURDF("plane.urdf")
    
    def disconnect(self):
        """PyBullet 연결 해제."""
        if HAS_PYBULLET and self.physics_client is not None:
            p.disconnect()
            self.physics_client = None
    
    def reset(self):
        """환경 리셋."""
        if not HAS_PYBULLET:
            return
        
        # 기존 객체 제거
        for obj_id in self.objects.values():
            p.removeBody(obj_id)
        self.objects = {}
    
    def _create_box(
        self, 
        name: str,
        position: List[float],
        half_extents: List[float] = [0.05, 0.05, 0.05],
        mass: float = 1.0,
        color: List[float] = [1, 0, 0, 1],
    ) -> int:
        """박스 생성."""
        if not HAS_PYBULLET:
            return -1
        
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
        
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
        )
        
        self.objects[name] = body_id
        return body_id
    
    def _create_sphere(
        self,
        name: str,
        position: List[float],
        radius: float = 0.03,
        mass: float = 0.5,
        color: List[float] = [0, 0, 1, 1],
    ) -> int:
        """구 생성."""
        if not HAS_PYBULLET:
            return -1
        
        collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
        )
        
        self.objects[name] = body_id
        return body_id
    
    def _setup_stacking(self):
        """Stacking (supports) 시나리오 설정.
        
        블록 B (테이블) 위에 블록 A (컵)
        """
        # 아래 블록 (지지체)
        self._create_box(
            "support_block",
            position=[0, 0, 0.05],
            half_extents=[0.1, 0.1, 0.05],
            mass=10.0,  # 무거움
            color=[0.5, 0.5, 0.5, 1],
        )
        
        # 위 블록 (피지지체)
        self._create_box(
            "top_block",
            position=[0, 0, 0.15],
            half_extents=[0.05, 0.05, 0.05],
            mass=1.0,
            color=[1, 0, 0, 1],
        )
        
        return "support_block", "top_block"
    
    def _setup_container(self):
        """Container (contains) 시나리오 설정.
        
        컵(open box) 안에 구슬
        """
        # 컵 (5개의 박스로 구성)
        cup_pos = [0, 0, 0.05]
        wall_thickness = 0.01
        cup_size = 0.08
        
        # 바닥
        self._create_box(
            "cup_bottom",
            position=[cup_pos[0], cup_pos[1], cup_pos[2]],
            half_extents=[cup_size, cup_size, wall_thickness],
            mass=5.0,
            color=[0, 0.5, 0, 1],
        )
        
        # 구슬
        self._create_sphere(
            "ball",
            position=[0, 0, 0.1],
            radius=0.02,
            mass=0.3,
            color=[0, 0, 1, 1],
        )
        
        return "cup_bottom", "ball"
    
    def _setup_independent(self):
        """Independent 시나리오 설정.
        
        무관한 두 블록
        """
        self._create_box(
            "block_a",
            position=[-0.2, 0, 0.05],
            half_extents=[0.05, 0.05, 0.05],
            mass=1.0,
            color=[1, 0, 0, 1],
        )
        
        self._create_box(
            "block_b",
            position=[0.2, 0, 0.05],
            half_extents=[0.05, 0.05, 0.05],
            mass=1.0,
            color=[0, 0, 1, 1],
        )
        
        return "block_a", "block_b"
    
    def _wait_settle(self, steps: int = None):
        """물리 안정화 대기."""
        if not HAS_PYBULLET:
            return
        
        steps = steps or self.config.settle_steps
        for _ in range(steps):
            p.stepSimulation()
    
    def _apply_action(self, obj_name: str, action_type: str) -> np.ndarray:
        """행동 적용.
        
        Returns:
            action 벡터 (6-DoF: force + torque)
        """
        action_vec = np.zeros(6)
        
        if not HAS_PYBULLET:
            return action_vec
        
        obj_id = self.objects.get(obj_name)
        if obj_id is None:
            return action_vec
        
        if action_type == "push":
            # 수평 push
            force = [self.config.push_force, 0, 0]
            p.applyExternalForce(
                obj_id, -1, force, [0, 0, 0],
                p.WORLD_FRAME
            )
            action_vec[:3] = force
            
        elif action_type == "remove":
            # 지지체 제거 (빠르게 아래로 이동)
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            new_pos = [pos[0], pos[1], pos[2] - self.config.remove_distance]
            p.resetBasePositionAndOrientation(obj_id, new_pos, orn)
            action_vec[2] = -self.config.remove_distance
            
        elif action_type == "lift":
            # 들어올리기
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            new_pos = [pos[0], pos[1], pos[2] + self.config.lift_height]
            p.resetBasePositionAndOrientation(obj_id, new_pos, orn)
            action_vec[2] = self.config.lift_height
            
        elif action_type == "tilt":
            # 기울이기 (컨테이너용)
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            # 30도 기울임
            new_orn = p.getQuaternionFromEuler([0.5, 0, 0])
            p.resetBasePositionAndOrientation(obj_id, pos, new_orn)
            action_vec[3] = 0.5  # roll
        
        return action_vec
    
    def collect_episode(
        self,
        scenario: str,
        action_type: str = "push",
    ) -> EpisodeData:
        """에피소드 수집.
        
        Args:
            scenario: "supports", "contains", "independent"
            action_type: "push", "remove", "lift", "tilt"
            
        Returns:
            EpisodeData with before/after states
        """
        if not HAS_PYBULLET:
            return self._mock_episode(scenario, action_type)
        
        self.reset()
        
        # 시나리오 설정
        if scenario == "supports":
            obj_i, obj_j = self._setup_stacking()
        elif scenario == "contains":
            obj_i, obj_j = self._setup_container()
        else:
            obj_i, obj_j = self._setup_independent()
        
        # 안정화
        self._wait_settle()
        
        # Before 상태
        state_i_before = ObjectState.from_pybullet(self.objects[obj_i], self.physics_client)
        state_j_before = ObjectState.from_pybullet(self.objects[obj_j], self.physics_client)
        
        # 행동 적용
        action = self._apply_action(obj_i, action_type)
        
        # 시뮬레이션 진행
        for _ in range(self.config.episode_steps):
            p.stepSimulation()
        
        # After 상태
        state_i_after = ObjectState.from_pybullet(self.objects[obj_i], self.physics_client)
        state_j_after = ObjectState.from_pybullet(self.objects[obj_j], self.physics_client)
        
        # 결과 계산
        delta_j = np.concatenate([
            state_j_after.position - state_j_before.position,
            state_j_after.velocity - state_j_before.velocity,
        ])
        
        # 낙하 판정: z가 크게 감소하면 떨어짐
        fell = (state_j_after.position[2] - state_j_before.position[2]) < -0.05
        
        # 접촉 유지: 변화가 작으면 유지
        contact_maintained = np.linalg.norm(delta_j[:3]) < 0.02
        
        # 안정성 변화
        stability_change = state_j_before.position[2] - state_j_after.position[2]
        
        return EpisodeData(
            scenario=scenario,
            action_type=action_type,
            state_i_before=state_i_before,
            state_j_before=state_j_before,
            state_i_after=state_i_after,
            state_j_after=state_j_after,
            action=action,
            delta_j=delta_j,
            fell=fell,
            contact_maintained=contact_maintained,
            stability_change=stability_change,
        )
    
    def _mock_episode(
        self,
        scenario: str,
        action_type: str,
    ) -> EpisodeData:
        """PyBullet 없을 때 목업 데이터 생성."""
        
        # 목업 상태
        state_i_before = ObjectState(
            position=np.array([0, 0, 0.05]),
            orientation=np.array([0, 0, 0, 1]),
            velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
        )
        
        state_j_before = ObjectState(
            position=np.array([0, 0, 0.15]),
            orientation=np.array([0, 0, 0, 1]),
            velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
        )
        
        # 시나리오별 결과
        if scenario == "supports" and action_type == "remove":
            # 지지체 제거 → 낙하
            delta_j = np.array([0, 0, -0.1, 0, 0, -1.0])
            fell = True
        elif scenario == "supports":
            # 다른 행동 → 같이 움직임
            delta_j = np.array([0.05, 0, 0, 0.1, 0, 0])
            fell = False
        elif scenario == "contains":
            # 컨테이너 → 내용물 따라감
            delta_j = np.array([0.02, 0.01, 0, 0, 0, 0])
            fell = False
        else:
            # 무관 → 변화 없음
            delta_j = np.zeros(6)
            fell = False
        
        state_i_after = ObjectState(
            position=state_i_before.position + np.array([0.1, 0, 0]),
            orientation=state_i_before.orientation,
            velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
        )
        
        state_j_after = ObjectState(
            position=state_j_before.position + delta_j[:3],
            orientation=state_j_before.orientation,
            velocity=delta_j[3:],
            angular_velocity=np.zeros(3),
        )
        
        return EpisodeData(
            scenario=scenario,
            action_type=action_type,
            state_i_before=state_i_before,
            state_j_before=state_j_before,
            state_i_after=state_i_after,
            state_j_after=state_j_after,
            action=np.array([0.1, 0, 0, 0, 0, 0]),
            delta_j=delta_j,
            fell=fell,
            contact_maintained=not fell,
            stability_change=delta_j[2] if fell else 0,
        )
    
    def collect_dataset(
        self,
        num_episodes: int = 100,
        scenarios: List[str] = ["supports", "contains", "independent"],
        actions: List[str] = ["push", "remove"],
    ) -> List[EpisodeData]:
        """데이터셋 수집.
        
        Args:
            num_episodes: 에피소드 수
            scenarios: 시나리오 목록
            actions: 행동 목록
            
        Returns:
            EpisodeData 리스트
        """
        dataset = []
        
        for i in range(num_episodes):
            scenario = np.random.choice(scenarios)
            action = np.random.choice(actions)
            
            episode = self.collect_episode(scenario, action)
            dataset.append(episode)
            
            if (i + 1) % 20 == 0:
                print(f"  Collected {i+1}/{num_episodes} episodes")
        
        return dataset


class PhysicsRelationDataset:
    """물리 시뮬레이션 기반 관계 데이터셋 (PyTorch)."""
    
    def __init__(self, episodes: List[EpisodeData]):
        self.episodes = episodes
        self._preprocess()
    
    def _preprocess(self):
        """에피소드를 텐서로 변환."""
        self.states_i = []
        self.states_j = []
        self.actions = []
        self.deltas = []
        self.signatures = []
        self.scenarios = []
        
        for ep in self.episodes:
            tensors = ep.to_tensors()
            self.states_i.append(tensors["state_i"])
            self.states_j.append(tensors["state_j"])
            self.actions.append(tensors["action"])
            self.deltas.append(tensors["delta_j"])
            self.signatures.append(tensors["outcome_signature"])
            self.scenarios.append(ep.scenario)
        
        self.states_i = torch.stack(self.states_i)
        self.states_j = torch.stack(self.states_j)
        self.actions = torch.stack(self.actions)
        self.deltas = torch.stack(self.deltas)
        self.signatures = torch.stack(self.signatures)
    
    def __len__(self):
        return len(self.episodes)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """배치 샘플링."""
        indices = np.random.choice(len(self), batch_size, replace=True)
        return {
            "state_i": self.states_i[indices],
            "state_j": self.states_j[indices],
            "action": self.actions[indices],
            "delta_j": self.deltas[indices],
            "signature": self.signatures[indices],
        }


def demo_pybullet_simulation():
    """PyBullet 시뮬레이션 데모."""
    print("=" * 60)
    print("PyBullet Relation Simulation Demo")
    print("=" * 60)
    
    env = RelationSimEnvironment()
    
    if HAS_PYBULLET:
        env.connect()
    
    print("\n1. Supports 시나리오 (remove action):")
    ep = env.collect_episode("supports", "remove")
    print(f"   Before j position: {ep.state_j_before.position}")
    print(f"   After j position: {ep.state_j_after.position}")
    print(f"   Fell: {ep.fell}")
    print(f"   Delta z: {ep.delta_j[2]:.4f}")
    
    print("\n2. Independent 시나리오 (push action):")
    ep = env.collect_episode("independent", "push")
    print(f"   Before j position: {ep.state_j_before.position}")
    print(f"   After j position: {ep.state_j_after.position}")
    print(f"   Delta z: {ep.delta_j[2]:.4f}")
    
    print("\n3. 데이터셋 수집:")
    dataset = env.collect_dataset(num_episodes=50)
    print(f"   Total episodes: {len(dataset)}")
    
    # 시나리오별 통계
    by_scenario = {}
    for ep in dataset:
        if ep.scenario not in by_scenario:
            by_scenario[ep.scenario] = {"count": 0, "fell": 0}
        by_scenario[ep.scenario]["count"] += 1
        if ep.fell:
            by_scenario[ep.scenario]["fell"] += 1
    
    print("\n   시나리오별 통계:")
    for scenario, stats in by_scenario.items():
        fell_rate = stats["fell"] / stats["count"] * 100
        print(f"     {scenario}: {stats['count']} eps, {fell_rate:.1f}% fell")
    
    if HAS_PYBULLET:
        env.disconnect()
    
    # PyTorch 데이터셋 변환
    print("\n4. PyTorch 데이터셋:")
    torch_dataset = PhysicsRelationDataset(dataset)
    batch = torch_dataset.sample(8)
    print(f"   state_i shape: {batch['state_i'].shape}")
    print(f"   delta_j shape: {batch['delta_j'].shape}")
    print(f"   signature shape: {batch['signature'].shape}")
    
    print("\n" + "=" * 60)
    print("핵심 인사이트:")
    print("  - 실제 물리 시뮬레이션에서 관계 데이터 수집")
    print("  - remove action → supports 관계에서만 낙하 발생")
    print("  - 라벨 없이 outcome으로 관계 구별 가능")
    print("=" * 60)


if __name__ == "__main__":
    demo_pybullet_simulation()
