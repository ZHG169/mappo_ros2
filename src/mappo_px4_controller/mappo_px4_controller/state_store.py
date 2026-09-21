"""Typed state containers used by the ROS 2/PX4 controller."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


Vector3 = Tuple[float, float, float]


@dataclass
class DefenderState:
    """Latest state of one defender vehicle in ROS ENU coordinates."""

    vehicle_id: int
    position_enu: Vector3 = (0.0, 0.0, 0.0)
    velocity_enu: Vector3 = (0.0, 0.0, 0.0)
    yaw_enu: float = 0.0
    mode: str = 'PATROL'
    target_id_1based: int = -1
    armed: bool = False
    nav_state: int = 0
    position_valid: bool = False
    px4_timestamp_us: int = 0


@dataclass
class AttackerState:
    """Latest observed state of one attacker in ROS ENU coordinates."""

    attacker_id_0based: int
    position_enu: Vector3 = (0.0, 0.0, 0.0)
    velocity_enu: Vector3 = (0.0, 0.0, 0.0)
    alive: bool = True
    state: str = 'UNKNOWN'
    t_hit_sec: Optional[float] = None
    last_update_sec: float = 0.0


@dataclass
class TrackerState:
    """Estimated state and validity for one tracked attacker."""

    attacker_id_0based: int
    estimated_position_enu: Vector3 = (0.0, 0.0, 0.0)
    estimated_velocity_enu: Vector3 = (0.0, 0.0, 0.0)
    valid: bool = False
    last_update_sec: float = 0.0


@dataclass
class AssignmentState:
    """Latest MAPPO assignment for one defender."""

    defender_id_0based: int
    target_id_1based: int = -1
    velocity_target_mps: float = 4.0
    action_index: int = -1
    updated_at_sec: float = 0.0


@dataclass
class ControllerState:
    """All mutable controller state, kept out of module-level globals."""

    defenders: Dict[int, DefenderState] = field(default_factory=dict)
    attackers: Dict[int, AttackerState] = field(default_factory=dict)
    trackers: Dict[int, TrackerState] = field(default_factory=dict)
    assignments: Dict[int, AssignmentState] = field(default_factory=dict)
