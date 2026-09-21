"""Coordinate conversions between PX4 NED/FRD and ROS ENU/FLU."""

from math import pi
from typing import Sequence, Tuple


Vector3 = Tuple[float, float, float]


def _vector3(values: Sequence[float]) -> Vector3:
    if len(values) != 3:
        raise ValueError('expected a three-element vector')
    return float(values[0]), float(values[1]), float(values[2])


def ned_to_enu(vector_ned: Sequence[float]) -> Vector3:
    """Convert (north, east, down) to (east, north, up)."""

    north, east, down = _vector3(vector_ned)
    return east, north, -down


def enu_to_ned(vector_enu: Sequence[float]) -> Vector3:
    """Convert (east, north, up) to (north, east, down)."""

    east, north, up = _vector3(vector_enu)
    return north, east, -up


def frd_to_flu(vector_frd: Sequence[float]) -> Vector3:
    """Convert (forward, right, down) to (forward, left, up)."""

    forward, right, down = _vector3(vector_frd)
    return forward, -right, -down


def flu_to_frd(vector_flu: Sequence[float]) -> Vector3:
    """Convert (forward, left, up) to (forward, right, down)."""

    forward, left, up = _vector3(vector_flu)
    return forward, -left, -up


def wrap_to_pi(angle_rad: float) -> float:
    """Wrap an angle to [-pi, pi)."""

    return (float(angle_rad) + pi) % (2.0 * pi) - pi


def yaw_ned_to_enu(yaw_ned_rad: float) -> float:
    """Convert PX4 NED yaw to ROS ENU yaw."""

    return wrap_to_pi(pi / 2.0 - float(yaw_ned_rad))


def yaw_enu_to_ned(yaw_enu_rad: float) -> float:
    """Convert ROS ENU yaw to PX4 NED yaw."""

    return wrap_to_pi(pi / 2.0 - float(yaw_enu_rad))
