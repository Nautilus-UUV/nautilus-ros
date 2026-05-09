"""Math helpers used by the control stack.

Two families: quaternion <-> Euler conversions (ZYX Tait-Bryan) and a
saturating clamp. Everything else has been retired with the divetest
heritage code that originally consumed it.
"""

import numpy as np


def quaternion_to_roll_pitch(qx: float, qy: float, qz: float, qw: float) -> tuple:
    """Extract roll and pitch (radians) from a quaternion (x, y, z, w).

    ZYX Tait-Bryan convention.
    """
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    return roll, pitch


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw (rotation around Z, radians) from a quaternion (x, y, z, w).

    ZYX Tait-Bryan convention; pairs with ``quaternion_to_roll_pitch``.
    """
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple:
    """Convert (roll, pitch, yaw) in radians to a quaternion (x, y, z, w).

    ZYX Tait-Bryan convention; round-trips through
    ``quaternion_to_roll_pitch`` + ``quaternion_to_yaw``.
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return float(qx), float(qy), float(qz), float(qw)


def clamp(val: float, lo: float, hi: float) -> float:
    """Saturate ``val`` into ``[lo, hi]``."""
    if val > hi:
        return hi
    if val < lo:
        return lo
    return val
