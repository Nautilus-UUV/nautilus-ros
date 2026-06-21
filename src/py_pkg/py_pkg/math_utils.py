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


def gravity_to_roll_pitch(ax: float, ay: float, az: float) -> tuple:
    """Infer (roll, pitch) in radians from a body-frame accelerometer reading.

    At low dynamics the accelerometer measures the gravity reaction (specific
    force), which in a level NED body frame (x forward, y right, z DOWN) points
    "up" -- i.e. along ``-z``: ``(0, 0, -g)``. Tilting the body rotates that
    vector, and the tilt falls straight out of its components -- no integration,
    no filter state. Yaw is unobservable from gravity alone (rotating about the
    vertical doesn't move the vector), so only roll and pitch are recovered.

    Same ZYX Tait-Bryan convention as ``rpy_to_quaternion`` /
    ``quaternion_to_roll_pitch``: for a body at (roll, pitch, any yaw) the NED
    specific force is ``(g·sin p, -g·cos p·sin r, -g·cos p·cos r)``, which
    inverts to ``roll = atan2(-ay, -az)`` and ``pitch = atan2(ax, hypot(ay, az))``.
    (NED specific force is just the negation of the FLU one, so this is the FLU
    inverse applied to ``-a``; roll comes out identical, pitch flips sign.) The
    ratios make the result independent of the vector's magnitude, so no
    normalization (or knowledge of g) is needed.
    """
    roll = float(np.arctan2(-ay, -az))
    pitch = float(np.arctan2(ax, np.hypot(ay, az)))
    return roll, pitch


def clamp(val: float, lo: float, hi: float) -> float:
    """Saturate ``val`` into ``[lo, hi]``."""
    if val > hi:
        return hi
    if val < lo:
        return lo
    return val


def deadband_snap(val: float, zero_below: float, snap_to: float, limit: float) -> float:
    """Shape a signed command through a deadband, then saturate.

    Magnitudes under ``zero_below`` are suppressed to 0; magnitudes in
    ``[zero_below, snap_to)`` are pushed up to ``±snap_to`` (the minimum
    value the actuator runs at reliably); everything else passes through,
    saturated into ``[-limit, limit]``. Sign is preserved. Invariant
    (not enforced): ``0 <= zero_below <= snap_to <= limit``.

    With ``zero_below == snap_to == 0`` this reduces to a plain
    ``clamp(val, -limit, limit)``.
    """
    mag = abs(val)
    if mag < zero_below:
        return 0.0
    if mag < snap_to:
        return snap_to if val > 0 else -snap_to
    return clamp(val, -limit, limit)


def span_band_guards(
    lo_endpoint: float, hi_endpoint: float, band: float
) -> tuple[float, float]:
    """Inset both ends of a range by ``band`` fraction of its span.

    Returns ``(low_guard, high_guard) = (lo + band*span, hi - band*span)``,
    where ``span = hi_endpoint - lo_endpoint``. Used to stop actuating once a
    reading is within ``band`` of an endpoint -- the last sliver of travel does
    no useful work (e.g. dead-heading the BCU pump against a full/empty tank).
    With ``band == 0`` the guards collapse onto the endpoints themselves.
    """
    span = hi_endpoint - lo_endpoint
    return lo_endpoint + band * span, hi_endpoint - band * span
