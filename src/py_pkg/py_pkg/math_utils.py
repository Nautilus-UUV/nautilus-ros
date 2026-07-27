"""Math helpers used by the control stack.

Two families: quaternion <-> Euler conversions (ZYX Tait-Bryan) and a
saturating clamp. Everything else has been retired with the divetest
heritage code that originally consumed it.
"""

import math


def quaternion_to_roll_pitch(qx: float, qy: float, qz: float, qw: float) -> tuple:
    """Extract roll and pitch (radians) from a quaternion (x, y, z, w).

    ZYX Tait-Bryan convention.
    """
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    return roll, pitch


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw (rotation around Z, radians) from a quaternion (x, y, z, w).

    ZYX Tait-Bryan convention; pairs with ``quaternion_to_roll_pitch``.
    """
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple:
    """Convert (roll, pitch, yaw) in radians to a quaternion (x, y, z, w).

    ZYX Tait-Bryan convention; round-trips through
    ``quaternion_to_roll_pitch`` + ``quaternion_to_yaw``.
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


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
    roll = math.atan2(-ay, -az)
    pitch = math.atan2(ax, math.hypot(ay, az))
    return roll, pitch


def clamp(val: float, lo: float, hi: float) -> float:
    """Saturate ``val`` into ``[lo, hi]``."""
    if val > hi:
        return hi
    if val < lo:
        return lo
    return val


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


def tank_limits_valid(empty_pa, full_pa) -> bool:
    """True when both tank endpoints are registered and usable.

    "Usable" means present (not None) and ``0 < empty < full`` -- a
    partially-decoded DiveInit reads 0.0, which must not count as registered.
    Shared by the BCU clamp and the lifeguard so the safety gate cannot drift.
    """
    if empty_pa is None or full_pa is None:
        return False
    return 0.0 < empty_pa < full_pa


def at_span_endpoint(
    value: float | None,
    lo_endpoint: float | None,
    hi_endpoint: float | None,
    band: float,
    *,
    toward_low: bool,
    toward_high: bool,
) -> bool:
    """True when ``value`` has reached the ``band`` guard on a side being
    pushed toward -- the point past which more actuation does no useful work.

    ``toward_low`` / ``toward_high`` say which endpoint the current command
    heads for; a side that isn't being pushed toward never reports reached, so
    flow *away* from a touched limit always passes. With unregistered or
    inverted endpoints (see ``tank_limits_valid``) or a missing reading there is
    no guard to sit on, so the answer is False.

    One home for the tank-endpoint predicate: ``TankLimitGuard`` (both its stop
    and release checks) and the lifeguard's blow stand-down share it, so the two
    safety gates cannot drift apart in their arithmetic.
    """
    if value is None or not tank_limits_valid(lo_endpoint, hi_endpoint):
        return False
    low_guard, high_guard = span_band_guards(lo_endpoint, hi_endpoint, band)
    return (toward_low and value <= low_guard) or (toward_high and value >= high_guard)
