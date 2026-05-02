"""
Sensor unit conversions for UUV topics.

These helpers turn raw values from the registry's sensor topics
(e.g. ``UUVTopics.EXTERNAL_PRESSURE``) into physically meaningful
units used by control code.

The depth control stack operates natively in **gauge pressure (Pa)** —
``EXTERNAL_PRESSURE`` is published as absolute Pa, and a single
``gauge_pressure_pa`` step at ingress is the only conversion needed.
``pressure_to_depth`` / ``depth_to_pressure_pa`` exist for logging,
test scaffolding, and non-control scripts that report in metres.

This module holds environmental constants and the pure-math
conversion functions that use them. Robot-specific physical
constants (pump, bladder, motors) live in `robot_specs.py`.
"""

from py_pkg.robot_specs import PUMP_EFFICIENCY, VOLUME_PER_REV_M3

# Standard atmosphere (Pa) — pressure at the water surface, subtracted
# off the absolute reading from the external pressure sensor.
ATMOSPHERIC_PRESSURE_PA = 101_325.0

# Fresh-water density (kg/m^3). Override for salt water if needed.
WATER_DENSITY_KG_M3 = 1_025.0

# Gravitational acceleration (m/s^2).
GRAVITY_M_S2 = 9.806

# Hydrostatic pressure rise per metre of submersion (Pa/m). The control
# stack uses gauge pressure as its primary state; this constant is the
# only place depth-in-metres ↔ pressure-in-Pa conversions get scaled.
WATER_PRESSURE_GRADIENT_PA_PER_M = WATER_DENSITY_KG_M3 * GRAVITY_M_S2

SECONDS_PER_MINUTE = 60


def gauge_pressure_pa(
    absolute_pa: float,
    atmospheric_pa: float = ATMOSPHERIC_PRESSURE_PA,
) -> float:
    """Return absolute pressure (Pa) minus atmospheric (Pa).

    Gauge pressure is the depth controller's native state: 0 at the
    surface, positive when submerged. Sign matches the Z-positive-down
    convention used by the rest of the control stack.
    """
    return absolute_pa - atmospheric_pa


def depth_to_pressure_pa(
    depth_m: float,
    density: float = WATER_DENSITY_KG_M3,
    atmospheric_pa: float = ATMOSPHERIC_PRESSURE_PA,
) -> float:
    """Inverse of :func:`pressure_to_depth` — absolute Pa for a given depth.

    Used by tests and logging that reason in metres. Control code
    should stay in gauge Pa and avoid the round trip.
    """
    return atmospheric_pa + depth_m * density * GRAVITY_M_S2


def pressure_to_depth(
    pressure_pa: float,
    density: float = WATER_DENSITY_KG_M3,
    atmospheric_pa: float = ATMOSPHERIC_PRESSURE_PA,
) -> float:
    """
    Convert absolute fluid pressure (Pa) to depth (metres, positive downward).

    depth = (P_abs - P_atm) / (rho * g)

    Kept for log/UI display and for non-control integration scripts
    that report depth in metres. The depth controller uses
    :func:`gauge_pressure_pa` instead.
    """
    return (pressure_pa - atmospheric_pa) / (density * GRAVITY_M_S2)


def q_to_rpm(q: float, bladder_volume: float) -> float:
    """
    Convert bladder flow-rate ratio (1/s) to motor RPM.

    :param q: bladder flow rate as a fraction of total volume per second (1/s)
    :param bladder_volume: bladder volume (m^3)
    :return: motor speed (RPM)
    """
    flow_rate = q * bladder_volume  # m^3/s
    return SECONDS_PER_MINUTE / (VOLUME_PER_REV_M3 * PUMP_EFFICIENCY) * flow_rate
