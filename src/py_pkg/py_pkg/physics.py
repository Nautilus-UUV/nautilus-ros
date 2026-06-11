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

from py_pkg.robot_specs import VOLUME_PER_REV_M3

# Standard atmosphere (Pa) — the FALLBACK gauge reference. The operator
# can register the actual surface pressure pre-dive (DIVE_INIT, see
# `SurfaceReference`); until that happens, conversions subtract this
# constant off the absolute reading from the external pressure sensor.
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

    Kept for log/UI display where depth is reported in metres. The
    depth controller uses :func:`gauge_pressure_pa` instead.
    """
    return (pressure_pa - atmospheric_pa) / (density * GRAVITY_M_S2)


class SurfaceReference:
    """The gauge reference a pressure-consuming node converts against.

    Weather and altitude move the real surface pressure a few kPa away
    from the standard atmosphere — enough to matter when "surfaced" is
    defined as half a metre of water. The operator registers the actual
    surface reading pre-dive (UI Initialize → DIVE_INIT); every node
    holds one of these and converts through :meth:`gauge`, so all
    consumers shift to the registered reference on the same latched
    message and stay in a single frame. Until a registration arrives
    (or if it's garbage), the standard atmosphere applies.
    """

    def __init__(self) -> None:
        self._surface_pa: float | None = None

    def register(self, surface_pa: float) -> bool:
        """Adopt a registered surface pressure; returns True if accepted.

        Non-positive values are rejected and leave the current reference
        untouched: a partially-filled DiveInit decodes missing fields as
        0.0, and silently adopting that would shift the whole gauge
        frame by ~101 kPa.
        """
        if not surface_pa > 0.0:
            return False
        self._surface_pa = float(surface_pa)
        return True

    def register_logged(self, surface_pa: float, logger) -> bool:
        """:meth:`register` plus the standard accept/reject log lines.

        Every DIVE_INIT consumer wants the same outcome logging; keeping
        the wording here means a policy or message change lands in one
        place. ``logger`` is any object with ``info``/``error`` (a node
        logger) so this module stays ROS-free.
        """
        if self.register(surface_pa):
            logger.info(f"dive init: gauge reference = {self.reference_pa:.0f} Pa")
            return True
        logger.error(
            f"dive init: surface pressure {surface_pa:.0f} Pa "
            "rejected -- keeping previous reference"
        )
        return False

    @property
    def reference_pa(self) -> float:
        """Current reference: registered surface, else standard atmosphere."""
        if self._surface_pa is not None:
            return self._surface_pa
        return ATMOSPHERIC_PRESSURE_PA

    def gauge(self, absolute_pa: float) -> float:
        """Absolute Pa → gauge Pa against the current reference."""
        return gauge_pressure_pa(absolute_pa, atmospheric_pa=self.reference_pa)


def q_to_rpm(q: float, bladder_volume: float, pump_efficiency: float) -> float:
    """
    Convert bladder flow-rate ratio (1/s) to motor RPM.

    :param q: bladder flow rate as a fraction of total volume per second (1/s)
    :param bladder_volume: bladder volume (m^3)
    :param pump_efficiency: volumetric efficiency the controller assumes
        when inverting flow → RPM. Lives on DepthPlantModel so MC sweeps
        can perturb controller-vs-actual pump efficiency.
    :return: motor speed (RPM)
    """
    flow_rate = q * bladder_volume  # m^3/s
    return SECONDS_PER_MINUTE / (VOLUME_PER_REV_M3 * pump_efficiency) * flow_rate
