"""Pure BCU plant-transient models shared by the sim HAL bridge and tests.

Ground-truth *plant* dynamics, not controller models: the sim bridge
(`nautilus_hal.bridges.bcu_sim_bridge`) drives these with commanded RPM to
shape what the simulated hardware actually does. Parameters live on
``scenarios.spec.rig.PlantSpec`` and arrive via ROS params; fitted values
come from the 2026-06-24 lake test (provenance:
UG-anomaly_detection/analysis/lake_test_jun24/pump_transient_fit.json).

Kept free of rclpy / nautilus_hal imports so Tier 1 tests exercise the
dynamics directly (same split as `sensor_noise.py`).
"""

import functools
from collections import deque
from typing import Callable

from py_pkg.math_utils import clamp


class PumpDynamics:
    """Commanded RPM -> effective shaft RPM: dead time + slew-rate limit.

    The lake feedback traces show the pump holds still for a dead time
    after a command, then ramps at a constant rate to the target — no
    exponential tail — so the model is a pure transport delay feeding a
    slew limiter, not a first-order lag.

    ``delay_s <= 0`` disables the dead time; ``slew_rpm_per_s <= 0``
    disables the ramp. Both disabled is exact passthrough (the
    pre-calibration bridge behavior).
    """

    def __init__(self, delay_s: float, slew_rpm_per_s: float):
        self.delay_s = float(delay_s)
        self.slew_rpm_per_s = float(slew_rpm_per_s)
        # (t_s, commanded_rpm) samples awaiting the dead time. Before any
        # sample is old enough the delayed target is 0.0 (pump at rest).
        self._history: deque[tuple[float, float]] = deque()
        self._eff_rpm = 0.0

    def step(self, t_s: float, commanded_rpm: float, dt_s: float) -> float:
        """Advance to time ``t_s`` with the current command; return eff RPM.

        ``dt_s`` is the caller's elapsed time since its previous step
        (the slew budget for this tick). Negative ``dt_s`` is treated
        as zero.
        """
        commanded_rpm = float(commanded_rpm)

        if self.delay_s <= 0.0:
            target = commanded_rpm
        else:
            self._history.append((t_s, commanded_rpm))
            cutoff = t_s - self.delay_s
            # Newest sample that has aged past the dead time becomes the
            # target; drop everything older than it.
            while len(self._history) >= 2 and self._history[1][0] <= cutoff:
                self._history.popleft()
            if self._history[0][0] <= cutoff:
                target = self._history[0][1]
            else:
                target = 0.0

        if self.slew_rpm_per_s <= 0.0:
            self._eff_rpm = target
        else:
            budget = self.slew_rpm_per_s * max(dt_s, 0.0)
            error = target - self._eff_rpm
            self._eff_rpm += clamp(error, -budget, budget)
        return self._eff_rpm


def tank_pressure_linear(
    volume_m3: float,
    bladder_min_m3: float,
    bladder_max_m3: float,
    tank_pressure_empty_pa: float,
    tank_pressure_full_pa: float,
) -> float:
    """Today's tank sensor map: linear in the tank's oil level.

    Oil in the bladder is oil OUT of the tank, so tank pressure runs
    inverse to bladder fill: full bladder -> drained tank -> empty
    endpoint; empty bladder -> full tank -> full endpoint.
    """
    span = bladder_max_m3 - bladder_min_m3
    frac = (bladder_max_m3 - volume_m3) / span if span > 0 else 0.0
    frac = clamp(frac, 0.0, 1.0)
    return tank_pressure_empty_pa + frac * (
        tank_pressure_full_pa - tank_pressure_empty_pa
    )


def tank_pressure_gaslaw(
    volume_m3: float,
    bladder_min_m3: float,
    bladder_max_m3: float,
    tank_pressure_empty_pa: float,
    tank_pressure_full_pa: float,
    air_volume_m3: float | None = None,
) -> float:
    """Isothermal air-cushion tank map: hyperbolic in the tank's oil level.

    The tank sensor reads the pressure of a trapped air cushion above the
    oil; pushing oil back into the tank compresses it (Boyle's law), so
    the curve is flat near tank-empty and steep near tank-full — the
    shape the lake traces show and the linear map cannot.

    ``air_volume_m3`` is the cushion volume at the tank-EMPTY endpoint
    (bladder full). ``None`` pins the cushion so the curve passes through
    BOTH calibrated endpoints exactly:

        V_A = span * p_full / (p_full - p_empty)
        p(V) = p_empty * V_A / (V_A - oil_in_tank)

    with ``oil_in_tank = bladder_max - V`` and ``span = bladder_max -
    bladder_min``. A finite ``air_volume_m3`` frees the curvature (one
    fitted dof); it must exceed ``span`` or the cushion would vanish
    inside the operating range.
    """
    span = bladder_max_m3 - bladder_min_m3
    if span <= 0:
        return tank_pressure_empty_pa
    if air_volume_m3 is None:
        if tank_pressure_full_pa <= tank_pressure_empty_pa:
            return tank_pressure_empty_pa
        air_volume_m3 = (
            span
            * tank_pressure_full_pa
            / (tank_pressure_full_pa - tank_pressure_empty_pa)
        )
    elif air_volume_m3 <= span:
        raise ValueError(
            f"air_volume_m3 ({air_volume_m3}) must exceed the bladder span "
            f"({span}) or the cushion vanishes inside the operating range"
        )
    volume_m3 = clamp(volume_m3, bladder_min_m3, bladder_max_m3)
    oil_in_tank = bladder_max_m3 - volume_m3
    return tank_pressure_empty_pa * air_volume_m3 / (air_volume_m3 - oil_in_tank)


def make_tank_pressure_map(
    shape: str,
    bladder_min_m3: float,
    bladder_max_m3: float,
    tank_pressure_empty_pa: float,
    tank_pressure_full_pa: float,
    air_volume_m3: float | None = None,
) -> Callable[[float], float]:
    """Bind a tank-pressure map once at startup: ``volume_m3 -> Pa``.

    The per-tick consumer (the BCU sim bridge's telemetry timer) then
    calls the returned function with just the bladder volume — no shape
    dispatch or parameter re-validation per tick. ``air_volume_m3``
    ``None`` or ``<= 0`` means the pinned cushion (ROS params have no
    null, so the wire encodes None as 0.0). An unknown ``shape`` or an
    invalid free cushion raises here, at bind time, not on the first
    telemetry tick.
    """
    if air_volume_m3 is not None and air_volume_m3 <= 0.0:
        air_volume_m3 = None
    if shape == "linear":
        return functools.partial(
            tank_pressure_linear,
            bladder_min_m3=bladder_min_m3,
            bladder_max_m3=bladder_max_m3,
            tank_pressure_empty_pa=tank_pressure_empty_pa,
            tank_pressure_full_pa=tank_pressure_full_pa,
        )
    if shape == "gaslaw":
        fn = functools.partial(
            tank_pressure_gaslaw,
            bladder_min_m3=bladder_min_m3,
            bladder_max_m3=bladder_max_m3,
            tank_pressure_empty_pa=tank_pressure_empty_pa,
            tank_pressure_full_pa=tank_pressure_full_pa,
            air_volume_m3=air_volume_m3,
        )
        # Probe call: surfaces a too-small free cushion (the guard in
        # tank_pressure_gaslaw) at startup instead of on the first tick.
        fn(bladder_min_m3)
        return fn
    raise ValueError(f"unknown tank_map_shape {shape!r}; expected linear or gaslaw")
