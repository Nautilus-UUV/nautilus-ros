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
import logging
from collections import deque
from typing import Callable

from py_pkg.math_utils import clamp
from py_pkg.robot_specs import BCU_MOTOR_VALVE_MASK

_LOG = logging.getLogger(__name__)


class PumpDynamics:
    """Commanded RPM -> effective shaft RPM: dead time + slew-rate limit.

    The lake feedback traces show the pump holds still for a dead time
    after a command, then ramps at a constant rate to the target — no
    exponential tail — so the model is a pure transport delay feeding a
    slew limiter, not a first-order lag.

    ``delay_s <= 0`` disables the dead time; ``slew_rpm_per_s <= 0``
    disables the ramp. Both disabled is exact passthrough (the
    pre-calibration bridge behavior).

    ``overshoot_frac > 0`` adds the EPOS4 velocity-loop crest the lake
    feedback shows on hardware (shaft peaks at 3113 on a 3000 command,
    frac ~= 0.038): on a new nonzero delayed target the ramp aims past
    it by ``overshoot_frac`` of the step, crests, then settles back to
    the target at the same slew rate. Spin-down to 0 never overshoots.
    ``overshoot_frac <= 0`` leaves every code path bit-identical to the
    plain delay+slew model.
    """

    def __init__(
        self,
        delay_s: float,
        slew_rpm_per_s: float,
        overshoot_frac: float = 0.0,
    ):
        self.delay_s = float(delay_s)
        self.slew_rpm_per_s = float(slew_rpm_per_s)
        self.overshoot_frac = float(overshoot_frac)
        # (t_s, commanded_rpm) samples awaiting the dead time. Before any
        # sample is old enough the delayed target is 0.0 (pump at rest).
        self._history: deque[tuple[float, float]] = deque()
        self._eff_rpm = 0.0
        # Overshoot state: the last delayed target we aimed for and, while
        # an excursion is in flight, the crest the ramp is heading to.
        self._prev_target = 0.0
        self._peak: float | None = None

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

        if self.overshoot_frac > 0.0:
            if target != self._prev_target:
                # New setpoint: aim past a nonzero target by a fraction of
                # the step; a stop command ramps straight to 0.
                self._peak = (
                    target + self.overshoot_frac * (target - self._eff_rpm)
                    if target != 0.0
                    else None
                )
                self._prev_target = target
            slew_to = self._peak if self._peak is not None else target
        else:
            slew_to = target

        if self.slew_rpm_per_s <= 0.0:
            self._eff_rpm = slew_to
        else:
            budget = self.slew_rpm_per_s * max(dt_s, 0.0)
            error = slew_to - self._eff_rpm
            self._eff_rpm += clamp(error, -budget, budget)
        if self._peak is not None and abs(self._peak - self._eff_rpm) < 1e-9:
            self._peak = None  # crest reached; settle back toward the target
        return self._eff_rpm


def pump_flow_active(rpm: float, valves: int) -> bool:
    """True iff the pump moves oil: shaft turning AND the motor way open.

    The single hydraulic gate shared by the BCU sim bridge's flow
    integral and the anomaly label bridge's ``active`` flag — a closed
    motor valve deadheads the pump (shaft spins, no flow), so both
    consumers must agree on what "actuating" means. Callers pick the
    rpm flavour (effective shaft rpm for the plant, commanded rpm for
    the label's approximation).
    """
    return rpm != 0 and bool(valves & BCU_MOTOR_VALVE_MASK)


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

    The return value is clamped to ``[empty, full]``: the calibrated
    endpoints are the physical range of the real tank (~97.8–195.5 kPa
    on hardware), and a free cushion smaller than the pinned one would
    otherwise run the hyperbola arbitrarily far past ``full`` at the
    ``bladder_min`` rail.
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
    p = tank_pressure_empty_pa * air_volume_m3 / (air_volume_m3 - oil_in_tank)
    if tank_pressure_full_pa > tank_pressure_empty_pa:
        p = clamp(p, tank_pressure_empty_pa, tank_pressure_full_pa)
    return p


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
        # A free cushion below the pinned one would drive the hyperbola
        # past the full endpoint at the bladder_min rail; the map clamps
        # that, but the config is inconsistent — say so once at bind time.
        span = bladder_max_m3 - bladder_min_m3
        if (
            air_volume_m3 is not None
            and span > 0
            and tank_pressure_full_pa > tank_pressure_empty_pa
        ):
            unclamped = (
                tank_pressure_empty_pa * air_volume_m3 / (air_volume_m3 - span)
            )
            if unclamped > tank_pressure_full_pa:
                _LOG.warning(
                    "gaslaw tank map: air_volume_m3=%.4e reaches %.0f Pa at "
                    "bladder_min, past tank_pressure_full_pa=%.0f — output "
                    "clamped to the calibrated interval (use 0.0 to pin the "
                    "cushion through both endpoints)",
                    air_volume_m3,
                    unclamped,
                    tank_pressure_full_pa,
                )
        return fn
    raise ValueError(f"unknown tank_map_shape {shape!r}; expected linear or gaslaw")
