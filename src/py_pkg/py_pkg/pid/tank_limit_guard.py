"""Latching tank-endpoint cutoff for the BCU command.

Sits between ``solve_bcu_command`` and the ``BcuCommandGate`` in
``bcu_node.control_loop``. It stops the pump and shuts the valves once the oil
tank has been driven within a small band of an endpoint -- the last sliver of
travel just dead-heads the pump against an effectively full/empty tank.

The naive version of this was a bare per-tick threshold
(``clamp_to_tank_limits`` below). A bench test exposed its failure mode: with
the vehicle held on the bench it can't dive, so the depth error never shrinks
and the controller demands "fill the tank" forever. The loop drove the tank
straight to the guard and parked exactly on it -- and sitting on that switching
threshold, ordinary tank-pressure sensor noise crossed it back and forth every
tick, so the cutoff flip-flopped between stop and go and the motor valve
chattered.

``TankLimitGuard`` cures that the same way ``BcuCommandGate`` cures the
near-setpoint dither: hysteresis plus a latch. Once a fill (or drain) command
reaches the ``stop_band`` guard, the guard latches that endpoint and holds the
command at zero. It only lets go again when one of:

  - the tank retreats past a *wider* ``release_band`` guard -- the hysteresis
    gap, sized wider than the sensor noise so merely parking at the stop guard
    can never release it, or
  - the command reverses to pull oil the other way (flow away from a touched
    limit is always allowed straight through), or
  - ``reset`` -- a fresh dive registration or a mission stop.

State lives only in the latch, so it is trivially Tier-1 testable against a
synthetic series. With unregistered or invalid limits the guard is inert: it
drops the latch and passes the command through untouched, exactly like the
clamp it wraps.
"""

from py_pkg.math_utils import span_band_guards


def _at_tank_endpoint(
    draining: bool,
    filling: bool,
    tank_pa: float | None,
    tank_empty_pa: float | None,
    tank_full_pa: float | None,
    band: float,
) -> bool:
    """True when the tank has reached the ``band`` guard on the side a command
    is pushing toward -- the point past which more flow just dead-heads the pump.

    ``draining`` heads toward ``tank_empty_pa`` (positive bus RPM); ``filling``
    toward ``tank_full_pa`` (negative RPM or the passive vent). With
    unregistered or inverted limits there is no guard to sit on, so it is False.
    The limits come from the pre-dive Initialize (DIVE_INIT). Both the bare
    clamp and the latching guard share this one threshold.
    """
    if tank_pa is None or tank_empty_pa is None or tank_full_pa is None:
        return False
    if tank_empty_pa <= 0.0 or tank_full_pa <= 0.0 or tank_empty_pa >= tank_full_pa:
        return False
    low_guard, high_guard = span_band_guards(tank_empty_pa, tank_full_pa, band)
    return (draining and tank_pa <= low_guard) or (filling and tank_pa >= high_guard)


def clamp_to_tank_limits(
    pump_rpm: int,
    motor_open: int,
    free_open: int,
    tank_pa: float | None,
    tank_empty_pa: float | None,
    tank_full_pa: float | None,
    band: float = 0.10,
) -> tuple[int, int, int]:
    """Stop commanding oil flow once the tank is within ``band`` of an endpoint.

    The tank runs inverse to the bladder: positive bus RPM inflates the bladder
    and drains the tank toward ``tank_empty_pa``; negative RPM fills it toward
    ``tank_full_pa``. The last stretch of travel just dead-heads the pump
    against an effectively full/empty tank, so we quit early.
    """
    draining = pump_rpm > 0
    filling = pump_rpm < 0 or bool(free_open)
    if _at_tank_endpoint(draining, filling, tank_pa, tank_empty_pa, tank_full_pa, band):
        return 0, 0, 0
    return pump_rpm, motor_open, free_open


class TankLimitGuard:
    """Latch + release hysteresis over the bare per-tick tank-endpoint cutoff.

    ``stop_band`` is the inner (narrower) guard the latch engages on;
    ``release_band`` is the wider guard the tank must retreat past before the
    latch lets go. ``release_band >= stop_band`` puts the release guard at or
    inside the stop guard, so the two never coincide unless explicitly set
    equal (no hysteresis).
    """

    def __init__(self, stop_band: float = 0.10, release_band: float = 0.12) -> None:
        self.stop_band = float(stop_band)
        self.release_band = float(release_band)
        self.reset()

    def reset(self) -> None:
        """Drop any latched endpoint -- back to fully transparent."""
        self._latched: str | None = None

    def apply(
        self,
        pump_rpm: int,
        motor_open: int,
        free_open: int,
        tank_pa: float | None,
        tank_empty_pa: float | None,
        tank_full_pa: float | None,
    ) -> tuple[int, int, int]:
        """Condition one raw BCU command through the latched tank cutoff.

        Inputs are the ``solve_bcu_command`` output plus the live tank pressure
        and the registered endpoints. Returns ``(pump_rpm, motor_open,
        free_open)`` -- zeroed while latched against an endpoint the command is
        still pushing toward, otherwise passed through.
        """
        # Unregistered / invalid limits -> the clamp is inert anyway, so the
        # latch is meaningless. Drop it and pass the command straight through.
        if (
            tank_pa is None
            or tank_empty_pa is None
            or tank_full_pa is None
            or tank_empty_pa <= 0.0
            or tank_full_pa <= 0.0
            or tank_empty_pa >= tank_full_pa
        ):
            self._latched = None
            return pump_rpm, motor_open, free_open

        # Which endpoint this command heads for, same convention as the clamp:
        # positive RPM drains toward empty; negative RPM or the passive vent
        # fills toward full.
        draining = pump_rpm > 0
        filling = pump_rpm < 0 or bool(free_open)

        # Release first, so a reversal or a retreat frees the latch on the very
        # tick it happens. The stop holds until the tank backs off past the
        # wider release guard, or the command stops pushing inward.
        if self._latched is not None:
            held_at_release = _at_tank_endpoint(
                draining, filling, tank_pa, tank_empty_pa, tank_full_pa, self.release_band
            )
            still_pushing_in = filling if self._latched == "full" else draining
            if not still_pushing_in or not held_at_release:
                self._latched = None

        # Engage: an unlatched command actively pushing into an endpoint latches
        # that side the moment it reaches the (narrower) stop guard.
        if self._latched is None and (draining or filling):
            if _at_tank_endpoint(
                draining, filling, tank_pa, tank_empty_pa, tank_full_pa, self.stop_band
            ):
                self._latched = "full" if filling else "empty"

        # Hold the command at zero while latched and still pushing inward;
        # anything else (idle, or flow away from the latched end) passes.
        if self._latched == "full" and filling:
            return 0, 0, 0
        if self._latched == "empty" and draining:
            return 0, 0, 0
        return pump_rpm, motor_open, free_open
