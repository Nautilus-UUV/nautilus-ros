"""Latching tank-endpoint cutoff for the BCU command.

Sits between ``solve_bcu_command`` and the wire in ``bcu_node.control_loop``. It
stops the pump and shuts the valves once the oil tank has been driven within
``stop_band`` of an endpoint -- the last sliver of travel just dead-heads the
pump against an effectively full/empty tank. This is the stop condition the
bang-bang missions run into on every leg: pump one way until the tank rails,
then coast until the mission's next leg reverses the command.

The naive version of this was a bare per-tick threshold (the since-removed
``clamp_to_tank_limits``). A bench test exposed its failure mode: with the
vehicle held on the bench it can't dive, so the depth error never shrinks and
the controller demands "fill the tank" forever. The loop drove the tank straight
to the guard and parked exactly on it -- and sitting on that switching threshold,
ordinary tank-pressure sensor noise crossed it back and forth every tick, so the
cutoff flip-flopped between stop and go and the motor valve chattered.

The latch cures that. Once a fill (or drain) command reaches the guard, the
guard latches that endpoint and holds the command at zero. Noise on ``tank_pa``
cannot release it, because the only two ways out are:

  - the command reverses to pull oil the other way (flow away from a touched
    limit is always allowed straight through), or
  - ``reset`` -- a fresh dive registration or a mission stop.

There is deliberately no retreat-based release: while latched the pump is
stopped and both valves are shut, so the tank cannot move on its own, and a
reversal is exactly what a mission leg change produces.

State lives only in the latch, so it is trivially Tier-1 testable against a
synthetic series. With unregistered or invalid limits the guard is inert: it
drops the latch and passes the command through untouched.
"""

from py_pkg.math_utils import at_span_endpoint, tank_limits_valid


class TankLimitGuard:
    """Latching cutoff over the bare per-tick tank-endpoint threshold.

    ``stop_band`` is a fraction of the empty->full span; the default lives on
    ``DepthSpec.tank_stop_band`` so there is one source of truth for the band a
    scenario can retune.
    """

    def __init__(self, stop_band: float) -> None:
        self.stop_band = float(stop_band)
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
        # Unregistered / invalid limits -> the cutoff is inert anyway, so the
        # latch is meaningless. Drop it and pass the command straight through.
        if tank_pa is None or not tank_limits_valid(tank_empty_pa, tank_full_pa):
            self._latched = None
            return pump_rpm, motor_open, free_open

        # Which endpoint this command heads for: positive RPM drains toward
        # empty; negative RPM or the passive vent fills toward full.
        draining = pump_rpm > 0
        filling = pump_rpm < 0 or bool(free_open)

        # Release first, so a reversal frees the latch on the very tick it
        # happens; the stop otherwise holds until the command stops pushing in.
        if self._latched is not None:
            still_pushing_in = filling if self._latched == "full" else draining
            if not still_pushing_in:
                self._latched = None

        # Engage: an unlatched command actively pushing into an endpoint latches
        # that side the moment the tank reaches the guard.
        if self._latched is None and (draining or filling):
            at_guard = at_span_endpoint(
                tank_pa,
                tank_empty_pa,
                tank_full_pa,
                self.stop_band,
                toward_low=draining,
                toward_high=filling,
            )
            if at_guard:
                self._latched = "full" if filling else "empty"

        # Still latched here means "still pushing into the endpoint": the
        # release block above drops the latch the moment the command reverses,
        # and the engage block only latches the side the command pushes toward.
        # So a live latch is always the stop, and everything else -- idle, or
        # flow away from the endpoint -- passes.
        if self._latched is not None:
            return 0, 0, 0
        return pump_rpm, motor_open, free_open
