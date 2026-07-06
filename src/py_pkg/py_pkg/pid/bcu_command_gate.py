"""Anti-chatter conditioning for the BCU wire command.

Sits between the controller solve (``solve_bcu_command``) and the wire
publish in ``bcu_node.control_loop``. It exists because a bench test found
that commanding a depth ~= the current depth left the loop dithering on
sensor-quantisation noise: the pump reversed at full magnitude and the
valves toggled open/closed every control tick, which is mechanically
abusive. The gate kills that at the source with three coupled rules:

  - **Error deadband + arm/disarm hysteresis.** While the depth error is
    inside the (narrow) disarm band the pump is held idle -- 0 rpm, valves
    closed. It only re-arms once the error grows past the (wider) arm band.
    Between the two it keeps whatever state it is already in, so noise
    dither across the setpoint can't flip the pump on and off.

  - **Minimum valve dwell.** The valve bitmask changes at most once per
    ``min_valve_dwell_s``, so the solenoids never chatter no matter how
    fast the controller's intent flips. Protects the valves in every
    regime, not just near the setpoint.

  - **Pump/valve consistency.** The pump is never run against a closed
    valve (it stays 0 until the motor valve has actually opened), and a
    disarm stops oil flow *immediately* even while the dwell still holds a
    valve open -- an open valve carrying no flow is harmless, but moving
    oil after a stop is not.

Pure and clock-injected: ``apply`` takes the current time, so it is
trivially Tier-1 testable against a synthetic time series. With
``error_arm_pa == error_disarm_pa == 0`` and ``min_valve_dwell_s == 0`` it
collapses to a pass-through identical to the pre-gate behaviour, which is
the clean way to disable it for sims that want the raw command.
"""

# Valves closed (motor_open, free_open) -- the disarmed/hold state. The other
# states the BCU commands are (1, 0) motor/pump-flow and (0, 1) free/vent;
# both-open is never produced by select_pump_and_valves.
_CLOSED = (0, 0)


class BcuCommandGate:
    """Hysteresis + dwell conditioner for ``(pump_rpm, motor_open, free_open)``."""

    def __init__(
        self,
        error_arm_pa: float,
        error_disarm_pa: float,
        min_valve_dwell_s: float,
    ) -> None:
        self.error_arm_pa = float(error_arm_pa)
        self.error_disarm_pa = float(error_disarm_pa)
        self.min_valve_dwell_s = float(min_valve_dwell_s)
        self.reset()

    def reset(self) -> None:
        """Back to construction state: disarmed, valves closed, dwell clear.

        Disarmed-on-reset means that if a fresh mission starts already near
        its target the pump stays idle until the error genuinely exceeds the
        arm band, rather than kicking once before settling.
        """
        self._armed = False
        self._valves = _CLOSED
        self._last_change_s: float | None = None

    def apply(
        self,
        error_pa: float,
        pump_rpm: int,
        motor_open: int,
        free_open: int,
        now_s: float,
    ) -> tuple[int, int, int]:
        """Condition one raw BCU command into the gated wire command.

        ``error_pa`` is the depth error (target - current, gauge Pa); the
        rest is the raw ``solve_bcu_command`` output and the current time.
        Returns ``(pump_rpm, motor_open, free_open)`` ready for the wire.
        """
        self._update_arming(abs(error_pa))

        # Disarmed -> hold position: valves closed, pump idle. Armed -> take
        # the controller's intended valve state and pump command.
        if self._armed:
            desired = (1 if motor_open else 0, 1 if free_open else 0)
            desired_pump = int(pump_rpm)
        else:
            desired = _CLOSED
            desired_pump = 0

        # Dwell-limit the valve bitmask: only adopt a new state once the
        # current one has been held long enough.
        if desired != self._valves and self._dwell_elapsed(now_s):
            self._valves = desired
            self._last_change_s = now_s
        held_motor, held_free = self._valves

        # Run the pump only when the *held* valve state actually carries the
        # motor flow path. This both forbids dead-heading against a closed
        # valve and zeroes the pump the instant we disarm (desired_pump is 0
        # then, so even a still-open motor valve carries no flow).
        out_pump = desired_pump if held_motor else 0
        return out_pump, held_motor, held_free

    def _update_arming(self, error_mag: float) -> None:
        # Hysteresis: leave the armed band only past disarm, re-enter only
        # past arm (arm >= disarm), so noise between the two can't toggle it.
        if self._armed:
            if error_mag <= self.error_disarm_pa:
                self._armed = False
        elif error_mag >= self.error_arm_pa:
            self._armed = True

    def _dwell_elapsed(self, now_s: float) -> bool:
        if self.min_valve_dwell_s <= 0.0 or self._last_change_s is None:
            return True
        return (now_s - self._last_change_s) >= self.min_valve_dwell_s
