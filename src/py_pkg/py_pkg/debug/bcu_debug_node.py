"""Manual BCU driver -- bypasses pathfinding + the depth PID.

Drives the BCU actuator topics (``/bcu/rpm`` and ``/bcu/valves``) by hand so
an operator can poke the pump and valves. Contention with depth_node is avoided by the
mission being stopped first -- the UI publishes ``/command``=false when the
operator engages a manual command, which makes depth_node emit one safe-stop
and go silent -- so this node is free to own the BCU. While a command is held
we heartbeat it; when idle we go silent (after a short trailing-zero flush) so
depth_node can reclaim the wire on the next mission.

``DEBUG_RESET`` is the operator's red all-stop: cancel any pump session, close
valves, cancel an in-progress emergency surface, command 0 RPM + closed valves
once, then go silent.

Command surfaces:

* Emergency surface (``DEBUG_EMERGENCY_SURFACE``) -- the safety path. Blow
  ballast: pump at ``EMERGENCY_SURFACE_RPM`` with valve 2 (the motor way)
  open, continuously. Deliberately dumb -- no surfaced check, no timeout, no
  pressure feedback of any kind -- so it cannot be argued out of surfacing by
  a bad sensor. Only an explicit ``False`` (or ``DEBUG_RESET``) stands it
  down. Both the operator's slider and the lifeguard failsafe land here.

* Pump for X seconds (``DEBUG_BCU_RPM``) -- run the motor at a requested RPM
  for a bounded window, then stop. Pumping needs valve 2 (the motor way) open
  to carry the flow; if it isn't, we warn (but still pump -- this is a debug
  knob, and the operator UI carries the same warning).

* Pump until tank pressure target (``DEBUG_BCU_RPM_UNTIL_PRESSURE``) --
  closed-loop sibling of the timed pump: run the motor at a requested RPM
  until ``/bcu/pressure`` crosses ``target_pressure_pa`` in the direction
  the sign of the RPM implies (positive inflates -> tank drops -> stop
  when tank_pa <= target; negative deflates -> tank rises -> stop when
  tank_pa >= target). The target is in the tank sensor's own frame (tank
  relative to hull, the frame ``/bcu/pressure`` reports) -- read it off
  the live stream, don't hand-convert from absolute. Mutually exclusive
  with the timed pump on this node.

None of the pump commands on THIS node consult the operator's registered
tank limits (DIVE_INIT) -- a timed or pressure-targeted pump can dead-head
at a tank endpoint until the operator stops it or ``MAX_PUMP_S`` fires.
Deliberate: debug tools execute what the operator says. The tank smarts
live elsewhere: the depth PID clamps its own output, and the MQTT bridge
stands down an emergency blow (lifeguard- or slider-triggered alike) by
publishing ``DEBUG_EMERGENCY_SURFACE``=false once the tank is within 10%
of the registered empty endpoint. Keeping that check out of THIS node is
what keeps the emergency surface impossible to argue out of surfacing
with a bad sensor.

* Valves (``DEBUG_BCU_VALVES``) -- latch a valve bitmask (bit0=valve2/motor,
  bit1=valve1/free).
  We only touch ``/bcu/valves`` once a valve command has actually arrived, so
  a pump-only session never slams the valves shut underneath the operator.
  Sending mask 0 closes them.

While a command is held we re-publish it at 10 Hz. The MQTT bridge's egress
side is a rate-limit-and-drop throttle, so a lone publish can lose the race; a
steady heartbeat -- plus a short trailing-zero flush when a command ends --
keeps the topic fresh on the wire (and the STM fed), so the reset-to-0 reliably
reaches the UI strip chart before we fall silent.

The timed pump duration is clamped to ``MAX_PUMP_S`` so a typo'd UI input
(3000 instead of 3) can't leave the motor running far longer than intended.
The same cap is the runaway backstop for the pressure-targeted mode.
"""

import rclpy
from nautilus_msgs.msg import BcuPumpCommand, BcuPumpUntilPressureCommand
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, Int16, Int32, UInt8

from py_pkg.robot_specs import BCU_MOTOR_VALVE_MASK
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
    spin_node,
)

MAX_PUMP_S = 300.0

# Re-publish cadence while a command is active. 10 Hz matches depth_node's
# control loop and the MQTT egress throttle, so each tick refreshes the
# bridge's per-topic clock and the next sample is always allowed through.
PUBLISH_PERIOD_S = 0.1

# Emergency-surface pump speed. Shared by the operator's slider and the
# lifeguard failsafe -- one behavior, one constant; Tier 2/3 tests assert it
# on the wire.
EMERGENCY_SURFACE_RPM = 3000

# How many ticks to keep carrying the terminal 0/closed after a command ends
# before going silent. 5 ticks @ 10 Hz = 0.5 s, comfortably above the egress
# throttle so the reset-to-0 always lands on the UI chart.
FLUSH_TICKS = 5


def _clamp_duration(duration_s: float, max_s: float = MAX_PUMP_S) -> float:
    """Clamp the user-requested pump duration to [0, max_s]."""
    if duration_s <= 0.0:
        return 0.0
    if duration_s > max_s:
        return float(max_s)
    return float(duration_s)


class BcuDebugNode(Node):
    """Operator-driven BCU pump/valve knob with an emergency-surface mode."""

    def __init__(self) -> None:
        super().__init__("bcu_debug")
        self._rpm_pub = create_publisher_for_topic(self, UUVTopics.BCU_RPM)
        self._valves_pub = create_publisher_for_topic(self, UUVTopics.BCU_VALVES)

        create_subscription_for_topic(self, UUVTopics.DEBUG_BCU_RPM, self._on_rpm_cmd)
        create_subscription_for_topic(
            self,
            UUVTopics.DEBUG_BCU_RPM_UNTIL_PRESSURE,
            self._on_rpm_until_pressure_cmd,
        )
        create_subscription_for_topic(
            self, UUVTopics.DEBUG_BCU_VALVES, self._on_valve_cmd
        )
        create_subscription_for_topic(
            self, UUVTopics.DEBUG_EMERGENCY_SURFACE, self._on_emergency_cmd
        )
        # Tank pressure is the stop signal for the pressure-targeted pump mode.
        create_subscription_for_topic(
            self, UUVTopics.BCU_PRESSURE, self._on_tank_pressure
        )
        # Red all-stop from the operator UI.
        create_subscription_for_topic(self, UUVTopics.DEBUG_RESET, self._on_reset)

        # A pump "session" is either timed (set by DEBUG_BCU_RPM) or
        # pressure-targeted (set by DEBUG_BCU_RPM_UNTIL_PRESSURE). Both share
        # _held_rpm + _pump_deadline_s; the pressure variant additionally
        # holds _target_pressure_pa. The two modes are mutually exclusive --
        # arming either replaces the other -- and _clear_pump_state() drops
        # them as a unit so the invariant survives every stop path.
        self._held_rpm: int = 0
        self._pump_deadline_s: float | None = None
        self._target_pressure_pa: int | None = None

        # Latched valve bitmask. None means we don't own /bcu/valves (depth_node
        # has it); an int means we're holding that mask.
        self._valve_mask: int | None = None

        # Emergency surface.
        self._emergency = False

        # Latest tank pressure in sensor-frame Pa; None until the first reading.
        # Drives the stop check for the pressure-targeted pump mode.
        self._tank_pa: float | None = None

        # Trailing-zero flush countdown. When a session ends (or on reset) we
        # carry the terminal 0/closed for a few ticks so the throttled MQTT
        # egress reliably lands it, then go silent.
        self._flush_ticks: int = 0

        self._timer = self.create_timer(PUBLISH_PERIOD_S, self._on_tick)

    # --- command callbacks ----------------------------------------------

    def _on_reset(self, _msg: Empty) -> None:
        # Red all-stop: cancel any pump session, drop valves, cancel an
        # in-progress emergency surface, and command 0 RPM + closed valves once.
        # Arm the trailing-zero flush so the terminal 0/closed reliably lands on
        # the throttled egress, then go silent.
        self._emergency = False
        self._clear_pump_state()
        self._valve_mask = None
        self._publish_rpm(0)
        self._publish_valves(0)
        self._arm_flush()
        self.get_logger().info("debug reset -- BCU all-stop (0 RPM, valves closed)")

    def _on_tank_pressure(self, msg: Int32) -> None:
        # BCU_PRESSURE is the tank pressure in Pa, in the sensor's own frame
        # (tank relative to hull, ~0.7-1.5 barg over the working range). We
        # just cache it; the stop check runs in _on_tick alongside the
        # deadline check so the decision is on the same heartbeat as the RPM
        # re-publish.
        self._tank_pa = float(msg.data)

    def _on_rpm_cmd(self, msg: BcuPumpCommand) -> None:
        if self._emergency:
            self.get_logger().warn("pump command ignored -- emergency surface active")
            return

        duration = _clamp_duration(float(msg.duration_s))
        if duration == 0.0:
            # Explicit stop: zero the motor, clear the window, flush, go silent.
            self._clear_pump_state()
            self._publish_rpm(0)
            self._arm_flush()
            return

        # New timed window supersedes any pressure-targeted session.
        self._clear_pump_state()
        self._held_rpm = int(msg.rpm)
        self._pump_deadline_s = self._now_s() + duration
        if self._held_rpm != 0 and not self._motor_valve_open():
            self.get_logger().warn(
                f"pumping at {self._held_rpm} rpm with valve 2 not commanded "
                "open -- open valve 2 (motor way) to carry the flow"
            )
        self._publish_rpm(self._held_rpm)

    def _on_rpm_until_pressure_cmd(self, msg: BcuPumpUntilPressureCommand) -> None:
        if self._emergency:
            self.get_logger().warn(
                "pump-until-pressure ignored -- emergency surface active"
            )
            return

        rpm = int(msg.rpm)
        target = int(msg.target_pressure_pa)

        if rpm == 0:
            # Treat as explicit stop, same as duration=0 on the timed command.
            self._clear_pump_state()
            self._publish_rpm(0)
            self._arm_flush()
            return

        if self._tank_pa is None:
            self.get_logger().warn(
                "pump-until-pressure ignored -- no tank-pressure sample yet "
                "(waiting on /bcu/pressure)"
            )
            return

        # No feasibility pre-check. Just run in the commanded direction and let
        # the per-tick stop condition end it when the tank reading crosses the
        # target. If the target sits on the wrong side (unreachable in the
        # pumped direction), it runs until the operator stops it or the
        # MAX_PUMP_S backstop fires.
        self._clear_pump_state()
        self._held_rpm = rpm
        self._target_pressure_pa = target
        self._pump_deadline_s = self._now_s() + MAX_PUMP_S
        if not self._motor_valve_open():
            self.get_logger().warn(
                f"pumping at {self._held_rpm} rpm with valve 2 not commanded "
                "open -- open valve 2 (motor way) to carry the flow"
            )
        self._publish_rpm(self._held_rpm)

    def _on_valve_cmd(self, msg: UInt8) -> None:
        if self._emergency:
            self.get_logger().warn("valve command ignored -- emergency surface active")
            return

        mask = int(msg.data) & 0b11
        if mask == 0:
            # Close up; flush the terminal closed-valves out, then go silent.
            self._publish_valves(0)
            self._valve_mask = None
            self._arm_flush()
            return

        self._valve_mask = mask
        self._publish_valves(mask)

    def _on_emergency_cmd(self, msg: Bool) -> None:
        engage = bool(msg.data)
        if engage and not self._emergency:
            self._emergency = True
            self.get_logger().warn("EMERGENCY SURFACE engaged -- blowing ballast")
            # Push the first blow-ballast sample now rather than waiting a tick.
            self._tick_emergency()
        elif not engage and self._emergency:
            self._emergency = False
            self._clear_pump_state()
            self._valve_mask = None
            self._publish_rpm(0)
            self._publish_valves(0)
            self._arm_flush()
            self.get_logger().info("emergency surface cancelled")

    # --- periodic heartbeat ---------------------------------------------

    def _on_tick(self) -> None:
        now = self._now_s()
        # Emergency always acts -- it's the safety path.
        if self._emergency:
            self._tick_emergency()
            return

        # Pump session bookkeeping. The pressure-target check runs first so
        # the operator's stop condition wins over the backstop on the same
        # tick they coincide. Either stop reason clears the whole session
        # (dropping _held_rpm to 0) and arms the trailing-zero flush so the
        # stop carries out as a 0 before we go silent.
        if self._pump_deadline_s is not None:
            target_reached = (
                self._target_pressure_pa is not None
                and self._tank_pa is not None
                and (
                    (
                        self._held_rpm > 0
                        and self._tank_pa <= float(self._target_pressure_pa)
                    )
                    or (
                        self._held_rpm < 0
                        and self._tank_pa >= float(self._target_pressure_pa)
                    )
                )
            )
            if target_reached:
                self.get_logger().info(
                    f"tank pressure target reached "
                    f"({self._tank_pa:.0f} Pa, target {self._target_pressure_pa} Pa) "
                    "-- pump stopped"
                )
                self._clear_pump_state()
                self._arm_flush()
            elif now >= self._pump_deadline_s:
                if self._target_pressure_pa is not None:
                    self.get_logger().warn(
                        f"pump-until-pressure hit the {MAX_PUMP_S:.0f} s runaway "
                        f"backstop before reaching target {self._target_pressure_pa} Pa "
                        f"(tank={self._tank_pa if self._tank_pa is not None else 'n/a'} Pa) "
                        "-- pump stopped"
                    )
                self._clear_pump_state()
                self._arm_flush()

        # Drive the wire only while a command is actually held; otherwise go
        # silent so depth_node can own the BCU. A pump session is live while
        # _pump_deadline_s is set; a valve hold while _valve_mask is set.
        pump_live = self._pump_deadline_s is not None
        valves_held = self._valve_mask is not None
        if pump_live or valves_held:
            # Heartbeat the held command. _held_rpm is the active pump setpoint
            # (0 for a valve-only hold); re-assert the mask if we hold valves.
            self._publish_rpm(self._held_rpm)
            if self._valve_mask:
                self._publish_valves(self._valve_mask)
        elif self._flush_ticks > 0:
            # A session/reset just ended: carry the terminal 0 out for a few
            # ticks so the rate-throttled /bcu/rpm egress reliably lands it,
            # then fall silent. RPM only -- /bcu/valves is on-change/retained
            # (the close was already delivered) and re-publishing it here would
            # clobber a pump-only session that never touched the valves.
            self._flush_ticks -= 1
            self._publish_rpm(0)
        # else: fully idle -> publish nothing.

    def _tick_emergency(self) -> None:
        # Continuously blow ballast through valve 2 (the motor way). No
        # surfaced check, no timeout -- only an explicit cancel (False) or
        # DEBUG_RESET stands it down.
        self._publish_rpm(EMERGENCY_SURFACE_RPM)
        self._publish_valves(BCU_MOTOR_VALVE_MASK)

    # --- helpers --------------------------------------------------------

    def _clear_pump_state(self) -> None:
        """Drop any held pump session (timed or pressure-targeted).

        Does not publish; callers decide whether to also push a zero RPM
        sample on the wire (most do) or just reset bookkeeping (e.g. when
        the very next line is going to publish a new RPM anyway).
        """
        self._held_rpm = 0
        self._pump_deadline_s = None
        self._target_pressure_pa = None

    def _arm_flush(self) -> None:
        """Start the trailing-zero flush so a terminal stop reaches the egress."""
        self._flush_ticks = FLUSH_TICKS

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _motor_valve_open(self) -> bool:
        return self._valve_mask is not None and bool(
            self._valve_mask & BCU_MOTOR_VALVE_MASK
        )

    def _publish_rpm(self, rpm: int) -> None:
        out = Int16()
        out.data = int(rpm)
        self._rpm_pub.publish(out)

    def _publish_valves(self, mask: int) -> None:
        out = UInt8()
        out.data = int(mask) & 0xFF
        self._valves_pub.publish(out)

    def destroy_node(self) -> bool:
        # Best-effort: stop the motor on shutdown so we don't leave a stale
        # RPM on the wire if we're killed mid-command.
        try:
            self._publish_rpm(0)
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BcuDebugNode()
    spin_node(node)


if __name__ == "__main__":
    main()
