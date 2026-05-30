"""Manual BCU driver -- bypasses pathfinding + the depth PID.

Drives the contested BCU actuator topics (``/bcu/rpm`` and ``/bcu/valves``)
by hand so an operator can poke the pump and valves. Whether it's *allowed*
to touch the wire is owned entirely by the operator's manual-override slider:
``CONTROL_MANUAL_OVERRIDE`` is raised by the mission UI (through the MQTT
bridge), depth_node stands down while it's True, and this node only relays /
heartbeats its commands during that window. When the override drops we zero
the motor once and hand ``/bcu/rpm`` + ``/bcu/valves`` back to depth_node.

Command surfaces:

* Emergency surface (``DEBUG_EMERGENCY_SURFACE``) -- the one exception to the
  override gate: it always acts (safety path). Blow ballast: full positive
  RPM with valve 1 open until the external pressure says we're at the surface,
  then stop the pump. The UI raises the override before arming it, so
  depth_node is already standing down. It also re-pumps if the vehicle sinks
  back below the surface threshold. Send ``False`` to stand down.

* Pump for X seconds (``DEBUG_BCU_RPM``) -- run the motor at a requested RPM
  for a bounded window, then stop. Pumping needs valve 1 open to carry the
  flow; if it isn't, we warn (but still pump -- this is a debug knob, and
  the operator UI carries the same warning).

* Pump until tank pressure target (``DEBUG_BCU_RPM_UNTIL_PRESSURE``) --
  closed-loop sibling of the timed pump: run the motor at a requested RPM
  until ``/bcu/pressure`` crosses ``target_pressure_pa`` in the direction
  the sign of the RPM implies (positive inflates -> tank drops -> stop
  when tank_pa <= target; negative deflates -> tank rises -> stop when
  tank_pa >= target). Mutually exclusive with the timed pump on this node:
  arming either replaces the other. The same ``MAX_PUMP_S`` cap fires as
  a safety floor if the target is never reached. If the tank is already
  past the target for the requested direction, we refuse and zero the motor
  rather than running uselessly.

* Valves (``DEBUG_BCU_VALVES``) -- latch a valve bitmask (bit0=v1, bit1=v2).
  We only touch ``/bcu/valves`` once a valve command has actually arrived, so
  a pump-only session never slams the valves shut underneath the operator.
  Sending mask 0 closes them.

While a command is held we re-publish it at 10 Hz. The MQTT bridge's egress
side is a rate-limit-and-drop throttle, so a lone publish can lose the race
against depth_node's stream; a steady heartbeat keeps the topic fresh on the
wire (and the STM fed) for as long as the operator asked.

The pump duration is clamped to ``MAX_PUMP_S`` so a typo'd UI input (3000
instead of 3) can't leave the motor running for half an hour.
"""

import rclpy
from nautilus_msgs.msg import BcuPumpCommand, BcuPumpUntilPressureCommand
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Int16, Int32, UInt8

from py_pkg.physics import gauge_pressure_pa
from py_pkg.robot_specs import BCU_MOTOR_MAX_RPM
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)

MAX_PUMP_S = 30.0

# Re-publish cadence while a command is active. 10 Hz matches depth_node's
# control loop and the MQTT egress throttle, so each tick refreshes the
# bridge's per-topic clock and the next sample is always allowed through.
PUBLISH_PERIOD_S = 0.1

# Cap on a single emergency-surface burst so a stuck dive (never reaches the
# surface) doesn't pump the motor flat-out indefinitely. Generous: a normal
# ascent surfaces well inside this. After it expires we stop pumping; the
# operator sees we didn't surface and intervenes.
EMERGENCY_MAX_S = 120.0

# Gauge pressure at/under which we call it "surfaced" (~0.2 m of fresh water).
SURFACE_GAUGE_EPS_PA = 2000.0

# Valve bitmask with valve 1 (the pump flow path, "motor way") open.
VALVE1_OPEN_MASK = 0b01


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
        create_subscription_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE, self._on_pressure
        )
        # Tank pressure is the stop signal for the pressure-targeted pump mode.
        create_subscription_for_topic(
            self, UUVTopics.BCU_PRESSURE, self._on_tank_pressure
        )
        # The operator's manual-override slider gates everything except the
        # emergency path: we only drive the wire while this is True.
        create_subscription_for_topic(
            self, UUVTopics.CONTROL_MANUAL_OVERRIDE, self._on_override
        )

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
        self._emergency_deadline_s: float | None = None
        self._emergency_timeout_warned = False
        # Latest external pressure as gauge Pa; None until the first reading.
        self._gauge_pa: float | None = None
        # Latest tank pressure in absolute Pa; None until the first reading.
        # Drives the stop check for the pressure-targeted pump mode.
        self._tank_pa: float | None = None
        # Whether the operator slider currently has us in manual mode.
        self._manual_override = False

        self._timer = self.create_timer(PUBLISH_PERIOD_S, self._on_tick)

    # --- command callbacks ----------------------------------------------

    def _on_override(self, msg: Bool) -> None:
        active = bool(msg.data)
        if active == self._manual_override:
            return
        self._manual_override = active
        if not active:
            # Slider off -> hand /bcu/rpm + /bcu/valves back to depth_node.
            # Drop any held pump/valve state and zero the motor once so we
            # leave a clean wire; emergency is left untouched (it ignores the
            # gate and the UI keeps the override up while surfacing).
            self._clear_pump_state()
            self._valve_mask = None
            self._publish_rpm(0)
        self.get_logger().info(
            f"manual override {'engaged' if active else 'released'} -- "
            f"bcu_debug {'driving' if active else 'silent'}"
        )

    def _on_pressure(self, msg: Int32) -> None:
        # EXTERNAL_PRESSURE is absolute Pa; the surface check works in gauge.
        self._gauge_pa = gauge_pressure_pa(float(msg.data))

    def _on_tank_pressure(self, msg: Int32) -> None:
        # BCU_PRESSURE is the absolute tank pressure in Pa. We just cache it;
        # the stop check runs in _on_tick alongside the deadline check so the
        # decision is on the same heartbeat as the RPM re-publish.
        self._tank_pa = float(msg.data)

    def _on_rpm_cmd(self, msg: BcuPumpCommand) -> None:
        if self._emergency:
            self.get_logger().warn("pump command ignored -- emergency surface active")
            return
        if not self._manual_override:
            self.get_logger().warn(
                "pump command ignored -- manual override is off (depth_node owns the BCU)"
            )
            return

        duration = _clamp_duration(float(msg.duration_s))
        if duration == 0.0:
            # Explicit stop: zero the motor and clear the window.
            self._clear_pump_state()
            self._publish_rpm(0)
            return

        # New timed window supersedes any pressure-targeted session.
        self._clear_pump_state()
        self._held_rpm = int(msg.rpm)
        self._pump_deadline_s = self._now_s() + duration
        if self._held_rpm != 0 and not self._valve1_open():
            self.get_logger().warn(
                f"pumping at {self._held_rpm} rpm with valve 1 not commanded "
                "open -- open valve 1 (motor way) to carry the flow"
            )
        self._publish_rpm(self._held_rpm)

    def _on_rpm_until_pressure_cmd(self, msg: BcuPumpUntilPressureCommand) -> None:
        if self._emergency:
            self.get_logger().warn(
                "pump-until-pressure ignored -- emergency surface active"
            )
            return
        if not self._manual_override:
            self.get_logger().warn(
                "pump-until-pressure ignored -- manual override is off "
                "(depth_node owns the BCU)"
            )
            return

        rpm = int(msg.rpm)
        target = int(msg.target_pressure_pa)

        if rpm == 0:
            # Treat as explicit stop, same as duration=0 on the timed command.
            self._clear_pump_state()
            self._publish_rpm(0)
            return

        if self._tank_pa is None:
            self.get_logger().warn(
                "pump-until-pressure ignored -- no tank-pressure sample yet "
                "(waiting on /bcu/pressure)"
            )
            return

        # Refuse to pump if the tank is already past the target in the direction
        # this RPM would push -- it would just hit the safety floor doing nothing.
        already_past = (rpm > 0 and self._tank_pa <= float(target)) or (
            rpm < 0 and self._tank_pa >= float(target)
        )
        if already_past:
            self.get_logger().warn(
                f"pump-until-pressure refused -- tank already at "
                f"{self._tank_pa:.0f} Pa, past target {target} Pa for "
                f"rpm={rpm}; not pumping"
            )
            self._clear_pump_state()
            self._publish_rpm(0)
            return

        # New pressure-targeted session supersedes any timed window.
        self._clear_pump_state()
        self._held_rpm = rpm
        self._target_pressure_pa = target
        self._pump_deadline_s = self._now_s() + MAX_PUMP_S
        if not self._valve1_open():
            self.get_logger().warn(
                f"pumping at {self._held_rpm} rpm with valve 1 not commanded "
                "open -- open valve 1 (motor way) to carry the flow"
            )
        self._publish_rpm(self._held_rpm)

    def _on_valve_cmd(self, msg: UInt8) -> None:
        if self._emergency:
            self.get_logger().warn("valve command ignored -- emergency surface active")
            return
        if not self._manual_override:
            self.get_logger().warn(
                "valve command ignored -- manual override is off (depth_node owns the BCU)"
            )
            return

        mask = int(msg.data) & 0b11
        if mask == 0:
            # Close up and hand /bcu/valves back to depth_node.
            self._publish_valves(0)
            self._valve_mask = None
            return

        self._valve_mask = mask
        self._publish_valves(mask)

    def _on_emergency_cmd(self, msg: Bool) -> None:
        engage = bool(msg.data)
        if engage and not self._emergency:
            self._emergency = True
            self._emergency_deadline_s = self._now_s() + EMERGENCY_MAX_S
            self._emergency_timeout_warned = False
            self.get_logger().warn("EMERGENCY SURFACE engaged -- blowing ballast")
            # Push the first blow-ballast sample now rather than waiting a tick.
            self._tick_emergency(self._now_s())
        elif not engage and self._emergency:
            self._emergency = False
            self._emergency_deadline_s = None
            self._clear_pump_state()
            self._valve_mask = None
            self._publish_rpm(0)
            self._publish_valves(0)
            self.get_logger().info("emergency surface cancelled")

    # --- periodic heartbeat ---------------------------------------------

    def _on_tick(self) -> None:
        now = self._now_s()
        # Emergency always acts -- it's the safety path and ignores the gate.
        if self._emergency:
            self._tick_emergency(now)
            return

        # Outside emergency we only touch the wire while the operator slider
        # has us in manual mode; otherwise depth_node owns the BCU.
        if not self._manual_override:
            return

        # Pump session bookkeeping. The pressure-target check runs first so
        # the operator's stop condition wins over the safety floor on the same
        # tick they coincide. Either stop reason clears the whole session so
        # the next callback (timed or targeted) starts from a clean slate.
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
                self._publish_rpm(0)
            elif now >= self._pump_deadline_s:
                if self._target_pressure_pa is not None:
                    self.get_logger().warn(
                        f"pump-until-pressure hit the {MAX_PUMP_S:.0f} s safety "
                        f"cap before reaching target {self._target_pressure_pa} Pa "
                        f"(tank={self._tank_pa if self._tank_pa is not None else 'n/a'} Pa) "
                        "-- pump stopped"
                    )
                self._clear_pump_state()
                self._publish_rpm(0)
            else:
                self._publish_rpm(self._held_rpm)

        # Re-assert held valves (None == not ours; 0 was already published on
        # the close command and the mask cleared).
        if self._valve_mask:
            self._publish_valves(self._valve_mask)

    def _tick_emergency(self, now: float) -> None:
        timed_out = (
            self._emergency_deadline_s is not None and now >= self._emergency_deadline_s
        )
        surfaced = self._gauge_pa is not None and self._gauge_pa <= SURFACE_GAUGE_EPS_PA

        if timed_out:
            if not self._emergency_timeout_warned:
                self.get_logger().warn(
                    "emergency surface timed out before reaching the surface -- "
                    "pump stopped; cancel to release"
                )
                self._emergency_timeout_warned = True
            self._publish_rpm(0)
            self._publish_valves(0)
            return

        if surfaced:
            # At the surface: stop pumping and close valves to hold the full
            # bladder. If we sink back past the threshold we'll re-pump.
            self._publish_rpm(0)
            self._publish_valves(0)
            return

        # Still down: blow ballast -- full inflate through valve 1.
        self._publish_rpm(BCU_MOTOR_MAX_RPM)
        self._publish_valves(VALVE1_OPEN_MASK)

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

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _valve1_open(self) -> bool:
        return self._valve_mask is not None and bool(
            self._valve_mask & VALVE1_OPEN_MASK
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
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
