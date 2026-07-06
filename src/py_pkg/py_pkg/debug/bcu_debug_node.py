"""Manual BCU driver -- drives ``/bcu/rpm`` + ``/bcu/valves`` by hand,
bypassing pathfinding and the depth PID.

The UI publishes ``/command``=false when a manual command is engaged, so
bcu_node safe-stops and frees the BCU for us; we heartbeat a held command and
fall silent when idle so bcu_node can reclaim it on the next mission.

Command surfaces:

* Emergency surface (``DEBUG_EMERGENCY_SURFACE``) -- blow ballast continuously
  at ``EMERGENCY_SURFACE_RPM`` with valve 2 (motor way) open. Deliberately
  dumb (no surfaced/timeout/pressure check) so a bad sensor can't argue it out
  of surfacing; only an explicit ``False`` or ``DEBUG_RESET`` stands it down.
* Pump for X seconds (``DEBUG_BCU_RPM``) -- run at a requested RPM for a
  bounded window, then stop.
* Pump until pressure (``DEBUG_BCU_RPM_UNTIL_PRESSURE``) -- run until
  ``/bcu/pressure`` (sensor-frame) crosses ``target_pressure_pa`` in the
  direction the RPM sign implies; mutually exclusive with the timed pump.
* Valves (``DEBUG_BCU_VALVES``) -- latch a bitmask (bit0=valve2/motor,
  bit1=valve1/free), mask 0 closes; only touched once a valve command arrives.
* Reset (``DEBUG_RESET``) -- red all-stop: cancel everything, command 0 RPM +
  closed valves once, go silent.

No pump command here consults the operator's tank limits, so a pump can
dead-head at a tank endpoint until stopped or ``MAX_PUMP_S`` fires -- deliberate;
the tank smarts live elsewhere (depth PID self-clamps, the MQTT bridge stands an
emergency blow down near the empty endpoint), which is what keeps this
emergency surface unarguable.

A held command re-publishes at 10 Hz because the MQTT egress is a
rate-limit-and-drop throttle; the heartbeat plus a trailing-zero flush keeps
the topic fresh and lands the reset-to-0 on the UI before we fall silent.
"""

import rclpy
from nautilus_msgs.msg import BcuPumpCommand, BcuPumpUntilPressureCommand
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, Int16, Int32, UInt8

from py_pkg.debug import FLUSH_TICKS, PUBLISH_PERIOD_S
from py_pkg.math_utils import clamp
from py_pkg.robot_specs import BCU_MOTOR_VALVE_MASK
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
    now_s,
    spin_node,
)

# Runaway backstop: caps the timed pump and the pressure-targeted mode so a
# typo'd UI input (3000 instead of 3) can't leave the motor running for ages.
MAX_PUMP_S = 300.0

# Emergency-surface pump speed, shared by the operator's slider and the
# lifeguard failsafe. Tier 2/3 tests assert it on the wire.
EMERGENCY_SURFACE_RPM = 3000


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
        create_subscription_for_topic(self, UUVTopics.DEBUG_RESET, self._on_reset)

        # A pump session is either timed or pressure-targeted; both share
        # _held_rpm + _pump_deadline_s, the pressure variant also holds
        # _target_pressure_pa. The two are mutually exclusive (arming either
        # replaces the other) and _clear_pump_state() drops them as a unit.
        self._held_rpm: int = 0
        self._pump_deadline_s: float | None = None
        self._target_pressure_pa: int | None = None

        # Latched valve bitmask. None means we don't own /bcu/valves (bcu_node
        # has it); an int means we're holding that mask.
        self._valve_mask: int | None = None

        self._emergency = False

        # Latest tank pressure (sensor-frame Pa), None until the first reading.
        self._tank_pa: float | None = None

        # Trailing-zero flush countdown; see _on_tick.
        self._flush_ticks: int = 0

        self._timer = self.create_timer(PUBLISH_PERIOD_S, self._on_tick)

    # --- helpers --------------------------------------------------------

    def _publish_rpm(self, rpm: int) -> None:
        out = Int16()
        out.data = int(rpm)
        self._rpm_pub.publish(out)

    def _publish_valves(self, mask: int) -> None:
        out = UInt8()
        out.data = int(mask) & 0xFF
        self._valves_pub.publish(out)

    def _motor_valve_open(self) -> bool:
        return self._valve_mask is not None and bool(
            self._valve_mask & BCU_MOTOR_VALVE_MASK
        )

    def _warn_valve_closed(self) -> None:
        self.get_logger().warn(
            f"pumping at {self._held_rpm} rpm with valve 2 not commanded open "
            "-- open valve 2 (motor way) to carry the flow"
        )

    def _clear_pump_state(self) -> None:
        # Drop any held pump session. Does not publish -- callers decide whether
        # to push a zero (most do) or just reset before publishing a new RPM.
        self._held_rpm = 0
        self._pump_deadline_s = None
        self._target_pressure_pa = None

    def _arm_flush(self) -> None:
        self._flush_ticks = FLUSH_TICKS

    def _stop_pump(self) -> None:
        """Zero the motor, drop the pump session, and flush the terminal 0 out."""
        self._clear_pump_state()
        self._publish_rpm(0)
        self._arm_flush()

    def _pressure_target_reached(self) -> bool:
        # Tank crosses target in the direction the RPM sign implies: positive
        # inflates and drops the tank reading (stop at/below target), negative
        # is the mirror. False until both a target and a sample exist.
        if self._target_pressure_pa is None or self._tank_pa is None:
            return False
        target = float(self._target_pressure_pa)
        return (self._held_rpm > 0 and self._tank_pa <= target) or (
            self._held_rpm < 0 and self._tank_pa >= target
        )

    # --- command callbacks ----------------------------------------------

    def _on_reset(self, _msg: Empty) -> None:
        # Red all-stop: cancel any session (pump or emergency), drop valves,
        # command 0 RPM + closed valves once, then flush and go silent.
        self._emergency = False
        self._clear_pump_state()
        self._valve_mask = None
        self._publish_rpm(0)
        self._publish_valves(0)
        self._arm_flush()
        self.get_logger().info("debug reset -- BCU all-stop (0 RPM, valves closed)")

    def _on_tank_pressure(self, msg: Int32) -> None:
        # Cache only; the stop check runs in _on_tick so the decision rides the
        # same heartbeat as the RPM re-publish.
        self._tank_pa = float(msg.data)

    def _on_rpm_cmd(self, msg: BcuPumpCommand) -> None:
        if self._emergency:
            self.get_logger().warn("pump command ignored -- emergency surface active")
            return

        duration = clamp(float(msg.duration_s), 0.0, MAX_PUMP_S)
        if duration == 0.0:
            self._stop_pump()  # explicit stop
            return

        # New timed window supersedes any pressure-targeted session.
        self._clear_pump_state()
        self._held_rpm = int(msg.rpm)
        self._pump_deadline_s = now_s(self) + duration
        if self._held_rpm != 0 and not self._motor_valve_open():
            self._warn_valve_closed()
        self._publish_rpm(self._held_rpm)

    def _on_rpm_until_pressure_cmd(self, msg: BcuPumpUntilPressureCommand) -> None:
        if self._emergency:
            self.get_logger().warn(
                "pump-until-pressure ignored -- emergency surface active"
            )
            return

        rpm = int(msg.rpm)
        if rpm == 0:
            self._stop_pump()  # explicit stop
            return

        if self._tank_pa is None:
            self.get_logger().warn(
                "pump-until-pressure ignored -- no tank-pressure sample yet "
                "(waiting on /bcu/pressure)"
            )
            return

        self._clear_pump_state()
        self._held_rpm = rpm
        self._target_pressure_pa = int(msg.target_pressure_pa)
        self._pump_deadline_s = now_s(self) + MAX_PUMP_S
        if not self._motor_valve_open():
            self._warn_valve_closed()
        self._publish_rpm(self._held_rpm)

    def _on_valve_cmd(self, msg: UInt8) -> None:
        if self._emergency:
            self.get_logger().warn("valve command ignored -- emergency surface active")
            return

        mask = int(msg.data) & 0b11
        if mask == 0:
            # Close up, flush the terminal closed-valves out, then go silent.
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
        # Emergency always acts -- it's the safety path.
        if self._emergency:
            self._tick_emergency()
            return

        # Pump session bookkeeping. The pressure-target check runs before the
        # deadline so the operator's stop condition wins on the tick they
        # coincide. Either reason clears the session and arms the flush.
        if self._pump_deadline_s is not None:
            if self._pressure_target_reached():
                self.get_logger().info(
                    f"tank pressure target reached "
                    f"({self._tank_pa:.0f} Pa, target {self._target_pressure_pa} Pa) "
                    "-- pump stopped"
                )
                self._clear_pump_state()
                self._arm_flush()
            elif now_s(self) >= self._pump_deadline_s:
                if self._target_pressure_pa is not None:
                    self.get_logger().warn(
                        f"pump-until-pressure hit the {MAX_PUMP_S:.0f} s runaway "
                        f"backstop before reaching target {self._target_pressure_pa} Pa "
                        f"(tank={self._tank_pa if self._tank_pa is not None else 'n/a'} Pa) "
                        "-- pump stopped"
                    )
                self._clear_pump_state()
                self._arm_flush()

        # Drive the wire only while a command is held; otherwise go silent.
        if self._pump_deadline_s is not None or self._valve_mask is not None:
            self._publish_rpm(self._held_rpm)  # 0 for a valve-only hold
            if self._valve_mask:
                self._publish_valves(self._valve_mask)
        elif self._flush_ticks > 0:
            # A session just ended: carry the terminal 0 out for a few ticks
            # so the throttled egress reliably lands it, then fall silent.
            self._flush_ticks -= 1
            self._publish_rpm(0)
        # else: fully idle -> publish nothing.

    def _tick_emergency(self) -> None:
        # Continuously blow ballast through valve 2 (the motor way).
        self._publish_rpm(EMERGENCY_SURFACE_RPM)
        self._publish_valves(BCU_MOTOR_VALVE_MASK)

    def destroy_node(self) -> bool:
        # Best-effort: stop the motor on shutdown so we don't leave a stale RPM
        # on the wire if we're killed mid-command.
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
