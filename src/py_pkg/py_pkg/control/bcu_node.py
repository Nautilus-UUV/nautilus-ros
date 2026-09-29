#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import Pose
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_msgs.msg import Int16, UInt8

from py_pkg.control.bcu_safe_stop_burst import BcuSafeStopBurst
from py_pkg.control.run_end import subscribe_run_end
from py_pkg.control.tank_limit_guard import TankLimitGuard
from py_pkg.math_utils import tank_limits_valid
from py_pkg.robot_specs import (
    BCU_DEEP_THRESHOLD_PA,
    BCU_FREE_VALVE_MASK,
    BCU_MOTOR_VALVE_MASK,
)
from py_pkg.scenarios.compile import bcu_spec_from_node
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
    now_s,
    spin_node,
)

# How long to keep re-asserting the safe-stop after a stop (or at boot)
# before the loop goes silent.
STOP_REASSERT_S = 1.0

# How long a manual BCU command keeps bcu_node off the wire (no reassert burst).
# >= STOP_REASSERT_S, and only needs to span the cross-topic delivery skew
# between the paired /command=false and the manual command; large is safe because
# bcu_debug owns the wire for the whole manual session anyway.
MANUAL_HOLD_S = 1.5


def solve_bcu_command(
    error_pa: float,
    current_pressure_pa: float,
    *,
    pump_rpm: int,
    deadband_pa: float,
    deep_threshold_pa: float = BCU_DEEP_THRESHOLD_PA,
) -> tuple[int, int, int]:
    """Turn one depth error into a raw BCU wire command -- bang-bang.

    The pump has exactly three states: full speed one way, full speed the
    other, or off. Nothing modulates, so nothing can dither: the command only
    changes when the sign of the error changes, which happens when the mission
    advances to its next leg.

    ``error_pa`` is ``target - current`` in gauge Pa (Z-positive-down, so a
    positive error means the target is deeper than we are). Returns
    ``(pump_rpm, motor_open, free_open)`` ready for the wire -- motor_open is
    valve 2 (bit0), free_open is valve 1 (bit1).

    Sign convention, unchanged from the PID this replaces and load-bearing
    downstream: positive bus RPM inflates the bladder (the vehicle rises) and
    so drains oil out of the tank; negative deflates (sinks) and fills the
    tank. ``TankLimitGuard`` reads the same signs off this return value.

    ``deadband_pa`` is the one thing keeping the pump off when there is nothing
    to do. It is not hysteresis and it does not latch -- it exists because
    SURFACE targets gauge 0 Pa, where the reading bobs across zero on sensor
    noise; without it the sign of the error would flip on that noise and the
    pump would start filling the tank at the surface. Sized well below every
    mission's arrival tolerance, so it never truncates a leg.

    Below ``deep_threshold_pa`` the pump can still push oil out into the
    bladder; above it the surrounding water wins, so a descend command opens
    valve 1 (the free/bypass way) and lets ambient pressure squeeze oil back
    into the tank on its own instead of dead-heading the pump.

    The tank-endpoint cutoff is applied downstream by ``TankLimitGuard``, which
    needs the raw direction this returns, so it is not folded in here.
    """
    if error_pa > deadband_pa:
        # Target is deeper: deflate the bladder to sink.
        if current_pressure_pa > deep_threshold_pa:
            return 0, 0, 1  # passive vent; the pump has no authority this deep
        return -pump_rpm, 1, 0
    if error_pa < -deadband_pa:
        # Target is shallower: inflate the bladder to rise.
        return pump_rpm, 1, 0
    # Close enough: freeze the bladder where it is.
    return 0, 0, 0


class BCUNode(Node):
    def __init__(self):
        super().__init__("bcu_node")

        # Reentrant group (private -- NOT rclpy's Node.default_callback_group)
        # so the manual-drive subscriptions can fire while the control timer is
        # running: being in a group of their own is what lets a manual command
        # cancel an in-flight safe-stop burst instead of flickering against it.
        # Inert under the single-threaded rclpy.spin() the node ships with; it
        # matters the moment this runs on a MultiThreadedExecutor.
        self._manual_cb_group = ReentrantCallbackGroup()

        cfg = bcu_spec_from_node(self)
        self._pump_rpm = cfg.pump_rpm
        self._deadband_pa = cfg.deadband_pa

        # Latching tank-endpoint cutoff: holds the pump/valves stopped once the
        # tank is parked at a limit. Latching (rather than a bare per-tick
        # threshold) is what stops tank-sensor noise chattering the valves while
        # we sit on the guard -- the bench-test bug.
        self._tank_guard = TankLimitGuard(stop_band=cfg.tank_stop_band)

        # Safe-stop re-assert burst with a manual-yield: republishes 0 RPM +
        # valves closed for a bounded burst after a stop so the STM latches the
        # zero, but cancels the burst when the operator drives the BCU by hand
        # (bcu_debug owns the same wire) so the two can't flicker against each
        # other. Driven from control_loop (tick) and _begin_safe_stop / the debug
        # subscriptions below.
        # BcuSafeStopBurst owns the >= 1 floor on the count.
        self._safe_stop = BcuSafeStopBurst(
            reassert_count=round(cfg.frequency_hz * STOP_REASSERT_S),
            manual_hold_s=MANUAL_HOLD_S,
        )

        # Held None until the first POSITION_TARGET arrives
        self.target_pressure_pa: float | None = None
        self.current_pressure_pa = 0.0

        # Tank endpoints from the operator's pre-dive Initialize (DIVE_INIT).
        self._tank_pa: float | None = None
        self._tank_empty_pa: float | None = None
        self._tank_full_pa: float | None = None

        # Log throttle
        self._target_log_every_n = 10
        self._target_cb_count = 0

        self.bcu_controller_rpm_publisher = create_publisher_for_topic(
            self, UUVTopics.BCU_RPM
        )

        self.bcu_valves_publisher = create_publisher_for_topic(
            self, UUVTopics.BCU_VALVES
        )

        # Pressure setpoint is `position.z` of POSITION_TARGET, in gauge
        # Pa (Z-positive-down: deeper = higher gauge pressure).
        create_subscription_for_topic(
            self, UUVTopics.POSITION_TARGET, self.target_pose_callback
        )

        # Depth measurement rides POSITION_ESTIMATION.position.z (gauge Pa,
        # Z-positive-down), already gauged by attitude_node.
        create_subscription_for_topic(
            self, UUVTopics.POSITION_ESTIMATION, self.current_pose_callback
        )

        # Tank pressure feeds the endpoint cutoff; the dive registration
        # carries the limits it compares against.
        create_subscription_for_topic(
            self, UUVTopics.BCU_PRESSURE, self._on_tank_pressure
        )

        create_subscription_for_topic(self, UUVTopics.DIVE_INIT, self._on_dive_init)

        # Either way a run ends, the controller drops its target, emits one
        # safe-stop and goes silent (freeing the BCU wire for a debug node).
        subscribe_run_end(self, self._stop)

        # Yield the wire whenever a debug/operator path drives the BCU. We don't
        # act on the payload -- a message on any of the manual-drive surfaces just
        # means someone (the operator, or the lifeguard's auto emergency-surface)
        # is driving, so we cancel any pending safe-stop burst rather than flicker
        # against it. The surface set (which excludes DEBUG_RESET) is owned by the
        # registry; see UUVTopics.BCU_MANUAL_DRIVE_TOPICS.
        for manual_topic in UUVTopics.BCU_MANUAL_DRIVE_TOPICS:
            create_subscription_for_topic(
                self,
                manual_topic,
                self._on_manual_activity,
                callback_group=self._manual_cb_group,
            )

        self.control_timer = self.create_timer(
            1.0 / cfg.frequency_hz, self.control_loop
        )

        # Leave the BCU wire unambiguously at 0/closed at boot. This runs
        # synchronously here, before spin_node, so the boot burst arms before any
        # subscription callback (incl. a latched manual command) can fire.
        self._begin_safe_stop()

        self.get_logger().info("Depth control node started (bang-bang).")

    def target_pose_callback(self, msg: Pose):
        self.target_pressure_pa = float(msg.position.z)
        self._target_cb_count += 1
        if self._target_cb_count % self._target_log_every_n == 0:
            self.get_logger().info(
                f"Updated target pressure: {self.target_pressure_pa} Pa"
            )

    def current_pose_callback(self, msg: Pose):
        # position.z is the depth measurement in gauge Pa.
        self.current_pressure_pa = float(msg.position.z)

    def _on_tank_pressure(self, msg):
        # Tank pressure in the sensor's own frame (tank relative to hull).
        self._tank_pa = float(msg.data)

    def _on_dive_init(self, msg):
        # Only the tank endpoints are ours now -- the surface-pressure gauge
        # reference moved to attitude_node, which publishes already-gauged depth.
        self._tank_empty_pa = float(msg.tank_empty_pa)
        self._tank_full_pa = float(msg.tank_full_pa)
        # Fresh endpoints invalidate any latch held against the old ones.
        self._tank_guard.reset()
        # Same predicate the guard itself gates on.
        limits_ok = tank_limits_valid(self._tank_empty_pa, self._tank_full_pa)
        log = self.get_logger().info if limits_ok else self.get_logger().error
        log(
            "dive init: "
            f"tank empty/full = {self._tank_empty_pa:.0f}/{self._tank_full_pa:.0f} Pa "
            f"({'ok' if limits_ok else 'invalid -- cutoff stays inert'})"
        )

    def _publish_bcu_stop(self) -> None:
        zero_rpm = Int16()
        zero_rpm.data = 0
        self.bcu_controller_rpm_publisher.publish(zero_rpm)
        valves_off = UInt8()
        valves_off.data = 0
        self.bcu_valves_publisher.publish(valves_off)

    def _begin_safe_stop(self) -> None:
        # Arm the re-assert burst and emit the first safe-stop now -- unless a
        # manual command holds the wire, in which case begin_stop yields and we
        # publish nothing (bcu_debug is driving and we'd only flicker against it).
        if self._safe_stop.begin_stop(now_s(self)):
            self._publish_bcu_stop()

    def _on_manual_activity(self, _msg) -> None:
        # A message on any BCU debug command topic: the operator/lifeguard is
        # driving the BCU. Cancel any pending safe-stop burst so we don't fight it.
        self._safe_stop.note_manual(now_s(self))

    def _stop(self, reason: str) -> None:
        """Drop the target, wipe controller state, park the wire. Idempotent."""
        # Edge-trigger: only the running->stopped transition does the stop work.
        # target_pressure_pa is None already means "stopped", so a repeated stop
        # (the UI sends /command=false before every manual command, and a
        # completion may land on a stack the operator already stopped) is a
        # no-op -- otherwise each one would re-arm the burst and chatter the wire
        # against the manual command.
        if self.target_pressure_pa is None:
            return
        self._tank_guard.reset()
        self.target_pressure_pa = None
        self._begin_safe_stop()
        self.get_logger().info(
            f"{reason} -> safe-stop (or yield to manual), "
            "BCU going silent (fresh state)."
        )

    def control_loop(self):
        if self.target_pressure_pa is None:
            # No target. Keep re-asserting the safe-stop for the burst window so
            # the STM latches the zero, then go silent so a debug node can own the
            # BCU wire with no contention. tick() returns False once the burst is
            # spent or a manual command has cancelled it.
            if self._safe_stop.tick():
                self._publish_bcu_stop()
            return

        # The whole control law: one stateless solve, then the tank cutoff.
        error_pa = self.target_pressure_pa - self.current_pressure_pa
        pump_rpm, motor_open, free_open = solve_bcu_command(
            error_pa,
            self.current_pressure_pa,
            pump_rpm=self._pump_rpm,
            deadband_pa=self._deadband_pa,
        )

        # Tank-endpoint cutoff: stop (and latch) the command once the tank is
        # parked at a registered limit, so it can't dead-head the pump or
        # chatter the valves on noise that straddles the guard.
        pump_rpm, motor_open, free_open = self._tank_guard.apply(
            pump_rpm,
            motor_open,
            free_open,
            self._tank_pa,
            self._tank_empty_pa,
            self._tank_full_pa,
        )

        rpm_msg = Int16()
        rpm_msg.data = pump_rpm
        valves_msg = UInt8()
        valves_msg.data = (BCU_MOTOR_VALVE_MASK if motor_open else 0) | (
            BCU_FREE_VALVE_MASK if free_open else 0
        )

        # RPM very shortly before valves: same callback, no sleep -- the publish
        # ordering on the wire follows the call order here.
        self.bcu_controller_rpm_publisher.publish(rpm_msg)
        self.bcu_valves_publisher.publish(valves_msg)
        self.get_logger().debug(
            f"error={error_pa:.0f} Pa -> pump {pump_rpm} rpm, "
            f"valves {valves_msg.data:#04b} (bit0=motor/valve2, bit1=free/valve1)"
        )

    def destroy_node(self) -> bool:
        # Best-effort safe-stop on teardown so a clean shutdown leaves the
        # pump at 0 / valves closed rather than whatever it last commanded.
        try:
            self._publish_bcu_stop()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    bcu_node = BCUNode()
    spin_node(bcu_node)


if __name__ == "__main__":
    main()
