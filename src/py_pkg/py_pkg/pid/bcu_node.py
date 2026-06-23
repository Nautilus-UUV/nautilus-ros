#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import Pose
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_msgs.msg import Bool, Int16, UInt8

from py_pkg.math_utils import deadband_snap
from py_pkg.physics import q_to_rpm
from py_pkg.pid import depth_control_system as ControlSystem
from py_pkg.pid.bcu_command_gate import BcuCommandGate
from py_pkg.pid.tank_limit_guard import TankLimitGuard
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
    spin_node,
)

# How long to keep re-asserting the safe-stop after a stop (or at boot)
# before the loop goes silent.
STOP_REASSERT_S = 1.0


def select_pump_and_valves(
    current_pressure_pa: float,
    q: float,
    pump_rpm: int,
    deep_threshold_pa: float,
) -> tuple[int, int, int]:
    """Decide what the pump and valves should do for the next control step.

    If we're already deep enough that the surrounding water pressure is above
    ``deep_threshold_pa`` AND the controller is asking to go deeper
    still (``q > 0``), we just open
    valve 1 (the free/bypass way) and let the ambient pressure
    squeeze oil out of the bladder back into the tank on its own.

    Otherwise: when the pump is actually running,
    valve 2 (the motor way) is open to carry the flow; when the pump is
    idle, both valves stay shut so the bladder holds whatever volume it
    has.

    Returns ``(pump_rpm, motor_open, free_open)`` -- motor_open is bit0
    of the wire bitmask (valve 2), free_open is bit1 (valve 1).
    """
    deep = current_pressure_pa > deep_threshold_pa
    wants_to_descend = q > 0
    if deep and wants_to_descend:
        return 0, 0, 1
    if pump_rpm != 0:
        return pump_rpm, 1, 0
    return pump_rpm, 0, 0


def solve_bcu_command(
    q: float,
    current_pressure_pa: float,
    *,
    bladder_volume_m3: float,
    pump_efficiency: float,
    min_rpm: int,
    min_operating_rpm: int,
    max_rpm: int,
    deep_threshold_pa: float = BCU_DEEP_THRESHOLD_PA,
) -> tuple[int, int, int]:
    """Turn a controller flow demand ``q`` into a raw BCU wire command.

    The bladder-actuation path as one functional chain:

        q  --q_to_rpm-->            motor RPM (sign carries the fill direction)
           --deadband_snap-->       motor RPM, snapped out of the dead pump band
           --negate-->              pump bus RPM (the bus runs inverse to fill)
           --select_pump_and_valves--> (pump_rpm, motor_open, free_open)

    ``q`` is the fraction of bladder volume to move per second; positive fills
    the bladder (sink). Returns ``(pump_rpm, motor_open, free_open)`` ready for
    the wire -- motor_open is valve 2 (bit0), free_open is valve 1 (bit1). The
    tank-endpoint cutoff is applied downstream by ``TankLimitGuard``, which
    needs the raw direction this returns, so it is not folded in here.
    """
    motor_rpm = deadband_snap(
        q_to_rpm(q, bladder_volume_m3, pump_efficiency),
        min_rpm,
        min_operating_rpm,
        max_rpm,
    )
    pump_rpm = int(-motor_rpm)
    return select_pump_and_valves(
        current_pressure_pa, q, pump_rpm, deep_threshold_pa
    )


class BCUNode(Node):
    def __init__(self):
        super().__init__("bcu_node")

        # Reentrant callback group so subscriptions and the control timer
        # can run concurrently.
        self.callback_group = ReentrantCallbackGroup()

        cfg = bcu_spec_from_node(self)
        self.control_system = ControlSystem.DepthControlSystem(cfg)
        self._initial_proportion_full = cfg.plant_model.initial_proportion_full
        self.current_bladder_level = self._initial_proportion_full
        self.bladder_volume = cfg.plant_model.bladder_nominal_m3
        self._min_rpm = cfg.plant_model.min_rpm
        self._min_operating_rpm = cfg.plant_model.min_operating_rpm
        self._max_rpm = cfg.plant_model.max_rpm
        self._pump_efficiency = cfg.plant_model.pump_efficiency

        # Latching tank-endpoint cutoff: holds the pump/valves stopped once the
        # tank is parked at a limit, with release hysteresis so sensor noise on
        # the guard threshold can't chatter the valves (the bench-test bug).
        self._tank_guard = TankLimitGuard(
            stop_band=cfg.tank_stop_band,
            release_band=cfg.tank_release_band,
        )

        # Anti-chatter gate between solve_bcu_command and the wire: error
        # deadband + arm/disarm hysteresis + minimum valve dwell.
        self._gate = BcuCommandGate(
            error_arm_pa=cfg.error_arm_pa,
            error_disarm_pa=cfg.error_disarm_pa,
            min_valve_dwell_s=cfg.min_valve_dwell_s,
        )

        # Safe-stop re-assert burst: how many idle ticks still publish 0 RPM
        # + valves closed before the loop goes silent. Decremented in
        # control_loop while target is None; topped up by _begin_safe_stop.
        self._stop_reassert_count = max(1, round(cfg.frequency_hz * STOP_REASSERT_S))
        self._stop_reassert_remaining = 0

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
            self, UUVTopics.BCU_RPM, callback_group=self.callback_group
        )

        self.bcu_valves_publisher = create_publisher_for_topic(
            self, UUVTopics.BCU_VALVES, callback_group=self.callback_group
        )

        # Pressure setpoint is `position.z` of POSITION_TARGET, in gauge
        # Pa (Z-positive-down: deeper = higher gauge pressure).
        self.target_pose_subscriber = create_subscription_for_topic(
            self,
            UUVTopics.POSITION_TARGET,
            self.target_pose_callback,
            callback_group=self.callback_group,
        )

        # Depth measurement rides POSITION_ESTIMATION.position.z (gauge Pa,
        # Z-positive-down), already gauged by attitude_node.
        self.position_estimation_subscriber = create_subscription_for_topic(
            self,
            UUVTopics.POSITION_ESTIMATION,
            self.current_pose_callback,
            callback_group=self.callback_group,
        )

        # Tank pressure feeds the output clamp; the dive registration
        # carries the limits it compares against (plus the surface
        # pressure for the gauge reference).
        self.tank_pressure_subscriber = create_subscription_for_topic(
            self,
            UUVTopics.BCU_PRESSURE,
            self._on_tank_pressure,
            callback_group=self.callback_group,
        )

        self.dive_init_subscriber = create_subscription_for_topic(
            self,
            UUVTopics.DIVE_INIT,
            self._on_dive_init,
            callback_group=self.callback_group,
        )

        # The mission run/stop signal. On /command=false the controller drops
        # its target, emits one safe-stop and goes silent (freeing the BCU wire
        # for a debug node).
        self.command_subscriber = create_subscription_for_topic(
            self,
            UUVTopics.COMMAND,
            self._on_command,
            callback_group=self.callback_group,
        )

        self.control_timer = self.create_timer(
            1.0 / cfg.frequency_hz,
            self.control_loop,
            callback_group=self.callback_group,
        )

        # Leave the BCU wire unambiguously at 0/closed at boot.
        self._begin_safe_stop()

        self.get_logger().info("Depth control node started.")

    def target_pose_callback(self, msg: Pose):
        self.target_pressure_pa = float(msg.position.z)
        self.control_system.target_pressure_pa = self.target_pressure_pa
        self._target_cb_count += 1
        if self._target_cb_count % self._target_log_every_n == 0:
            self.get_logger().info(
                f"Updated target pressure: {self.target_pressure_pa} Pa"
            )

    def current_pose_callback(self, msg: Pose):
        # position.z is the depth measurement in gauge Pa.
        self.current_pressure_pa = float(msg.position.z)
        self.get_logger().debug(
            f"Received current pressure: {self.current_pressure_pa} Pa"
        )

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
        # Mirrors the clamp's own sanity check.
        limits_ok = 0.0 < self._tank_empty_pa < self._tank_full_pa
        log = self.get_logger().info if limits_ok else self.get_logger().error
        log(
            "dive init: "
            f"tank empty/full = {self._tank_empty_pa:.0f}/{self._tank_full_pa:.0f} Pa "
            f"({'ok' if limits_ok else 'invalid -- clamp stays inert'})"
        )

    def _publish_bcu_stop(self) -> None:
        zero_rpm = Int16()
        zero_rpm.data = 0
        self.bcu_controller_rpm_publisher.publish(zero_rpm)
        valves_off = UInt8()
        valves_off.data = 0
        self.bcu_valves_publisher.publish(valves_off)

    def _begin_safe_stop(self) -> None:
        # Emit one safe-stop now and arm the re-assert burst so control_loop
        # keeps publishing 0/closed for ~STOP_REASSERT_S.
        self._publish_bcu_stop()
        self._stop_reassert_remaining = self._stop_reassert_count

    def _on_command(self, msg: Bool) -> None:
        # /command=true (start) is a no-op for the controller
        if bool(msg.data):
            return
        self.control_system.reset()
        self._gate.reset()
        self._tank_guard.reset()
        self.current_bladder_level = self._initial_proportion_full
        self.target_pressure_pa = None
        self._begin_safe_stop()
        self.get_logger().info(
            "stop -> safe-stop burst emitted, BCU going silent (fresh state)."
        )

    def control_loop(self):
        if self.target_pressure_pa is None:
            # No target. Keep re-asserting the safe-stop for the burst window
            # so the STM latches the zero, then go silent so a debug node can
            # own the BCU wire with no contention.
            if self._stop_reassert_remaining > 0:
                self._publish_bcu_stop()
                self._stop_reassert_remaining -= 1
            return

        # The PID is the only stateful step: it integrates over time to turn the
        # current pressure into a flow demand q. Everything downstream is the
        # pure solve_bcu_command pipeline.
        now = self.get_clock().now().nanoseconds / 1e9
        q = self.control_system.calc_acc(self.current_pressure_pa, now)

        pump_rpm, motor_open, free_open = solve_bcu_command(
            q,
            self.current_pressure_pa,
            bladder_volume_m3=self.bladder_volume,
            pump_efficiency=self._pump_efficiency,
            min_rpm=self._min_rpm,
            min_operating_rpm=self._min_operating_rpm,
            max_rpm=self._max_rpm,
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

        # Anti-chatter conditioning: deadband + hysteresis on the depth error,
        # and a minimum valve dwell. Holds the pump idle near the setpoint and
        # stops the valves toggling every tick.
        error_pa = self.target_pressure_pa - self.current_pressure_pa
        pump_rpm, motor_open, free_open = self._gate.apply(
            error_pa, pump_rpm, motor_open, free_open, now_s=now
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
            f"q={q:.4f} Hz -> pump {pump_rpm} rpm, "
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
