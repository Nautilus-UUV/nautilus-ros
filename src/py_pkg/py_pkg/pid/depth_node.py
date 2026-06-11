#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import Pose
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_msgs.msg import Bool, Int16, UInt8

from py_pkg.math_utils import deadband_snap
from py_pkg.physics import SurfaceReference, q_to_rpm
from py_pkg.pid import depth_control_system as ControlSystem
from py_pkg.robot_specs import (
    BCU_DEEP_THRESHOLD_PA,
    BCU_FREE_VALVE_MASK,
    BCU_MOTOR_VALVE_MASK,
)
from py_pkg.scenarios.compile import depth_spec_from_node
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
    spin_node,
)


def select_pump_and_valves(
    current_pressure_pa: float,
    q: float,
    pump_rpm: int,
    deep_threshold_pa: float,
) -> tuple[int, int, int]:
    """Decide what the pump and valves should do for the next control step.

    There's one special case worth pulling out: if we're already deep
    enough that the surrounding water pressure is above
    ``deep_threshold_pa`` AND the controller is asking to go deeper
    still (``q > 0``), we don't run the pump at all. We just open
    valve 1 (the free/bypass way) and let the high ambient pressure
    squeeze oil out of the bladder back into the tank on its own. The
    bladder deflates, the glider displaces less water, and we sink —
    without spending any pump energy.

    Otherwise the rule is simple: when the pump is actually running,
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


def clamp_to_tank_limits(
    pump_rpm: int,
    motor_open: int,
    free_open: int,
    tank_pa: float | None,
    tank_empty_pa: float | None,
    tank_full_pa: float | None,
) -> tuple[int, int, int]:
    """Stop commanding oil flow past a registered tank endpoint.

    The tank runs inverse to the bladder: positive bus RPM inflates the
    bladder and drains the tank toward ``tank_empty_pa``; negative RPM --
    or the passive free-valve vent, where ambient pressure squeezes oil
    back on its own -- fills it toward ``tank_full_pa``. Once the tank
    reading reaches the endpoint the commanded flow is headed for, there
    is no oil left to move (or no room left to put it), so we zero the
    pump and shut both valves rather than dead-head against the stop.

    Only flow TOWARD a touched limit is clamped. A command in the other
    direction passes through untouched on the very next tick -- this is
    a per-tick output clamp, not a latch.

    The limits come from the operator's pre-dive Initialize (DIVE_INIT),
    sampled off the same /bcu/pressure stream this function compares
    against. Until they're registered -- or if a registration is
    nonsense (non-positive, or empty >= full, e.g. from a half-filled
    payload decoding as zeros) -- the clamp stays inert.
    """
    if tank_pa is None or tank_empty_pa is None or tank_full_pa is None:
        return pump_rpm, motor_open, free_open
    if tank_empty_pa <= 0.0 or tank_full_pa <= 0.0 or tank_empty_pa >= tank_full_pa:
        return pump_rpm, motor_open, free_open

    draining_tank = pump_rpm > 0
    filling_tank = pump_rpm < 0 or bool(free_open)
    if draining_tank and tank_pa <= tank_empty_pa:
        return 0, 0, 0
    if filling_tank and tank_pa >= tank_full_pa:
        return 0, 0, 0
    return pump_rpm, motor_open, free_open


class DepthControlNode(Node):
    def __init__(self):
        super().__init__("depth_control_node")

        # Reentrant callback group so subscriptions and the control timer
        # can run concurrently.
        self.callback_group = ReentrantCallbackGroup()

        cfg = depth_spec_from_node(self)
        self.control_system = ControlSystem.DepthControlSystem(cfg)
        self._initial_proportion_full = cfg.plant_model.initial_proportion_full
        self.current_bladder_level = self._initial_proportion_full
        self.bladder_volume = cfg.plant_model.bladder_nominal_m3
        self._min_rpm = cfg.plant_model.min_rpm
        self._min_operating_rpm = cfg.plant_model.min_operating_rpm
        self._max_rpm = cfg.plant_model.max_rpm
        self._pump_efficiency = cfg.plant_model.pump_efficiency
        self.current_time = self.get_clock().now().nanoseconds / 1e9

        self.control_output = 0.0
        self.motor_rpm = 0.0
        # Held None until the first POSITION_TARGET arrives
        self.target_pressure_pa: float | None = None
        self.current_pressure_pa = 0.0

        # Pre-dive registrations (DIVE_INIT). The gauge reference falls
        # back to the standard atmosphere until the operator registers
        # the real surface pressure; the tank limits stay None -- and the
        # output clamp inert -- until both endpoints are registered.
        self._surface_ref = SurfaceReference()
        self._tank_pa: float | None = None
        self._tank_empty_pa: float | None = None
        self._tank_full_pa: float | None = None
        self._tank_clamp_active = False

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
        # Pa (Z-positive-down: deeper = higher gauge pressure). Pose's
        # position.z is reused as a pressure channel — the depth
        # controller's contract; pathfinding/EKF emit accordingly.
        self.target_pose_subscriber = create_subscription_for_topic(
            self,
            UUVTopics.POSITION_TARGET,
            self.target_pose_callback,
            callback_group=self.callback_group,
        )

        self.pressure_external_subscriber = create_subscription_for_topic(
            self,
            UUVTopics.EXTERNAL_PRESSURE,
            self.current_pressure_callback,
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
        # for a debug node); /command=true is a no-op here -- we just wait for
        # pathfinding's next POSITION_TARGET.
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
        self._publish_bcu_stop()

        self.get_logger().info("Depth control node started.")

    def target_pose_callback(self, msg: Pose):
        self.target_pressure_pa = float(msg.position.z)
        self.control_system.target_pressure_pa = self.target_pressure_pa
        self._target_cb_count += 1
        if self._target_cb_count % self._target_log_every_n == 0:
            self.get_logger().info(
                f"Updated target pressure: {self.target_pressure_pa} Pa"
            )

    def current_pressure_callback(self, msg):
        # EXTERNAL_PRESSURE is absolute Pa; the controller works in gauge,
        # referenced to the registered surface pressure (standard
        # atmosphere until the operator initializes).
        self.current_pressure_pa = self._surface_ref.gauge(float(msg.data))
        self.get_logger().debug(
            f"Received current pressure: {self.current_pressure_pa} Pa"
        )

    def _on_tank_pressure(self, msg):
        # Tank pressure in the sensor's own frame (tank relative to hull),
        # the same stream the registered limits were sampled from. Cached
        # here; the clamp decision runs in control_loop.
        self._tank_pa = float(msg.data)

    def _on_dive_init(self, msg):
        self._tank_empty_pa = float(msg.tank_empty_pa)
        self._tank_full_pa = float(msg.tank_full_pa)
        self._surface_ref.register_logged(
            float(msg.surface_pressure_pa), self.get_logger()
        )
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

    def _on_command(self, msg: Bool) -> None:
        # /command=true (start) is a no-op for the controller -- it just waits
        # for pathfinding's next POSITION_TARGET. /command=false (stop) wipes
        # controller state, drops the target, and publishes ONE safe-stop
        # (0 RPM + valves closed) before control_loop goes silent. That single
        # safe-stop is mandatory: the STM has no staleness watchdog and re-ships
        # the last value forever, so going silent without zeroing first would
        # leave the last mission RPM latched on the wire.
        if bool(msg.data):
            return
        self.control_system.reset()
        self.current_bladder_level = self._initial_proportion_full
        self.control_output = 0.0
        self.motor_rpm = 0.0
        self.target_pressure_pa = None
        self._publish_bcu_stop()
        self.get_logger().info(
            "stop -> safe-stop emitted, BCU going silent (fresh state)."
        )

    def control_loop(self):
        if self.target_pressure_pa is None:
            # No target -> go silent. We do NOT stream a zero hold here: the
            # safe-stop is emitted once on the /command=false stop (and once at
            # boot), after which staying off the wire lets a debug node own the
            # BCU with no contention.
            return

        self.current_time = self.get_clock().now().nanoseconds / 1e9
        self.control_output = self.control_system.calc_acc(
            self.current_pressure_pa, self.current_time
        )

        msg = Int16()
        self.motor_rpm = q_to_rpm(
            self.control_output, self.bladder_volume, self._pump_efficiency
        )
        # Shape the RPM through the pump deadband: commands below min_rpm are
        # suppressed to 0, commands between min_rpm and min_operating_rpm are
        # pushed up to the minimum speed the pump runs at reliably, and
        # everything else passes through saturated to +/-max_rpm.
        self.motor_rpm = deadband_snap(
            self.motor_rpm, self._min_rpm, self._min_operating_rpm, self._max_rpm
        )
        # The controller's q sign is opposite the bus convention: q > 0 means
        # descend (deflate the bladder), but on the BCU bus a positive RPM
        # inflates (rise). Negate so a descend command goes out as negative RPM.
        pump_rpm = int(-1 * self.motor_rpm)

        pump_rpm, motor_open, free_open = select_pump_and_valves(
            self.current_pressure_pa,
            self.control_output,
            pump_rpm,
            BCU_DEEP_THRESHOLD_PA,
        )

        # Last word before the wire: don't command oil flow past a
        # registered tank endpoint (inert until the operator initializes).
        clamped = clamp_to_tank_limits(
            pump_rpm,
            motor_open,
            free_open,
            self._tank_pa,
            self._tank_empty_pa,
            self._tank_full_pa,
        )
        clamp_engaged = clamped != (pump_rpm, motor_open, free_open)
        if clamp_engaged != self._tank_clamp_active:
            self._tank_clamp_active = clamp_engaged
            if clamp_engaged:
                self.get_logger().warning(
                    f"tank at a registered limit (tank={self._tank_pa:.0f} Pa, "
                    f"empty={self._tank_empty_pa:.0f}, full={self._tank_full_pa:.0f}) "
                    "-- pump clamped to 0, valves closed"
                )
            else:
                self.get_logger().info("tank limit clamp released")
        pump_rpm, motor_open, free_open = clamped

        msg.data = pump_rpm
        valves_msg = UInt8()
        valves_msg.data = (BCU_MOTOR_VALVE_MASK if motor_open else 0) | (
            BCU_FREE_VALVE_MASK if free_open else 0
        )

        # RPM very shortly before valves: same callback, no sleep — the
        # publish ordering on the wire follows the call order here.
        self.bcu_controller_rpm_publisher.publish(msg)
        self.bcu_valves_publisher.publish(valves_msg)
        self.get_logger().debug(
            f"Fraction of bladder volume filled per second in Hz: {self.control_output}"
        )
        self.get_logger().debug(f"Command to motor in RPM: {self.motor_rpm}")
        self.get_logger().debug(
            f"Valves bitmask (bit0=motor/valve2, bit1=free/valve1): "
            f"{valves_msg.data:#04b}"
        )


def main(args=None):
    rclpy.init(args=args)
    depth_control_node = DepthControlNode()
    spin_node(depth_control_node)


if __name__ == "__main__":
    main()
