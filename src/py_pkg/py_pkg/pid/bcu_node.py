#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from std_msgs.msg import Bool, Int16, UInt8

from py_pkg.math_utils import deadband_snap, span_band_guards, tank_limits_valid
from py_pkg.physics import q_to_rpm
from py_pkg.robot_specs import (
    BCU_DEEP_THRESHOLD_PA,
    BCU_FREE_VALVE_MASK,
    BCU_MOTOR_VALVE_MASK,
)
from py_pkg.scenarios.compile import bcu_spec_from_node
from py_pkg.utils_controls import PIDController
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
    now_s,
    spin_node,
)


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

    The tank runs inverse to the bladder: positive bus RPM inflates the
    bladder and drains the tank toward ``tank_empty_pa``; negative RPM --
    -- fills it toward ``tank_full_pa``. The last stretch of
    travel is just dead-heads the pump against a tank that's effectively full/empty,
    so we quit early.

    The limits come from the pre-dive Initialize (DIVE_INIT).
    """
    if tank_pa is None or not tank_limits_valid(tank_empty_pa, tank_full_pa):
        return pump_rpm, motor_open, free_open

    low_guard, high_guard = span_band_guards(tank_empty_pa, tank_full_pa, band)
    draining_tank = pump_rpm > 0
    filling_tank = pump_rpm < 0 or bool(free_open)
    if draining_tank and tank_pa <= low_guard:
        return 0, 0, 0
    if filling_tank and tank_pa >= high_guard:
        return 0, 0, 0
    return pump_rpm, motor_open, free_open


def solve_bcu_command(
    q: float,
    current_pressure_pa: float,
    tank_pa: float | None,
    tank_empty_pa: float | None,
    tank_full_pa: float | None,
    *,
    bladder_volume_m3: float,
    pump_efficiency: float,
    min_rpm: int,
    min_operating_rpm: int,
    max_rpm: int,
    deep_threshold_pa: float = BCU_DEEP_THRESHOLD_PA,
) -> tuple[int, int, int]:
    """Turn a controller flow demand ``q`` into a BCU wire command.

    The whole bladder-actuation path as one functional chain:

        q  --q_to_rpm-->            motor RPM (sign carries the fill direction)
           --deadband_snap-->       motor RPM, snapped out of the dead pump band
           --negate-->              pump bus RPM (the bus runs inverse to fill)
           --select_pump_and_valves--> (pump_rpm, motor_open, free_open)
           --clamp_to_tank_limits-->   same triple, zeroed near a tank endpoint

    ``q`` is the fraction of bladder volume to move per second; positive fills
    the bladder (sink). Returns ``(pump_rpm, motor_open, free_open)`` ready for
    the wire -- motor_open is valve 2 (bit0), free_open is valve 1 (bit1).
    """
    motor_rpm = deadband_snap(
        q_to_rpm(q, bladder_volume_m3, pump_efficiency),
        min_rpm,
        min_operating_rpm,
        max_rpm,
    )
    pump_rpm = int(-motor_rpm)
    pump_rpm, motor_open, free_open = select_pump_and_valves(
        current_pressure_pa, q, pump_rpm, deep_threshold_pa
    )
    return clamp_to_tank_limits(
        pump_rpm, motor_open, free_open, tank_pa, tank_empty_pa, tank_full_pa
    )


class BCUNode(Node):
    def __init__(self):
        super().__init__("bcu_node")

        cfg = bcu_spec_from_node(self)
        # Outer depth loop: a single PID tracks pressure entirely in gauge Pa
        # and outputs a bladder flow ratio q in 1/s. Positive q = deflate the
        # bladder, sink; negative q = inflate, rise. A single PID is enough
        # because flow -> volume -> buoyancy -> depth behaves like a damped
        # double integrator. The q -> pump RPM wire conversion (with its sign
        # flip: positive bus RPM inflates) lives in solve_bcu_command.
        pp = cfg.pid_pressure
        self.pid_pressure = PIDController(
            kp=pp.kp,
            ki=pp.ki,
            kd=pp.kd,
            integral_limits=pp.integral_limits,
            output_limits=pp.output_limits,
            derivative_filter=pp.derivative_filter,
        )
        self.bladder_volume = cfg.plant_model.bladder_nominal_m3
        self._min_rpm = cfg.plant_model.min_rpm
        self._min_operating_rpm = cfg.plant_model.min_operating_rpm
        self._max_rpm = cfg.plant_model.max_rpm
        self._pump_efficiency = cfg.plant_model.pump_efficiency

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

        # Tank pressure feeds the output clamp; the dive registration
        # carries the limits it compares against (plus the surface
        # pressure for the gauge reference).
        create_subscription_for_topic(
            self, UUVTopics.BCU_PRESSURE, self._on_tank_pressure
        )

        create_subscription_for_topic(self, UUVTopics.DIVE_INIT, self._on_dive_init)

        # The mission run/stop signal. On /command=false the controller drops
        # its target, emits one safe-stop and goes silent (freeing the BCU wire
        # for a debug node).
        create_subscription_for_topic(self, UUVTopics.COMMAND, self._on_command)

        self.control_timer = self.create_timer(
            1.0 / cfg.frequency_hz, self.control_loop
        )

        # Leave the BCU wire unambiguously at 0/closed at boot.
        self._publish_bcu_stop()

        self.get_logger().info("Depth control node started.")

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
        # Same predicate the clamp itself gates on.
        limits_ok = tank_limits_valid(self._tank_empty_pa, self._tank_full_pa)
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
        # the last value forever..
        if bool(msg.data):
            return
        self.pid_pressure.reset()
        self.target_pressure_pa = None
        self._publish_bcu_stop()
        self.get_logger().info(
            "stop -> safe-stop emitted, BCU going silent (fresh state)."
        )

    def control_loop(self):
        if self.target_pressure_pa is None:
            # No target -> go silent. Lets a debug node own the
            # BCU with no contention.
            return

        # The PID is the only stateful step: it integrates over time to turn the
        # current pressure into a flow demand q. Everything downstream is the
        # pure solve_bcu_command pipeline.
        q = self.pid_pressure.update(
            self.target_pressure_pa, self.current_pressure_pa, now_s(self)
        )

        pump_rpm, motor_open, free_open = solve_bcu_command(
            q,
            self.current_pressure_pa,
            self._tank_pa,
            self._tank_empty_pa,
            self._tank_full_pa,
            bladder_volume_m3=self.bladder_volume,
            pump_efficiency=self._pump_efficiency,
            min_rpm=self._min_rpm,
            min_operating_rpm=self._min_operating_rpm,
            max_rpm=self._max_rpm,
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


def main(args=None):
    rclpy.init(args=args)
    bcu_node = BCUNode()
    spin_node(bcu_node)


if __name__ == "__main__":
    main()
