#!/usr/bin/env python3
"""Attitude Control Unit node: bang-bang pitch + PID roll.

DEPRECATED / NOT IMPLEMENTED IN SIM: the simulated ACU actuator has been
removed — the glider_nautilus model is now a static, symmetric, BCU-only
vehicle (the pitch/roll mass-shifter joints are fixed and the sim ACU bridge
is gone). This node is retained for the real-hardware path only; it still
drives ``/acu/pitch`` / ``/acu/roll`` into can_com_node, but those outputs
have no consumer in simulation.
"""

import math

import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from std_msgs.msg import Int16

from py_pkg.math_utils import quaternion_to_roll_pitch
from py_pkg.control.acu_axis_controller import AxisController
from py_pkg.control.run_end import subscribe_run_end
from py_pkg.robot_specs import ACU_PITCH_MM_PER_M, ACU_ROLL_CDEG_PER_DEG
from py_pkg.scenarios.compile import acu_pitch_spec_from_node, acu_roll_spec_from_node
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
    now_s,
    spin_node,
)


class ACUControlNode(Node):
    def __init__(self):
        super().__init__("acu_control_node")

        # Bang-bang extremes on the wire. The pitch spec gives us the
        # operational soft-saturation in metres; the wire format is mm.
        # "Front" is the most negative end of the stroke (mass forward),
        # "back" is the least negative (mass aft).
        pitch_cfg = acu_pitch_spec_from_node(self)
        self._acu_pitch_back_mm = int(
            round(pitch_cfg.output_limits[0] * ACU_PITCH_MM_PER_M)
        )
        self._acu_pitch_front_mm = int(
            round(pitch_cfg.output_limits[1] * ACU_PITCH_MM_PER_M)
        )

        roll_cfg = acu_roll_spec_from_node(self)
        self.roll_axis = AxisController(
            kp=roll_cfg.kp,
            ki=roll_cfg.ki,
            kd=roll_cfg.kd,
            command_tolerance=roll_cfg.command_tolerance,
            integral_limits=roll_cfg.integral_limits,
            output_limits=roll_cfg.output_limits,
            derivative_filter=roll_cfg.derivative_filter,
        )

        # Bang-bang state: gate the first command on having both a
        # depth reading and a setpoint, so we don't pick a side from
        # uninitialized zeros. Depth (gauge Pa) arrives on
        # POSITION_ESTIMATION.position.z, already gauged by attitude_node.
        self.current_pressure_pa: float | None = None
        self.target_pressure_pa: float | None = None

        # Roll PID state.
        self.target_roll_deg = 0.0
        self.current_roll_deg = 0.0

        self.pitch_pub = create_publisher_for_topic(self, UUVTopics.ACU_PITCH)
        self.roll_pub = create_publisher_for_topic(self, UUVTopics.ACU_ROLL)

        create_subscription_for_topic(
            self, UUVTopics.POSITION_TARGET, self.target_pose_callback
        )
        # Single vehicle-state input: roll (off the quaternion) for the roll PID
        # and gauge depth (position.z) for the bang-bang pitch leg select.
        create_subscription_for_topic(
            self, UUVTopics.POSITION_ESTIMATION, self.current_pose_callback
        )

        # Either way a run ends, the controller drops its target, neutralizes
        # the ACU once, and gates control_loop off (silent) so acu_debug can own
        # the wire.
        subscribe_run_end(self, self._stop)

        self.control_timer = self.create_timer(
            1.0 / roll_cfg.frequency_hz, self.control_loop
        )

        # Leave the ACU wire at neutral at boot, before any target arrives.
        self._publish_acu_neutral()

        self.get_logger().info("ACU control node started (bang-bang pitch, PID roll).")

    def target_pose_callback(self, msg: Pose):
        # POSITION_TARGET.position.z carries the target pressure in Pa
        # (pathfinding's TRIM convention). Roll comes off the quaternion
        # the same way as before.
        self.target_pressure_pa = float(msg.position.z)
        q = msg.orientation
        roll, _ = quaternion_to_roll_pitch(q.x, q.y, q.z, q.w)
        self.target_roll_deg = math.degrees(roll)

    def current_pose_callback(self, msg: Pose):
        # Roll feeds the roll PID; gauge depth (position.z) feeds the bang-bang
        # pitch leg select. Pitch off the pose is deliberately ignored.
        q = msg.orientation
        roll, _ = quaternion_to_roll_pitch(q.x, q.y, q.z, q.w)
        self.current_roll_deg = math.degrees(roll)
        self.current_pressure_pa = float(msg.position.z)

    def _publish_acu_neutral(self) -> None:
        pitch = Int16()
        pitch.data = 0
        self.pitch_pub.publish(pitch)
        roll = Int16()
        roll.data = 0
        self.roll_pub.publish(roll)

    def _stop(self, reason: str) -> None:
        """Drop the target, wipe the roll PID, emit ONE neutral, go silent.

        The single neutral matters because the STM re-ships the last value
        forever (no staleness watchdog), so silence alone would leave the last
        mission attitude latched on the wire.
        """
        # Edge-trigger, mirroring bcu_node: only the running->stopped
        # transition emits. target_pressure_pa is None already means "stopped",
        # and the ACU wire has no arbiter -- last writer wins at the STM. The UI
        # sends /command=false before EVERY manual command, so without this
        # guard each one would drop a neutral on top of the operator's
        # pitch/roll and undo the command they just sent. (The BCU needs the
        # same guard for a different reason -- to avoid re-arming its safe-stop
        # burst -- but the failure mode here is the more direct one: a
        # controller that is not running is still overwriting the wire.)
        if self.target_pressure_pa is None:
            return
        self.roll_axis.reset()
        self.target_pressure_pa = None
        self.target_roll_deg = 0.0
        self.current_roll_deg = 0.0
        self._publish_acu_neutral()
        self.get_logger().info(
            f"{reason} -> neutral emitted, ACU going silent (fresh state)."
        )

    def control_loop(self):
        if self.target_pressure_pa is None:
            # No active mission target -> stay off /acu/pitch and /acu/roll
            # entirely (roll PID included) so acu_debug can own the wire after a
            # stop. The one neutral sample was already emitted on stop/boot.
            # (Mirrors bcu_node's no-target gate -- same None sentinel.)
            return
        self._update_pitch()
        self._update_roll()

    def _update_pitch(self):
        if self.current_pressure_pa is None or self.target_pressure_pa is None:
            return
        diving = self.current_pressure_pa < self.target_pressure_pa
        msg = Int16()
        msg.data = self._acu_pitch_back_mm if diving else self._acu_pitch_front_mm
        self.pitch_pub.publish(msg)

    def _update_roll(self):
        self.roll_axis.update_sensor(self.current_roll_deg)
        roll_cmd = self.roll_axis.update(self.target_roll_deg, now_s(self))
        if roll_cmd is not None:
            msg = Int16()
            msg.data = int(round(roll_cmd * ACU_ROLL_CDEG_PER_DEG))
            self.roll_pub.publish(msg)

    def destroy_node(self) -> bool:
        # Best-effort neutral on teardown, mirroring bcu_node's safe-stop. The
        # STM latches the last value it was sent and has no staleness watchdog,
        # so a node that exits mid-mission -- Ctrl-C, a launch shutdown, a
        # crash-restart -- would otherwise leave the mass shifter parked at the
        # last commanded pitch/roll with nothing left running to move it.
        # Unconditional (not edge-triggered): _stop's guard exists to avoid
        # fighting a live manual driver, but on teardown there is no later
        # command from us to fight with.
        try:
            self._publish_acu_neutral()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ACUControlNode()
    spin_node(node)


if __name__ == "__main__":
    main()
