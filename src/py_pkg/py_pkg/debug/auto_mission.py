"""Boot-time mission autostart: latched MissionCommand + start over the tether.

Fires on launch with no external input -- hence the ``auto_`` prefix and the
``debug/`` home. Publishes one ``MissionCommand`` on ``PATH`` and, after a short
delay, one start
(``std_msgs/Bool`` true) on ``COMMAND``, then stays alive for the rest of the
launch.

Why a long-lived node instead of ``ros2 topic pub --once``: ``PATH`` and
``COMMAND`` both ride ``UUVQoS.COMMAND`` (RELIABLE + TRANSIENT_LOCAL), so a
single publish is latched. But TRANSIENT_LOCAL only keeps that latched sample
alive while the *publisher* is alive -- the moment a ``--once`` process exits,
its sample is gone. A ``pathfinding_node`` that finished DDS discovery after
the publisher had already exited got nothing and sat forever on "waiting for
/path", which intermittently stalled whole sweep runs. Keeping this node up for
the launch lifetime means the latch persists and any late-joining subscriber
still receives both messages.

We publish ``PATH`` exactly once and never on a repeat timer on purpose:
``pathfinding._on_path`` re-creates the mission and resets its state machine on
*every* message, so periodically re-publishing would clobber an already-running
mission back to LOADED. Latch-and-hold delivers the sample to late joiners
without that side effect.

When the launch supplies the scenario's tank endpoints
(``dive_init_tank_empty_pa`` / ``dive_init_tank_full_pa``), one latched
``DiveInit`` goes out before ``PATH`` -- the sim surrogate for the operator
UI's Initialize button, arming ``bcu_node``'s tank-limit clamp through the
same wire path hardware uses. ``surface_pressure_pa`` rides as 0.0, which
``SurfaceReference.register`` rejects, so the standard-atmosphere gauge
reference stays untouched. Defaults (0.0/0.0) publish nothing.
"""

import rclpy
from nautilus_msgs.msg import DiveInit, MissionCommand
from rclpy.node import Node
from std_msgs.msg import Bool

from py_pkg.math_utils import tank_limits_valid
from py_pkg.uuv_ros_core import UUVTopics, create_publisher_for_topic, spin_node


class AutoMission(Node):
    """Publishes a latched MissionCommand on /path then a latched start on /command."""

    def __init__(self) -> None:
        super().__init__("auto_mission")

        self.declare_parameter("mission_id", 1)
        self.declare_parameter("target_pressure_pa", 0.0)
        self.declare_parameter("angle_rad", 0.0)
        self.declare_parameter("n_oscillations", 0)
        self.declare_parameter("dwell_s", 0.0)
        self.declare_parameter("n_steps", 1)
        self.declare_parameter("dive_init_tank_empty_pa", 0.0)
        self.declare_parameter("dive_init_tank_full_pa", 0.0)
        # Gap between /path and /command so the mission is loaded before the
        # start lands. Order isn't strictly required -- pathfinding re-checks
        # its preconditions on every /path and pressure message -- but it keeps
        # the intent obvious and matches the old OnProcessExit sequencing.
        self.declare_parameter("start_delay_s", 2.0)

        mission_id = self.get_parameter("mission_id").get_parameter_value().integer_value
        target_pressure_pa = (
            self.get_parameter("target_pressure_pa").get_parameter_value().double_value
        )
        angle_rad = self.get_parameter("angle_rad").get_parameter_value().double_value
        n_oscillations = (
            self.get_parameter("n_oscillations").get_parameter_value().integer_value
        )
        dwell_s = self.get_parameter("dwell_s").get_parameter_value().double_value
        n_steps = self.get_parameter("n_steps").get_parameter_value().integer_value
        tank_empty_pa = (
            self.get_parameter("dive_init_tank_empty_pa")
            .get_parameter_value()
            .double_value
        )
        tank_full_pa = (
            self.get_parameter("dive_init_tank_full_pa")
            .get_parameter_value()
            .double_value
        )
        start_delay_s = (
            self.get_parameter("start_delay_s").get_parameter_value().double_value
        )

        self._path_pub = create_publisher_for_topic(self, UUVTopics.PATH)
        self._command_pub = create_publisher_for_topic(self, UUVTopics.COMMAND)

        # The publisher must outlive the publish for the latch to persist --
        # created unconditionally so the attribute always exists.
        self._dive_init_pub = create_publisher_for_topic(self, UUVTopics.DIVE_INIT)
        if tank_limits_valid(tank_empty_pa, tank_full_pa):
            init = DiveInit()
            init.surface_pressure_pa = 0.0
            init.tank_empty_pa = float(tank_empty_pa)
            init.tank_full_pa = float(tank_full_pa)
            self._dive_init_pub.publish(init)
            self.get_logger().info(
                f"Published latched DiveInit on {UUVTopics.DIVE_INIT}: "
                f"tank_empty_pa={init.tank_empty_pa}, tank_full_pa={init.tank_full_pa}"
            )

        cmd = MissionCommand()
        cmd.mission_id = int(mission_id)
        cmd.target_pressure_pa = float(target_pressure_pa)
        cmd.angle_rad = float(angle_rad)
        # MissionCommand still carries the pre-rename field name on the wire.
        cmd.n_resurfaces = int(n_oscillations)
        cmd.dwell_s = float(dwell_s)
        cmd.n_steps = int(n_steps)
        self._path_pub.publish(cmd)
        self.get_logger().info(
            f"Published latched MissionCommand on {UUVTopics.PATH}: "
            f"mission_id={cmd.mission_id}, target_pressure_pa={cmd.target_pressure_pa}, "
            f"angle_rad={cmd.angle_rad}, n_oscillations={cmd.n_resurfaces}, "
            f"dwell_s={cmd.dwell_s}, n_steps={cmd.n_steps}"
        )

        # One-shot: a periodic timer we cancel on first fire.
        self._start_timer = self.create_timer(start_delay_s, self._emit_start)

    def _emit_start(self) -> None:
        self._start_timer.cancel()
        msg = Bool()
        msg.data = True
        self._command_pub.publish(msg)
        self.get_logger().info(f"Published latched start (true) on {UUVTopics.COMMAND}.")


def main(args=None):
    rclpy.init(args=args)
    node = AutoMission()
    spin_node(node)


if __name__ == "__main__":
    main()
