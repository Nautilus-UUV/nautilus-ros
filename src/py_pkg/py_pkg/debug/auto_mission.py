"""Boot-time mission autostart: latched MissionCommand + start over the tether.

Fires on launch with no external input -- hence the ``auto_`` prefix and the
``debug/`` home next to ``auto_bcu_oscillator``. Publishes one
``MissionCommand`` on ``PATH`` and, after a short delay, one ``start`` on
``COMMAND``, then stays alive for the rest of the launch.

Why a long-lived node instead of ``ros2 topic pub --once``: ``PATH`` and
``COMMAND`` both ride ``UUVQoS.COMMAND`` (RELIABLE + TRANSIENT_LOCAL), so a
single publish is latched. But TRANSIENT_LOCAL only keeps that latched sample
alive while the *publisher* is alive -- the moment a ``--once`` process exits,
its sample is gone. A ``pathfinding_node`` that finished DDS discovery after
the publisher had already exited got nothing and sat forever on "waiting for
/path", which intermittently stalled whole sweep runs. Keeping this node up for
the launch lifetime means the latch persists and any late-joining subscriber
still receives both messages. This mirrors how ``auto_bcu_oscillator`` holds
``CONTROL_MANUAL_OVERRIDE`` latched.

We publish ``PATH`` exactly once and never on a repeat timer on purpose:
``pathfinding._on_path`` re-creates the mission and resets its state machine on
*every* message, so periodically re-publishing would clobber an already-running
mission back to LOADED. Latch-and-hold delivers the sample to late joiners
without that side effect.
"""

import rclpy
from nautilus_msgs.msg import MissionCommand
from rclpy.node import Node
from std_msgs.msg import String

from py_pkg.uuv_ros_core import UUVTopics, create_publisher_for_topic, spin_node


class AutoMission(Node):
    """Publishes a latched MissionCommand on /path then a latched start on /command."""

    def __init__(self) -> None:
        super().__init__("auto_mission")

        self.declare_parameter("mission_id", 1)
        self.declare_parameter("target_pressure_pa", 0.0)
        self.declare_parameter("angle_rad", 0.0)
        self.declare_parameter("n_resurfaces", 0)
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
        n_resurfaces = (
            self.get_parameter("n_resurfaces").get_parameter_value().integer_value
        )
        start_delay_s = (
            self.get_parameter("start_delay_s").get_parameter_value().double_value
        )

        self._path_pub = create_publisher_for_topic(self, UUVTopics.PATH)
        self._command_pub = create_publisher_for_topic(self, UUVTopics.COMMAND)

        cmd = MissionCommand()
        cmd.mission_id = int(mission_id)
        cmd.target_pressure_pa = float(target_pressure_pa)
        cmd.angle_rad = float(angle_rad)
        cmd.n_resurfaces = int(n_resurfaces)
        self._path_pub.publish(cmd)
        self.get_logger().info(
            f"Published latched MissionCommand on {UUVTopics.PATH}: "
            f"mission_id={cmd.mission_id}, target_pressure_pa={cmd.target_pressure_pa}, "
            f"angle_rad={cmd.angle_rad}, n_resurfaces={cmd.n_resurfaces}"
        )

        # One-shot: a periodic timer we cancel on first fire.
        self._start_timer = self.create_timer(start_delay_s, self._emit_start)

    def _emit_start(self) -> None:
        self._start_timer.cancel()
        msg = String()
        msg.data = "start"
        self._command_pub.publish(msg)
        self.get_logger().info(f"Published latched 'start' on {UUVTopics.COMMAND}.")


def main(args=None):
    rclpy.init(args=args)
    node = AutoMission()
    spin_node(node)


if __name__ == "__main__":
    main()
