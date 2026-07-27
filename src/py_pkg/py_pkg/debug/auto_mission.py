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

With ``wait_for_sim_ready`` true (the sim launches pass it), the whole
sequence additionally holds until the ``nautilus_hal`` sim_ready_gate's
latched ``/sim/ready`` -- the world is spawned paused and only unpaused
once every required node is discovered, so the mission can never start
against a half-built graph or a still-falling vehicle.
"""

import rclpy
from nautilus_msgs.msg import DiveInit, MissionCommand
from rclpy.node import Node
from std_msgs.msg import Bool

from py_pkg.debug.mission_fields import MISSION_FIELDS
from py_pkg.math_utils import tank_limits_valid
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
    spin_node,
)

# The mission fields live in a ROS-free leaf module so the launch file can
# import the table without dragging rclpy and the message registry into the
# launch process. See py_pkg/debug/mission_fields.py.


class AutoMission(Node):
    """Publishes a latched MissionCommand on /path then a latched start on /command."""

    def __init__(self) -> None:
        super().__init__("auto_mission")

        for param, _field, _cast, default in MISSION_FIELDS:
            self.declare_parameter(param, default)
        self.declare_parameter("dive_init_tank_empty_pa", 0.0)
        self.declare_parameter("dive_init_tank_full_pa", 0.0)
        # Gap between /path and /command so the mission is loaded before the
        # start lands. Order isn't strictly required -- pathfinding re-checks
        # its preconditions on every /path and pressure message -- but it keeps
        # the intent obvious and matches the old OnProcessExit sequencing.
        self.declare_parameter("start_delay_s", 2.0)
        # Hold the whole mission sequence until the sim_ready_gate's
        # latched /sim/ready lands (paused-spawn bringup: the world is
        # only unpaused once every required node is discovered). False =
        # publish on the legacy fixed timers — for compositions without
        # the gate.
        self.declare_parameter("wait_for_sim_ready", False)

        self._path_pub = create_publisher_for_topic(self, UUVTopics.PATH)
        self._command_pub = create_publisher_for_topic(self, UUVTopics.COMMAND)
        # The publisher must outlive the publish for the latch to persist --
        # created unconditionally so the attribute always exists.
        self._dive_init_pub = create_publisher_for_topic(self, UUVTopics.DIVE_INIT)

        # Build the latched messages here, where the parameters are read.
        # _publish_mission then only puts them on the wire -- either now or
        # when /sim/ready lands -- so deferring the sequence needs no
        # parallel copy of the mission fields.
        tank_empty_pa = self.get_parameter("dive_init_tank_empty_pa").value
        tank_full_pa = self.get_parameter("dive_init_tank_full_pa").value
        self._dive_init: DiveInit | None = None
        if tank_limits_valid(tank_empty_pa, tank_full_pa):
            self._dive_init = DiveInit()
            self._dive_init.surface_pressure_pa = 0.0
            self._dive_init.tank_empty_pa = float(tank_empty_pa)
            self._dive_init.tank_full_pa = float(tank_full_pa)

        self._cmd = MissionCommand()
        for param, field, cast, _default in MISSION_FIELDS:
            setattr(self._cmd, field, cast(self.get_parameter(param).value))

        self._start_delay_s = self.get_parameter("start_delay_s").value
        self._published = False
        self._ready_sub = None
        if self.get_parameter("wait_for_sim_ready").value:
            # COMMAND-profile latch: subscribing after the gate opened
            # still delivers the ready sample.
            self._ready_sub = create_subscription_for_topic(
                self, UUVTopics.SIM_READY, self._on_sim_ready
            )
            self.get_logger().info(
                f"auto_mission: holding mission until {UUVTopics.SIM_READY}"
            )
        else:
            self._publish_mission()

    def _on_sim_ready(self, msg: Bool) -> None:
        if msg.data and not self._published:
            self._publish_mission()

    def _publish_mission(self) -> None:
        self._published = True
        if self._dive_init is not None:
            self._dive_init_pub.publish(self._dive_init)
            self.get_logger().info(
                f"Published latched DiveInit on {UUVTopics.DIVE_INIT}: "
                f"tank_empty_pa={self._dive_init.tank_empty_pa}, "
                f"tank_full_pa={self._dive_init.tank_full_pa}"
            )

        cmd = self._cmd
        self._path_pub.publish(cmd)
        # Logged under the operator-facing parameter names, from the same table
        # that filled the message -- so a new field shows up here for free.
        fields = ", ".join(
            f"{param}={getattr(cmd, field)}"
            for param, field, _cast, _default in MISSION_FIELDS
        )
        self.get_logger().info(
            f"Published latched MissionCommand on {UUVTopics.PATH}: {fields}"
        )

        # One-shot: a periodic timer we cancel on first fire.
        self._start_timer = self.create_timer(self._start_delay_s, self._emit_start)

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
