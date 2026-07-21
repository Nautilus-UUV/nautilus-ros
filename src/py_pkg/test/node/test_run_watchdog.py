"""Tier 2 in-process rclpy test for RunWatchdogNode.

Spins the node alongside a tester that feeds /command, /mission/complete, and
synthetic ground-truth odometry on the node's *default* odom topic (so the
default stays exercised). The node's exit_fn is a recorder instead of
sys.exit -- the conclude path must be observable without SystemExit killing
pytest -- and the deadlines/graces are shrunk to wall-clock test scale via
parameter overrides.
"""

import json
import time

import pytest
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool

from py_pkg.uuv_ros_core import UUVQoS, UUVTopics, create_publisher_for_topic
from py_pkg.watchdog.run_watchdog_node import (
    EXIT_ABORT_FLOATER,
    EXIT_MISSION_COMPLETE,
    RunWatchdogNode,
)

from conftest import NodeHarness

# The node's default odom_topic: bridge.launch.py's ground-truth record
# topic f"/model/{model_name}/odometry" at the canonical model name.
# Deliberately not overridden, so a drifted default breaks this file.
ODOM_TOPIC = "/model/glider_nautilus/odometry"

Z_SPAWN = -5.0
DIVE_DEADLINE_S = 0.5
STALL_GRACE_S = 0.5
GRACE_S = 0.1


class _WatchdogTesterNode(Node):
    """Feeds /command, /mission/complete, and ground-truth odometry."""

    def __init__(self):
        super().__init__("run_watchdog_tester")
        self.command_pub = create_publisher_for_topic(self, UUVTopics.COMMAND)
        self.complete_pub = create_publisher_for_topic(self, UUVTopics.MISSION_COMPLETE)
        # Same sensor-data QoS the node subscribes with.
        self.odom_pub = self.create_publisher(
            Odometry, ODOM_TOPIC, UUVQoS.SENSOR_STREAM
        )

    def publish_command(self, start: bool) -> None:
        msg = Bool()
        msg.data = bool(start)
        self.command_pub.publish(msg)

    def publish_complete(self) -> None:
        msg = Bool()
        msg.data = True
        self.complete_pub.publish(msg)

    def publish_odom_z(self, z_m: float) -> None:
        msg = Odometry()
        msg.pose.pose.position.z = float(z_m)
        self.odom_pub.publish(msg)


class WatchdogHarness(NodeHarness):
    """NodeHarness specialised for RunWatchdogNode: exit recording + odom
    feeding on top of the generic scaffolding."""

    def __init__(self, verdict_path: str):
        self.exits: list[int] = []
        super().__init__(
            lambda: RunWatchdogNode(
                exit_fn=self.exits.append,
                parameter_overrides=[
                    Parameter(
                        "dive_deadline_s", Parameter.Type.DOUBLE, DIVE_DEADLINE_S
                    ),
                    Parameter("stall_grace_s", Parameter.Type.DOUBLE, STALL_GRACE_S),
                    Parameter("verdict_path", Parameter.Type.STRING, verdict_path),
                    Parameter("complete_grace_s", Parameter.Type.DOUBLE, GRACE_S),
                    Parameter("abort_grace_s", Parameter.Type.DOUBLE, GRACE_S),
                ],
            ),
            _WatchdogTesterNode,
        )

    def feed_surface_odom_until(self, predicate, timeout: float) -> None:
        """Publish surface-bobbing odometry (never dives) while spinning,
        until ``predicate()`` is truthy or ``timeout`` elapses."""
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.tester.publish_odom_z(Z_SPAWN)
            self.spin_for(0.05)


@pytest.fixture
def make_harness():
    """Factory so each test picks its own verdict_path (including "")."""
    harnesses: list[WatchdogHarness] = []

    def _make(verdict_path: str) -> WatchdogHarness:
        harness = WatchdogHarness(verdict_path)
        harnesses.append(harness)
        return harness

    try:
        yield _make
    finally:
        for harness in harnesses:
            harness.shutdown()


class TestRunWatchdog:
    def test_floater_verdict_writes_json_and_exits(self, make_harness, tmp_path):
        path = tmp_path / "run_verdict.json"
        h = make_harness(str(path))
        h.publish_command(True)
        h.feed_surface_odom_until(lambda: h.exits, timeout=4.0)

        assert h.exits == [EXIT_ABORT_FLOATER]
        data = json.loads(path.read_text())
        assert set(data) == {"verdict", "t_wall_s", "dive_m", "drawup_m"}
        assert data["verdict"] == "abort_floater"
        assert data["dive_m"] < 2.0
        assert data["drawup_m"] < 1.0

    def test_complete_beats_abort(self, make_harness, tmp_path):
        path = tmp_path / "run_verdict.json"
        h = make_harness(str(path))
        h.publish_command(True)
        # Completion lands first (latched, so discovery can't drop it) ...
        h.publish_complete()
        h.spin_until(lambda: h.node._concluded is not None, timeout=2.0)
        # ... then floater-grade odometry past the deadline cannot override
        # the concluded verdict: first verdict wins, exit code stays 0.
        h.feed_surface_odom_until(lambda: h.exits, timeout=4.0)

        assert h.exits == [EXIT_MISSION_COMPLETE]
        assert json.loads(path.read_text())["verdict"] == "mission_complete"

    def test_armed_only_after_command_true(self, make_harness, tmp_path):
        path = tmp_path / "run_verdict.json"
        h = make_harness(str(path))
        # No /command yet: surface odometry well past the dive deadline must
        # not produce a verdict (the plausibility clock has not started).
        h.feed_surface_odom_until(lambda: False, timeout=2 * DIVE_DEADLINE_S)
        assert not h.exits
        assert h.node._concluded is None
        assert not path.exists()

        t_armed = time.monotonic()
        h.publish_command(True)
        h.feed_surface_odom_until(lambda: h.exits, timeout=4.0)
        assert h.exits == [EXIT_ABORT_FLOATER]
        # The deadline ran from arming, not from node start: the pre-arm
        # feeding already exceeded it, so an early-started clock would have
        # fired sooner than this.
        assert time.monotonic() - t_armed >= DIVE_DEADLINE_S

    def test_empty_verdict_path_writes_nothing(self, make_harness, tmp_path):
        h = make_harness("")
        h.publish_command(True)
        h.feed_surface_odom_until(lambda: h.exits, timeout=4.0)

        # Concludes and exits normally -- just without a verdict file.
        assert h.exits == [EXIT_ABORT_FLOATER]
        assert list(tmp_path.iterdir()) == []
