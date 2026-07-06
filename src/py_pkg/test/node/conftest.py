"""Tier 2 scaffolding for in-process rclpy tests.

Spins the node-under-test alongside a tester node in a SingleThreadedExecutor.
Tester nodes use uuv_ros_core factories so QoS auto-matches whatever the
node-under-test expects — manual create_publisher / create_subscription calls
with mismatched QoS will silently drop messages.

Layout:
* ``NodeHarness`` — generic: holds a node-under-test + tester node, drives
  them with an executor, and exposes ``spin_for`` / ``spin_until`` helpers.
* ``_BCUTesterNode`` / ``_ACUTesterNode`` — per-node tester surfaces
  declaring the inbound publishers and outbound subscriptions specific to
  each node-under-test.
* ``bcu_node_harness`` / ``acu_node_harness`` — function-scoped pytest
  fixtures wiring node-under-test class + tester class into ``NodeHarness``.
"""

import math
import os
import time

import pytest
import rclpy
from geometry_msgs.msg import Pose
from nautilus_msgs.msg import (
    BcuPumpCommand,
    BcuPumpUntilPressureCommand,
    DiveInit,
    MissionCommand,
)
from py_pkg.debug.acu_debug_node import AcuDebugNode
from py_pkg.debug.bcu_debug_node import BcuDebugNode
from py_pkg.math_utils import rpy_to_quaternion
from py_pkg.path.pathfinding import PathfindingNode
from py_pkg.pid.acu_node import ACUControlNode
from py_pkg.pid.bcu_node import BCUNode
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, Int16, Int32, UInt8


def _isolated_ros_domain_id() -> int:
    """Per-PID ROS_DOMAIN_ID. Tests use production topic names, so any
    sibling rclpy participant on the default domain (stray `ros2` CLI,
    leftover Gazebo, parallel pytest) would inject traffic into the
    harness and corrupt assertions."""
    return (os.getpid() % 101) + 1


@pytest.fixture(scope="session", autouse=True)
def rclpy_session():
    """Init/shutdown rclpy once per session, on an isolated ROS_DOMAIN_ID."""
    os.environ["ROS_DOMAIN_ID"] = str(_isolated_ros_domain_id())
    rclpy.init()
    yield
    rclpy.shutdown()


class NodeHarness:
    """Generic Tier 2 harness: node-under-test + tester node + executor.

    Owns lifecycle (executor add/remove, destroy_node, executor shutdown).
    Subclasses or callers supply the node-under-test and a tester node;
    the tester node owns the test-side publishers/subscribers.
    """

    def __init__(self, node_under_test_cls, tester_cls):
        # Both arguments are zero-arg callables: a Node subclass, or a
        # factory lambda when the node needs constructor arguments.
        self.node = node_under_test_cls()
        self.tester = tester_cls()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor.add_node(self.tester)

    def __getattr__(self, name):
        """Delegate unknown attribute lookups to the tester node.

        Only consulted for names not found normally, so members defined
        on a subclass (e.g. transforming properties) still win. Private
        and dunder names are refused so pickling and pytest introspection
        get a clean AttributeError instead of recursing into a
        half-constructed harness.
        """
        if name.startswith("_"):
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        tester = self.__dict__.get("tester")
        if tester is None:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        return getattr(tester, name)

    def spin_for(self, duration_s: float, slice_s: float = 0.02) -> None:
        """Spin the executor for at least ``duration_s`` of wall time."""
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=slice_s)

    def spin_until(self, predicate, timeout: float = 2.0, slice_s: float = 0.02):
        """Spin until ``predicate()`` is truthy or ``timeout`` elapses."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            self.executor.spin_once(timeout_sec=slice_s)
        if not predicate():
            raise TimeoutError(f"predicate did not become true within {timeout}s")

    def shutdown(self) -> None:
        try:
            self.executor.remove_node(self.node)
            self.executor.remove_node(self.tester)
        finally:
            self.node.destroy_node()
            self.tester.destroy_node()
            self.executor.shutdown()


# ---------------------------------------------------------------------------
# Depth node harness
# ---------------------------------------------------------------------------


class _BCUTesterNode(Node):
    """Drives BCUNode and captures BCU_RPM + BCU_VALVES emissions."""

    def __init__(self):
        super().__init__("bcu_node_tester")
        self.received_rpm: list[int] = []
        self.received_valves: list[int] = []

        self.target_pose_pub = create_publisher_for_topic(
            self, UUVTopics.POSITION_TARGET
        )
        # Depth measurement now rides POSITION_ESTIMATION.position.z (gauge Pa),
        # already gauged by attitude_node -- bcu_node no longer subscribes to
        # EXTERNAL_PRESSURE or holds a SurfaceReference.
        self.estimation_pose_pub = create_publisher_for_topic(
            self, UUVTopics.POSITION_ESTIMATION
        )
        self.tank_pressure_pub = create_publisher_for_topic(
            self, UUVTopics.BCU_PRESSURE
        )
        self.dive_init_pub = create_publisher_for_topic(self, UUVTopics.DIVE_INIT)
        self.command_pub = create_publisher_for_topic(self, UUVTopics.COMMAND)
        self.bcu_rpm_sub = create_subscription_for_topic(
            self, UUVTopics.BCU_RPM, self._on_rpm
        )
        self.bcu_valves_sub = create_subscription_for_topic(
            self, UUVTopics.BCU_VALVES, self._on_valves
        )

    def _on_rpm(self, msg: Int16) -> None:
        self.received_rpm.append(int(msg.data))

    def _on_valves(self, msg: UInt8) -> None:
        self.received_valves.append(int(msg.data))

    def publish_target_pressure(self, value_pa: float) -> None:
        # bcu_node treats position.z as the gauge-pressure setpoint
        # (Pa). Other Pose fields are zeroed.
        msg = Pose()
        msg.position.z = float(value_pa)
        self.target_pose_pub.publish(msg)

    def publish_depth_gauge(self, gauge_pa: float) -> None:
        # The depth measurement: POSITION_ESTIMATION.position.z carries the
        # gauge pressure (Pa, Z-positive-down) attitude_node already computed.
        # bcu_node reads it straight off position.z -- no ingress conversion.
        msg = Pose()
        msg.position.z = float(gauge_pa)
        self.estimation_pose_pub.publish(msg)

    def publish_tank_pressure(self, value_pa: int) -> None:
        # BCU_PRESSURE in the tank sensor's own frame; feeds the output clamp.
        msg = Int32()
        msg.data = int(value_pa)
        self.tank_pressure_pub.publish(msg)

    def publish_dive_init(
        self,
        surface_pressure_pa: float = 0.0,
        tank_empty_pa: float = 0.0,
        tank_full_pa: float = 0.0,
    ) -> None:
        msg = DiveInit()
        msg.surface_pressure_pa = float(surface_pressure_pa)
        msg.tank_empty_pa = float(tank_empty_pa)
        msg.tank_full_pa = float(tank_full_pa)
        self.dive_init_pub.publish(msg)

    def publish_command(self, start: bool) -> None:
        # bcu_node subscribes to /command and resets to a safe-silent state
        # on false (the old CONTROL_RESET path, now folded into /command).
        msg = Bool()
        msg.data = bool(start)
        self.command_pub.publish(msg)


@pytest.fixture
def bcu_node_harness():
    """Function-scoped harness. Tears both nodes down on exit."""
    harness = NodeHarness(BCUNode, _BCUTesterNode)
    try:
        yield harness
    finally:
        harness.shutdown()


# ---------------------------------------------------------------------------
# ACU node harness
# ---------------------------------------------------------------------------


class _ACUTesterNode(Node):
    """Drives ACUControlNode and captures ACU_PITCH (Int16 mm) /
    ACU_ROLL (Int16 cdeg).

    POSITION_TARGET carries the roll setpoint in its orientation and the
    *target gauge pressure* on position.z (pathfinding's TRIM convention).
    POSITION_ESTIMATION is now the single vehicle-state input: orientation =
    current roll for the roll PID, position.z = current gauge depth (Pa) the
    bang-bang pitch loop compares against. attitude_node owns the
    absolute->gauge conversion, so there's no EXTERNAL_PRESSURE/DIVE_INIT path
    on the node anymore."""

    def __init__(self):
        super().__init__("acu_node_tester")
        self.received_pitch_mm: list[int] = []
        self.received_roll_cdeg: list[int] = []

        self.target_pose_pub = create_publisher_for_topic(
            self, UUVTopics.POSITION_TARGET
        )
        self.estimation_pose_pub = create_publisher_for_topic(
            self, UUVTopics.POSITION_ESTIMATION
        )
        self.command_pub = create_publisher_for_topic(self, UUVTopics.COMMAND)
        self.pitch_sub = create_subscription_for_topic(
            self, UUVTopics.ACU_PITCH, self._on_pitch
        )
        self.roll_sub = create_subscription_for_topic(
            self, UUVTopics.ACU_ROLL, self._on_roll
        )

    def _on_pitch(self, msg: Int16) -> None:
        self.received_pitch_mm.append(int(msg.data))

    def _on_roll(self, msg: Int16) -> None:
        self.received_roll_cdeg.append(int(msg.data))

    @staticmethod
    def _pose(roll_deg: float, pressure_pa: float = 0.0) -> Pose:
        # position.z doubles as the pressure channel: a target gauge pressure
        # on POSITION_TARGET, a current gauge depth on POSITION_ESTIMATION.
        # Roll-only quaternion (pitch=yaw=0) — pitch isn't part of the ACU
        # target/estimation surface anymore now that pitch is driven by
        # pressure error rather than a target attitude.
        qx, qy, qz, qw = rpy_to_quaternion(math.radians(roll_deg), 0.0, 0.0)
        msg = Pose()
        msg.position.z = float(pressure_pa)
        msg.orientation.x = float(qx)
        msg.orientation.y = float(qy)
        msg.orientation.z = float(qz)
        msg.orientation.w = float(qw)
        return msg

    def publish_target(
        self, roll_deg: float = 0.0, target_pressure_pa: float = 0.0
    ) -> None:
        self.target_pose_pub.publish(self._pose(roll_deg, target_pressure_pa))

    def publish_current_attitude(
        self, roll_deg: float = 0.0, gauge_pa: float = 0.0
    ) -> None:
        # The single vehicle-state pose: roll (orientation) + gauge depth
        # (position.z) the bang-bang pitch leg select reads directly.
        self.estimation_pose_pub.publish(self._pose(roll_deg, gauge_pa))

    def publish_command(self, start: bool) -> None:
        # acu_node subscribes to /command and resets to a safe-silent state
        # on false (the old CONTROL_RESET path, now folded into /command).
        msg = Bool()
        msg.data = bool(start)
        self.command_pub.publish(msg)


@pytest.fixture
def acu_node_harness():
    """Function-scoped harness. Tears both nodes down on exit."""
    harness = NodeHarness(ACUControlNode, _ACUTesterNode)
    try:
        yield harness
    finally:
        harness.shutdown()


# ---------------------------------------------------------------------------
# Pathfinding node harness
# ---------------------------------------------------------------------------


class _PathfindingTesterNode(Node):
    """Drives PathfindingNode and captures POSITION_TARGET emissions."""

    def __init__(self):
        super().__init__("pathfinding_node_tester")
        self.received_targets: list = []

        self.estimation_pub = create_publisher_for_topic(
            self, UUVTopics.POSITION_ESTIMATION
        )
        self.command_pub = create_publisher_for_topic(self, UUVTopics.COMMAND)
        self.path_pub = create_publisher_for_topic(self, UUVTopics.PATH)
        self.target_sub = create_subscription_for_topic(
            self, UUVTopics.POSITION_TARGET, self._on_target
        )

    def _on_target(self, msg: Pose) -> None:
        self.received_targets.append(msg)

    def publish_pose_estimation(self, x: float, y: float, z: float) -> None:
        # position.z is gauge depth (Pa); pathfinding reads it straight off the
        # pose (attitude_node already gauged it) -- no absolute->gauge step here.
        msg = Pose()
        msg.position.x = float(x)
        msg.position.y = float(y)
        msg.position.z = float(z)
        msg.orientation.w = 1.0
        self.estimation_pub.publish(msg)

    def publish_depth_gauge(self, gauge_pa: float) -> None:
        # Single depth path now: the gauge value attitude_node would have
        # computed, delivered on POSITION_ESTIMATION.position.z. Both gates
        # _tick and feeds the mission's pressure-driven phases.
        self.publish_pose_estimation(0.0, 0.0, gauge_pa)

    def publish_command(self, start: bool) -> None:
        msg = Bool()
        msg.data = bool(start)
        self.command_pub.publish(msg)

    def publish_mission_command(
        self,
        mission_id: int,
        target_pressure_pa: float = 0.0,
        angle_rad: float = 0.0,
        n_resurfaces: int = 0,
    ) -> None:
        msg = MissionCommand()
        msg.mission_id = int(mission_id)
        msg.target_pressure_pa = float(target_pressure_pa)
        msg.angle_rad = float(angle_rad)
        msg.n_resurfaces = int(n_resurfaces)
        self.path_pub.publish(msg)


@pytest.fixture
def pathfinding_node_harness():
    """Function-scoped harness. Tears both nodes down on exit."""
    harness = NodeHarness(PathfindingNode, _PathfindingTesterNode)
    try:
        yield harness
    finally:
        harness.shutdown()


# ---------------------------------------------------------------------------
# BCU debug node harness
# ---------------------------------------------------------------------------


class _BcuDebugTesterNode(Node):
    """Drives BcuDebugNode and captures BCU_RPM + BCU_VALVES emissions
    with timestamps.

    There's no override gate now -- the node drives the wire whenever it holds
    a command. The tester publishes DEBUG_RESET to exercise the red all-stop.

    The debug node owns a duration timer, so tests need to reason about
    *when* each rpm value arrives -- a passthrough subscriber that only
    keeps the values would lose the "did the stop fire on time" signal."""

    def __init__(self):
        super().__init__("bcu_debug_tester")
        self.received: list[tuple[float, int]] = []
        self.received_valves: list[int] = []

        self.cmd_pub = create_publisher_for_topic(self, UUVTopics.DEBUG_BCU_RPM)
        self.pump_until_pub = create_publisher_for_topic(
            self, UUVTopics.DEBUG_BCU_RPM_UNTIL_PRESSURE
        )
        self.tank_pressure_pub = create_publisher_for_topic(
            self, UUVTopics.BCU_PRESSURE
        )
        self.valves_cmd_pub = create_publisher_for_topic(
            self, UUVTopics.DEBUG_BCU_VALVES
        )
        self.emergency_pub = create_publisher_for_topic(
            self, UUVTopics.DEBUG_EMERGENCY_SURFACE
        )
        self.reset_pub = create_publisher_for_topic(self, UUVTopics.DEBUG_RESET)
        self.rpm_sub = create_subscription_for_topic(
            self, UUVTopics.BCU_RPM, self._on_rpm
        )
        self.valves_sub = create_subscription_for_topic(
            self, UUVTopics.BCU_VALVES, self._on_valves
        )

    def _on_rpm(self, msg: Int16) -> None:
        self.received.append((time.monotonic(), int(msg.data)))

    def _on_valves(self, msg: UInt8) -> None:
        self.received_valves.append(int(msg.data))

    def publish_pump(self, rpm: int, duration_s: float) -> None:
        msg = BcuPumpCommand()
        msg.rpm = int(rpm)
        msg.duration_s = float(duration_s)
        self.cmd_pub.publish(msg)

    def publish_pump_until_pressure(self, rpm: int, target_pressure_pa: int) -> None:
        msg = BcuPumpUntilPressureCommand()
        msg.rpm = int(rpm)
        msg.target_pressure_pa = int(target_pressure_pa)
        self.pump_until_pub.publish(msg)

    def publish_tank_pressure(self, value_pa: int) -> None:
        msg = Int32()
        msg.data = int(value_pa)
        self.tank_pressure_pub.publish(msg)

    def publish_valves(self, mask: int) -> None:
        msg = UInt8()
        msg.data = int(mask)
        self.valves_cmd_pub.publish(msg)

    def publish_emergency(self, active: bool) -> None:
        msg = Bool()
        msg.data = bool(active)
        self.emergency_pub.publish(msg)

    def publish_reset(self) -> None:
        self.reset_pub.publish(Empty())


class BcuDebugNodeHarness(NodeHarness):
    """NodeHarness specialised for BcuDebugNode + _BcuDebugTesterNode.

    ``received_rpm`` transforms rather than forwards -- it strips the
    timestamps off the tester's ``received`` samples -- so it stays a
    real member instead of riding the base ``__getattr__`` delegation.
    """

    def __init__(self):
        super().__init__(BcuDebugNode, _BcuDebugTesterNode)

    @property
    def received_rpm(self) -> list[int]:
        return [rpm for _, rpm in self.tester.received]


@pytest.fixture
def bcu_debug_node_harness():
    """Function-scoped harness. Tears both nodes down on exit."""
    harness = BcuDebugNodeHarness()
    try:
        yield harness
    finally:
        harness.shutdown()


# ---------------------------------------------------------------------------
# ACU debug node harness
# ---------------------------------------------------------------------------


class _AcuDebugTesterNode(Node):
    """Drives AcuDebugNode and captures ACU_PITCH / ACU_ROLL emissions.

    There's no override gate now -- the node holds whatever pitch/roll it's
    commanded. The tester publishes DEBUG_RESET to exercise the release to
    neutral + silent."""

    def __init__(self):
        super().__init__("acu_debug_tester")
        self.received_pitch_mm: list[int] = []
        self.received_roll_cdeg: list[int] = []

        self.pitch_cmd_pub = create_publisher_for_topic(self, UUVTopics.DEBUG_ACU_PITCH)
        self.roll_cmd_pub = create_publisher_for_topic(self, UUVTopics.DEBUG_ACU_ROLL)
        self.reset_pub = create_publisher_for_topic(self, UUVTopics.DEBUG_RESET)
        self.pitch_sub = create_subscription_for_topic(
            self, UUVTopics.ACU_PITCH, self._on_pitch
        )
        self.roll_sub = create_subscription_for_topic(
            self, UUVTopics.ACU_ROLL, self._on_roll
        )

    def _on_pitch(self, msg: Int16) -> None:
        self.received_pitch_mm.append(int(msg.data))

    def _on_roll(self, msg: Int16) -> None:
        self.received_roll_cdeg.append(int(msg.data))

    def publish_pitch(self, mm: int) -> None:
        msg = Int16()
        msg.data = int(mm)
        self.pitch_cmd_pub.publish(msg)

    def publish_roll(self, cdeg: int) -> None:
        msg = Int16()
        msg.data = int(cdeg)
        self.roll_cmd_pub.publish(msg)

    def publish_reset(self) -> None:
        self.reset_pub.publish(Empty())


@pytest.fixture
def acu_debug_node_harness():
    """Function-scoped harness. Tears both nodes down on exit."""
    harness = NodeHarness(AcuDebugNode, _AcuDebugTesterNode)
    try:
        yield harness
    finally:
        harness.shutdown()
