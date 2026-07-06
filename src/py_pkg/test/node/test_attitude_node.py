"""Tier 2 in-process rclpy tests for AttitudeNode.

AttitudeNode is the vehicle-state estimator that replaced the EKF: it turns the
prefiltered IMU's gravity reaction into (roll, pitch) and gauges the external
pressure into depth, publishing both on one Pose (POSITION_ESTIMATION).

Black-box: the tester publishes IMU_FILTERED (sensor_msgs/Imu, a tilted gravity
accel vector + REP-145 no-orientation flag), EXTERNAL_PRESSURE (Int32 absolute
Pa) and DIVE_INIT (registers the gauge reference), then subscribes
POSITION_ESTIMATION and asserts on the emitted Pose.

Contracts pinned here:
* orientation decodes (quaternion_to_roll_pitch / quaternion_to_yaw) to the
  roll/pitch implied by the injected gravity vector, with yaw == 0;
* position.z == gauge pressure (absolute - registered surface);
* the publish GATE holds -- nothing is emitted until BOTH a first IMU and a
  first external-pressure sample have arrived.

The injected gravity vector for a target (roll, pitch) is built the same way as
test/unit/test_math_utils.py: world up (-Z in the NED body frame, magnitude g)
expressed in the body frame via the project's own rpy_to_quaternion, so the test
rides the same convention the estimator inverts.
"""

import math

import pytest
from geometry_msgs.msg import Pose
from py_pkg.attitude.attitude_node import AttitudeNode
from py_pkg.math_utils import (
    quaternion_to_roll_pitch,
    quaternion_to_yaw,
)
from py_pkg.physics import ATMOSPHERIC_PRESSURE_PA
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Int32

from nautilus_msgs.msg import DiveInit

from _attitude_helpers import specific_force as _specific_force
from conftest import NodeHarness


class _AttitudeTesterNode(Node):
    """Drives AttitudeNode: feeds IMU_FILTERED + EXTERNAL_PRESSURE + DIVE_INIT,
    captures POSITION_ESTIMATION poses."""

    def __init__(self):
        super().__init__("attitude_node_tester")
        self.received_poses: list[Pose] = []

        self.imu_pub = create_publisher_for_topic(self, UUVTopics.IMU_FILTERED)
        self.external_pressure_pub = create_publisher_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE
        )
        self.dive_init_pub = create_publisher_for_topic(self, UUVTopics.DIVE_INIT)
        self.estimation_sub = create_subscription_for_topic(
            self, UUVTopics.POSITION_ESTIMATION, self._on_estimation
        )

    def _on_estimation(self, msg: Pose) -> None:
        self.received_poses.append(msg)

    def publish_imu(self, roll_deg: float = 0.0, pitch_deg: float = 0.0) -> None:
        # The prefiltered IMU carries the gravity reaction (specific force) in
        # linear_acceleration and -- per REP-145 -- flags orientation as
        # unavailable (covariance[0] == -1). attitude_node reads only the accel.
        ax, ay, az = _specific_force(
            math.radians(roll_deg), math.radians(pitch_deg)
        )
        msg = Imu()
        msg.orientation_covariance[0] = -1.0
        msg.linear_acceleration.x = float(ax)
        msg.linear_acceleration.y = float(ay)
        msg.linear_acceleration.z = float(az)
        self.imu_pub.publish(msg)

    def publish_external_pressure(self, value_pa: int) -> None:
        msg = Int32()
        msg.data = int(value_pa)
        self.external_pressure_pub.publish(msg)

    def publish_dive_init(self, surface_pressure_pa: float) -> None:
        msg = DiveInit()
        msg.surface_pressure_pa = float(surface_pressure_pa)
        self.dive_init_pub.publish(msg)


class AttitudeNodeHarness(NodeHarness):
    """NodeHarness specialised for AttitudeNode + _AttitudeTesterNode."""

    def __init__(self):
        super().__init__(AttitudeNode, _AttitudeTesterNode)

    @property
    def received_poses(self) -> list[Pose]:
        return self.tester.received_poses

    def publish_imu(self, roll_deg: float = 0.0, pitch_deg: float = 0.0) -> None:
        self.tester.publish_imu(roll_deg=roll_deg, pitch_deg=pitch_deg)

    def publish_external_pressure(self, value_pa: int) -> None:
        self.tester.publish_external_pressure(value_pa)

    def publish_dive_init(self, surface_pressure_pa: float) -> None:
        self.tester.publish_dive_init(surface_pressure_pa)


@pytest.fixture
def attitude_node_harness():
    """Function-scoped harness. Tears both nodes down on exit."""
    harness = AttitudeNodeHarness()
    try:
        yield harness
    finally:
        harness.shutdown()


def _decode(pose: Pose) -> tuple[float, float, float]:
    """(roll, pitch, yaw) in radians from a Pose's orientation."""
    q = pose.orientation
    roll, pitch = quaternion_to_roll_pitch(q.x, q.y, q.z, q.w)
    yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
    return roll, pitch, yaw


def _emit_pose(h, absolute_pa, roll_deg=0.0, pitch_deg=0.0) -> Pose:
    """Get one published Pose out of the node, race-free.

    The IMU callback is the sole publisher and only fires the gate once
    ``_have_pressure`` is set, so we land the pressure sample first (spin to
    process it), *then* the IMU sample that crosses the gate. Returns the last
    received Pose."""
    h.publish_external_pressure(int(absolute_pa))
    h.spin_until(lambda: h.node._have_pressure, timeout=1.0)
    h.publish_imu(roll_deg=roll_deg, pitch_deg=pitch_deg)
    h.spin_until(lambda: len(h.received_poses) >= 1, timeout=1.5)
    return h.received_poses[-1]


class TestWiringSmoke:
    """Construction + topic graph wiring."""

    def test_node_constructs(self, attitude_node_harness):
        assert attitude_node_harness.node is not None

    def test_imu_filtered_subscription_present(self, attitude_node_harness):
        names = [s.topic_name for s in attitude_node_harness.node.subscriptions]
        assert "/imu/filtered" in names

    def test_external_pressure_subscription_present(self, attitude_node_harness):
        names = [s.topic_name for s in attitude_node_harness.node.subscriptions]
        assert "/external/pressure" in names

    def test_dive_init_subscription_present(self, attitude_node_harness):
        names = [s.topic_name for s in attitude_node_harness.node.subscriptions]
        assert "/init/dive" in names

    def test_position_estimation_publisher_present(self, attitude_node_harness):
        names = [p.topic_name for p in attitude_node_harness.node.publishers]
        assert "/position/estimation" in names


class TestAttitudeFromGravity:
    """The published Pose orientation decodes to the roll/pitch implied by the
    injected gravity vector, with yaw == 0 (gravity can't observe yaw)."""

    def test_level_is_zero_attitude(self, attitude_node_harness):
        h = attitude_node_harness
        pose = _emit_pose(h, ATMOSPHERIC_PRESSURE_PA, roll_deg=0.0, pitch_deg=0.0)
        roll, pitch, yaw = _decode(pose)
        assert math.degrees(roll) == pytest.approx(0.0, abs=1e-3)
        assert math.degrees(pitch) == pytest.approx(0.0, abs=1e-3)
        assert yaw == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("roll_deg", [-30.0, 15.0, 45.0])
    def test_roll_recovered(self, attitude_node_harness, roll_deg):
        h = attitude_node_harness
        pose = _emit_pose(h, ATMOSPHERIC_PRESSURE_PA, roll_deg=roll_deg)
        roll, pitch, yaw = _decode(pose)
        assert math.degrees(roll) == pytest.approx(roll_deg, abs=1e-2)
        assert math.degrees(pitch) == pytest.approx(0.0, abs=1e-2)
        assert yaw == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("pitch_deg", [-25.0, 10.0, 40.0])
    def test_pitch_recovered(self, attitude_node_harness, pitch_deg):
        h = attitude_node_harness
        pose = _emit_pose(h, ATMOSPHERIC_PRESSURE_PA, pitch_deg=pitch_deg)
        roll, pitch, yaw = _decode(pose)
        assert math.degrees(roll) == pytest.approx(0.0, abs=1e-2)
        assert math.degrees(pitch) == pytest.approx(pitch_deg, abs=1e-2)
        assert yaw == pytest.approx(0.0, abs=1e-6)

    def test_combined_roll_pitch_recovered(self, attitude_node_harness):
        h = attitude_node_harness
        pose = _emit_pose(h, ATMOSPHERIC_PRESSURE_PA, roll_deg=20.0, pitch_deg=-15.0)
        roll, pitch, yaw = _decode(pose)
        assert math.degrees(roll) == pytest.approx(20.0, abs=1e-2)
        assert math.degrees(pitch) == pytest.approx(-15.0, abs=1e-2)
        assert yaw == pytest.approx(0.0, abs=1e-6)


class TestDepthGauging:
    """position.z == gauge pressure (absolute - registered surface), with
    position.x/y pinned at 0 (a single IMU can't observe horizontal position)."""

    def test_z_is_gauge_against_standard_atmosphere(self, attitude_node_harness):
        # No DIVE_INIT -> standard atmosphere is the reference.
        h = attitude_node_harness
        absolute_pa = int(ATMOSPHERIC_PRESSURE_PA) + 60_000
        pose = _emit_pose(h, absolute_pa)
        expected_gauge = absolute_pa - ATMOSPHERIC_PRESSURE_PA
        assert pose.position.z == pytest.approx(expected_gauge, abs=1e-3)
        assert pose.position.x == pytest.approx(0.0, abs=1e-9)
        assert pose.position.y == pytest.approx(0.0, abs=1e-9)

    def test_surface_reads_zero_gauge(self, attitude_node_harness):
        h = attitude_node_harness
        pose = _emit_pose(h, ATMOSPHERIC_PRESSURE_PA)
        assert pose.position.z == pytest.approx(0.0, abs=1e-3)

    def test_registered_surface_shifts_gauge(self, attitude_node_harness):
        # DIVE_INIT registers a surface 10 kPa above standard; z gauges against
        # that, so the same absolute reads 10 kPa shallower than before.
        h = attitude_node_harness
        surface_pa = ATMOSPHERIC_PRESSURE_PA + 10_000.0
        absolute_pa = int(surface_pa) + 40_000
        h.publish_dive_init(surface_pa)
        h.spin_until(
            lambda: h.node._surface_ref.reference_pa == pytest.approx(surface_pa),
            timeout=1.0,
        )
        pose = _emit_pose(h, absolute_pa)
        assert pose.position.z == pytest.approx(absolute_pa - surface_pa, abs=1e-3)


class TestPublishGate:
    """Nothing is emitted until BOTH a first IMU and a first external-pressure
    sample have arrived -- a fabricated z=0 before real pressure would read to a
    depth consumer as "at the surface"."""

    def test_imu_only_emits_nothing(self, attitude_node_harness):
        h = attitude_node_harness
        for _ in range(8):
            h.publish_imu(roll_deg=10.0)
            h.spin_for(0.05)
        assert h.received_poses == []
        assert h.node._have_imu is True
        assert h.node._have_pressure is False

    def test_pressure_only_emits_nothing(self, attitude_node_harness):
        # Pressure paces nothing -- only the IMU callback publishes -- so with
        # no IMU the node stays silent even with pressure in hand.
        h = attitude_node_harness
        for _ in range(8):
            h.publish_external_pressure(int(ATMOSPHERIC_PRESSURE_PA))
            h.spin_for(0.05)
        assert h.received_poses == []
        assert h.node._have_pressure is True
        assert h.node._have_imu is False

    def test_emits_once_both_inputs_seen(self, attitude_node_harness):
        h = attitude_node_harness
        h.publish_imu()
        h.spin_for(0.1)
        assert h.received_poses == [], "no pressure yet -> still gated"
        # Land the pressure sample, then the next IMU sample crosses the gate.
        h.publish_external_pressure(int(ATMOSPHERIC_PRESSURE_PA))
        h.spin_until(lambda: h.node._have_pressure, timeout=1.0)
        h.publish_imu()
        h.spin_until(lambda: len(h.received_poses) >= 1, timeout=1.5)
        assert len(h.received_poses) >= 1
