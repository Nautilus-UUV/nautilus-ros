"""Vehicle state estimator: gravity-tilt attitude + gauged depth.

The glider accelerates slowly enough that the accelerometer's specific force is
essentially the gravity reaction, so roll and pitch fall straight out of the
prefiltered accel vector (``math_utils.gravity_to_roll_pitch``, NED body frame:
x forward, y right, z down) with no filter state. Yaw is unobservable from
gravity alone, so it's pinned to 0.

Depth rides the same Pose. External pressure (absolute Pa) is gauged against the
surface reference the operator registers at dive-init and written into
``position.z`` gauge Pa.

Inputs  -> output:
    /imu/filtered (Imu)        -> roll, pitch  (orientation, yaw=0)
    /external/pressure (Int32) -> gauged -> position.z (gauge Pa)
    /init/dive (DiveInit)      -> registers the gauge reference
                               -> /position/estimation (geometry_msgs/Pose)

Publish is gated until BOTH a first IMU sample and a first external-pressure
sample have arrived, so ``position.z`` is never a fabricated 0 -- which
``bcu_node`` would read as "at the surface" and could dive on.
"""

import math

import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node

from py_pkg.math_utils import gravity_to_roll_pitch, rpy_to_quaternion
from py_pkg.physics import SurfaceReference
from py_pkg.uuv_ros_core.node_factory import (
    create_publisher_for_topic,
    create_subscription_for_topic,
)
from py_pkg.uuv_ros_core.node_runtime import spin_node
from py_pkg.uuv_ros_core.topics import UUVTopics


class AttitudeNode(Node):
    """Estimate roll/pitch from gravity and depth from external pressure."""

    def __init__(self):
        super().__init__("attitude_node")

        # Latest attitude (from the IMU) and gauged depth (from external
        # pressure), each latched as it arrives; the IMU stream paces publishing.
        self._roll = 0.0
        self._pitch = 0.0
        self._gauge_pa = 0.0
        self._have_imu = False
        self._have_pressure = False

        # Owns the absolute->gauge conversion for the whole stack now. Defaults
        # to the standard atmosphere until the operator registers the real
        # surface pressure (DIVE_INIT), exactly like the per-node references it
        # replaces.
        self._surface_ref = SurfaceReference()

        # Throttle the position log to ~1 Hz against the ~50 Hz IMU stream.
        self._log_every_n = 50
        self._cb_count = 0

        create_subscription_for_topic(self, UUVTopics.IMU_FILTERED, self._on_imu)
        create_subscription_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE, self._on_external_pressure
        )
        create_subscription_for_topic(self, UUVTopics.DIVE_INIT, self._on_dive_init)
        self._pub = create_publisher_for_topic(self, UUVTopics.POSITION_ESTIMATION)

        self.get_logger().info(
            "Attitude/state estimator started: "
            f"{UUVTopics.IMU_FILTERED} + {UUVTopics.EXTERNAL_PRESSURE} -> "
            f"{UUVTopics.POSITION_ESTIMATION}"
        )

    def _on_imu(self, msg):
        a = msg.linear_acceleration
        self._roll, self._pitch = gravity_to_roll_pitch(a.x, a.y, a.z)
        self._have_imu = True
        # The IMU is the high-rate driver: publish one pose per IMU sample using
        # the latest gauged depth.
        self._publish()

    def _on_external_pressure(self, msg):
        # EXTERNAL_PRESSURE is absolute Pa; gauge against the registered surface
        # reference (standard atmosphere until DIVE_INIT). Cached here; the IMU
        # callback does the publishing.
        self._gauge_pa = self._surface_ref.gauge(float(msg.data))
        self._have_pressure = True

    def _on_dive_init(self, msg):
        surface_pa = float(msg.surface_pressure_pa)
        if self._surface_ref.register(surface_pa):
            self.get_logger().info(
                f"dive init: gauge reference = {self._surface_ref.reference_pa:.0f} Pa"
            )
        else:
            self.get_logger().error(
                f"dive init: surface pressure {surface_pa:.0f} Pa "
                "rejected -- keeping previous reference"
            )

    def _publish(self):
        # Gate: never emit a fabricated z=0 before the first real pressure -- a
        # depth consumer would read it as "at the surface".
        if not (self._have_imu and self._have_pressure):
            return

        qx, qy, qz, qw = rpy_to_quaternion(self._roll, self._pitch, 0.0)
        pose = Pose()
        # position.x/y stay 0 -- a single IMU can't observe horizontal position.
        pose.position.z = self._gauge_pa
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        self._pub.publish(pose)

        self._cb_count += 1
        if self._cb_count % self._log_every_n == 0:
            self.get_logger().info(
                f"roll={math.degrees(self._roll):.1f} deg "
                f"pitch={math.degrees(self._pitch):.1f} deg "
                f"depth={self._gauge_pa:.0f} Pa gauge"
            )


def main(args=None):
    rclpy.init(args=args)
    node = AttitudeNode()
    spin_node(node)


if __name__ == "__main__":
    main()
