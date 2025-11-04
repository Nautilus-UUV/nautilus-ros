"""
QoS profiles optimized for underwater vehicle operations.

Example:
-------
    from uuv_ros_utils.qos_profiles import PolarisQoS

    publisher = self.create_publisher(
        Float32, topic, PolarisQoS.SAFETY_CRITICAL
    )

"""

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from .topics import PolarisTopics


class PolarisQoS:
    """QoS profiles for underwater operations."""

    # Critical safety systems
    SAFETY_CRITICAL = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        depth=50,
    )

    # Control systems
    CONTROL = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)

    # High-frequency sensors
    SENSOR_STREAM = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=5)

    # Commands and missions
    COMMAND = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        depth=20,
    )


# Topic QoS mapping
TOPIC_QOS_MAP = {
    # Safety critical
    PolarisTopics.INTERNAL_LEAK: PolarisQoS.SAFETY_CRITICAL,
    PolarisTopics.INTERNAL_PRESSURE: PolarisQoS.SAFETY_CRITICAL,
    PolarisTopics.EXTERNAL_PRESSURE: PolarisQoS.SAFETY_CRITICAL,
    # High-frequency sensors
    PolarisTopics.IMU_LEFT: PolarisQoS.SENSOR_STREAM,
    PolarisTopics.IMU_RIGHT: PolarisQoS.SENSOR_STREAM,
    # Commands
    PolarisTopics.COMMAND: PolarisQoS.COMMAND,
    PolarisTopics.PATH: PolarisQoS.COMMAND,
    # Everything else: control
    PolarisTopics.INTERNAL_TEMPERATURE: PolarisQoS.CONTROL,
    PolarisTopics.INTERNAL_HUMIDITY: PolarisQoS.CONTROL,
    PolarisTopics.EXTERNAL_TEMPERATURE: PolarisQoS.CONTROL,
    PolarisTopics.BCU_PRESSURE: PolarisQoS.CONTROL,
    PolarisTopics.BCU_FLOW_RATE: PolarisQoS.CONTROL,
    PolarisTopics.BCU_RPM: PolarisQoS.CONTROL,
    PolarisTopics.ACU_TILT: PolarisQoS.CONTROL,
    PolarisTopics.ACU_ROLL: PolarisQoS.CONTROL,
    PolarisTopics.ACU_TILT_STEPS: PolarisQoS.CONTROL,
    PolarisTopics.ACU_ROLL_STEPS: PolarisQoS.CONTROL,
    PolarisTopics.ACU_FEEDBACK_OFFSET: PolarisQoS.CONTROL,
    PolarisTopics.ACU_FEEDBACK_ANGLE: PolarisQoS.CONTROL,
    PolarisTopics.IMU_FILTERED_LEFT: PolarisQoS.CONTROL,
    PolarisTopics.IMU_FILTERED_RIGHT: PolarisQoS.CONTROL,
    PolarisTopics.POSITION_TARGET: PolarisQoS.CONTROL,
    PolarisTopics.POSITION_ESTIMATION: PolarisQoS.CONTROL,
    PolarisTopics.CAN_OUT: PolarisQoS.CONTROL,
}
