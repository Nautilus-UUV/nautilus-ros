"""
QoS profiles optimized for underwater vehicle operations.

Example:
-------
    from uuv_ros_core.qos_profiles import UUVQoS

    publisher = self.create_publisher(
        Float32, topic, UUVQoS.SAFETY_CRITICAL
    )

"""

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from .topics import UUVTopics


class UUVQoS:
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
    UUVTopics.INTERNAL_LEAK: UUVQoS.SAFETY_CRITICAL,
    UUVTopics.INTERNAL_PRESSURE: UUVQoS.SAFETY_CRITICAL,
    UUVTopics.EXTERNAL_PRESSURE: UUVQoS.SAFETY_CRITICAL,
    # High-frequency sensors
    UUVTopics.IMU_LEFT: UUVQoS.SENSOR_STREAM,
    UUVTopics.IMU_RIGHT: UUVQoS.SENSOR_STREAM,
    # Commands
    UUVTopics.COMMAND: UUVQoS.COMMAND,
    UUVTopics.PATH: UUVQoS.COMMAND,
    UUVTopics.DEBUG_BCU_RPM: UUVQoS.COMMAND,
    UUVTopics.DEBUG_BCU_VALVES: UUVQoS.COMMAND,
    UUVTopics.DEBUG_ACU_PITCH: UUVQoS.COMMAND,
    UUVTopics.DEBUG_ACU_ROLL: UUVQoS.COMMAND,
    UUVTopics.DEBUG_EMERGENCY_SURFACE: UUVQoS.COMMAND,
    # Latched control-state flags; TRANSIENT_LOCAL via COMMAND profile so a
    # late-joining depth_node / acu_node (or the operator UI) sees the current
    # override state immediately.
    UUVTopics.CONTROL_MANUAL_OVERRIDE: UUVQoS.COMMAND,
    UUVTopics.CONTROL_ACU_OVERRIDE: UUVQoS.COMMAND,
    # A reset is a transient event, not a state to latch — RELIABLE but
    # volatile, so a controller that (re)starts later doesn't replay an old
    # reset. The controllers are already fresh on construction anyway.
    UUVTopics.CONTROL_RESET: UUVQoS.CONTROL,
    # Everything else: control
    UUVTopics.INTERNAL_TEMPERATURE: UUVQoS.CONTROL,
    UUVTopics.INTERNAL_HUMIDITY: UUVQoS.CONTROL,
    UUVTopics.EXTERNAL_TEMPERATURE: UUVQoS.CONTROL,
    UUVTopics.BCU_PRESSURE: UUVQoS.CONTROL,
    UUVTopics.BCU_VOLUME: UUVQoS.CONTROL,
    UUVTopics.BCU_FLOW_RATE: UUVQoS.CONTROL,
    UUVTopics.BCU_RPM: UUVQoS.CONTROL,
    UUVTopics.BCU_VALVES: UUVQoS.CONTROL,
    UUVTopics.BCU_FEEDBACK_RPM: UUVQoS.CONTROL,
    UUVTopics.BCU_FEEDBACK_VALVES: UUVQoS.CONTROL,
    UUVTopics.ACU_PITCH: UUVQoS.CONTROL,
    UUVTopics.ACU_ROLL: UUVQoS.CONTROL,
    UUVTopics.ACU_FEEDBACK_OFFSET: UUVQoS.CONTROL,
    UUVTopics.ACU_FEEDBACK_ANGLE: UUVQoS.CONTROL,
    UUVTopics.IMU_FILTERED_LEFT: UUVQoS.CONTROL,
    UUVTopics.IMU_FILTERED_RIGHT: UUVQoS.CONTROL,
    UUVTopics.POSITION_TARGET: UUVQoS.CONTROL,
    UUVTopics.POSITION_ESTIMATION: UUVQoS.CONTROL,
    UUVTopics.CAN_OUT: UUVQoS.CONTROL,
    UUVTopics.STATUS_LIVENESS: UUVQoS.CONTROL,
}
