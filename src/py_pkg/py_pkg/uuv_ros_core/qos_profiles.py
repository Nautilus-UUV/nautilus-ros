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
    UUVTopics.IMU: UUVQoS.SENSOR_STREAM,
    # Commands
    UUVTopics.COMMAND: UUVQoS.COMMAND,
    UUVTopics.PATH: UUVQoS.COMMAND,
    # Latched state from a long-lived node (RELIABLE + TRANSIENT_LOCAL): a
    # run watchdog that discovers pathfinding late still sees completion.
    UUVTopics.MISSION_COMPLETE: UUVQoS.COMMAND,
    # Pre-dive registration is state, not an event: TRANSIENT_LOCAL latches
    # the last Initialize so a controller that (re)starts mid-deployment
    # still sees the registered values.
    UUVTopics.DIVE_INIT: UUVQoS.COMMAND,
    UUVTopics.DEBUG_BCU_RPM: UUVQoS.COMMAND,
    UUVTopics.DEBUG_BCU_RPM_UNTIL_PRESSURE: UUVQoS.COMMAND,
    UUVTopics.DEBUG_BCU_VALVES: UUVQoS.COMMAND,
    UUVTopics.DEBUG_ACU_PITCH: UUVQoS.COMMAND,
    UUVTopics.DEBUG_ACU_ROLL: UUVQoS.COMMAND,
    UUVTopics.DEBUG_EMERGENCY_SURFACE: UUVQoS.COMMAND,
    # The debug all-stop is a transient event, not a state to latch — RELIABLE
    # but volatile, so a debug node that (re)starts later doesn't replay an old
    # reset. Volatile is safe here precisely because the debug nodes construct
    # in the silent/idle state: one that's down at publish time isn't driving
    # the wire and comes up safe, so a missed reset can't strand an actuator.
    UUVTopics.DEBUG_RESET: UUVQoS.CONTROL,
    # Everything else: control
    UUVTopics.INTERNAL_TEMPERATURE: UUVQoS.CONTROL,
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
    UUVTopics.IMU_FILTERED: UUVQoS.CONTROL,
    UUVTopics.POSITION_TARGET: UUVQoS.CONTROL,
    UUVTopics.POSITION_ESTIMATION: UUVQoS.CONTROL,
    UUVTopics.STATUS_LIVENESS: UUVQoS.CONTROL,
    # Ground-truth labels must not be lost: RELIABLE (CONTROL), not a
    # best-effort sensor stream.
    UUVTopics.ANOMALY_LABEL: UUVQoS.CONTROL,
}
