"""
Message type mappings for UUV topics.

Example:
-------
    from uuv_ros_core.message_types import TOPIC_MESSAGE_MAP
    from uuv_ros_core.topics import UUVTopics

    msg_type = TOPIC_MESSAGE_MAP[UUVTopics.BCU_FLOW_RATE]

"""

from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import Pose
from nautilus_msgs.msg import (
    AnomalyLabel,
    BcuPumpCommand,
    BcuPumpUntilPressureCommand,
    DiveInit,
    MissionCommand,
)
from sensor_msgs.msg import Imu, Temperature
from std_msgs.msg import (
    Bool,
    Empty,
    Float32,
    Int16,
    Int32,
    UInt8,
    UInt8MultiArray,
)

from .topics import UUVTopics

TOPIC_MESSAGE_MAP = {
    UUVTopics.INTERNAL_TEMPERATURE: Temperature,
    UUVTopics.INTERNAL_PRESSURE: Int32,
    UUVTopics.INTERNAL_LEAK: UInt8MultiArray,
    UUVTopics.EXTERNAL_TEMPERATURE: Temperature,
    UUVTopics.EXTERNAL_PRESSURE: Int32,
    UUVTopics.BCU_PRESSURE: Int32,
    UUVTopics.BCU_VOLUME: Int32,
    UUVTopics.BCU_FLOW_RATE: Float32,
    UUVTopics.BCU_RPM: Int16,
    # Bit 0 = valve 2 (motor way), bit 1 = valve 1 (free way); 1 = open, 0 = closed.
    UUVTopics.BCU_VALVES: UInt8,
    UUVTopics.BCU_FEEDBACK_RPM: Int16,
    UUVTopics.BCU_FEEDBACK_VALVES: UInt8,
    UUVTopics.ACU_PITCH: Int16,
    UUVTopics.ACU_ROLL: Int16,
    UUVTopics.ACU_FEEDBACK_OFFSET: Float32,
    UUVTopics.ACU_FEEDBACK_ANGLE: Float32,
    UUVTopics.IMU: Imu,
    UUVTopics.IMU_FILTERED: Imu,
    UUVTopics.POSITION_TARGET: Pose,
    UUVTopics.POSITION_ESTIMATION: Pose,
    UUVTopics.PATH: MissionCommand,
    # Mission run/stop: true = start the loaded mission, false = stop and reset
    # the stack to its clean initial state.
    UUVTopics.COMMAND: Bool,
    UUVTopics.DIVE_INIT: DiveInit,
    UUVTopics.DEBUG_BCU_RPM: BcuPumpCommand,
    UUVTopics.DEBUG_BCU_RPM_UNTIL_PRESSURE: BcuPumpUntilPressureCommand,
    # Same wire types as the actuator topics these debug injection points feed.
    UUVTopics.DEBUG_BCU_VALVES: UInt8,
    UUVTopics.DEBUG_ACU_PITCH: Int16,
    UUVTopics.DEBUG_ACU_ROLL: Int16,
    UUVTopics.DEBUG_EMERGENCY_SURFACE: Bool,
    UUVTopics.DEBUG_RESET: Empty,
    UUVTopics.STATUS_LIVENESS: DiagnosticArray,
    UUVTopics.ANOMALY_LABEL: AnomalyLabel,
}
