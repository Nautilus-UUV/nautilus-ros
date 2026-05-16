"""
Message type mappings for UUV topics.

Example:
-------
    from uuv_ros_core.message_types import TOPIC_MESSAGE_MAP
    from uuv_ros_core.topics import UUVTopics

    msg_type = TOPIC_MESSAGE_MAP[UUVTopics.BCU_FLOW_RATE]

"""

from can_msgs.msg import Frame
from geometry_msgs.msg import Pose
from nautilus_msgs.msg import BcuPumpCommand, MissionCommand
from sensor_msgs.msg import Imu, Temperature
from std_msgs.msg import (
    Bool,
    Float32,
    Int16,
    Int32,
    String,
    UInt8,
    UInt8MultiArray,
)

from .topics import UUVTopics

TOPIC_MESSAGE_MAP = {
    UUVTopics.INTERNAL_TEMPERATURE: Temperature,
    UUVTopics.INTERNAL_PRESSURE: Int32,
    UUVTopics.INTERNAL_LEAK: UInt8MultiArray,
    UUVTopics.INTERNAL_HUMIDITY: Float32,
    UUVTopics.EXTERNAL_TEMPERATURE: Temperature,
    UUVTopics.EXTERNAL_PRESSURE: Int32,
    UUVTopics.BCU_PRESSURE: Int32,
    UUVTopics.BCU_VOLUME: Int32,
    UUVTopics.BCU_FLOW_RATE: Float32,
    UUVTopics.BCU_RPM: Int16,
    # Bit 0 = valve 1, bit 1 = valve 2; 1 = open, 0 = closed.
    UUVTopics.BCU_VALVES: UInt8,
    UUVTopics.ACU_PITCH: Int16,
    UUVTopics.ACU_ROLL: Int16,
    UUVTopics.ACU_FEEDBACK_OFFSET: Float32,
    UUVTopics.ACU_FEEDBACK_ANGLE: Float32,
    UUVTopics.IMU_LEFT: Imu,
    UUVTopics.IMU_RIGHT: Imu,
    UUVTopics.IMU_FILTERED_LEFT: Imu,
    UUVTopics.IMU_FILTERED_RIGHT: Imu,
    UUVTopics.POSITION_TARGET: Pose,
    UUVTopics.POSITION_ESTIMATION: Pose,
    UUVTopics.PATH: MissionCommand,
    UUVTopics.COMMAND: String,
    UUVTopics.CAN_OUT: Frame,
    UUVTopics.DEBUG_BCU_RPM: BcuPumpCommand,
    UUVTopics.CONTROL_MANUAL_OVERRIDE: Bool,
}
