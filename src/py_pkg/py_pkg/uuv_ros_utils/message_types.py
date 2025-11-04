"""
Message type mappings for Polaris UUV topics.

Example:
-------
    from uuv_ros_utils.message_types import TOPIC_MESSAGE_MAP
    from uuv_ros_utils.topics import PolarisTopics

    msg_type = TOPIC_MESSAGE_MAP[PolarisTopics.BCU_FLOW_RATE]

"""

from sensor_msgs.msg import Temperature, Imu
from std_msgs.msg import Int32, Float32, String, UInt8MultiArray, Float32MultiArray
from geometry_msgs.msg import Point
from can_msgs.msg import Frame
from .topics import PolarisTopics


TOPIC_MESSAGE_MAP = {
    PolarisTopics.INTERNAL_TEMPERATURE: Temperature,
    PolarisTopics.INTERNAL_PRESSURE: Int32,
    PolarisTopics.INTERNAL_LEAK: UInt8MultiArray,
    PolarisTopics.INTERNAL_HUMIDITY: Float32,
    PolarisTopics.EXTERNAL_TEMPERATURE: Temperature,
    PolarisTopics.EXTERNAL_PRESSURE: Int32,
    PolarisTopics.BCU_PRESSURE: Int32,
    PolarisTopics.BCU_FLOW_RATE: Float32,
    PolarisTopics.BCU_RPM: Int32,
    PolarisTopics.ACU_TILT: Float32,
    PolarisTopics.ACU_ROLL: Float32,
    PolarisTopics.ACU_TILT_STEPS: Int32,
    PolarisTopics.ACU_ROLL_STEPS: Int32,
    PolarisTopics.ACU_FEEDBACK_OFFSET: Float32,
    PolarisTopics.ACU_FEEDBACK_ANGLE: Float32,
    PolarisTopics.IMU_LEFT: Imu,
    PolarisTopics.IMU_RIGHT: Imu,
    PolarisTopics.IMU_FILTERED_LEFT: Imu,
    PolarisTopics.IMU_FILTERED_RIGHT: Imu,
    PolarisTopics.POSITION_TARGET: Point,
    PolarisTopics.POSITION_ESTIMATION: Point,
    PolarisTopics.PATH: Float32MultiArray,
    PolarisTopics.COMMAND: String,
    PolarisTopics.CAN_OUT: Frame,
}
