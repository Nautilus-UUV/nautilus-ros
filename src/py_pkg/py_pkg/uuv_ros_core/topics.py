"""
Centralized topic definitions for UUV system.

Single source of truth for all topic names.

Example:
-------
    from uuv_ros_core.topics import UUVTopics

    self.publisher = self.create_publisher(
        Float32, UUVTopics.BCU_FLOW_RATE, 10
    )

"""


class UUVTopics:
    """Topic constants for UUV system."""

    # Internal sensors
    INTERNAL_TEMPERATURE = "/internal/temperature"
    INTERNAL_PRESSURE = "/internal/pressure"
    INTERNAL_LEAK = "/internal/leak"
    INTERNAL_HUMIDITY = "/internal/humidity"

    # External sensors
    EXTERNAL_TEMPERATURE = "/external/temperature"
    EXTERNAL_PRESSURE = "/external/pressure"

    # Buoyancy Control Unit (BCU)
    BCU_PRESSURE = "/bcu/pressure"
    BCU_VOLUME = "/bcu/volume"
    BCU_FLOW_RATE = "/bcu/flow_rate"
    BCU_RPM = "/bcu/rpm"

    # Attitude Control Unit (ACU)
    ACU_PITCH = "/acu/pitch"
    ACU_ROLL = "/acu/roll"
    ACU_FEEDBACK_OFFSET = "/acu/feedback/offset"
    ACU_FEEDBACK_ANGLE = "/acu/feedback/angle"

    # IMU data
    IMU_LEFT = "/imu/left"
    IMU_RIGHT = "/imu/right"
    IMU_FILTERED_LEFT = "/imu/filtered/left"
    IMU_FILTERED_RIGHT = "/imu/filtered/right"

    # Navigation and control
    POSITION_TARGET = "/position/target"
    POSITION_ESTIMATION = "/position/estimation"
    PATH = "/path"
    COMMAND = "/command"

    # CAN communication
    CAN_OUT = "/can/out"
