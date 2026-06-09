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
    BCU_VALVES = "/bcu/valves"
    BCU_RPM = "/bcu/rpm"
    BCU_FEEDBACK_RPM = "/bcu/feedback/rpm"
    BCU_FEEDBACK_VALVES = "/bcu/feedback/valves"

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

    # Debug / bench overrides
    DEBUG_BCU_RPM = "/debug/bcu/rpm"
    DEBUG_BCU_RPM_UNTIL_PRESSURE = "/debug/bcu/rpm_until_pressure"
    DEBUG_BCU_VALVES = "/debug/bcu/valves"
    DEBUG_ACU_PITCH = "/debug/acu/pitch"
    DEBUG_ACU_ROLL = "/debug/acu/roll"
    DEBUG_EMERGENCY_SURFACE = "/debug/emergency_surface"
    # All-stop for the debug nodes: zero RPM, close valves, neutral ACU, cancel
    # any emergency surface, then go silent. Published by the operator's red
    # Reset button. The mission/controller stop rides /command=false instead.
    DEBUG_RESET = "/debug/reset"

    # System status
    # Per-subsystem health, one DiagnosticArray published by the liveness node
    # from a freshness watchdog over the steady glider-side feedback/sensor
    # streams.
    STATUS_LIVENESS = "/status/liveness"
