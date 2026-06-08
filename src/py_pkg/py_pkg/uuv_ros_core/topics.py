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
    # Actuator feedback the BCU bridge pings back so an idle pump/valves still
    # prove they're alive. RPM is the fault-adjusted effective value; valves
    # echo the latest commanded bitmask (bit0=v1, bit1=v2).
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

    # Debug / bench overrides (manual injection points; bypass closed-loop control)
    DEBUG_BCU_RPM = "/debug/bcu/rpm"
    # Manual BCU valve state, full bitmask (bit0=v1, bit1=v2). Owned by
    # bcu_debug while a valve command is active.
    DEBUG_BCU_VALVES = "/debug/bcu/valves"
    # Manual ACU setpoints, same wire formats as the actuator topics they
    # feed: pitch in mm, roll in centidegrees. Held by acu_debug.
    DEBUG_ACU_PITCH = "/debug/acu/pitch"
    DEBUG_ACU_ROLL = "/debug/acu/roll"
    # Blow ballast and surface now. True engages (max inflate until at the
    # surface); False cancels. The operator UI raises the override first.
    DEBUG_EMERGENCY_SURFACE = "/debug/emergency_surface"

    # Control-graph coordination
    # Operator-owned manual-mode flag: the mission UI raises it (through the
    # MQTT bridge) when entering manual control. While True, depth_node stands
    # down on /bcu/rpm + /bcu/valves and bcu_debug is the only BCU driver;
    # while False, bcu_debug stays silent and depth_node owns the wire.
    CONTROL_MANUAL_OVERRIDE = "/control/manual_override"
    # ACU twin of CONTROL_MANUAL_OVERRIDE. Kept separate so the two loops can
    # be silenced independently if ever needed; the operator UI drives both
    # from one slider. acu_node honours it; acu_debug gates on it.
    CONTROL_ACU_OVERRIDE = "/control/acu_override"
    # One-shot "drop everything and go fresh" signal. depth_node / acu_node
    # clear their target back to None and wipe controller state (integrators,
    # filters, publish guards) so they sit exactly as they did at boot before
    # any mission. pathfinding emits it when the Do-Nothing mission starts.
    CONTROL_RESET = "/control/reset"

    # System status
    # Per-subsystem health, one DiagnosticArray published by the liveness node
    # from a freshness watchdog over the steady glider-side feedback/sensor
    # streams.
    STATUS_LIVENESS = "/status/liveness"
