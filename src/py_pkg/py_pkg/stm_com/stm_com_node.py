"""ROS 2 <-> STM32 serial bridge.

The STM32 owns the BCU motor driver and a clutch of housekeeping sensors
(pressures, temperatures, leak detect). It streams sensor frames up the
UART and treats every frame from us as a cue to take the latest RPM
setpoint -- so this node both pumps inbound telemetry out onto ROS topics
and ships the most recent BCU_RPM back down on each tick.

Wire format (matches the STM firmware's switch on variable id):

    0xAA | var_id (>H) | length (B) | payload (little-endian)

Outbound (Pi -> STM):
    0x2102  BCU_RPM        int16   motor RPM setpoint
    0x2106  VALVES_TARGET  uint8   desired valve bitmap (bit0=valve2/motor, bit1=valve1/free)

Inbound (STM -> Pi):
    0x2103  BCU_STATUS     int16   motor's currently reported RPM (~1 Hz)
    0x2107  VALVES_STATUS  uint8   actual valve bitmap the firmware reports back
    0x2400  EXT_PRESSURE   uint16  absolute, 100 Pa / LSB
    0x2401  TANK_PRESSURE  uint16  gauge relative to hull, 100 Pa / LSB
    0x2402  INT_PRESSURE   uint16  absolute, 100 Pa / LSB
    0x2410  EXT_TEMP       int16   0.01 °C / LSB
    0x2411  INT_TEMP       int16   0.01 °C / LSB
    0x2420  LEAKS          uint8   bitmask; any nonzero bit = water detected
    0x2430  ACCEL_X        int16   raw accelerometer counts (±6 g full scale)
    0x2431  ACCEL_Y        int16   raw accelerometer counts
    0x2432  ACCEL_Z        int16   raw accelerometer counts
    0x2433  GYRO_X         int16   raw gyro counts (±2000 °/s full scale)
    0x2434  GYRO_Y         int16   raw gyro counts
    0x2435  GYRO_Z         int16   raw gyro counts (last of the batch -> publish)

The six IMU words are raw sensor counts; physics.imu_counts_to_body turns them
into an SI sensor_msgs/Imu (m/s^2, rad/s) in the FLU body frame, published on
IMU_LEFT -- the same topic the sim HAL bridge feeds, so the prefilter -> EKF ->
MQTT -> UI chain runs unchanged on hardware. The STM sends the six contiguously
each cycle, so we assemble + publish one Imu when GYRO_Z arrives. Crucially the
IMU frames are NOT treated as a setpoint-send cue (see _poll_serial): only the
housekeeping/status frames pace our downward RPM/valve TX, so a faster IMU
stream doesn't multiply what we push back to the STM.

Until the depth PID has published a single RPM (and the BCU a valve bitmap)
we send 0 -- safe default: motor off, valves closed, if the STM polls before
the control stack is up.
"""

import struct

import rclpy
import serial
from rclpy.node import Node
from sensor_msgs.msg import Imu, Temperature
from std_msgs.msg import Int16, Int32, UInt8, UInt8MultiArray

from py_pkg.physics import imu_counts_to_body
from py_pkg.robot_specs import (
    STM_BCU_RPM_SIGN,
    STM_PRESSURE_LSB_PA,
    STM_TEMPERATURE_LSB_C,
)
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
    spin_node
)

SYNC_BYTE = b"\xaa"
HEADER_FMT = ">HB"  # big-endian uint16 var_id, uint8 length
HEADER_LEN = struct.calcsize(HEADER_FMT)

BCU_RPM_VAR_ID = 0x2102
VALVES_TARGET_VAR_ID = 0x2106  # <B — uint8, us → STM, desired valve bitmap
VALVES_STATUS_VAR_ID = 0x2107  # <B — uint8, STM → us, actual valve bitmap
BCU_STATUS_VAR_ID = 0x2103  # <h — int16, STM → us, ~1 Hz measured motor RPM
EXT_PRESSURE_VAR_ID = 0x2400  # <H — uint16
TANK_PRESSURE_VAR_ID = 0x2401  # <H — uint16  → BCU_PRESSURE
INT_PRESSURE_VAR_ID = 0x2402  # <H — uint16
EXT_TEMP_VAR_ID = 0x2410  # <h — int16
INT_TEMP_VAR_ID = 0x2411  # <h — int16
LEAKS_VAR_ID = 0x2420  # <B — uint8

# IMU: six int16 count words, STM → us. Sent contiguously each cycle in this
# order, so GYRO_Z is the cue to assemble + publish one sensor_msgs/Imu.
ACCEL_X_VAR_ID = 0x2430  # <h — int16 raw accel count
ACCEL_Y_VAR_ID = 0x2431  # <h — int16
ACCEL_Z_VAR_ID = 0x2432  # <h — int16
GYRO_X_VAR_ID = 0x2433  # <h — int16 raw gyro count
GYRO_Y_VAR_ID = 0x2434  # <h — int16
GYRO_Z_VAR_ID = 0x2435  # <h — int16 (last of the batch)
IMU_VAR_IDS = (
    ACCEL_X_VAR_ID,
    ACCEL_Y_VAR_ID,
    ACCEL_Z_VAR_ID,
    GYRO_X_VAR_ID,
    GYRO_Y_VAR_ID,
    GYRO_Z_VAR_ID,
)

# Frame the published Imu is stamped with. Cosmetic for our consumers (the EKF
# reads accel/gyro directly), but kept descriptive and stable.
IMU_FRAME_ID = "imu_left"

PAYLOAD_FMT = "<h"  # little-endian signed 16-bit
BCU_RPM_PAYLOAD_LEN = struct.calcsize(PAYLOAD_FMT)

# Inbound payload shapes -- size-checked before each unpack.
UINT16_FMT = "<H"
INT16_FMT = "<h"
UINT8_FMT = "<B"
UINT16_LEN = struct.calcsize(UINT16_FMT)
INT16_LEN = struct.calcsize(INT16_FMT)
UINT8_LEN = struct.calcsize(UINT8_FMT)


class STMComNode(Node):
    """Bridges ``/bcu/rpm`` to the STM32 over UART."""

    def __init__(self) -> None:
        super().__init__("stm_com")

        self.declare_parameter("port", "/dev/serial0")
        self.declare_parameter("baud", 115200)
        self.declare_parameter("poll_period_s", 0.01)

        port = self.get_parameter("port").get_parameter_value().string_value
        baud = self.get_parameter("baud").get_parameter_value().integer_value
        poll_period = (
            self.get_parameter("poll_period_s").get_parameter_value().double_value
        )

        # timeout=0 -> non-blocking reads. We drive cadence from the ROS
        # timer instead so a stalled UART can't wedge the executor.
        self._ser = serial.Serial(port, baud, timeout=0)
        self.get_logger().info(f"stm_com opened {port} @ {baud}")

        self._latest_rpm: int = 0
        # Valve bitmap we beat down to the STM (bit0=valve2/motor, bit1=valve1/free; 1=open).
        # 0 until the BCU first commands -- safe default, valves closed.
        self._latest_valves: int = 0
        self._rx_buf = bytearray()
        # Latest raw IMU counts, accumulated across the six per-axis frames and
        # converted to one Imu when GYRO_Z lands.
        self._imu_counts = {var_id: 0 for var_id in IMU_VAR_IDS}

        create_subscription_for_topic(self, UUVTopics.BCU_RPM, self._on_rpm)
        create_subscription_for_topic(self, UUVTopics.BCU_VALVES, self._on_valves)
        self.create_timer(poll_period, self._poll_serial)

        # ==========================
        # --- SENSOR DATA
        self._ext_pressure_pub = create_publisher_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE
        )
        self._tank_pressure_pub = create_publisher_for_topic(
            self, UUVTopics.BCU_PRESSURE
        )
        self._int_pressure_pub = create_publisher_for_topic(
            self, UUVTopics.INTERNAL_PRESSURE
        )
        self._ext_temp_pub = create_publisher_for_topic(
            self, UUVTopics.EXTERNAL_TEMPERATURE
        )
        self._int_temp_pub = create_publisher_for_topic(
            self, UUVTopics.INTERNAL_TEMPERATURE
        )
        self._leak_pub = create_publisher_for_topic(self, UUVTopics.INTERNAL_LEAK)
        # Measured BCU motor RPM the STM beats back at ~1 Hz. Same topic the
        # sim HAL bridge publishes, so liveness + UI see the same shape on
        # hardware as in Gazebo.
        self._bcu_feedback_rpm_pub = create_publisher_for_topic(
            self, UUVTopics.BCU_FEEDBACK_RPM
        )
        # Actual valve bitmap the STM reports on 0x2107. Same topic the sim HAL
        # bridge echoes, so liveness + UI see the same shape sim vs hardware.
        self._bcu_feedback_valves_pub = create_publisher_for_topic(
            self, UUVTopics.BCU_FEEDBACK_VALVES
        )
        # Single hardware IMU -> IMU_LEFT (Imu + SENSOR_STREAM via the registry),
        # the same topic the sim bridge publishes and the prefilter consumes.
        self._imu_pub = create_publisher_for_topic(self, UUVTopics.IMU_LEFT)

    def _on_rpm(self, msg) -> None:
        self._latest_rpm = int(msg.data)

    def _on_valves(self, msg) -> None:
        # Mask to a byte like can_com does; the firmware re-applies VALVE_MASK.
        self._latest_valves = int(msg.data) & 0xFF

    def _poll_serial(self) -> None:
        # Drain whatever's queued in one shot. STM frames are tiny (4-5 B)
        # so even at 100 Hz polling the buffer never sits on more than a
        # handful of bytes.
        pending = self._ser.in_waiting
        if pending:
            self._rx_buf.extend(self._ser.read(pending))

        # Parse complete frames out of the buffer, leaving any tail for the
        # next tick. The STM is supposed to always lead with 0xAA, but if
        # the line ever desyncs we just drop bytes until we find one again.
        while self._rx_buf:
            sync_idx = self._rx_buf.find(SYNC_BYTE)
            if sync_idx < 0:
                self._rx_buf.clear()
                return
            if sync_idx:
                del self._rx_buf[:sync_idx]

            if len(self._rx_buf) < 1 + HEADER_LEN:
                return  # wait for the rest of the header
            var_id, length = struct.unpack(HEADER_FMT, self._rx_buf[1 : 1 + HEADER_LEN])

            frame_len = 1 + HEADER_LEN + length
            if len(self._rx_buf) < frame_len:
                return  # wait for the rest of the payload

            payload = bytes(self._rx_buf[1 + HEADER_LEN : frame_len])
            del self._rx_buf[:frame_len]

            # Housekeeping/status frames double as the poll cue to push the
            # latest actuator setpoints back down -- but IMU frames do NOT, so a
            # high-rate IMU stream can't multiply our downward RPM/valve TX.
            if self._dispatch(var_id, payload):
                self._send_setpoints()

    def _dispatch(self, var_id: int, payload: bytes) -> bool:
        """Route one frame to its publisher; return True if it's a setpoint cue.

        Every frame except the IMU stream paces our downward setpoint TX, so all
        the housekeeping/status branches (and unknown ids, preserving the prior
        behaviour) return True; the six IMU ids return False.
        """
        if var_id == EXT_PRESSURE_VAR_ID:
            self._publish_pressure(self._ext_pressure_pub, payload)
        elif var_id == TANK_PRESSURE_VAR_ID:
            # Gauge relative to hull, so on a sealed tank at hull pressure
            # it sits near zero until the BCU actually pumps. Forward it
            # anyway so BCU_PRESSURE stays populated for liveness.
            self._publish_pressure(self._tank_pressure_pub, payload)
        elif var_id == INT_PRESSURE_VAR_ID:
            self._publish_pressure(self._int_pressure_pub, payload)
        elif var_id == EXT_TEMP_VAR_ID:
            self._publish_temperature(self._ext_temp_pub, payload)
        elif var_id == INT_TEMP_VAR_ID:
            self._publish_temperature(self._int_temp_pub, payload)
        elif var_id == LEAKS_VAR_ID:
            self._publish_leaks(payload)
        elif var_id == BCU_STATUS_VAR_ID:
            self._publish_bcu_rpm_feedback(payload)
        elif var_id == VALVES_STATUS_VAR_ID:
            self._publish_valves_feedback(payload)
        elif var_id in IMU_VAR_IDS:
            self._handle_imu(var_id, payload)
            return False  # IMU frames never cue a setpoint send
        else:
            self.get_logger().debug(f"unknown var_id=0x{var_id:04x} len={len(payload)}")
        return True

    def _publish_pressure(self, pub, payload: bytes) -> None:
        if len(payload) != UINT16_LEN:
            self.get_logger().warning(f"pressure payload wrong length: {len(payload)}")
            return
        (raw,) = struct.unpack(UINT16_FMT, payload)
        pub.publish(Int32(data=raw * STM_PRESSURE_LSB_PA))

    def _publish_temperature(self, pub, payload: bytes) -> None:
        if len(payload) != INT16_LEN:
            self.get_logger().warning(
                f"temperature payload wrong length: {len(payload)}"
            )
            return
        (raw,) = struct.unpack(INT16_FMT, payload)
        msg = Temperature()
        msg.temperature = float(raw) * STM_TEMPERATURE_LSB_C
        pub.publish(msg)

    def _publish_bcu_rpm_feedback(self, payload: bytes) -> None:
        if len(payload) != INT16_LEN:
            self.get_logger().warning(
                f"bcu status payload wrong length: {len(payload)}"
            )
            return
        (rpm,) = struct.unpack(INT16_FMT, payload)
        # Flip back into the ROS-side convention so feedback matches the
        # commanded sign (see STM_BCU_RPM_SIGN -- the pump is wired reversed).
        self._bcu_feedback_rpm_pub.publish(Int16(data=STM_BCU_RPM_SIGN * rpm))

    def _publish_valves_feedback(self, payload: bytes) -> None:
        if len(payload) != UINT8_LEN:
            self.get_logger().warning(
                f"valve status payload wrong length: {len(payload)}"
            )
            return
        (bitmap,) = struct.unpack(UINT8_FMT, payload)
        self._bcu_feedback_valves_pub.publish(UInt8(data=bitmap))

    def _publish_leaks(self, payload: bytes) -> None:
        if len(payload) != UINT8_LEN:
            self.get_logger().warning(f"leak payload wrong length: {len(payload)}")
            return
        (raw,) = struct.unpack(UINT8_FMT, payload)
        self._leak_pub.publish(UInt8MultiArray(data=[raw]))
        if raw:
            self.get_logger().warning(f"leak detected, bitmask=0x{raw:02x}")

    def _handle_imu(self, var_id: int, payload: bytes) -> None:
        # Latch one axis' raw count; assemble + publish the whole Imu when the
        # last word of the batch (GYRO_Z) arrives.
        if len(payload) != INT16_LEN:
            self.get_logger().warning(
                f"imu payload wrong length: 0x{var_id:04x} len={len(payload)}"
            )
            return
        (self._imu_counts[var_id],) = struct.unpack(INT16_FMT, payload)
        if var_id == GYRO_Z_VAR_ID:
            self._publish_imu()

    def _publish_imu(self) -> None:
        c = self._imu_counts
        accel, gyro = imu_counts_to_body(
            c[ACCEL_X_VAR_ID],
            c[ACCEL_Y_VAR_ID],
            c[ACCEL_Z_VAR_ID],
            c[GYRO_X_VAR_ID],
            c[GYRO_Y_VAR_ID],
            c[GYRO_Z_VAR_ID],
        )
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = IMU_FRAME_ID
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = accel
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = gyro
        # This IMU streams no orientation; the REP-145 / sensor_msgs convention is
        # to flag that with orientation_covariance[0] = -1 so the EKF skips it.
        msg.orientation_covariance[0] = -1.0
        self._imu_pub.publish(msg)

    def _send_setpoints(self) -> None:
        # The STM treats every frame from us as a cue to latch the latest
        # setpoints, so we push RPM and valves together on each poll cue.
        self._send_rpm()
        self._send_valves()

    def _send_rpm(self) -> None:
        # Flip into the hardware's wiring polarity at the wire (see
        # STM_BCU_RPM_SIGN). The cached _latest_rpm stays in the ROS-side
        # convention; only the bytes on the UART carry the reversed sign.
        wire_rpm = STM_BCU_RPM_SIGN * self._latest_rpm
        packet = (
            SYNC_BYTE
            + struct.pack(HEADER_FMT, BCU_RPM_VAR_ID, BCU_RPM_PAYLOAD_LEN)
            + struct.pack(PAYLOAD_FMT, wire_rpm)
        )
        self._ser.write(packet)
        self.get_logger().debug(f"tx bcu rpm: {wire_rpm} (ros {self._latest_rpm})")

    def _send_valves(self) -> None:
        packet = (
            SYNC_BYTE
            + struct.pack(HEADER_FMT, VALVES_TARGET_VAR_ID, UINT8_LEN)
            + struct.pack(UINT8_FMT, self._latest_valves)
        )
        self._ser.write(packet)
        self.get_logger().debug(f"tx valves: {self._latest_valves:#04x}")

    def destroy_node(self) -> bool:
        try:
            self._ser.close()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = STMComNode()
    spin_node(node)


if __name__ == "__main__":
    main()
