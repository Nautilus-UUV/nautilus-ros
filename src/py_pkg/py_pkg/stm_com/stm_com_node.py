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

Inbound (STM -> Pi):
    0x2400  EXT_PRESSURE   uint16  absolute, 100 Pa / LSB
    0x2401  TANK_PRESSURE  uint16  gauge relative to hull, 100 Pa / LSB
    0x2402  INT_PRESSURE   uint16  absolute, 100 Pa / LSB
    0x2410  EXT_TEMP       int16   0.01 °C / LSB
    0x2411  INT_TEMP       int16   0.01 °C / LSB
    0x2420  LEAKS          uint8   bitmask; any nonzero bit = water detected

Until the depth PID has published a single RPM we send 0 -- safe default,
keeps the motor off if the STM polls before the control stack is up.
"""

import struct

import rclpy
import serial
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Temperature
from std_msgs.msg import Int32, UInt8MultiArray

from py_pkg.robot_specs import STM_PRESSURE_LSB_PA, STM_TEMPERATURE_LSB_C
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)

SYNC_BYTE = b"\xaa"
HEADER_FMT = ">HB"  # big-endian uint16 var_id, uint8 length
HEADER_LEN = struct.calcsize(HEADER_FMT)

BCU_RPM_VAR_ID = 0x2102
EXT_PRESSURE_VAR_ID = 0x2400  # <H — uint16
TANK_PRESSURE_VAR_ID = 0x2401  # <H — uint16  → BCU_PRESSURE
INT_PRESSURE_VAR_ID = 0x2402  # <H — uint16
EXT_TEMP_VAR_ID = 0x2410  # <h — int16
INT_TEMP_VAR_ID = 0x2411  # <h — int16
LEAKS_VAR_ID = 0x2420  # <B — uint8

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
        self._rx_buf = bytearray()

        create_subscription_for_topic(self, UUVTopics.BCU_RPM, self._on_rpm)
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

    def _on_rpm(self, msg) -> None:
        self._latest_rpm = int(msg.data)

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
            var_id, length = struct.unpack(
                HEADER_FMT, self._rx_buf[1 : 1 + HEADER_LEN]
            )

            frame_len = 1 + HEADER_LEN + length
            if len(self._rx_buf) < frame_len:
                return  # wait for the rest of the payload

            payload = bytes(self._rx_buf[1 + HEADER_LEN : frame_len])
            del self._rx_buf[:frame_len]

            self._dispatch(var_id, payload)
            # Every framed message from the STM is also the poll cue for
            # us to push the latest RPM setpoint back down.
            self._send_rpm()

    def _dispatch(self, var_id: int, payload: bytes) -> None:
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
        else:
            self.get_logger().debug(
                f"unknown var_id=0x{var_id:04x} len={len(payload)}"
            )

    def _publish_pressure(self, pub, payload: bytes) -> None:
        if len(payload) != UINT16_LEN:
            self.get_logger().warning(
                f"pressure payload wrong length: {len(payload)}"
            )
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

    def _publish_leaks(self, payload: bytes) -> None:
        if len(payload) != UINT8_LEN:
            self.get_logger().warning(f"leak payload wrong length: {len(payload)}")
            return
        (raw,) = struct.unpack(UINT8_FMT, payload)
        self._leak_pub.publish(UInt8MultiArray(data=[raw]))
        if raw:
            self.get_logger().warning(f"leak detected, bitmask=0x{raw:02x}")

    def _send_rpm(self) -> None:
        packet = (
            SYNC_BYTE
            + struct.pack(HEADER_FMT, BCU_RPM_VAR_ID, BCU_RPM_PAYLOAD_LEN)
            + struct.pack(PAYLOAD_FMT, self._latest_rpm)
        )
        self._ser.write(packet)
        self.get_logger().debug(f"tx bcu rpm: {self._latest_rpm}")

    def destroy_node(self) -> bool:
        try:
            self._ser.close()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = STMComNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
