"""ROS 2 -> STM32 serial bridge for the BCU motor.

The STM32 owns the BCU motor driver and polls us over UART for an updated
RPM setpoint. We subscribe to ``BCU_RPM`` (Int16, signed), cache the most
recent value, and ship it back down the wire every time the STM sends us
anything. The STM's reply is a small status byte we don't interpret yet --
its arrival is purely the cue to push our latest setpoint.

Wire format (matches the STM firmware's switch on variable id):

    0xAA  | var_id (>H) | length (B) | payload (little-endian)

For BCU RPM: var_id = 0x2102, length = 2, payload = ``struct.pack('<h', rpm)``.

Until the depth PID has published a single RPM we send 0 -- safe default,
keeps the motor off if the STM polls before the control stack is up.
"""

import struct

import rclpy
import serial
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from py_pkg.uuv_ros_core import UUVTopics, create_subscription_for_topic

SYNC_BYTE = b"\xaa"
HEADER_FMT = ">HB"  # big-endian uint16 var_id, uint8 length
HEADER_LEN = struct.calcsize(HEADER_FMT)

BCU_RPM_VAR_ID = 0x2102
BCU_RPM_PAYLOAD_FMT = "<h"  # little-endian signed 16-bit
BCU_RPM_PAYLOAD_LEN = struct.calcsize(BCU_RPM_PAYLOAD_FMT)


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
            _var_id, length = struct.unpack(
                HEADER_FMT, self._rx_buf[1 : 1 + HEADER_LEN]
            )

            frame_len = 1 + HEADER_LEN + length
            if len(self._rx_buf) < frame_len:
                return  # wait for the rest of the payload

            del self._rx_buf[:frame_len]
            self._send_rpm()

    def _send_rpm(self) -> None:
        packet = (
            SYNC_BYTE
            + struct.pack(HEADER_FMT, BCU_RPM_VAR_ID, BCU_RPM_PAYLOAD_LEN)
            + struct.pack(BCU_RPM_PAYLOAD_FMT, self._latest_rpm)
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
