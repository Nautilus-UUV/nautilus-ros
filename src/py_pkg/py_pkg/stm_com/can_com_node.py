"""ROS 2 -> CAN bus bridge for the Control Unit (CU) board.

The CAN-side twin of ``stm_com_node``. Where the STM bridge ships a single
RPM setpoint over UART, the CU board reads *everything* it drives off one
PDO frame: BCU rpm, both ACU axes, and the valve bitmask. So we subscribe to
those four actuator topics, cache the latest of each, and heartbeat the
packed frame onto SocketCAN at a steady rate. A continuous stream also means
the CU board can watchdog the link if it ever wants to.

Wire format (CAN ID 0x181, standard 11-bit, 8 data bytes, little-endian):

    bytes 0-1  ACU_pitch   uint16  (mm of prismatic travel)
    bytes 2-3  BCU         int16   (rpm; negative = deflate / sink)
    bytes 4-5  ACU_roll    int16   (centidegrees, +/-3000 = +/-30 deg)
    byte  6    Valves      uint8   bitmask (bit0=valve1, bit1=valve2; 1=open)
    byte  7    reserved    0       (spare for future BCU state info)

    struct.pack("<HhhBB", pitch_mm, bcu_rpm, acu_roll_cdeg, valves, 0)

We talk to the bus through a raw ``AF_CAN`` / ``CAN_RAW`` socket -- the same
syscalls ``cansend`` and python-can use under the hood, but with no extra
dependency. That ties us to Linux SocketCAN, which is exactly the deployment.
The bus *bitrate* is the OS's job (``ip link set can0 type can bitrate
125000``); we just attach to the already-up interface.

Until the control stack has published anything we send all-zeros: motor off,
valves closed, neutral attitude -- the safe default if the CU reads early.
"""

import socket
import struct

import rclpy
from rclpy.node import Node

from py_pkg.uuv_ros_core import UUVTopics, create_subscription_for_topic, spin_node

# <H h h B B> -> pitch(u16), bcu(i16), roll(i16), valves(u8), reserved(u8).
PDO_FMT = "<HhhBB"
# The kernel can_frame struct: 32-bit id, 1-byte dlc, 3 pad bytes, 8 data.
CAN_FRAME_FMT = "<IB3x8s"

# ACU_PITCH rides the wire as an unsigned 16-bit but the topic is a signed
# Int16, and struct would raise on a negative. Clamp into range -- correct
# for the whole tested span (pitch >= 0); the encoding for negative travel is
# a firmware-contract question still open, so don't abs() (that flips sense).
PITCH_MIN = 0
PITCH_MAX = 0xFFFF


class CANComNode(Node):
    """Heartbeats the actuator PDO (ID 0x181) onto SocketCAN."""

    def __init__(self) -> None:
        super().__init__("can_com")

        self.declare_parameter("channel", "can0")
        self.declare_parameter("can_id", 0x181)
        self.declare_parameter("tx_period_s", 1.0)  # 1 Hz heartbeat

        channel = self.get_parameter("channel").get_parameter_value().string_value
        self._can_id = (
            self.get_parameter("can_id").get_parameter_value().integer_value
        )
        tx_period = (
            self.get_parameter("tx_period_s").get_parameter_value().double_value
        )

        # Attach to the already-up SocketCAN interface. bind() raises if the
        # interface isn't there -- fail fast, same as stm_com on a missing
        # serial port. The node is launch-gated (enable_can_com:=) so sim and
        # dev hosts without a CAN device never reach this.
        self._sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self._sock.bind((channel,))
        self.get_logger().info(
            f"can_com bound {channel}, tx id 0x{self._can_id:X} "
            f"@ {1.0 / tx_period:.0f} Hz"
        )

        # Safe defaults until each topic first speaks.
        self._pitch_mm: int = 0
        self._bcu_rpm: int = 0
        self._roll_cdeg: int = 0
        self._valves: int = 0

        create_subscription_for_topic(self, UUVTopics.BCU_RPM, self._on_rpm)
        create_subscription_for_topic(self, UUVTopics.BCU_VALVES, self._on_valves)
        create_subscription_for_topic(self, UUVTopics.ACU_PITCH, self._on_pitch)
        create_subscription_for_topic(self, UUVTopics.ACU_ROLL, self._on_roll)

        self.create_timer(tx_period, self._send_pdo)

    # --- callbacks: cache only, no I/O ---------------------------------

    def _on_rpm(self, msg) -> None:
        self._bcu_rpm = int(msg.data)

    def _on_valves(self, msg) -> None:
        self._valves = int(msg.data) & 0xFF

    def _on_pitch(self, msg) -> None:
        self._pitch_mm = int(msg.data)

    def _on_roll(self, msg) -> None:
        self._roll_cdeg = int(msg.data)

    # --- tx ------------------------------------------------------------

    def _send_pdo(self) -> None:
        pitch = max(PITCH_MIN, min(PITCH_MAX, self._pitch_mm))
        payload = struct.pack(
            PDO_FMT, pitch, self._bcu_rpm, self._roll_cdeg, self._valves, 0
        )
        # 0x181 fits in 11 bits so the id needs no flags; an extended id would
        # OR in socket.CAN_EFF_FLAG here.
        frame = struct.pack(CAN_FRAME_FMT, self._can_id, len(payload), payload)
        try:
            self._sock.send(frame)
        except OSError as exc:
            # A full tx queue or a bus that just dropped shouldn't take the
            # timer down with it -- log and let the next tick try again.
            self.get_logger().warning(f"can tx failed: {exc}")
            return
        self.get_logger().debug(
            f"tx pdo pitch={pitch} rpm={self._bcu_rpm} "
            f"roll={self._roll_cdeg} valves={self._valves:#04x}"
        )

    def destroy_node(self) -> bool:
        try:
            self._sock.close()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CANComNode()
    spin_node(node)


if __name__ == "__main__":
    main()
