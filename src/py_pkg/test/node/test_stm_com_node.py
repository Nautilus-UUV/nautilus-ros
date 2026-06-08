"""Tier 2 in-process rclpy test for STMComNode.

Drives the node against a fake serial port (no real UART). Verifies the two new
valve paths: an inbound 0x2107 valve-status frame republishes on
``/bcu/feedback/valves``, and a ``/bcu/valves`` command produces an outbound
0x2106 valve-target frame on the next STM poll cue.

The harness lives here rather than in ``conftest.py`` because STMComNode opens a
serial port in ``__init__`` -- the generic ``NodeHarness`` constructs its node
with no args, so we monkeypatch ``serial.Serial`` to hand back a fake instead.
"""

import struct
import time

import pytest
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Int16, UInt8

import py_pkg.stm_com.stm_com_node as stm
from py_pkg.robot_specs import STM_BCU_RPM_SIGN
from py_pkg.stm_com.stm_com_node import (
    BCU_RPM_VAR_ID,
    BCU_STATUS_VAR_ID,
    HEADER_FMT,
    HEADER_LEN,
    STMComNode,
    SYNC_BYTE,
    VALVES_TARGET_VAR_ID,
    VALVES_STATUS_VAR_ID,
)
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)


def _frame(var_id: int, payload: bytes) -> bytes:
    """Encode one wire frame: SYNC | >HB header | payload."""
    return SYNC_BYTE + struct.pack(HEADER_FMT, var_id, len(payload)) + payload


def _decode_frames(blob: bytes) -> list[tuple[int, bytes]]:
    """Pull every complete (var_id, payload) frame out of a captured byte blob."""
    frames = []
    buf = bytearray(blob)
    while buf:
        idx = buf.find(SYNC_BYTE)
        if idx < 0:
            break
        del buf[:idx]
        if len(buf) < 1 + HEADER_LEN:
            break
        var_id, length = struct.unpack(HEADER_FMT, buf[1 : 1 + HEADER_LEN])
        end = 1 + HEADER_LEN + length
        if len(buf) < end:
            break
        frames.append((var_id, bytes(buf[1 + HEADER_LEN : end])))
        del buf[:end]
    return frames


class _FakeSerial:
    """Drop-in for serial.Serial: an in-memory RX buffer + captured TX bytes."""

    def __init__(self, *args, **kwargs):
        self._rx = bytearray()
        self.tx = bytearray()

    def inject(self, data: bytes) -> None:
        """Queue bytes for the node to read on its next poll."""
        self._rx.extend(data)

    @property
    def in_waiting(self) -> int:
        return len(self._rx)

    def read(self, n: int) -> bytes:
        chunk = bytes(self._rx[:n])
        del self._rx[:n]
        return chunk

    def write(self, data: bytes) -> int:
        self.tx.extend(data)
        return len(data)

    def close(self) -> None:
        pass


class _STMTesterNode(Node):
    """Captures /bcu/feedback/* and publishes the actuator commands."""

    def __init__(self):
        super().__init__("stm_com_tester")
        self.received_valves: list[int] = []
        self.received_feedback_rpm: list[int] = []
        self.valves_cmd_pub = create_publisher_for_topic(self, UUVTopics.BCU_VALVES)
        self.rpm_cmd_pub = create_publisher_for_topic(self, UUVTopics.BCU_RPM)
        create_subscription_for_topic(
            self, UUVTopics.BCU_FEEDBACK_VALVES, self._on_valves
        )
        create_subscription_for_topic(
            self, UUVTopics.BCU_FEEDBACK_RPM, self._on_feedback_rpm
        )

    def _on_valves(self, msg: UInt8) -> None:
        self.received_valves.append(int(msg.data))

    def _on_feedback_rpm(self, msg: Int16) -> None:
        self.received_feedback_rpm.append(int(msg.data))

    def command_valves(self, bitmap: int) -> None:
        self.valves_cmd_pub.publish(UInt8(data=bitmap))

    def command_rpm(self, rpm: int) -> None:
        self.rpm_cmd_pub.publish(Int16(data=rpm))


class STMComHarness:
    def __init__(self, fake: _FakeSerial):
        self.fake = fake
        self.node = STMComNode()
        self.tester = _STMTesterNode()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor.add_node(self.tester)

    def spin_for(self, duration_s: float, slice_s: float = 0.02) -> None:
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=slice_s)

    def spin_until(self, predicate, timeout: float = 2.0, slice_s: float = 0.02):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            self.executor.spin_once(timeout_sec=slice_s)
        if not predicate():
            raise TimeoutError(f"predicate did not become true within {timeout}s")

    def shutdown(self) -> None:
        try:
            self.executor.remove_node(self.node)
            self.executor.remove_node(self.tester)
        finally:
            self.node.destroy_node()
            self.tester.destroy_node()
            self.executor.shutdown()


@pytest.fixture
def stm_harness(monkeypatch):
    fake = _FakeSerial()
    # STMComNode does `serial.Serial(port, baud, timeout=0)` in __init__.
    monkeypatch.setattr(stm.serial, "Serial", lambda *a, **k: fake)
    harness = STMComHarness(fake)
    try:
        yield harness
    finally:
        harness.shutdown()


class TestSTMValves:
    def test_valve_status_frame_republished_as_feedback(self, stm_harness):
        h = stm_harness
        h.fake.inject(_frame(VALVES_STATUS_VAR_ID, struct.pack("<B", 0b11)))
        h.spin_until(lambda: len(h.tester.received_valves) > 0, timeout=2.0)
        assert h.tester.received_valves[-1] == 0b11

    def test_commanded_valves_sent_as_target_frame(self, stm_harness):
        h = stm_harness
        # Let the BCU_VALVES command reach the node's cache first...
        h.tester.command_valves(0b10)
        h.spin_for(0.1)
        # ...then any inbound frame is the poll cue that flushes setpoints down.
        h.fake.inject(_frame(VALVES_STATUS_VAR_ID, struct.pack("<B", 0)))
        h.spin_for(0.1)
        sent = _decode_frames(bytes(h.fake.tx))
        valve_targets = [p for vid, p in sent if vid == VALVES_TARGET_VAR_ID]
        assert valve_targets, "no 0x2106 valve-target frame was sent"
        assert struct.unpack("<B", valve_targets[-1])[0] == 0b10

    def test_rpm_still_sent_on_same_cue(self, stm_harness):
        # Guards the _send_setpoints refactor: valves didn't displace RPM.
        # The wire carries the hardware-polarity-flipped value (see
        # STM_BCU_RPM_SIGN), not the raw ROS-side command.
        h = stm_harness
        h.tester.command_rpm(123)
        h.spin_for(0.1)
        h.fake.inject(_frame(VALVES_STATUS_VAR_ID, struct.pack("<B", 0)))
        h.spin_for(0.1)
        sent = _decode_frames(bytes(h.fake.tx))
        rpm_frames = [p for vid, p in sent if vid == BCU_RPM_VAR_ID]
        assert rpm_frames, "no 0x2102 rpm frame was sent"
        assert struct.unpack("<h", rpm_frames[-1])[0] == STM_BCU_RPM_SIGN * 123


class TestSTMRpmSign:
    """The BCU pump is wired reversed on the bench, so stm_com flips the RPM
    sign at the wire (STM_BCU_RPM_SIGN) on both the commanded setpoint and the
    measured feedback, keeping the whole ROS graph on one convention."""

    def _last_rpm_on_wire(self, h) -> int:
        sent = _decode_frames(bytes(h.fake.tx))
        rpm_frames = [p for vid, p in sent if vid == BCU_RPM_VAR_ID]
        assert rpm_frames, "no 0x2102 rpm frame was sent"
        return struct.unpack("<h", rpm_frames[-1])[0]

    def test_commanded_rpm_sign_flipped_on_wire(self, stm_harness):
        h = stm_harness
        for ros_rpm in (500, -500):
            h.tester.command_rpm(ros_rpm)
            h.spin_for(0.1)
            h.fake.inject(_frame(VALVES_STATUS_VAR_ID, struct.pack("<B", 0)))
            h.spin_for(0.1)
            assert self._last_rpm_on_wire(h) == STM_BCU_RPM_SIGN * ros_rpm

    def test_feedback_rpm_sign_flipped(self, stm_harness):
        # 0x2103 BCU_STATUS carries the motor's measured RPM in the hardware's
        # reversed polarity; the node flips it back before publishing feedback.
        h = stm_harness
        h.fake.inject(_frame(BCU_STATUS_VAR_ID, struct.pack("<h", 800)))
        h.spin_until(lambda: len(h.tester.received_feedback_rpm) > 0, timeout=2.0)
        assert h.tester.received_feedback_rpm[-1] == STM_BCU_RPM_SIGN * 800
