"""Tier 3: the per-timestamp ground-truth anomaly label stream.

In-process tests of ``anomaly_label_bridge`` (no Gazebo): a nominal run
broadcasts an explicit ``active=false`` stream (downstream never infers
from absence); a sensor run carries channel/archetype with constant
``active=true``; and a bcu_pump run gates ``active`` on actual pump
actuation — last commanded ``/bcu/rpm`` nonzero AND the motor valve
open, the shared ``pump_flow_active`` gate — so idle phases of a
degraded-pump run are not mislabeled as anomalous data.

Don't run alongside any other sim/rclpy process on the host — the
production topic names overlap.
"""

import pytest

from ._sim_helpers import spin_until

pytestmark = pytest.mark.sim

# The dave sister repo may not be built (nautilus-ros-only CI): skip,
# don't fail, exactly like test_buoyancy_budget_parity.
pytest.importorskip("nautilus_hal")

import rclpy  # noqa: E402
from nautilus_hal.bridges.anomaly_label_bridge import AnomalyLabelBridge  # noqa: E402
from py_pkg.robot_specs import BCU_MOTOR_VALVE_MASK  # noqa: E402
from py_pkg.uuv_ros_core import (  # noqa: E402
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)
from rclpy.node import Node  # noqa: E402
from std_msgs.msg import Int16, UInt8  # noqa: E402

from ._bcu_bridge_harness import make_rig  # noqa: E402


class _LabelProbe(Node):
    def __init__(self):
        super().__init__("anomaly_label_probe")
        self.labels = []
        create_subscription_for_topic(self, UUVTopics.ANOMALY_LABEL, self.labels.append)
        self.rpm_pub = create_publisher_for_topic(self, UUVTopics.BCU_RPM)
        self.valves_pub = create_publisher_for_topic(self, UUVTopics.BCU_VALVES)


rig_nominal = make_rig(_LabelProbe, name="rig_nominal", bridge_cls=AnomalyLabelBridge)
rig_sensor = make_rig(
    _LabelProbe,
    init_args=[
        "--ros-args",
        "-p",
        "anomaly_class:=sensor",
        "-p",
        "channel:=tank_pressure",
        "-p",
        "archetype:=stuck",
    ],
    name="rig_sensor",
    bridge_cls=AnomalyLabelBridge,
)
rig_pump = make_rig(
    _LabelProbe,
    init_args=["--ros-args", "-p", "anomaly_class:=bcu_pump"],
    name="rig_pump",
    bridge_cls=AnomalyLabelBridge,
)


def _wait_fresh_labels(probe, executor, n=5, timeout_s=10.0):
    """Spin until `n` labels published *after* the call arrive; return them."""
    start = len(probe.labels)
    assert spin_until(
        executor, lambda: len(probe.labels) >= start + n, timeout_s=timeout_s
    ), "label stream stalled"
    return probe.labels[start:]


def test_nominal_run_streams_explicit_inactive_labels(rig_nominal):
    probe, executor = rig_nominal
    labels = _wait_fresh_labels(probe, executor)
    assert all(l.anomaly_class == "nominal" for l in labels)
    assert all(l.channel == "" and l.archetype == "" for l in labels)
    assert all(l.active is False for l in labels)
    # Stamped on the node clock, so bag consumers can align by time.
    assert all(l.header.stamp.sec > 0 for l in labels)


def test_sensor_run_labels_whole_run_active(rig_sensor):
    probe, executor = rig_sensor
    labels = _wait_fresh_labels(probe, executor)
    assert all(l.anomaly_class == "sensor" for l in labels)
    assert all(l.channel == "tank_pressure" for l in labels)
    assert all(l.archetype == "stuck" for l in labels)
    assert all(l.active is True for l in labels)


def test_bcu_pump_active_gates_on_rpm_and_motor_valve(rig_pump):
    probe, executor = rig_pump
    # Idle: fault present but not manifesting.
    labels = _wait_fresh_labels(probe, executor)
    assert all(l.anomaly_class == "bcu_pump" for l in labels)
    assert all(l.active is False for l in labels)

    # RPM alone is not actuation: the closed motor valve deadheads
    # the pump, so no flow and no data deviation.
    probe.rpm_pub.publish(Int16(data=3000))
    labels = _wait_fresh_labels(probe, executor)
    assert all(l.active is False for l in labels)

    # RPM + open motor valve = actuating -> active.
    probe.valves_pub.publish(UInt8(data=BCU_MOTOR_VALVE_MASK))
    assert spin_until(
        executor,
        lambda: probe.labels and probe.labels[-1].active is True,
        timeout_s=10.0,
    ), "label never went active under rpm+valve actuation"
    labels = _wait_fresh_labels(probe, executor)
    assert all(l.active is True for l in labels)

    # Commanded back to zero -> inactive again.
    probe.rpm_pub.publish(Int16(data=0))
    assert spin_until(
        executor,
        lambda: probe.labels and probe.labels[-1].active is False,
        timeout_s=10.0,
    ), "label never went inactive after rpm 0"


def test_unknown_class_fails_fast():
    rclpy.init(args=["--ros-args", "-p", "anomaly_class:=gremlins"])
    try:
        with pytest.raises(ValueError, match="anomaly_class"):
            AnomalyLabelBridge()
    finally:
        rclpy.shutdown()
