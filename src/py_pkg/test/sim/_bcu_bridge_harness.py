"""Shared harness for the in-process BCU-bridge tests (no Gazebo).

House rule: ``nautilus_hal``-importing tests are Tier 3 even without
Gazebo, so the consumers are marker-gated ``@pytest.mark.sim`` -- and
because this module imports ``nautilus_hal`` at the top level, they must
``pytest.importorskip("nautilus_hal")`` BEFORE importing it.

``BridgeProbe`` carries the non-obvious part of these tests: the
volume-state stand-in publisher that impersonates the ros_gz bridge's
bladder-volume echo, so the bridge's integrate-and-push path is armed
like in a real sim. ``make_rig`` builds the one-executor fixture and
``wait_wired`` is the arm-and-seed handshake. Defined once here so the
sibling bridge tests can't drift apart on the rig choreography or the
drive/visibility constants.
"""

import time

import pytest
import rclpy
from nautilus_hal.bridges.bcu_sim_bridge import BCUSimBridge
from nautilus_hal.constants import SimTopics
from py_pkg.scenarios.spec.rig import PlantSpec, SimSpec
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float64, Int16

from ._sim_helpers import spin_until

PLANT = PlantSpec()
SIM = SimSpec()

CMD_RATE_HZ = 10.0
HELD_RPM = 3000
# Same visibility threshold the pump-transient lake test uses: the shaft
# must have visibly spun up for downstream assertions to be non-vacuous.
MIN_VISIBLE_RPM = 100
# Seed the bladder mid-span so a driven run has headroom both ways.
SEED_VOLUME_M3 = (PLANT.bladder_min_m3 + PLANT.bladder_max_m3) / 2.0


class BridgeProbe(Node):
    """Drives BCU_RPM + the volume-state stand-in; records the feedback
    echo. Subclasses add their test-specific I/O on top."""

    def __init__(self, name: str = "bcu_bridge_probe"):
        super().__init__(name)
        self.fb_samples: list[tuple[float, int]] = []
        self.rpm_pub = create_publisher_for_topic(self, UUVTopics.BCU_RPM)
        # Stand-in for the ros_gz bridge's bladder-volume echo, so the
        # bridge's integrate-and-push path is armed like in a real sim.
        self.volume_state_pub = self.create_publisher(
            Float64,
            SimTopics.BUOYANCY_VOLUME_STATE.format(model_name=SIM.model_name),
            10,
        )
        create_subscription_for_topic(self, UUVTopics.BCU_FEEDBACK_RPM, self._on_fb)

    def _on_fb(self, msg: Int16) -> None:
        self.fb_samples.append((time.monotonic(), int(msg.data)))


def make_rig(probe_cls):
    """Module-level ``rig = make_rig(_MyProbe)`` gives a test file the
    shared bridge + probe + executor fixture with one teardown order."""

    @pytest.fixture()
    def rig():
        rclpy.init()
        bridge = BCUSimBridge()
        probe = probe_cls()
        executor = SingleThreadedExecutor()
        executor.add_node(bridge)
        executor.add_node(probe)
        yield probe, executor
        executor.remove_node(probe)
        executor.remove_node(bridge)
        probe.destroy_node()
        bridge.destroy_node()
        executor.shutdown()
        rclpy.shutdown()

    return rig


def wait_wired(probe: BridgeProbe, executor: SingleThreadedExecutor) -> None:
    # The bridge's echo reaching the probe proves the pub/sub graph is up.
    assert spin_until(
        executor, lambda: len(probe.fb_samples) > 0, timeout_s=10.0
    ), "feedback echo never arrived -- bridge not spinning?"
    probe.volume_state_pub.publish(Float64(data=SEED_VOLUME_M3))
