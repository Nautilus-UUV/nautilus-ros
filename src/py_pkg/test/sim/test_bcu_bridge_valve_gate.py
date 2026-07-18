"""Tier 3 regression: valve 2 (motor way) gates the BCU hydraulic path.

In-process bridge test on the shared ``_bcu_bridge_harness`` rig (no
Gazebo): the probe drives BCU_RPM + BCU_VALVES and records the flow
echo, the feedback RPM, and the volume commands the bridge pushes
toward Gazebo.

Hardware parity: the pump only carries oil through valve 2. The bridge
must deadhead a closed valve -- shaft spins (feedback echo live), zero
flow, zero transfer -- and carry/stop the transfer as the valve opens
and closes. bcu_debug's "open valve 2 (motor way) to carry the flow"
warning describes exactly this plant behavior.

Don't run alongside any other sim/rclpy process on the host -- the
production topic names overlap.
"""

import time

import pytest
from py_pkg.robot_specs import BCU_MOTOR_VALVE_MASK
from py_pkg.scenarios.spec.rig import BcuBridgeSpec
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Float32, Float64, Int16, UInt8

from ._sim_helpers import spin_for, spin_until, window

pytestmark = pytest.mark.sim

# The dave sister repo may not be built (nautilus-ros-only CI): skip,
# don't fail, exactly like test_bcu_bridge_feedback_decay.
pytest.importorskip("nautilus_hal")

from nautilus_hal.constants import SimTopics  # noqa: E402

from ._bcu_bridge_harness import (  # noqa: E402
    CMD_RATE_HZ,
    HELD_RPM,
    MIN_VISIBLE_RPM,
    PLANT,
    SEED_VOLUME_M3,
    SIM,
    BridgeProbe,
    make_rig,
    wait_wired,
)

# Hold commands past the dead time so the shaft transient fully lands.
HOLD_S = PLANT.pump_response_delay_s + 1.0
# The bridge steps the plant on its publish timer; give the gate a few
# of those ticks of grace to act after a valve edge.
GATE_GRACE_S = 5.0 / BcuBridgeSpec().publish_rate_hz


class _GateProbe(BridgeProbe):
    """Adds the valve command path; records the flow echo and the
    volume commands the bridge pushes toward Gazebo."""

    def __init__(self):
        super().__init__("bcu_valve_gate_probe")
        self.flow_samples: list[tuple[float, float]] = []
        self.volume_cmds: list[tuple[float, float]] = []
        self.valves_pub = create_publisher_for_topic(self, UUVTopics.BCU_VALVES)
        # The volume command the bridge pushes toward Gazebo == the transfer.
        self.create_subscription(
            Float64,
            SimTopics.BUOYANCY_COMMAND.format(model_name=SIM.model_name),
            self._on_volume_cmd,
            10,
        )
        create_subscription_for_topic(self, UUVTopics.BCU_FLOW_RATE, self._on_flow)

    def _on_flow(self, msg: Float32) -> None:
        self.flow_samples.append((time.monotonic(), float(msg.data)))

    def _on_volume_cmd(self, msg: Float64) -> None:
        self.volume_cmds.append((time.monotonic(), float(msg.data)))


rig = make_rig(_GateProbe)


def _wait_armed(probe: _GateProbe, executor: SingleThreadedExecutor) -> None:
    wait_wired(probe, executor)
    # The bridge starts pushing volume commands only once seeded.
    assert spin_until(
        executor, lambda: len(probe.volume_cmds) > 0, timeout_s=10.0
    ), "no volume command after seeding -- integrate-and-push path not armed?"


def _drive(
    probe: _GateProbe,
    executor: SingleThreadedExecutor,
    rpm: int,
    valve_mask: int,
    duration_s: float,
) -> None:
    period_s = 1.0 / CMD_RATE_HZ
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        probe.rpm_pub.publish(Int16(data=rpm))
        probe.valves_pub.publish(UInt8(data=valve_mask))
        spin_for(executor, period_s)


def test_closed_valve_deadheads(rig):
    """Valve 2 closed + RPM held: shaft spins, zero flow, zero transfer."""
    probe, executor = rig
    _wait_armed(probe, executor)

    _drive(probe, executor, HELD_RPM, 0, HOLD_S)

    # Non-vacuity: the shaft must have visibly spun up (the gate blocks
    # the hydraulic path, not the motor).
    peak = max(abs(v) for _, v in probe.fb_samples)
    assert peak >= MIN_VISIBLE_RPM, f"shaft never spun up (peak {peak} rpm)"

    nonzero_flow = [v for _, v in probe.flow_samples if v != 0.0]
    assert (
        nonzero_flow == []
    ), f"flow echo moved with valve 2 closed: {nonzero_flow[:5]!r}"
    moved = [v for _, v in probe.volume_cmds if abs(v - SEED_VOLUME_M3) > 1e-12]
    assert (
        moved == []
    ), f"bladder volume departed the seed with valve 2 closed: {moved[:5]!r}"


def test_open_valve_carries_flow_and_close_stops_it(rig):
    """Opening valve 2 carries the transfer; closing it stops the
    transfer on the next ticks even while the shaft is still spinning."""
    probe, executor = rig
    _wait_armed(probe, executor)

    # Open valve 2 and drive: flow + volume must move.
    _drive(probe, executor, HELD_RPM, BCU_MOTOR_VALVE_MASK, HOLD_S)
    open_flow = [v for _, v in probe.flow_samples if v > 0.0]
    assert open_flow, "no positive flow with valve 2 open and +RPM held"
    assert probe.volume_cmds[-1][1] > SEED_VOLUME_M3, (
        "bladder volume did not rise with valve 2 open: "
        f"last={probe.volume_cmds[-1][1]!r} seed={SEED_VOLUME_M3!r}"
    )

    # Close the valve, keep the RPM command held: deadhead again.
    t_close = time.monotonic()
    _drive(probe, executor, HELD_RPM, 0, GATE_GRACE_S)
    volume_at_close = probe.volume_cmds[-1][1]

    _drive(probe, executor, HELD_RPM, 0, HOLD_S)

    # The shaft is still commanded and spinning...
    recent_fb = window(probe.fb_samples, t_close + GATE_GRACE_S)
    assert (
        recent_fb and min(abs(v) for v in recent_fb) >= MIN_VISIBLE_RPM
    ), f"shaft stopped on valve close -- gate hit the motor: {recent_fb[-5:]!r}"
    # ...but past the grace window the flow is 0 and the volume is pinned.
    late_flow = window(probe.flow_samples, t_close + GATE_GRACE_S)
    assert late_flow and all(
        v == 0.0 for v in late_flow
    ), f"flow echo persisted after valve close: {late_flow[:5]!r}"
    drifted = [
        v
        for v in window(probe.volume_cmds, t_close + GATE_GRACE_S)
        if abs(v - volume_at_close) > 1e-12
    ]
    assert (
        drifted == []
    ), f"bladder volume kept moving after valve close: {drifted[:5]!r}"
