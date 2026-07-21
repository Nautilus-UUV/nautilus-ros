"""Tier 3: the BCU flow echo pins to 0.0 at a bladder rail (deadhead).

In-process bridge test on the shared ``_bcu_bridge_harness`` rig (no
Gazebo boot -- seconds, not minutes): seed the ros_gz volume echo AT
``bladder_max_m3``, command a filling pump with the motor valve open, and
prove the applied-flow echo (``/bcu/flow_rate``) reads exactly 0.0 once
the volume integrator pins at the rail -- the pump deadheads -- while the
feedback rpm echo (``/bcu/feedback/rpm``) keeps reporting the spinning
shaft.

Contract for the v2 "applied flow" semantics: once synced to Gazebo's
bladder volume the flow echo reports the flow the volume clamp actually
admitted, so at a rail (fill frozen) it must say 0.0 rather than the
open-loop ``rps * displacement`` the shaft would otherwise carry. The
feedback rpm echo is deliberately unaffected: the real STM reports shaft
speed, and a deadheaded pump still spins.

Don't run alongside any other sim/rclpy process on the host -- the
production topic names overlap.
"""

import time

import pytest
from std_msgs.msg import Float32, Float64, Int16, UInt8

from ._sim_helpers import spin_for, spin_until

pytestmark = pytest.mark.sim

# The dave sister repo may not be built (nautilus-ros-only CI): skip,
# don't fail, exactly like test_buoyancy_budget_parity.
pytest.importorskip("nautilus_hal")

from py_pkg.robot_specs import BCU_MOTOR_VALVE_MASK  # noqa: E402
from py_pkg.uuv_ros_core import (  # noqa: E402
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)

from ._bcu_bridge_harness import (  # noqa: E402
    HELD_RPM,
    MIN_VISIBLE_RPM,
    PLANT,
    BridgeProbe,
    make_rig,
)

# Dead time + slew up to a clearly-spinning shaft, plus generous slack.
SPINUP_BUDGET_S = (
    PLANT.pump_response_delay_s + MIN_VISIBLE_RPM / PLANT.pump_slew_rpm_per_s + 6.0
)


class _RailProbe(BridgeProbe):
    """Adds a valve-command publisher and a ``/bcu/flow_rate`` tap."""

    def __init__(self):
        super().__init__("bcu_rail_probe")
        self.flow_samples: list[tuple[float, float]] = []
        self.valves_pub = create_publisher_for_topic(self, UUVTopics.BCU_VALVES)
        create_subscription_for_topic(self, UUVTopics.BCU_FLOW_RATE, self._on_flow)

    def _on_flow(self, msg: Float32) -> None:
        self.flow_samples.append((time.monotonic(), float(msg.data)))


rig_rail = make_rig(_RailProbe, name="rig_rail")


def test_flow_pins_to_zero_at_bladder_rail_while_shaft_spins(rig_rail):
    probe, executor = rig_rail

    # Arm on the feedback echo (proves the pub/sub graph is up), then seed
    # the bladder volume AT the max rail -- NOT the harness mid-span seed,
    # so the integrator pins immediately.
    assert spin_until(
        executor, lambda: len(probe.fb_samples) > 0, timeout_s=10.0
    ), "feedback echo never arrived -- bridge not spinning?"

    probe.valves_pub.publish(UInt8(data=BCU_MOTOR_VALVE_MASK))
    # Positive rpm fills the bladder (toward max), so any admitted flow
    # only presses harder against the rail -- the clamp holds it pinned.
    probe.rpm_pub.publish(Int16(data=HELD_RPM))
    probe.volume_state_pub.publish(Float64(data=PLANT.bladder_max_m3))

    # Wait for the shaft to visibly spin up past the dead time; by now the
    # volume seed has long been ingested, so the integrator is pinned.
    assert spin_until(
        executor,
        lambda: any(abs(v) >= MIN_VISIBLE_RPM for _, v in probe.fb_samples),
        timeout_s=SPINUP_BUDGET_S,
    ), f"shaft never spun up; trailing fb: {[v for _, v in probe.fb_samples][-10:]}"

    # Observe a fresh window with the shaft spinning and the fill railed.
    flow_mark = len(probe.flow_samples)
    fb_mark = len(probe.fb_samples)
    spin_for(executor, 2.0)

    new_flow = [v for _, v in probe.flow_samples[flow_mark:]]
    new_fb = [v for _, v in probe.fb_samples[fb_mark:]]

    assert len(new_flow) >= 5, f"too few flow samples in the pinned window: {new_flow}"
    # Applied flow is exactly 0.0: the clamp freezes current_volume at the
    # rail, so the (current - prev)/dt echo is a zero difference.
    assert all(
        v == 0.0 for v in new_flow
    ), f"flow echo did not pin at 0.0 at the bladder rail: {new_flow[:8]}"
    # ...while the feedback echo keeps reporting a spinning shaft (deadhead).
    assert new_fb, "no feedback samples in the pinned window"
    assert all(
        v != 0 for v in new_fb
    ), f"feedback rpm collapsed to 0 at the rail -- shaft not spinning? {new_fb[:8]}"
    assert (
        max(new_fb) >= MIN_VISIBLE_RPM
    ), f"shaft speed fell below the visibility floor at the rail: {new_fb[:8]}"
