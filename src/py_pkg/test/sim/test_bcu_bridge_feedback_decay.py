"""Tier 3 regression: the BCU feedback echo decays after commands stop.

In-process bridge test on the shared ``_bcu_bridge_harness`` rig (no
Gazebo): the probe drives BCU_RPM the way bcu_debug does -- a held
command at 10 Hz, a trailing-zero flush, then silence.

Regression for the stuck UI "Feedback RPM" gauge: the bridge used to
step PumpDynamics only inside rpm_callback, so when the commander went
silent ~0.5 s after its stop (bcu_debug's flush window) the 0 never aged
through the ~1 s transport delay -- the last effective RPM froze nonzero
and the 10 Hz feedback echo re-published the stale value forever. Plant
stepping now rides the publish timer, so the transient keeps evolving
after the last message and the echo ramps down to 0.

The flip side is pinned too: a nonzero command followed by silence keeps
the pump running (STM semantics -- stopping requires an explicit 0).

Don't run alongside any other sim/rclpy process on the host -- the
production topic names overlap.
"""

import time

import pytest
from std_msgs.msg import Int16

from ._sim_helpers import spin_for, spin_until

pytestmark = pytest.mark.sim

# The dave sister repo may not be built (nautilus-ros-only CI): skip,
# don't fail, exactly like test_buoyancy_budget_parity.
pytest.importorskip("nautilus_hal")

from ._bcu_bridge_harness import (  # noqa: E402
    CMD_RATE_HZ,
    HELD_RPM,
    MIN_VISIBLE_RPM,
    PLANT,
    BridgeProbe,
    make_rig,
    wait_wired,
)

# bcu_debug's stop shape: one immediate 0 plus FLUSH_TICKS trailing zeros.
FLUSH_ZEROS = 6
# Hold the command a little past the dead time so the shaft visibly
# spins up before the stop lands.
HOLD_S = PLANT.pump_response_delay_s + 0.5
# All samples in this trailing window must be 0 to call the echo settled.
QUIET_WINDOW_S = 0.5
# After the last message the echo must reach 0 within: the flush 0 aging
# through the dead time, plus slewing down from the highest speed the
# hold could have reached (target held nonzero for at most the hold plus
# the flush, so the down-ramp mirrors that), plus scheduling slack.
RAMP_DOWN_BUDGET_S = (
    PLANT.pump_response_delay_s + HOLD_S + FLUSH_ZEROS / CMD_RATE_HZ + 3.0
)


class _EchoProbe(BridgeProbe):
    """Adds the trailing-window echo view the settle assertions use."""

    def __init__(self):
        super().__init__("bcu_echo_probe")

    def recent_fb(self, window_s: float) -> list[int]:
        cutoff = time.monotonic() - window_s
        return [v for t, v in self.fb_samples if t >= cutoff]


rig = make_rig(_EchoProbe)


def _drive(probe: _EchoProbe, executor, cmds: list[int]):
    period_s = 1.0 / CMD_RATE_HZ
    for rpm in cmds:
        probe.rpm_pub.publish(Int16(data=rpm))
        spin_for(executor, period_s)


def test_echo_decays_to_zero_after_command_stream_stops(rig):
    """bcu_debug's hold + flush + silence must land the echo at 0."""
    probe, executor = rig
    wait_wired(probe, executor)

    _drive(
        probe,
        executor,
        [HELD_RPM] * int(HOLD_S * CMD_RATE_HZ) + [0] * FLUSH_ZEROS,
    )

    # Spin-up must be visible (the hold outlasted the dead time) --
    # otherwise the decay assertion below would pass vacuously.
    peak = max(abs(v) for _, v in probe.fb_samples)
    assert peak >= MIN_VISIBLE_RPM, f"echo never moved (peak {peak} rpm)"

    def _quiet() -> bool:
        recent = probe.recent_fb(QUIET_WINDOW_S)
        return len(recent) >= 3 and all(v == 0 for v in recent)

    assert spin_until(executor, _quiet, timeout_s=RAMP_DOWN_BUDGET_S), (
        "feedback echo never settled at 0 after the command stream "
        f"stopped; trailing samples: {[v for _, v in probe.fb_samples][-10:]}"
    )


def test_held_command_survives_publisher_silence(rig):
    """A nonzero command with no follow-up keeps the pump running.

    STM semantics: stopping requires an explicit 0 on the wire, not just
    the publisher falling silent. bcu_debug's trailing-zero flush is what
    lands the stop; this test proves the flush is load-bearing.
    """
    probe, executor = rig
    wait_wired(probe, executor)

    probe.rpm_pub.publish(Int16(data=HELD_RPM))
    # One dead time to age the command in, plus a second of slew.
    spin_for(executor, PLANT.pump_response_delay_s + 1.0)

    recent = probe.recent_fb(QUIET_WINDOW_S)
    assert recent, "no echo samples in the trailing window"
    assert min(abs(v) for v in recent) >= MIN_VISIBLE_RPM, (
        "pump did not keep spinning on a held command through publisher "
        f"silence; trailing samples: {recent!r}"
    )
