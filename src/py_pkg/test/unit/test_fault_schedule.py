"""Tier 1 for the fault onset/progression envelope (sensor_faults.FaultSchedule).

The schedule is the deterministic ``m(t) in [0, 1]`` gate that scales a
fault's one drawn severity in time. Every claim of its contract is pinned
here:
- the default schedule is constant 1.0 from the first call (v1 parity: a
  run with a default schedule behaves exactly like the original
  whole-run fault),
- step gates 0 -> 1 at the onset,
- ramp rises linearly to 1 over ``ramp_s`` then holds,
- intermittent cycles on/off per ``period_s`` / ``duty`` across periods,
- the epoch is latched by ``start`` OR lazily on the first call,
- ``active_elapsed`` is 0 before onset and grows after,
- an unknown shape / a negative onset raise at construction.
"""

from __future__ import annotations

import pytest
from py_pkg.sensor_faults import FaultSchedule


# ---------------------------------------------------------------------------
# Default schedule == constant 1.0 (v1 parity)
# ---------------------------------------------------------------------------


def test_default_schedule_is_constant_one_from_first_call():
    s = FaultSchedule()
    # No start(): the first multiplier call latches the epoch, and a
    # step-at-t=0 schedule reads 1.0 from that first sample onward.
    assert s.multiplier(0.0) == 1.0
    for t in (0.0, 0.5, 10.0, 1000.0):
        assert s.multiplier(t) == 1.0


def test_default_schedule_lazy_epoch_still_reads_one_at_first_sample():
    # Even when the first sample is at a large clock value, a default
    # (onset 0) schedule reads 1.0 immediately -- the whole-run behavior.
    s = FaultSchedule()
    assert s.multiplier(500.0) == 1.0
    assert s.multiplier(500.1) == 1.0


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


def test_step_gates_zero_then_one_at_onset():
    s = FaultSchedule(onset_s=5.0, shape="step")
    s.start(0.0)
    assert s.multiplier(0.0) == 0.0
    assert s.multiplier(4.999) == 0.0
    assert s.multiplier(5.0) == 1.0  # onset is inclusive
    assert s.multiplier(5.001) == 1.0
    assert s.multiplier(100.0) == 1.0


# ---------------------------------------------------------------------------
# Ramp
# ---------------------------------------------------------------------------


def test_ramp_rises_linearly_then_holds_at_one():
    s = FaultSchedule(onset_s=5.0, shape="ramp", ramp_s=10.0)
    s.start(0.0)
    assert s.multiplier(2.0) == 0.0  # pre-onset
    assert s.multiplier(5.0) == 0.0  # onset: ramp just beginning
    assert s.multiplier(10.0) == pytest.approx(0.5)  # halfway up the ramp
    assert s.multiplier(15.0) == pytest.approx(1.0)  # ramp complete
    assert s.multiplier(30.0) == pytest.approx(1.0)  # holds at 1


# ---------------------------------------------------------------------------
# Intermittent
# ---------------------------------------------------------------------------


def test_intermittent_on_off_pattern_across_several_periods():
    # duty 0.3 of a 10 s period => on for [0, 3) s of each period, off after.
    s = FaultSchedule(onset_s=0.0, shape="intermittent", period_s=10.0, duty=0.3)
    s.start(0.0)
    # First period.
    assert s.multiplier(0.0) == 1.0
    assert s.multiplier(2.999) == 1.0
    assert s.multiplier(3.0) == 0.0  # duty window is half-open [0, 3)
    assert s.multiplier(9.999) == 0.0
    # Second and third periods repeat the same on/off shape.
    assert s.multiplier(10.0) == 1.0
    assert s.multiplier(12.999) == 1.0
    assert s.multiplier(13.0) == 0.0
    assert s.multiplier(20.0) == 1.0
    assert s.multiplier(29.5) == 0.0


def test_intermittent_onset_shifts_the_cycle_origin():
    # onset delays the first period; the cycle counts from the onset.
    s = FaultSchedule(onset_s=5.0, shape="intermittent", period_s=10.0, duty=0.5)
    s.start(0.0)
    assert s.multiplier(4.0) == 0.0  # pre-onset
    assert s.multiplier(5.0) == 1.0  # onset -> first period on-window opens
    assert s.multiplier(9.999) == 1.0  # still within first 5 s on-window
    assert s.multiplier(10.0) == 0.0  # off-window
    assert s.multiplier(14.999) == 0.0
    assert s.multiplier(15.0) == 1.0  # next period on-window


# ---------------------------------------------------------------------------
# Epoch latching: explicit start() vs lazy first-call
# ---------------------------------------------------------------------------


def test_epoch_latched_explicitly_via_start():
    s = FaultSchedule(onset_s=5.0, shape="step")
    s.start(100.0)  # epoch pinned to a nonzero clock
    assert s.multiplier(104.999) == 0.0
    assert s.multiplier(105.0) == 1.0  # onset measured from the started epoch


def test_epoch_latched_lazily_on_first_call():
    s = FaultSchedule(onset_s=5.0, shape="step")
    # No start(): the first multiplier call pins the epoch to its own t.
    assert s.multiplier(100.0) == 0.0  # rel = 100 - 100 - 5 < 0
    assert s.multiplier(104.999) == 0.0
    assert s.multiplier(105.0) == 1.0


# ---------------------------------------------------------------------------
# active_elapsed
# ---------------------------------------------------------------------------


def test_active_elapsed_zero_before_onset_then_grows():
    s = FaultSchedule(onset_s=5.0, shape="step")
    s.start(0.0)
    assert s.active_elapsed(3.0) == 0.0  # pre-onset
    assert s.active_elapsed(5.0) == 0.0  # exactly at onset
    assert s.active_elapsed(8.0) == pytest.approx(3.0)
    assert s.active_elapsed(20.0) == pytest.approx(15.0)


def test_active_elapsed_latches_epoch_lazily():
    s = FaultSchedule(onset_s=2.0, shape="step")
    # First touch is active_elapsed, which must latch the epoch just like
    # multiplier would.
    assert s.active_elapsed(50.0) == 0.0  # rel = 50 - 50 - 2 < 0 -> 0
    assert s.active_elapsed(55.0) == pytest.approx(3.0)  # 55 - 50 - 2


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_unknown_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        FaultSchedule(shape="spike")


def test_negative_onset_raises():
    with pytest.raises(ValueError, match="onset_s"):
        FaultSchedule(onset_s=-1.0)


# ---------------------------------------------------------------------------
# blend: the one law every consumer applies the envelope through
# ---------------------------------------------------------------------------


def test_blend_is_healthy_before_onset_and_severity_after():
    s = FaultSchedule(onset_s=10.0)
    s.start(0.0)
    # Sensor-style: healthy 0.0, drawn severity 5.0.
    assert s.blend(0.0, 5.0, 5.0) == 0.0
    assert s.blend(0.0, 5.0, 15.0) == 5.0
    # Pump-style: healthy 1.0, drawn effectiveness 0.6. The site looks
    # inverted only because its healthy value is 1.0.
    assert s.blend(1.0, 0.6, 5.0) == 1.0
    assert s.blend(1.0, 0.6, 15.0) == pytest.approx(0.6)


def test_blend_tracks_the_ramp_between_the_two_ends():
    s = FaultSchedule(onset_s=0.0, shape="ramp", ramp_s=10.0)
    s.start(0.0)
    assert s.blend(1.0, 0.5, 5.0) == pytest.approx(0.75)  # halfway
    assert s.blend(0.0, 4.0, 5.0) == pytest.approx(2.0)


def test_blend_on_default_schedule_is_the_severity():
    # v1 parity: with no onset, the fault is felt at full severity from
    # the first sample, whichever end is "healthy".
    s = FaultSchedule()
    assert s.blend(1.0, 0.6, 0.0) == pytest.approx(0.6)
    assert s.blend(0.0, 7.0, 0.0) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Vocabulary parity with the wire schema
# ---------------------------------------------------------------------------


def test_shape_vocabulary_matches_the_wire_schema():
    # FAULT_SCHEDULE_SHAPES is the runtime model's own vocabulary (kept
    # pydantic-free); FaultScheduleShape is its wire spelling, and
    # anomaly.ONSET_SHAPES derives from that. This is the tripwire that
    # keeps a shape added on one side from being rejected on the other.
    from typing import get_args

    from py_pkg.scenarios.spec.rig import FaultScheduleShape
    from py_pkg.sensor_faults import FAULT_SCHEDULE_SHAPES

    assert FAULT_SCHEDULE_SHAPES == get_args(FaultScheduleShape)
