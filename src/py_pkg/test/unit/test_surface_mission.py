"""Tier 1 unit tests for SurfaceMission.

Pure logic — no rclpy, no executor, no sim. Pins the closed-loop
ascent contract: constant z=0 setpoint, identity orientation, and a
self-terminating dwell timer that needs `DWELL_AT_SURFACE_S` of
continuous surface observation before declaring `is_done`.

The executor calls `update(pressure)` then `is_done(mission_t)` per
tick; these tests interleave those calls in the same order.
"""

import math

import pytest
from py_pkg.path.missions.profile import SURFACE_THRESHOLD_PA, MissionState
from py_pkg.path.missions.surface import DWELL_AT_SURFACE_S, SurfaceMission

DEEP_PA = 60_000.0  # ~6 m gauge -- well above any reasonable surface threshold


class TestReferenceIsConstant:
    """Setpoint is gauge 0 Pa with identity attitude regardless of `mission_t`
    or any pressure samples seen so far."""

    def test_z_is_zero(self):
        m = SurfaceMission()
        m.start(MissionState())
        assert m.reference(0.0).position.z == pytest.approx(0.0)

    def test_orientation_is_unit_w(self):
        m = SurfaceMission()
        m.start(MissionState())
        q = m.reference(0.0).orientation
        assert q.x == pytest.approx(0.0)
        assert q.y == pytest.approx(0.0)
        assert q.z == pytest.approx(0.0)
        assert q.w == pytest.approx(1.0)

    def test_orientation_unit_norm(self):
        m = SurfaceMission()
        m.start(MissionState())
        q = m.reference(0.0).orientation
        norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        assert norm == pytest.approx(1.0)

    @pytest.mark.parametrize("t", [0.0, 1.0, 60.0, 3600.0, 1e9])
    def test_reference_constant_in_t(self, t):
        m = SurfaceMission()
        m.start(MissionState())
        ref = m.reference(t)
        assert ref.position.z == pytest.approx(0.0)
        assert ref.orientation.w == pytest.approx(1.0)

    def test_pressure_samples_do_not_change_setpoint(self):
        m = SurfaceMission()
        m.start(MissionState())
        for p in (DEEP_PA, 0.0, SURFACE_THRESHOLD_PA, -1.0):
            m.update(p)
            ref = m.reference(0.0)
            assert ref.position.z == pytest.approx(0.0)
            assert ref.orientation.w == pytest.approx(1.0)

    def test_command_parameters_are_ignored(self):
        # `target_pressure_pa`, `shallow_pressure_pa`, `angle_rad`,
        # `n_resurfaces` are mission-specific — SurfaceMission has none, and
        # any value passed must not bleed into the setpoint.
        m = SurfaceMission()
        m.start(
            MissionState(
                target_pressure_pa=80_000.0,
                shallow_pressure_pa=20_000.0,
                angle_rad=0.5,
                n_resurfaces=3,
            )
        )
        assert m.reference(0.0).position.z == pytest.approx(0.0)


class TestNotDoneBeforeSurfacing:
    """`is_done` only flips after the dwell completes; calls before any
    pressure ingress, or while pressure is above threshold, must return
    False forever."""

    def test_is_done_false_before_any_update(self):
        m = SurfaceMission()
        m.start(MissionState())
        # No update called -> no observation -> not at surface.
        assert m.is_done(0.0) is False
        assert m.is_done(1e6) is False

    def test_is_done_false_while_below_surface(self):
        m = SurfaceMission()
        m.start(MissionState())
        # Step through plausible mission timeline at depth.
        for t in (0.0, 1.0, 5.0, 30.0, 120.0, 1_000.0):
            m.update(DEEP_PA)
            assert m.is_done(t) is False


class TestDwellTimerCompletes:
    """Once `update` reports the glider at the surface, `is_done` must
    transition True after exactly `DWELL_AT_SURFACE_S` of mission-time
    elapsed since the first surfaced sample."""

    def test_done_after_full_dwell(self):
        m = SurfaceMission()
        m.start(MissionState())
        m.update(0.0)  # surfaced
        assert m.is_done(0.0) is False  # latches dwell start at t=0
        assert m.is_done(DWELL_AT_SURFACE_S - 0.1) is False
        # Executor calls update before is_done each tick.
        m.update(0.0)
        assert m.is_done(DWELL_AT_SURFACE_S) is True

    def test_threshold_boundary_counts_as_surfaced(self):
        m = SurfaceMission()
        m.start(MissionState())
        m.update(SURFACE_THRESHOLD_PA)  # exactly on threshold (<= predicate)
        assert m.is_done(0.0) is False
        m.update(SURFACE_THRESHOLD_PA)
        assert m.is_done(DWELL_AT_SURFACE_S) is True

    def test_just_above_threshold_does_not_count(self):
        m = SurfaceMission()
        m.start(MissionState())
        m.update(SURFACE_THRESHOLD_PA + 1.0)
        # Even after a long wait, never-surfaced means never-done.
        assert m.is_done(DWELL_AT_SURFACE_S * 10) is False


class TestDwellTimerResets:
    """A pressure spike above threshold mid-dwell (sea-state, controller
    overshoot) must restart the dwell from zero rather than carry over
    the partial accumulation."""

    def test_spike_above_threshold_resets_timer(self):
        m = SurfaceMission()
        m.start(MissionState())
        # Half the dwell at the surface.
        half = DWELL_AT_SURFACE_S / 2
        m.update(0.0)
        assert m.is_done(0.0) is False  # latch dwell start
        m.update(0.0)
        assert m.is_done(half) is False
        # Bob back below the surface — timer resets.
        m.update(DEEP_PA)
        assert m.is_done(half + 0.5) is False
        # Returning to surface starts a fresh dwell from this mission_t.
        resume_t = half + 1.0
        m.update(0.0)
        assert m.is_done(resume_t) is False  # new latch
        # Carrying over from the original start would have completed by now;
        # only the fresh dwell counts.
        m.update(0.0)
        assert m.is_done(resume_t + DWELL_AT_SURFACE_S - 0.1) is False
        m.update(0.0)
        assert m.is_done(resume_t + DWELL_AT_SURFACE_S) is True


class TestRestart:
    """Calling `start()` again must clear any previous dwell so the new
    mission-instance gets a clean slate."""

    def test_restart_clears_completed_dwell(self):
        m = SurfaceMission()
        m.start(MissionState())
        m.update(0.0)
        m.is_done(0.0)
        m.update(0.0)
        assert m.is_done(DWELL_AT_SURFACE_S) is True
        # Re-arm the same instance.
        m.start(MissionState())
        # No update yet -> _at_surface back to False -> not done.
        assert m.is_done(DWELL_AT_SURFACE_S * 10) is False
