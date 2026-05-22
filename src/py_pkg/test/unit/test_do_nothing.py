"""Tier 1 unit tests for DoNothingMission.

Pure logic — no rclpy, no executor, no sim. Pins the "control does
nothing" contract: `reference()` yields no setpoint (None) at every
mission_t, the mission never self-terminates, and it advertises
`resets_control_on_start` so the executor wipes the controllers back to
their fresh, no-mission state when it starts.
"""

import pytest

from py_pkg.path.missions import DoNothingMission, MissionId, create_mission
from py_pkg.path.missions.profile import MissionState


class TestReferenceIsNone:
    """No setpoint, ever — the executor publishes nothing, so the
    controllers fall back to their no-target safe hold."""

    @pytest.mark.parametrize("t", [0.0, 1.0, 60.0, 3600.0, 1e9])
    def test_reference_is_none(self, t):
        m = DoNothingMission()
        m.start(MissionState())
        assert m.reference(t) is None

    def test_pressure_samples_do_not_produce_a_setpoint(self):
        m = DoNothingMission()
        m.start(MissionState())
        for p in (0.0, 50_000.0, 150_000.0):
            m.update(p)
            assert m.reference(0.0) is None

    def test_command_parameters_are_ignored(self):
        m = DoNothingMission()
        m.start(
            MissionState(target_pressure_pa=80_000.0, angle_rad=0.5, n_resurfaces=3)
        )
        assert m.reference(0.0) is None


class TestNeverDone:
    """DoNothing holds until an external stop/abort; is_done is always
    False regardless of time or pressure history."""

    @pytest.mark.parametrize("t", [0.0, 1.0, 1e6, 1e9])
    def test_is_done_false(self, t):
        m = DoNothingMission()
        m.start(MissionState())
        assert m.is_done(t) is False

    def test_is_done_false_after_updates(self):
        m = DoNothingMission()
        m.start(MissionState())
        for t in (0.0, 10.0, 100.0):
            m.update(0.0)
            assert m.is_done(t) is False


class TestResetsControlOnStart:
    """The executor reads this flag to emit CONTROL_RESET on start."""

    def test_flag_is_true(self):
        assert DoNothingMission().resets_control_on_start is True


class TestFactoryRegistration:
    def test_create_mission_returns_do_nothing(self):
        m = create_mission(int(MissionId.DO_NOTHING))
        assert isinstance(m, DoNothingMission)

    def test_do_nothing_id_is_3(self):
        assert int(MissionId.DO_NOTHING) == 3
