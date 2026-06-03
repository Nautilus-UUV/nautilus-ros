"""Tier 2 in-process rclpy tests for PathfindingNode.

Black-box: drive the node via PATH (MissionCommand) / EXTERNAL_PRESSURE /
COMMAND and assert on what it publishes on POSITION_TARGET. The node is
now a thin mission dispatcher — the planner is gone, and per-mission
setpoint generation lives in `path/missions/`.
"""

import time

import pytest
from py_pkg.path.missions import MissionId
from py_pkg.path.missions import surface as surface_mod

# Mission ids used by the tests below.
SAWTOOTH = int(MissionId.SAWTOOTH)
TRIM = int(MissionId.TRIM_AND_NEUTRAL_BUOYANCY)
SURFACE = int(MissionId.SURFACE)
DO_NOTHING = int(MissionId.DO_NOTHING)

# Surface absolute pressure (Pa). gauge_pressure_pa() yields ~0 -> "at surface".
PRESSURE_AT_SURFACE_PA = 101_325
# Deep absolute pressure (Pa) -> ~6 m gauge, well above SURFACE_THRESHOLD_PA.
PRESSURE_AT_DEPTH_PA = 161_325


class TestWiringSmoke:
    def test_node_constructs(self, pathfinding_node_harness):
        assert pathfinding_node_harness.node is not None

    def test_position_estimation_subscription_present(self, pathfinding_node_harness):
        names = [s.topic_name for s in pathfinding_node_harness.node.subscriptions]
        assert "/position/estimation" in names

    def test_external_pressure_subscription_present(self, pathfinding_node_harness):
        names = [s.topic_name for s in pathfinding_node_harness.node.subscriptions]
        assert "/external/pressure" in names

    def test_command_subscription_present(self, pathfinding_node_harness):
        names = [s.topic_name for s in pathfinding_node_harness.node.subscriptions]
        assert "/command" in names

    def test_path_subscription_present(self, pathfinding_node_harness):
        names = [s.topic_name for s in pathfinding_node_harness.node.subscriptions]
        assert "/path" in names

    def test_position_target_publisher_present(self, pathfinding_node_harness):
        names = [p.topic_name for p in pathfinding_node_harness.node.publishers]
        assert "/position/target" in names

    def test_initial_mode_is_idle(self, pathfinding_node_harness):
        assert pathfinding_node_harness.node._mode == "IDLE"


class TestPathIngress:
    def test_known_mission_id_loads_mission(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_mission_command(
            SAWTOOTH, target_pressure_pa=200_000.0, angle_rad=0.5, n_resurfaces=2
        )
        h.spin_until(lambda: h.node._mode == "LOADED", timeout=1.0)
        assert h.node._mission is not None
        assert h.node._mission_cmd is not None
        assert h.node._mission_cmd.mission_id == SAWTOOTH
        assert h.node._mission_cmd.target_pressure_pa == pytest.approx(200_000.0)

    def test_unknown_mission_id_is_rejected(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        # uint8 max -> well outside the registry.
        h.publish_mission_command(255)
        h.spin_for(0.3)
        assert h.node._mode == "IDLE"
        assert h.node._mission is None


class TestStartPreconditions:
    def test_start_without_mission_emits_nothing(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_until(lambda: h.node._current_pressure_pa is not None, timeout=1.0)
        h.publish_command("start")
        h.spin_for(0.5)
        assert h.received_targets == []
        assert h.node._mode == "IDLE"
        # The start is queued, waiting for a mission to load.
        assert h.node._start_pending is True

    def test_start_without_pressure_stays_loaded(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_mission_command(TRIM, target_pressure_pa=50_000.0)
        h.spin_until(lambda: h.node._mode == "LOADED", timeout=1.0)
        h.publish_command("start")
        h.spin_for(0.5)
        # Start queued -> stays LOADED, never RUNNING, no emissions.
        assert h.node._mode == "LOADED"
        assert h.received_targets == []
        assert h.node._start_pending is True


class TestStartRaceTolerance:
    """`start` may race ahead of `/path` or pressure ingress (Issue #43);
    the node must queue the intent and fire it once preconditions hold."""

    def test_start_before_path_fires_on_path(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_command("start")
        h.spin_for(0.2)
        assert h.node._mode == "IDLE"
        assert h.node._start_pending is True

        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.publish_mission_command(TRIM, target_pressure_pa=50_000.0)
        h.spin_until(lambda: h.node._mode == "RUNNING", timeout=1.0)
        h.spin_until(lambda: len(h.received_targets) >= 1, timeout=1.0)
        assert h.node._start_pending is False

    def test_start_before_pressure_fires_on_pressure(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_mission_command(TRIM, target_pressure_pa=50_000.0)
        h.spin_until(lambda: h.node._mode == "LOADED", timeout=1.0)
        h.publish_command("start")
        h.spin_for(0.2)
        assert h.node._mode == "LOADED"
        assert h.node._start_pending is True

        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_until(lambda: h.node._mode == "RUNNING", timeout=1.0)
        h.spin_until(lambda: len(h.received_targets) >= 1, timeout=1.0)
        assert h.node._start_pending is False

    def test_redelivered_path_does_not_unseat_running_mission(
        self, pathfinding_node_harness
    ):
        # `/path` is latched, so the same MissionCommand can be redelivered on
        # discovery re-matching. Once we're RUNNING, a duplicate must be a no-op:
        # the original bug reloaded the mission, dropped back to LOADED and
        # nulled `_mission_t0_s`, which silently stranded `_tick` so the glider
        # got no setpoints and just drifted at spawn.
        h = pathfinding_node_harness
        h.publish_mission_command(TRIM, target_pressure_pa=50_000.0)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_until(lambda: h.node._current_pressure_pa is not None, timeout=1.0)
        h.publish_command("start")
        h.spin_until(lambda: h.node._mode == "RUNNING", timeout=1.0)
        t0_before = h.node._mission_t0_s
        assert t0_before is not None

        # Redeliver the identical mission, then keep feeding pressure so the
        # timer has everything it needs to publish.
        h.publish_mission_command(TRIM, target_pressure_pa=50_000.0)
        before = len(h.received_targets)
        for _ in range(8):
            h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
            h.spin_for(0.05)

        assert h.node._mode == "RUNNING"
        assert h.node._mission_t0_s == t0_before  # mission clock not reset
        assert len(h.received_targets) > before  # setpoints still flowing

    def test_stop_clears_pending_start(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_command("start")
        h.spin_until(lambda: h.node._start_pending is True, timeout=1.0)
        h.publish_command("stop")
        h.spin_until(lambda: h.node._mode == "STOPPED", timeout=1.0)
        assert h.node._start_pending is False

        h.publish_mission_command(TRIM, target_pressure_pa=50_000.0)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_for(0.5)
        # No auto-start after stop cleared the pending intent.
        assert h.node._mode == "LOADED"
        assert h.received_targets == []

    def test_abort_clears_pending_start(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_command("start")
        h.spin_until(lambda: h.node._start_pending is True, timeout=1.0)
        h.publish_command("abort")
        h.spin_until(lambda: h.node._mode == "IDLE", timeout=1.0)
        # Abort publishes one resurface Pose synchronously.
        h.spin_until(lambda: len(h.received_targets) >= 1, timeout=1.0)
        assert h.node._start_pending is False
        emissions_after_abort = len(h.received_targets)

        h.publish_mission_command(TRIM, target_pressure_pa=50_000.0)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_for(0.5)
        # No auto-start after abort cleared the pending intent.
        assert h.node._mode == "LOADED"
        assert len(h.received_targets) == emissions_after_abort


class TestStartHappyPath:
    def test_start_emits_at_10hz(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_mission_command(TRIM, target_pressure_pa=50_000.0)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_until(
            lambda: (
                h.node._mode == "LOADED" and h.node._current_pressure_pa is not None
            ),
            timeout=1.0,
        )
        h.publish_command("start")
        h.spin_until(lambda: h.node._mode == "RUNNING", timeout=1.0)
        # 10 Hz timer -> ~5 emissions in 0.5 s; 3 is the slack lower bound.
        h.spin_until(lambda: len(h.received_targets) >= 3, timeout=1.0)
        assert h.node._mode == "RUNNING"

    def test_running_target_carries_mission_setpoint(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        target_pa = 50_000.0
        h.publish_mission_command(TRIM, target_pressure_pa=target_pa)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_until(
            lambda: (
                h.node._mode == "LOADED" and h.node._current_pressure_pa is not None
            ),
            timeout=1.0,
        )
        h.publish_command("start")
        h.spin_until(lambda: len(h.received_targets) >= 1, timeout=1.0)
        # TRIM mission -> position.z is the operator-supplied gauge Pa.
        first = h.received_targets[0]
        assert first.position.z == pytest.approx(target_pa, abs=1e-3)
        assert first.orientation.w == pytest.approx(1.0, abs=1e-9)


class TestStopCommand:
    def test_stop_freezes_mode_and_halts_emissions(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_mission_command(TRIM, target_pressure_pa=50_000.0)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_until(
            lambda: (
                h.node._mode == "LOADED" and h.node._current_pressure_pa is not None
            ),
            timeout=1.0,
        )
        h.publish_command("start")
        h.spin_until(lambda: h.node._mode == "RUNNING", timeout=1.0)
        h.spin_until(lambda: len(h.received_targets) >= 1, timeout=1.0)

        h.publish_command("stop")
        h.spin_until(lambda: h.node._mode == "STOPPED", timeout=1.0)
        # Drain in-flight emissions queued before STOPPED was observed.
        h.spin_for(0.4)
        emissions_after_stop = len(h.received_targets)

        h.spin_for(1.0)
        assert len(h.received_targets) == emissions_after_stop


class TestAbortCommand:
    def test_abort_resets_state_and_publishes_resurface(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_mission_command(TRIM, target_pressure_pa=50_000.0)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_until(
            lambda: (
                h.node._mode == "LOADED" and h.node._current_pressure_pa is not None
            ),
            timeout=1.0,
        )
        h.publish_command("start")
        h.spin_until(lambda: h.node._mode == "RUNNING", timeout=1.0)
        h.spin_until(lambda: len(h.received_targets) >= 1, timeout=1.0)
        emissions_before_abort = len(h.received_targets)

        h.publish_command("abort")
        h.spin_until(lambda: h.node._mode == "IDLE", timeout=1.0)
        # Abort publishes one resurface Pose synchronously, then the timer
        # no-ops because mode is now IDLE.
        h.spin_until(
            lambda: len(h.received_targets) >= emissions_before_abort + 1,
            timeout=1.0,
        )
        assert h.node._mission is None
        assert h.node._mission_cmd is None

        # Drain anything still in-flight, then confirm timer is quiescent.
        h.spin_for(0.4)
        emissions_after_abort = len(h.received_targets)
        h.spin_for(0.6)
        assert len(h.received_targets) == emissions_after_abort

        # The abort emission is the resurface Pose: z=0, identity orientation.
        resurface = h.received_targets[emissions_before_abort]
        assert resurface.position.z == pytest.approx(0.0, abs=1e-9)
        assert resurface.orientation.w == pytest.approx(1.0, abs=1e-9)
        assert resurface.orientation.x == pytest.approx(0.0, abs=1e-9)
        assert resurface.orientation.y == pytest.approx(0.0, abs=1e-9)
        assert resurface.orientation.z == pytest.approx(0.0, abs=1e-9)


class TestSurfaceMission:
    """SURFACE drives Pose(z=0, identity) and self-terminates after a
    surface dwell. Tests shrink DWELL_AT_SURFACE_S so they stay fast —
    the dwell semantics are pinned in test_surface_mission.py (Tier 1)."""

    def test_running_target_is_z_zero_identity(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        # Stay below the surface threshold so the mission doesn't immediately
        # self-terminate before the test can assert on emissions.
        h.publish_mission_command(SURFACE)
        h.publish_external_pressure(PRESSURE_AT_DEPTH_PA)
        h.spin_until(
            lambda: (
                h.node._mode == "LOADED" and h.node._current_pressure_pa is not None
            ),
            timeout=1.0,
        )
        h.publish_command("start")
        h.spin_until(lambda: len(h.received_targets) >= 1, timeout=1.0)
        first = h.received_targets[0]
        assert first.position.z == pytest.approx(0.0, abs=1e-9)
        assert first.orientation.w == pytest.approx(1.0, abs=1e-9)
        assert first.orientation.x == pytest.approx(0.0, abs=1e-9)
        assert first.orientation.y == pytest.approx(0.0, abs=1e-9)
        assert first.orientation.z == pytest.approx(0.0, abs=1e-9)

    def test_self_terminates_after_surface_dwell(
        self, pathfinding_node_harness, monkeypatch
    ):
        # Shrink the dwell so we don't sleep 10 wall-clock seconds.
        monkeypatch.setattr(surface_mod, "DWELL_AT_SURFACE_S", 0.3)

        h = pathfinding_node_harness
        h.publish_mission_command(SURFACE)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_until(
            lambda: (
                h.node._mode == "LOADED" and h.node._current_pressure_pa is not None
            ),
            timeout=1.0,
        )
        h.publish_command("start")
        h.spin_until(lambda: h.node._mode == "RUNNING", timeout=1.0)
        # Keep feeding "at surface" pressure across the dwell window.
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and h.node._mode == "RUNNING":
            h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
            h.spin_for(0.05)
        # After the dwell, pathfinding_node returns to IDLE and clears state.
        h.spin_until(lambda: h.node._mode == "IDLE", timeout=1.0)
        assert h.node._mission is None
        assert h.node._mission_cmd is None

        # Drain in-flight emissions, then confirm the timer is quiescent.
        h.spin_for(0.4)
        idle_count = len(h.received_targets)
        h.spin_for(0.6)
        assert len(h.received_targets) == idle_count

    def test_does_not_terminate_below_surface(
        self, pathfinding_node_harness, monkeypatch
    ):
        monkeypatch.setattr(surface_mod, "DWELL_AT_SURFACE_S", 0.3)

        h = pathfinding_node_harness
        h.publish_mission_command(SURFACE)
        h.publish_external_pressure(PRESSURE_AT_DEPTH_PA)
        h.spin_until(
            lambda: (
                h.node._mode == "LOADED" and h.node._current_pressure_pa is not None
            ),
            timeout=1.0,
        )
        h.publish_command("start")
        h.spin_until(lambda: h.node._mode == "RUNNING", timeout=1.0)
        # Pump deep-pressure samples for several dwell-windows; mission must
        # stay RUNNING and keep emitting setpoints.
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            h.publish_external_pressure(PRESSURE_AT_DEPTH_PA)
            h.spin_for(0.05)
        assert h.node._mode == "RUNNING"
        assert len(h.received_targets) >= 5


class TestDoNothingMission:
    """DO_NOTHING keeps the node RUNNING but commands nothing: it emits a
    CONTROL_RESET on start (so the controllers go fresh) and then never
    publishes a POSITION_TARGET."""

    def test_start_emits_control_reset(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_mission_command(DO_NOTHING)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_until(
            lambda: (
                h.node._mode == "LOADED" and h.node._current_pressure_pa is not None
            ),
            timeout=1.0,
        )
        h.publish_command("start")
        h.spin_until(lambda: h.node._mode == "RUNNING", timeout=1.0)
        h.spin_until(lambda: len(h.received_resets) >= 1, timeout=1.0)
        assert len(h.received_resets) >= 1

    def test_running_emits_no_targets(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_mission_command(DO_NOTHING)
        h.publish_external_pressure(PRESSURE_AT_DEPTH_PA)
        h.spin_until(
            lambda: (
                h.node._mode == "LOADED" and h.node._current_pressure_pa is not None
            ),
            timeout=1.0,
        )
        h.publish_command("start")
        h.spin_until(lambda: h.node._mode == "RUNNING", timeout=1.0)
        # Keep feeding pressure across several tick windows; the mission must
        # stay RUNNING and never publish a setpoint (reference -> None).
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            h.publish_external_pressure(PRESSURE_AT_DEPTH_PA)
            h.spin_for(0.05)
        assert h.node._mode == "RUNNING"
        assert h.received_targets == []


class TestTickGating:
    def test_loaded_does_not_emit(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_mission_command(TRIM, target_pressure_pa=50_000.0)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_until(
            lambda: (
                h.node._mode == "LOADED" and h.node._current_pressure_pa is not None
            ),
            timeout=1.0,
        )
        # No `start` command -> mode stays LOADED, timer must no-op.
        h.spin_for(0.7)
        assert h.received_targets == []
        assert h.node._mode == "LOADED"
