"""Tier 2 in-process rclpy tests for PathfindingNode.

Black-box: drive the node via PATH (MissionCommand) / POSITION_ESTIMATION /
COMMAND and assert on what it publishes on POSITION_TARGET. The node is
now a thin mission dispatcher — the planner is gone, and per-mission
setpoint generation lives in `path/missions/`.

The current depth arrives already gauged on POSITION_ESTIMATION.position.z
(attitude_node owns the absolute->gauge conversion), so the tests feed gauge
Pa straight in via ``publish_depth_gauge`` -- pathfinding no longer subscribes
to EXTERNAL_PRESSURE or holds its own SurfaceReference.
"""

import time

import pytest
from py_pkg.path.missions import MissionId
from py_pkg.path.missions import surface as surface_mod

# Mission ids used by the tests below.
SAWTOOTH = int(MissionId.SAWTOOTH)
TRIM = int(MissionId.TRIM_AND_NEUTRAL_BUOYANCY)
SURFACE = int(MissionId.SURFACE)

# Surface gauge pressure (Pa): ~0 -> "at surface".
GAUGE_AT_SURFACE_PA = 0.0
# Deep gauge pressure (Pa) -> ~6 m, well above SURFACE_THRESHOLD_PA.
GAUGE_AT_DEPTH_PA = 60_000.0


class TestWiringSmoke:
    def test_node_constructs(self, pathfinding_node_harness):
        assert pathfinding_node_harness.node is not None

    def test_position_estimation_subscription_present(self, pathfinding_node_harness):
        names = [s.topic_name for s in pathfinding_node_harness.node.subscriptions]
        assert "/position/estimation" in names

    def test_command_subscription_present(self, pathfinding_node_harness):
        names = [s.topic_name for s in pathfinding_node_harness.node.subscriptions]
        assert "/command" in names

    def test_path_subscription_present(self, pathfinding_node_harness):
        names = [s.topic_name for s in pathfinding_node_harness.node.subscriptions]
        assert "/path" in names

    def test_position_target_publisher_present(self, pathfinding_node_harness):
        names = [p.topic_name for p in pathfinding_node_harness.node.publishers]
        assert "/position/target" in names

    def test_initial_state_is_idle(self, pathfinding_node_harness):
        # No mission loaded -> idle.
        assert pathfinding_node_harness.node._mission is None


class TestPathIngress:
    def test_known_mission_id_loads_mission(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_mission_command(
            SAWTOOTH,
            target_pressure_pa=200_000.0,
            shallow_pressure_pa=50_000.0,
            angle_rad=0.5,
            n_oscillations=2,
        )
        h.spin_until(lambda: h.node._mission is not None, timeout=1.0)
        assert h.node._mission is not None
        assert h.node._mission_cmd is not None
        assert h.node._mission_cmd.mission_id == SAWTOOTH
        assert h.node._mission_cmd.target_pressure_pa == pytest.approx(200_000.0)

    def test_sawtooth_fields_propagate_into_mission(self, pathfinding_node_harness):
        # The two-pressure sawtooth params must survive ingress on /path and
        # reach the mission's internal state at start() (the full propagation
        # frontend -> bridge -> /path -> pathfinding -> mission, ROS side).
        h = pathfinding_node_harness
        h.publish_mission_command(
            SAWTOOTH,
            target_pressure_pa=120_000.0,
            shallow_pressure_pa=40_000.0,
            angle_rad=0.6,
            n_oscillations=3,
        )
        h.publish_depth_gauge(GAUGE_AT_DEPTH_PA)
        h.spin_until(
            lambda: (
                h.node._mission is not None and h.node._current_pressure_pa is not None
            ),
            timeout=1.0,
        )
        # The command carries the new fields...
        assert h.node._mission_cmd.shallow_pressure_pa == pytest.approx(40_000.0)
        assert h.node._mission_cmd.n_oscillations == 3
        # ...and starting the mission threads them into the state machine.
        h.publish_command(True)
        h.spin_until(lambda: h.node._mission_t0_s is not None, timeout=1.0)
        assert h.node._mission._deep_pa == pytest.approx(120_000.0)
        assert h.node._mission._shallow_pa == pytest.approx(40_000.0)
        assert h.node._mission._n_oscillations == 3

    def test_unknown_mission_id_is_rejected(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        # uint8 max -> well outside the registry.
        h.publish_mission_command(255)
        h.spin_for(0.3)
        # Bad id rejected -> nothing loaded.
        assert h.node._mission is None


class TestStartPreconditions:
    def test_start_without_mission_emits_nothing(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_until(lambda: h.node._current_pressure_pa is not None, timeout=1.0)
        h.publish_command(True)
        h.spin_for(0.5)
        assert h.received_targets == []
        assert h.node._mission is None
        # The start is queued, waiting for a mission to load.
        assert h.node._run_requested is True

    def test_start_without_pressure_stays_loaded(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_mission_command(TRIM, target_pressure_pa=50_000.0)
        h.spin_until(lambda: h.node._mission is not None, timeout=1.0)
        h.publish_command(True)
        h.spin_for(0.5)
        # Start queued -> stays LOADED, never RUNNING, no emissions.
        assert h.node._mission is not None and h.node._mission_t0_s is None
        assert h.received_targets == []
        assert h.node._run_requested is True


class TestStartRaceTolerance:
    """`start` may race ahead of `/path` or pressure ingress (Issue #43);
    the node must queue the intent and fire it once preconditions hold."""

    def test_start_before_path_fires_on_path(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_command(True)
        h.spin_for(0.2)
        assert h.node._mission is None
        assert h.node._run_requested is True

        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.publish_mission_command(TRIM, target_pressure_pa=50_000.0)
        h.spin_until(lambda: h.node._mission_t0_s is not None, timeout=1.0)
        h.spin_until(lambda: len(h.received_targets) >= 1, timeout=1.0)
        assert h.node._mission_t0_s is not None

    def test_start_before_pressure_fires_on_pressure(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_mission_command(TRIM, target_pressure_pa=50_000.0)
        h.spin_until(lambda: h.node._mission is not None, timeout=1.0)
        h.publish_command(True)
        h.spin_for(0.2)
        assert h.node._mission is not None and h.node._mission_t0_s is None
        assert h.node._run_requested is True

        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_until(lambda: h.node._mission_t0_s is not None, timeout=1.0)
        h.spin_until(lambda: len(h.received_targets) >= 1, timeout=1.0)
        assert h.node._mission_t0_s is not None

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
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_until(lambda: h.node._current_pressure_pa is not None, timeout=1.0)
        h.publish_command(True)
        h.spin_until(lambda: h.node._mission_t0_s is not None, timeout=1.0)
        t0_before = h.node._mission_t0_s
        assert t0_before is not None

        # Redeliver the identical mission, then keep feeding pressure so the
        # timer has everything it needs to publish.
        h.publish_mission_command(TRIM, target_pressure_pa=50_000.0)
        before = len(h.received_targets)
        for _ in range(8):
            h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
            h.spin_for(0.05)

        assert h.node._mission_t0_s is not None
        assert h.node._mission_t0_s == t0_before  # mission clock not reset
        assert len(h.received_targets) > before  # setpoints still flowing

    def test_stop_clears_pending_start(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_command(True)
        h.spin_until(lambda: h.node._run_requested is True, timeout=1.0)
        h.publish_command(False)
        # No mission was ever loaded, so the node stays idle throughout; the
        # stop just drops the latched run intent. Wait on that clearing.
        h.spin_until(lambda: h.node._run_requested is False, timeout=1.0)
        assert h.node._mission is None

        h.publish_mission_command(TRIM, target_pressure_pa=50_000.0)
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_for(0.5)
        # The stop cleared the run intent, so the new /path loads the mission
        # but does not auto-fire -- it sits loaded-not-running.
        assert h.node._mission is not None and h.node._mission_t0_s is None
        assert h.received_targets == []


class TestStartHappyPath:
    def test_start_emits_at_10hz(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_mission_command(TRIM, target_pressure_pa=50_000.0)
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_until(
            lambda: (
                h.node._mission is not None and h.node._current_pressure_pa is not None
            ),
            timeout=1.0,
        )
        h.publish_command(True)
        h.spin_until(lambda: h.node._mission_t0_s is not None, timeout=1.0)
        # 10 Hz timer -> ~5 emissions in 0.5 s; 3 is the slack lower bound.
        h.spin_until(lambda: len(h.received_targets) >= 3, timeout=1.0)
        assert h.node._mission_t0_s is not None

    def test_running_target_carries_mission_setpoint(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        target_pa = 50_000.0
        h.publish_mission_command(TRIM, target_pressure_pa=target_pa)
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_until(
            lambda: (
                h.node._mission is not None and h.node._current_pressure_pa is not None
            ),
            timeout=1.0,
        )
        h.publish_command(True)
        h.spin_until(lambda: len(h.received_targets) >= 1, timeout=1.0)
        # TRIM mission -> position.z is the operator-supplied gauge Pa.
        first = h.received_targets[0]
        assert first.position.z == pytest.approx(target_pa, abs=1e-3)
        assert first.orientation.w == pytest.approx(1.0, abs=1e-9)


class TestPoseEstimationIngress:
    """POSITION_ESTIMATION.position.z (gauge Pa) flows straight into
    ``_current_pressure_pa`` -- attitude_node owns the gauge conversion now,
    so the node stores the value verbatim and uses it for phase detection."""

    def test_surface_gauge_lands_at_zero(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_until(lambda: h.node._current_pressure_pa is not None, timeout=1.0)
        assert h.node._current_pressure_pa == pytest.approx(0.0, abs=1e-3)

    def test_gauge_depth_stored_verbatim(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_depth_gauge(GAUGE_AT_DEPTH_PA)
        h.spin_until(
            lambda: (
                h.node._current_pressure_pa is not None
                and h.node._current_pressure_pa
                == pytest.approx(GAUGE_AT_DEPTH_PA, abs=1e-3)
            ),
            timeout=1.0,
        )


class TestStopCommand:
    def test_stop_idles_clears_mission_and_halts_emissions(
        self, pathfinding_node_harness
    ):
        h = pathfinding_node_harness
        h.publish_mission_command(TRIM, target_pressure_pa=50_000.0)
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_until(
            lambda: (
                h.node._mission is not None and h.node._current_pressure_pa is not None
            ),
            timeout=1.0,
        )
        h.publish_command(True)
        h.spin_until(lambda: h.node._mission_t0_s is not None, timeout=1.0)
        h.spin_until(lambda: len(h.received_targets) >= 1, timeout=1.0)

        # Stop -> clean IDLE: mission/path cleared, no further setpoints, and no
        # resurface Pose. The controllers reset themselves off the same
        # /command=false (pathfinding emits nothing to POSITION_TARGET).
        h.publish_command(False)
        h.spin_until(lambda: h.node._mission is None, timeout=1.0)
        assert h.node._mission is None
        assert h.node._mission_cmd is None
        # Drain in-flight emissions queued before IDLE was observed.
        h.spin_for(0.4)
        emissions_after_stop = len(h.received_targets)

        h.spin_for(1.0)
        assert len(h.received_targets) == emissions_after_stop


class TestSurfaceMission:
    """SURFACE drives Pose(z=0, identity) and self-terminates after a
    surface dwell. Tests shrink DWELL_AT_SURFACE_S so they stay fast —
    the dwell semantics are pinned in test_surface_mission.py (Tier 1)."""

    def test_running_target_is_z_zero_identity(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        # Stay below the surface threshold so the mission doesn't immediately
        # self-terminate before the test can assert on emissions.
        h.publish_mission_command(SURFACE)
        h.publish_depth_gauge(GAUGE_AT_DEPTH_PA)
        h.spin_until(
            lambda: (
                h.node._mission is not None and h.node._current_pressure_pa is not None
            ),
            timeout=1.0,
        )
        h.publish_command(True)
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
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_until(
            lambda: (
                h.node._mission is not None and h.node._current_pressure_pa is not None
            ),
            timeout=1.0,
        )
        h.publish_command(True)
        h.spin_until(lambda: h.node._mission_t0_s is not None, timeout=1.0)
        # Keep feeding "at surface" pressure across the dwell window.
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and h.node._mission_t0_s is not None:
            h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
            h.spin_for(0.05)
        # After the dwell, pathfinding_node returns to IDLE and clears state.
        h.spin_until(lambda: h.node._mission is None, timeout=1.0)
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
        h.publish_depth_gauge(GAUGE_AT_DEPTH_PA)
        h.spin_until(
            lambda: (
                h.node._mission is not None and h.node._current_pressure_pa is not None
            ),
            timeout=1.0,
        )
        h.publish_command(True)
        h.spin_until(lambda: h.node._mission_t0_s is not None, timeout=1.0)
        # Pump deep-pressure samples for several dwell-windows; mission must
        # stay RUNNING and keep emitting setpoints.
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            h.publish_depth_gauge(GAUGE_AT_DEPTH_PA)
            h.spin_for(0.05)
        assert h.node._mission_t0_s is not None
        assert len(h.received_targets) >= 5


class TestCompletionSafeStop:
    """On mission completion the node drives /command=false itself so the
    controllers run their safe-stop -- completion is not otherwise a stop
    signal, and without it the BCU would hold the last target forever.
    Exercised through SURFACE, which self-terminates on a (shrunk) dwell."""

    def test_completion_emits_command_false(
        self, pathfinding_node_harness, monkeypatch
    ):
        monkeypatch.setattr(surface_mod, "DWELL_AT_SURFACE_S", 0.3)

        h = pathfinding_node_harness
        h.publish_mission_command(SURFACE)
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_until(
            lambda: (
                h.node._mission is not None and h.node._current_pressure_pa is not None
            ),
            timeout=1.0,
        )
        # The only /command WE publish is this start.
        h.publish_command(True)
        h.spin_until(lambda: h.node._mission_t0_s is not None, timeout=1.0)
        # Feed "at surface" across the dwell so the mission completes.
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and h.node._mission is not None:
            h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
            h.spin_for(0.05)
        h.spin_until(lambda: h.node._mission is None, timeout=1.0)
        h.spin_for(0.2)  # let the completion /command=false reach the tester

        # We never published a stop, so a False in the stream came from the
        # node's completion safe-stop.
        assert False in h.received_commands, h.received_commands


class TestTickGating:
    def test_loaded_does_not_emit(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_mission_command(TRIM, target_pressure_pa=50_000.0)
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_until(
            lambda: (
                h.node._mission is not None and h.node._current_pressure_pa is not None
            ),
            timeout=1.0,
        )
        # No `start` command -> stays loaded-not-running, timer must no-op.
        h.spin_for(0.7)
        assert h.received_targets == []
        assert h.node._mission is not None and h.node._mission_t0_s is None
