"""Tier 2 in-process rclpy tests for PathfindingNode.

Black-box: drive the node via PATH (MissionCommand) / EXTERNAL_PRESSURE /
COMMAND and assert on what it publishes on POSITION_TARGET. The node is
now a thin mission dispatcher — the planner is gone, and per-mission
setpoint generation lives in `path/missions/`.
"""

import pytest

from py_pkg.path.missions import MissionId


# Mission ids used by the tests below.
SAWTOOTH = int(MissionId.SAWTOOTH)
TRIM = int(MissionId.TRIM_AND_NEUTRAL_BUOYANCY)

# Surface absolute pressure (Pa). gauge_pressure_pa() yields ~0 -> "at surface".
PRESSURE_AT_SURFACE_PA = 101_325


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
        h.publish_mission_command(SAWTOOTH, target_pressure_pa=200_000.0,
                                  angle_rad=0.5, n_resurfaces=2)
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

    def test_start_without_pressure_stays_loaded(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_mission_command(TRIM, target_pressure_pa=50_000.0)
        h.spin_until(lambda: h.node._mode == "LOADED", timeout=1.0)
        h.publish_command("start")
        h.spin_for(0.5)
        # Start aborted -> stays LOADED, never RUNNING, no emissions.
        assert h.node._mode == "LOADED"
        assert h.received_targets == []


class TestStartHappyPath:
    def test_start_emits_at_10hz(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_mission_command(TRIM, target_pressure_pa=50_000.0)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_until(
            lambda: h.node._mode == "LOADED"
            and h.node._current_pressure_pa is not None,
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
            lambda: h.node._mode == "LOADED"
            and h.node._current_pressure_pa is not None,
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
            lambda: h.node._mode == "LOADED"
            and h.node._current_pressure_pa is not None,
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
            lambda: h.node._mode == "LOADED"
            and h.node._current_pressure_pa is not None,
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


class TestTickGating:
    def test_loaded_does_not_emit(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_mission_command(TRIM, target_pressure_pa=50_000.0)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_until(
            lambda: h.node._mode == "LOADED"
            and h.node._current_pressure_pa is not None,
            timeout=1.0,
        )
        # No `start` command -> mode stays LOADED, timer must no-op.
        h.spin_for(0.7)
        assert h.received_targets == []
        assert h.node._mode == "LOADED"
