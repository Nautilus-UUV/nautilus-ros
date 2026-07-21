"""Tier 2 in-process rclpy tests for the boot-time mission autostart node.

`AutoMission` does all its work in the constructor: it optionally publishes
one latched `DiveInit` (only when the scenario supplied valid tank
endpoints), then one latched `MissionCommand` on `/path`. Both ride
`UUVQoS.COMMAND` (RELIABLE + TRANSIENT_LOCAL), so a subscriber that joins
*after* construction still receives them -- exactly the late-join guarantee
the node exists to provide.

There is no auto-mission harness in the shared node conftest (the node
takes no runtime input), so these tests build a minimal executor harness
locally. Parameter overrides are injected by wrapping ``Node.__init__`` for
the "auto_mission" node only -- the node reads its params and publishes
inside ``__init__``, so the overrides must be present before construction.
"""

import time

import pytest
from nautilus_msgs.msg import DiveInit, MissionCommand  # noqa: F401 (msg types)
from py_pkg.debug.auto_mission import AutoMission
from py_pkg.uuv_ros_core import UUVTopics, create_subscription_for_topic
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter


class _InitSubscriber(Node):
    """Late-joining subscriber on DIVE_INIT + PATH (both latched)."""

    def __init__(self):
        super().__init__("auto_mission_tester")
        self.dive_inits: list = []
        self.missions: list = []
        create_subscription_for_topic(self, UUVTopics.DIVE_INIT, self._on_dive_init)
        create_subscription_for_topic(self, UUVTopics.PATH, self._on_path)

    def _on_dive_init(self, msg) -> None:
        self.dive_inits.append(msg)

    def _on_path(self, msg) -> None:
        self.missions.append(msg)


class _AutoMissionHarness:
    """Owns an executor + the nodes added to it; tears them all down."""

    def __init__(self):
        self.executor = SingleThreadedExecutor()
        self._nodes: list[Node] = []

    def add(self, node: Node) -> Node:
        self.executor.add_node(node)
        self._nodes.append(node)
        return node

    def spin_for(self, duration_s: float, slice_s: float = 0.02) -> None:
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=slice_s)

    def spin_until(
        self, predicate, timeout: float = 3.0, slice_s: float = 0.02
    ) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            self.executor.spin_once(timeout_sec=slice_s)
        return bool(predicate())

    def shutdown(self) -> None:
        for node in self._nodes:
            try:
                self.executor.remove_node(node)
            except Exception:
                pass
        for node in self._nodes:
            node.destroy_node()
        self.executor.shutdown()


@pytest.fixture
def am_harness():
    harness = _AutoMissionHarness()
    try:
        yield harness
    finally:
        harness.shutdown()


def _make_auto_mission(monkeypatch, **overrides) -> AutoMission:
    """Construct AutoMission with ROS parameter overrides.

    AutoMission doesn't forward constructor kwargs, so overrides are
    injected by wrapping ``Node.__init__`` for the "auto_mission" node only.
    """
    real_init = Node.__init__
    params = [Parameter(name, value=value) for name, value in overrides.items()]

    def patched(self, *args, **kwargs):
        if args and args[0] == "auto_mission":
            kwargs["parameter_overrides"] = params
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(Node, "__init__", patched)
    return AutoMission()


# Defined first so it can never see a latched DiveInit leaked from the
# endpoint test's (now-destroyed) publisher.
def test_default_params_publish_no_dive_init_but_still_publish_path(am_harness):
    # Default tank endpoints are 0.0/0.0 -> tank_limits_valid is False, so
    # NO DiveInit is published. The MissionCommand still goes out.
    auto = am_harness.add(AutoMission())
    sub = am_harness.add(_InitSubscriber())

    got_path = am_harness.spin_until(lambda: len(sub.missions) >= 1, timeout=3.0)
    assert got_path, "MissionCommand must still be published at default params"

    # Grace period: no DiveInit may ever arrive.
    am_harness.spin_for(0.5)
    assert sub.dive_inits == [], "default params must not publish a DiveInit"


def test_endpoints_publish_one_latched_dive_init_before_path(am_harness, monkeypatch):
    auto = _make_auto_mission(
        monkeypatch,
        dive_init_tank_empty_pa=97_800.0,
        dive_init_tank_full_pa=190_000.0,
        dwell_s=7.0,
        n_steps=4,
        mission_id=1,
        target_pressure_pa=60_000.0,
        # Keep the one-shot /command start out of the spin window; these
        # tests assert only on the latched DiveInit + /path.
        start_delay_s=100.0,
    )
    am_harness.add(auto)
    # Subscriber created AFTER the node published in its constructor: proves
    # the TRANSIENT_LOCAL latch reaches a late joiner.
    sub = am_harness.add(_InitSubscriber())

    got = am_harness.spin_until(lambda: sub.dive_inits and sub.missions, timeout=3.0)
    assert got, "late subscriber did not receive the latched DiveInit + MissionCommand"

    # Grace period to catch any erroneous duplicate latched samples.
    am_harness.spin_for(0.3)

    assert len(sub.dive_inits) == 1, "exactly one latched DiveInit expected"
    di = sub.dive_inits[0]
    assert di.tank_empty_pa == pytest.approx(97_800.0)
    assert di.tank_full_pa == pytest.approx(190_000.0)
    # surface_pressure_pa rides as 0.0 (SurfaceReference.register rejects it).
    assert di.surface_pressure_pa == pytest.approx(0.0)

    assert len(sub.missions) == 1, "exactly one latched MissionCommand expected"
    cmd = sub.missions[0]
    assert cmd.dwell_s == pytest.approx(7.0)
    assert cmd.n_steps == 4
