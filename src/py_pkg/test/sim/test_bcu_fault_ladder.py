"""Tier 3 (in-process) test for the monotonic BCU fault ladder.

Constructs ``BCUFaultInjector`` in-process — no Gazebo — and drives its
stepping logic directly. It imports from ``nautilus_hal``, so per the
project convention it is marker-gated ``@pytest.mark.sim`` even though it
never launches a simulator. Opt in with ``pytest -m sim test/sim/`` after
sourcing the workspace install.

Locks the degradation contract: the pump effectiveness walks 100 -> 80 ->
60 -> 40 -> 20 -> 0 % in equal 1/num_levels steps, the level only ever
increases (no recovery), latches at num_levels, the telemetry publishes the
true latched level regardless of pump activity, and mttf_sec <= 0 disables
stepping entirely.
"""

import random

import pytest
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

from nautilus_hal.injectors.fault_injection import BCUFaultInjector


@pytest.fixture
def node():
    rclpy.init()
    n = None
    try:
        n = Node("fault_ladder_test_node")
        yield n
    finally:
        if n is not None:
            n.destroy_node()
        rclpy.shutdown()


def _make_injector(node, mttf_sec, num_levels=5, seed=0):
    # A seeded RNG makes the stepping deterministic; tick_period is irrelevant
    # here because we drive _maybe_degrade() by hand rather than via timers.
    return BCUFaultInjector(
        node,
        fault_topic="/bcu/rpm/fault",
        mttf_sec=mttf_sec,
        num_levels=num_levels,
        rng=random.Random(seed),
    )


@pytest.mark.sim
def test_starts_healthy(node):
    inj = _make_injector(node, mttf_sec=60.0)
    assert inj.level == 0
    assert inj.effectiveness() == 1.0
    assert inj.apply(1000.0) == 1000.0


@pytest.mark.sim
def test_effectiveness_ladder(node):
    # Force each level and check the 100/80/60/40/20/0 % ladder exactly.
    inj = _make_injector(node, mttf_sec=60.0, num_levels=5)
    expected_rpm = [1000.0, 800.0, 600.0, 400.0, 200.0, 0.0]
    for level, rpm in enumerate(expected_rpm):
        inj.level = level
        assert inj.effectiveness() == pytest.approx((5 - level) / 5)
        assert inj.apply(1000.0) == pytest.approx(rpm)


@pytest.mark.sim
def test_monotonic_and_latches(node):
    # A short MTTF makes stepping likely each tick; drive many ticks and
    # confirm the level never decreases and never exceeds num_levels.
    inj = _make_injector(node, mttf_sec=2.0, num_levels=5, seed=1)
    prev = inj.level
    for _ in range(2000):
        inj._maybe_degrade()
        assert inj.level >= prev, "degradation level must never decrease"
        assert inj.level <= inj.num_levels
        prev = inj.level
    # With MTTF=2 s over 2000 ticks the ladder should bottom out.
    assert inj.level == inj.num_levels
    assert inj.effectiveness() == 0.0


@pytest.mark.sim
def test_mttf_zero_disables(node):
    # mttf_sec <= 0 => step probability 0 => the actuator stays healthy.
    inj = _make_injector(node, mttf_sec=0.0)
    assert inj._step_prob == 0.0
    for _ in range(5000):
        inj._maybe_degrade()
    assert inj.level == 0
    assert inj.effectiveness() == 1.0


@pytest.mark.sim
def test_telemetry_reports_true_level(node):
    # _publish_state emits the current latched level, with no idle masking.
    inj = _make_injector(node, mttf_sec=60.0)
    received = []
    node.create_subscription(Int32, "/bcu/rpm/fault", lambda m: received.append(m.data), 10)

    inj.level = 3
    inj._publish_state()
    rclpy.spin_once(node, timeout_sec=1.0)

    assert received and received[-1] == 3
