"""Tier 3 (in-process) test for the BCU tank-pressure synthesis.

Constructs ``BCUSimBridge`` in-process — no Gazebo — and exercises
``tank_pressure_pa`` directly. It imports from ``nautilus_hal``, so per the
project convention it is marker-gated ``@pytest.mark.sim`` even though it
never launches a simulator. Opt in with ``pytest -m sim test/sim/`` after
sourcing the workspace install.

Locks the bladder-fill -> tank-pressure contract: tank pressure runs INVERSE
to bladder fill (the tank feeds the bladder, so a full bladder = a drained
tank), mapped linearly onto the dive tests' 0.7-1.5 barg interval, plus the
optional hull partial-vacuum offset.
"""

import pytest
import rclpy

from nautilus_hal.bridges.bcu_sim_bridge import BCUSimBridge


@pytest.fixture
def bridge():
    rclpy.init()
    node = None
    try:
        node = BCUSimBridge()
        yield node
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


@pytest.mark.sim
def test_default_endpoints_match_dive_test_interval(bridge):
    # The 0.7-1.5 barg interval the dive tests reported, expressed in gauge Pa.
    assert bridge.tank_pressure_empty_pa == 70_000.0
    assert bridge.tank_pressure_full_pa == 150_000.0
    assert bridge.tank_pressure_vacuum_offset_pa == 0.0


@pytest.mark.sim
def test_empty_and_full_endpoints(bridge):
    # Empty bladder = oil sits in the tank = highest tank pressure.
    bridge.latest_volume_m3 = bridge.bladder_min_m3
    assert bridge.tank_pressure_pa() == int(bridge.tank_pressure_full_pa)

    # Full bladder = oil pushed out into the bladder = drained tank = lowest.
    bridge.latest_volume_m3 = bridge.bladder_max_m3
    assert bridge.tank_pressure_pa() == int(bridge.tank_pressure_empty_pa)


@pytest.mark.sim
def test_midpoint_is_linear(bridge):
    bridge.latest_volume_m3 = 0.5 * (bridge.bladder_min_m3 + bridge.bladder_max_m3)
    expected = int(0.5 * (bridge.tank_pressure_empty_pa + bridge.tank_pressure_full_pa))
    assert bridge.tank_pressure_pa() == expected  # 110_000 at defaults


@pytest.mark.sim
def test_monotonic_decreasing_with_fill(bridge):
    span = bridge.bladder_max_m3 - bridge.bladder_min_m3
    readings = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        bridge.latest_volume_m3 = bridge.bladder_min_m3 + frac * span
        readings.append(bridge.tank_pressure_pa())
    # More oil in the bladder = less left in the tank = lower tank pressure,
    # strictly.
    assert readings == sorted(readings, reverse=True)
    assert len(set(readings)) == len(readings)


@pytest.mark.sim
def test_volume_outside_operating_range_saturates(bridge):
    # Below min / above max clamp to the endpoints; never over/undershoot.
    # Below min = even emptier bladder = fuller tank = high endpoint.
    bridge.latest_volume_m3 = bridge.bladder_min_m3 - 0.001
    assert bridge.tank_pressure_pa() == int(bridge.tank_pressure_full_pa)

    bridge.latest_volume_m3 = bridge.bladder_max_m3 + 0.001
    assert bridge.tank_pressure_pa() == int(bridge.tank_pressure_empty_pa)


@pytest.mark.sim
def test_vacuum_offset_shifts_reading(bridge):
    # Empty bladder reads the full (tank-full-of-oil) endpoint; the offset
    # shifts whatever the base reading is.
    bridge.latest_volume_m3 = bridge.bladder_min_m3
    bridge.tank_pressure_vacuum_offset_pa = 5_000.0
    assert bridge.tank_pressure_pa() == int(bridge.tank_pressure_full_pa) + 5_000


@pytest.mark.sim
def test_swapping_endpoints_inverts_direction(bridge):
    # Swapping empty<->full is the one-line direction flip.
    high = bridge.tank_pressure_full_pa  # 150_000 by default
    bridge.tank_pressure_empty_pa, bridge.tank_pressure_full_pa = (
        bridge.tank_pressure_full_pa,
        bridge.tank_pressure_empty_pa,
    )
    # A full bladder normally reads the low (drained-tank) pressure; after the
    # swap it reads the high one instead.
    bridge.latest_volume_m3 = bridge.bladder_max_m3
    assert bridge.tank_pressure_pa() == int(high)  # 150_000
