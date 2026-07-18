"""Tier 1 coverage for the BCU plant-transient models (plant_dynamics.py).

- PumpDynamics: passthrough when disabled, dead-time timing, slew ramps
  (both signs, mid-ramp reversal, irregular dt), history pruning.
- Tank maps: linear extraction matches the legacy bridge math; gaslaw
  pins both endpoints, is monotonic, curves the right way, degrades to
  linear for a huge cushion, clamps out-of-range volumes.
- make_tank_pressure_map: matches the direct functions, <= 0 cushion
  means pinned, rejects unknown shapes / bad cushions at bind time.
"""

from __future__ import annotations

import pytest
from py_pkg.plant_dynamics import (
    PumpDynamics,
    make_tank_pressure_map,
    tank_pressure_gaslaw,
    tank_pressure_linear,
)

V_MIN = 0.001
V_MAX = 0.00245
P_EMPTY = 97_800.0
P_FULL = 190_000.0


# ---------------------------------------------------------------------------
# PumpDynamics
# ---------------------------------------------------------------------------


def _run(pump: PumpDynamics, commands: list[tuple[float, float]]) -> list[float]:
    """Feed (t_s, rpm) samples; dt derived from consecutive timestamps."""
    out = []
    last_t = commands[0][0]
    for t, rpm in commands:
        out.append(pump.step(t, rpm, t - last_t))
        last_t = t
    return out


def test_disabled_is_exact_passthrough():
    pump = PumpDynamics(delay_s=0.0, slew_rpm_per_s=0.0)
    for t, cmd in [(0.0, 3000.0), (0.1, -3000.0), (0.2, 0.0), (5.0, 1234.5)]:
        assert pump.step(t, cmd, 0.1) == cmd


def test_dead_time_holds_then_releases():
    pump = PumpDynamics(delay_s=1.0, slew_rpm_per_s=0.0)
    # Command steps to 3000 at t=0; the pump must hold 0 until t >= 1.0.
    ticks = [(round(0.1 * i, 3), 3000.0) for i in range(15)]
    out = _run(pump, ticks)
    for (t, _), eff in zip(ticks, out):
        if t < 1.0:
            assert eff == 0.0, f"moved during dead time at t={t}"
        else:
            assert eff == 3000.0, f"not released at t={t}"


def test_dead_time_tracks_command_history_not_latest():
    pump = PumpDynamics(delay_s=1.0, slew_rpm_per_s=0.0)
    # 3000 commanded during [0, 0.5), then 0 after: the delayed target
    # must replay that same sequence one second later.
    seq = [(0.0, 3000.0), (0.4, 3000.0), (0.5, 0.0), (1.0, 0.0), (1.4, 0.0), (1.5, 0.0)]
    out = _run(pump, seq)
    assert out[:2] == [0.0, 0.0]  # still in dead time
    assert out[3] == 3000.0  # t=1.0 sees the t=0.0 command
    assert out[4] == 3000.0  # t=1.4 sees the t=0.4 command
    assert out[5] == 0.0  # t=1.5 sees the t=0.5 zero


def test_slew_ramps_at_constant_rate_both_signs():
    pump = PumpDynamics(delay_s=0.0, slew_rpm_per_s=500.0)
    # 1 s ticks toward +3000: 500 rpm per tick, 6 ticks to arrive.
    for i in range(1, 7):
        assert pump.step(float(i), 3000.0, 1.0) == pytest.approx(min(500.0 * i, 3000.0))
    # Hold at target once reached.
    assert pump.step(7.0, 3000.0, 1.0) == pytest.approx(3000.0)
    # Reverse toward -3000 at the same rate.
    assert pump.step(8.0, -3000.0, 1.0) == pytest.approx(2500.0)
    assert pump.step(9.0, -3000.0, 1.0) == pytest.approx(2000.0)


def test_mid_ramp_reversal_turns_around_immediately():
    pump = PumpDynamics(delay_s=0.0, slew_rpm_per_s=1000.0)
    pump.step(1.0, 3000.0, 1.0)  # -> 1000
    pump.step(2.0, 3000.0, 1.0)  # -> 2000
    assert pump.step(3.0, 0.0, 1.0) == pytest.approx(1000.0)
    assert pump.step(4.0, 0.0, 1.0) == pytest.approx(0.0)


def test_irregular_dt_scales_the_slew_budget():
    pump = PumpDynamics(delay_s=0.0, slew_rpm_per_s=1000.0)
    assert pump.step(0.1, 3000.0, 0.1) == pytest.approx(100.0)
    assert pump.step(0.6, 3000.0, 0.5) == pytest.approx(600.0)
    # Zero / negative dt must not move the ramp.
    assert pump.step(0.6, 3000.0, 0.0) == pytest.approx(600.0)
    assert pump.step(0.6, 3000.0, -1.0) == pytest.approx(600.0)


def test_combined_delay_then_ramp():
    pump = PumpDynamics(delay_s=1.5, slew_rpm_per_s=500.0)
    ticks = [(round(0.5 * i, 3), 3000.0) for i in range(10)]
    out = _run(pump, ticks)
    # Nothing until t >= 1.5; from then on 250 rpm per 0.5 s tick
    # (t=1.5 is the first tick whose delayed sample has aged out).
    for (t, _), eff in zip(ticks, out):
        if t < 1.5:
            expected = 0.0
        else:
            n_ticks = round((t - 1.5) / 0.5) + 1
            expected = min(250.0 * n_ticks, 3000.0)
        assert eff == pytest.approx(expected), f"t={t}"


def test_history_stays_pruned():
    pump = PumpDynamics(delay_s=1.0, slew_rpm_per_s=0.0)
    for i in range(10_000):
        pump.step(i * 0.1, 3000.0, 0.1)
    # Dead time spans 10 ticks at 10 Hz; the deque must not grow unbounded.
    assert len(pump._history) <= 12


# ---------------------------------------------------------------------------
# Tank maps
# ---------------------------------------------------------------------------


def test_linear_matches_legacy_bridge_math():
    # Reference: bcu_sim_bridge.tank_pressure_pa (inverse-linear oil map).
    for v in [V_MIN, 0.0015, 0.002, V_MAX]:
        span = V_MAX - V_MIN
        frac = max(0.0, min(1.0, (V_MAX - v) / span))
        expected = P_EMPTY + frac * (P_FULL - P_EMPTY)
        assert tank_pressure_linear(v, V_MIN, V_MAX, P_EMPTY, P_FULL) == pytest.approx(
            expected
        )


def test_linear_clamps_out_of_range():
    assert tank_pressure_linear(0.0, V_MIN, V_MAX, P_EMPTY, P_FULL) == pytest.approx(
        P_FULL
    )
    assert tank_pressure_linear(0.005, V_MIN, V_MAX, P_EMPTY, P_FULL) == pytest.approx(
        P_EMPTY
    )


def test_gaslaw_pinned_hits_both_endpoints_exactly():
    assert tank_pressure_gaslaw(V_MAX, V_MIN, V_MAX, P_EMPTY, P_FULL) == pytest.approx(
        P_EMPTY, abs=1e-9
    )
    assert tank_pressure_gaslaw(V_MIN, V_MIN, V_MAX, P_EMPTY, P_FULL) == pytest.approx(
        P_FULL, abs=1e-9
    )


def test_gaslaw_monotonic_and_curved_the_right_way():
    vs = [V_MIN + i * (V_MAX - V_MIN) / 20 for i in range(21)]
    ps = [tank_pressure_gaslaw(v, V_MIN, V_MAX, P_EMPTY, P_FULL) for v in vs]
    # Pressure falls as the bladder fills (oil leaves the tank).
    assert all(a > b for a, b in zip(ps, ps[1:]))
    # Air-cushion curvature: flat near tank-empty (bladder full), steep
    # near tank-full (bladder empty) => |slope| shrinks as volume rises,
    # and sits below the linear map's constant slope at the full-bladder
    # end, above it at the empty-bladder end.
    slope_lo = ps[0] - ps[1]  # near bladder-empty (tank full): steep
    slope_hi = ps[-2] - ps[-1]  # near bladder-full (tank empty): flat
    linear_step = (P_FULL - P_EMPTY) / 20
    assert slope_lo > linear_step > slope_hi
    # Between the pinned endpoints the hyperbola sits BELOW the line.
    mid = tank_pressure_gaslaw(0.001725, V_MIN, V_MAX, P_EMPTY, P_FULL)
    assert mid < tank_pressure_linear(0.001725, V_MIN, V_MAX, P_EMPTY, P_FULL)


def test_gaslaw_large_cushion_degrades_to_linear():
    span = V_MAX - V_MIN
    for v in [V_MIN, 0.0015, 0.002, V_MAX]:
        lin = tank_pressure_linear(v, V_MIN, V_MAX, P_EMPTY, P_FULL)
        # A huge cushion linearizes Boyle's law, but its pinned endpoints
        # differ: anchor the comparison via the empty endpoint + slope.
        big = 1000.0 * span
        gas = tank_pressure_gaslaw(v, V_MIN, V_MAX, P_EMPTY, P_FULL, air_volume_m3=big)
        # Slope of the big-cushion curve: p_empty * oil / V_A to first order.
        expected = P_EMPTY * (1.0 + (V_MAX - v) / big)
        assert gas == pytest.approx(expected, rel=1e-5)
        assert abs(gas - P_EMPTY) < abs(lin - P_EMPTY) + 1e-9


def test_gaslaw_free_cushion_must_exceed_span():
    with pytest.raises(ValueError, match="exceed"):
        tank_pressure_gaslaw(
            0.002, V_MIN, V_MAX, P_EMPTY, P_FULL, air_volume_m3=V_MAX - V_MIN
        )


def test_gaslaw_clamps_out_of_range():
    lo = tank_pressure_gaslaw(0.0, V_MIN, V_MAX, P_EMPTY, P_FULL)
    hi = tank_pressure_gaslaw(0.005, V_MIN, V_MAX, P_EMPTY, P_FULL)
    assert lo == pytest.approx(P_FULL, abs=1e-9)
    assert hi == pytest.approx(P_EMPTY, abs=1e-9)


# ---------------------------------------------------------------------------
# make_tank_pressure_map (the bridge's bind-once entry point)
# ---------------------------------------------------------------------------


def test_map_factory_matches_direct_functions():
    linear = make_tank_pressure_map("linear", V_MIN, V_MAX, P_EMPTY, P_FULL)
    # air_volume_m3 <= 0 is the pinned cushion (the ROS-param encoding
    # of "no free cushion"), so it must match the direct-call default.
    gas = make_tank_pressure_map(
        "gaslaw", V_MIN, V_MAX, P_EMPTY, P_FULL, air_volume_m3=0.0
    )
    for v in [V_MIN, 0.0015, 0.002, V_MAX]:
        assert linear(v) == tank_pressure_linear(v, V_MIN, V_MAX, P_EMPTY, P_FULL)
        assert gas(v) == tank_pressure_gaslaw(v, V_MIN, V_MAX, P_EMPTY, P_FULL)


def test_map_factory_rejects_unknown_shape():
    with pytest.raises(ValueError, match="tank_map_shape"):
        make_tank_pressure_map("cubic", V_MIN, V_MAX, P_EMPTY, P_FULL)


def test_map_factory_validates_free_cushion_at_bind_time():
    with pytest.raises(ValueError, match="exceed"):
        make_tank_pressure_map(
            "gaslaw", V_MIN, V_MAX, P_EMPTY, P_FULL, air_volume_m3=V_MAX - V_MIN
        )
