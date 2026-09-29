"""Tier 1 unit tests for the BCU bang-bang law (`control/bcu_node.py`).

`solve_bcu_command` is the entire control law: one depth error in, one
`(pump_rpm, motor_open, free_open)` wire command out. Pure, no rclpy.

The contracts pinned here, in the order they matter:

  - **Three states, nothing between.** The pump is at `+pump_rpm`,
    `-pump_rpm`, or 0. Any intermediate value would mean something is
    modulating, which is exactly what this law exists to remove.
  - **Sign.** Positive bus RPM inflates the bladder and the vehicle
    rises; negative deflates and it sinks. `TankLimitGuard` reads the
    same signs off this return value, and the STM wire flip
    (`STM_BCU_RPM_SIGN`) is applied downstream of both, so getting this
    backwards would drive the vehicle away from every target.
  - **Deadband.** Inside it the pump is off and both valves shut. This is
    what keeps the pump silent at the surface, where SURFACE's 0 Pa
    target sits in the middle of the sensor's noise band.
  - **Deep passive vent.** Past `deep_threshold_pa` a descend command
    stops the pump and opens the free/bypass valve instead of
    dead-heading against ambient pressure.
"""

import pytest
from py_pkg.control.bcu_node import solve_bcu_command
from py_pkg.robot_specs import BCU_DEEP_THRESHOLD_PA
from py_pkg.scenarios.spec.control import DepthSpec

SPEC = DepthSpec()
PUMP_RPM = SPEC.pump_rpm
DEADBAND = SPEC.deadband_pa

SHALLOW = 10_000.0  # gauge Pa, well under the deep threshold
DEEP = BCU_DEEP_THRESHOLD_PA + 10_000.0


def _solve(error_pa, current_pa=SHALLOW):
    return solve_bcu_command(
        error_pa,
        current_pa,
        pump_rpm=PUMP_RPM,
        deadband_pa=DEADBAND,
    )


class TestDirection:
    """Which way the pump runs, given the sign of target - current."""

    def test_target_deeper_deflates_the_bladder(self):
        # Positive error = target below us = sink = negative bus RPM.
        pump, motor, free = _solve(+50_000.0)
        assert pump == -PUMP_RPM
        assert (motor, free) == (1, 0)

    def test_target_shallower_inflates_the_bladder(self):
        # Negative error = target above us = rise = positive bus RPM.
        pump, motor, free = _solve(-50_000.0)
        assert pump == +PUMP_RPM
        assert (motor, free) == (1, 0)

    @pytest.mark.parametrize("error", [2001.0, 5_000.0, 50_000.0, 1e9])
    def test_magnitude_is_constant_regardless_of_error(self, error):
        # The defining property of bang-bang: a 0.2 m error and a 100 m
        # error command exactly the same thing. Nothing is proportional.
        assert _solve(error)[0] == -PUMP_RPM
        assert _solve(-error)[0] == +PUMP_RPM

    def test_only_three_pump_states_exist(self):
        commanded = {
            _solve(e)[0]
            for e in (
                -1e9,
                -9000.0,
                -2001.0,
                -2000.0,
                -1.0,
                0.0,
                1.0,
                2000.0,
                2001.0,
                1e9,
            )
        }
        assert commanded == {-PUMP_RPM, 0, +PUMP_RPM}


class TestDeadband:
    """Inside the deadband the bladder freezes: pump off, both valves shut."""

    @pytest.mark.parametrize("error", [0.0, 1.0, -1.0, 1999.0, -1999.0])
    def test_inside_deadband_is_fully_idle(self, error):
        assert _solve(error) == (0, 0, 0)

    def test_boundary_is_inclusive_of_idle(self):
        # |error| == deadband is still idle; the pump only runs strictly
        # outside, so a reading parked exactly on the edge cannot run it.
        assert _solve(+DEADBAND) == (0, 0, 0)
        assert _solve(-DEADBAND) == (0, 0, 0)

    def test_just_outside_deadband_commands_full_speed(self):
        assert _solve(DEADBAND + 1.0)[0] == -PUMP_RPM
        assert _solve(-DEADBAND - 1.0)[0] == +PUMP_RPM

    def test_surface_bob_cannot_start_the_pump(self):
        # SURFACE targets gauge 0 Pa and the reading bobs across zero on
        # 100 Pa sensor quantisation. Every one of those samples must
        # leave the wire idle -- this is the regression the deadband is
        # here for. (error = target - current = -current.)
        for current in (-300.0, -100.0, 0.0, 100.0, 300.0, 500.0):
            assert _solve(-current, current_pa=max(current, 0.0)) == (0, 0, 0)

    def test_zero_deadband_reduces_to_pure_sign(self):
        # The off switch: with no deadband every non-zero error runs the
        # pump, and only an exactly-zero error idles it.
        def bare(error):
            return solve_bcu_command(error, SHALLOW, pump_rpm=PUMP_RPM, deadband_pa=0.0)

        assert bare(1e-9)[0] == -PUMP_RPM
        assert bare(-1e-9)[0] == +PUMP_RPM
        assert bare(0.0) == (0, 0, 0)


class TestDeepPassiveVent:
    """Past the deep threshold the pump cannot push oil out; ambient does it."""

    def test_deep_descend_vents_instead_of_pumping(self):
        pump, motor, free = _solve(+50_000.0, current_pa=DEEP)
        assert pump == 0
        assert (motor, free) == (0, 1)  # free/bypass valve only

    def test_deep_ascend_still_pumps(self):
        # Inflating deep is what the pump is FOR; only the descend
        # direction hands over to ambient pressure.
        pump, motor, free = _solve(-50_000.0, current_pa=DEEP)
        assert pump == +PUMP_RPM
        assert (motor, free) == (1, 0)

    def test_deep_inside_deadband_is_still_idle(self):
        # The vent is a descend-command branch, not a depth branch: with
        # nothing to do, deep or shallow, everything stays shut.
        assert _solve(0.0, current_pa=DEEP) == (0, 0, 0)

    def test_threshold_is_strict(self):
        # Exactly at the threshold the pump still has authority.
        at = solve_bcu_command(
            +50_000.0,
            BCU_DEEP_THRESHOLD_PA,
            pump_rpm=PUMP_RPM,
            deadband_pa=DEADBAND,
        )
        assert at == (-PUMP_RPM, 1, 0)


class TestValveBits:
    """The two valves are never both open, and never open with no reason."""

    @pytest.mark.parametrize(
        "error,current",
        [
            (+50_000.0, SHALLOW),
            (-50_000.0, SHALLOW),
            (+50_000.0, DEEP),
            (-50_000.0, DEEP),
            (0.0, SHALLOW),
            (0.0, DEEP),
        ],
    )
    def test_valves_are_mutually_exclusive(self, error, current):
        _, motor, free = _solve(error, current_pa=current)
        assert not (motor and free)
        assert motor in (0, 1) and free in (0, 1)

    def test_pump_never_runs_against_shut_valves(self):
        for error, current in (
            (+50_000.0, SHALLOW),
            (-50_000.0, SHALLOW),
            (+50_000.0, DEEP),
            (0.0, SHALLOW),
        ):
            pump, motor, _ = _solve(error, current_pa=current)
            assert pump == 0 or motor == 1
