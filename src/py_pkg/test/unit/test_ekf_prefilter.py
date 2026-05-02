"""Tier 1 unit tests for the EMA math in py_pkg.ekf_prefilter.

The prefilter is a ROS node, but its smoothing logic is the pure
function `ema(self, x_new, x_prev)` that only depends on `self.alpha`.
We exercise it as an unbound method with a SimpleNamespace stand-in for
`self` to keep these tests Tier 1 — no rclpy.init(), no ROS context.

Two goals:

1. *High-level idea*: EMA must pass DC unchanged, must reduce to the
   first sample on init, must respect the alpha=0/1 endpoints, and
   must converge to a step input over time.

2. *Implementation*: catches accidental sign flips, off-by-one in the
   first-sample bypass, or alpha mis-application (alpha vs 1-alpha).
"""

from types import SimpleNamespace

import pytest
from py_pkg.ekf_prefilter.ekf_prefilter import EkfPrefilter


def ema(alpha, x_new, x_prev):
    """Call EkfPrefilter.ema as an unbound method with a fake self."""
    return EkfPrefilter.ema(SimpleNamespace(alpha=alpha), x_new, x_prev)


class TestFirstSample:
    """When prev is None the EMA must seed itself with x_new directly,
    not blend it against an implicit zero — otherwise the filter takes
    several samples to wind up to its true input."""

    def test_first_sample_returns_input_unchanged_at_alpha_half(self):
        assert ema(0.5, 7.5, None) == pytest.approx(7.5)

    def test_first_sample_returns_input_unchanged_at_alpha_zero(self):
        # Even with maximum smoothing, the very first sample is the only
        # information we have — return it.
        assert ema(0.0, 3.0, None) == pytest.approx(3.0)

    def test_first_sample_handles_negative_input(self):
        assert ema(0.5, -2.5, None) == pytest.approx(-2.5)

    def test_first_sample_handles_zero(self):
        assert ema(0.5, 0.0, None) == pytest.approx(0.0)


class TestAlphaEndpoints:
    """alpha=1: no smoothing, output = input. alpha=0: max smoothing,
    output stays at the previous value (filter is frozen)."""

    def test_alpha_one_passes_input_through(self):
        assert ema(1.0, 10.0, 0.0) == pytest.approx(10.0)
        assert ema(1.0, -3.0, 100.0) == pytest.approx(-3.0)

    def test_alpha_zero_freezes_at_previous(self):
        # New input is completely ignored; the output is the previous value.
        assert ema(0.0, 10.0, 4.2) == pytest.approx(4.2)
        assert ema(0.0, -99.0, 4.2) == pytest.approx(4.2)


class TestBlendingMath:
    """Verify the formula a*x_new + (1-a)*x_prev (and not the inverse)."""

    def test_alpha_half_is_arithmetic_mean(self):
        assert ema(0.5, 10.0, 0.0) == pytest.approx(5.0)
        assert ema(0.5, 4.0, 8.0) == pytest.approx(6.0)

    def test_alpha_quarter_weights_prev_more(self):
        # alpha=0.25 means the new sample contributes 25%, the old 75%.
        # If alpha and (1-alpha) were swapped this test would catch it.
        assert ema(0.25, 100.0, 0.0) == pytest.approx(25.0)
        assert ema(0.25, 0.0, 100.0) == pytest.approx(75.0)

    def test_alpha_three_quarters_weights_new_more(self):
        assert ema(0.75, 100.0, 0.0) == pytest.approx(75.0)
        assert ema(0.75, 0.0, 100.0) == pytest.approx(25.0)


class TestDCInvariance:
    """A constant input must produce a constant output (gain = 1 at DC).
    This is the most important property: the filter must not bias
    accelerometer or gyro measurements when the underlying signal is
    steady, otherwise the EKF gets a wrong mean."""

    @pytest.mark.parametrize("alpha", [0.1, 0.25, 0.5, 0.75, 0.9])
    def test_constant_input_gives_constant_output(self, alpha):
        c = 9.81
        # Seed and then run many steps with the constant input.
        y = ema(alpha, c, None)
        for _ in range(50):
            y = ema(alpha, c, y)
        assert y == pytest.approx(c)


class TestStepResponse:
    """A step input must converge monotonically toward the new value."""

    def test_step_response_converges_at_alpha_half(self):
        alpha = 0.5
        # Seed with zero input
        y = ema(alpha, 0.0, None)
        # Now apply a unit step
        ys = []
        for _ in range(20):
            y = ema(alpha, 1.0, y)
            ys.append(y)
        # First sample is 0.5 (= alpha * 1 + (1-alpha) * 0)
        assert ys[0] == pytest.approx(0.5)
        # Strictly monotone increasing
        for i in range(1, len(ys)):
            assert ys[i] > ys[i - 1]
        # Converges to the step value
        assert ys[-1] == pytest.approx(1.0, abs=1e-5)

    def test_smaller_alpha_responds_more_slowly(self):
        # After N samples of a step input, alpha=0.1 must be further from
        # the step value than alpha=0.5. This pins down the *direction*
        # of the alpha→smoothness tradeoff.
        def run_step(alpha, n):
            y = ema(alpha, 0.0, None)
            for _ in range(n):
                y = ema(alpha, 1.0, y)
            return y

        slow = run_step(0.1, 5)
        fast = run_step(0.5, 5)
        assert slow < fast < 1.0


class TestNoSideEffects:
    """The ema function must not depend on any state besides self.alpha
    and its arguments. If a refactor adds hidden state (e.g. internal
    buffer), this catches it by exercising it through a fresh
    SimpleNamespace each call."""

    def test_repeated_calls_with_same_inputs_give_same_output(self):
        a, b, c = 0.3, 5.0, 2.0
        first = ema(a, b, c)
        # Construct a brand-new fake self each time
        for _ in range(10):
            assert ema(a, b, c) == pytest.approx(first)
