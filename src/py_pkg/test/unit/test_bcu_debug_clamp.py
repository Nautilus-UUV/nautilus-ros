"""Tier 1: pure-logic clamp on the manual pump duration."""

from py_pkg.debug.bcu_debug_node import MAX_PUMP_S, _clamp_duration


def test_clamp_within_range_passes_through():
    assert _clamp_duration(5.0) == 5.0


def test_clamp_zero_returns_zero():
    assert _clamp_duration(0.0) == 0.0


def test_clamp_negative_returns_zero():
    # Operator typo'd a minus sign -- treat as a no-op pump, never as
    # "run forever" or a giant positive value.
    assert _clamp_duration(-2.0) == 0.0


def test_clamp_huge_caps_at_max():
    assert _clamp_duration(9999.0) == MAX_PUMP_S


def test_clamp_at_max_is_unchanged():
    assert _clamp_duration(MAX_PUMP_S) == MAX_PUMP_S


def test_custom_max_overrides_default():
    assert _clamp_duration(50.0, max_s=10.0) == 10.0
