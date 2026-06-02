"""Tier 1 gates for the hydrodynamic forward map (doc §14).

G1 identity, G2 scalar band, G3 no-double-counting, sign sanity, the full
per-knob single-knob sweep (§14.4), the §14.5 round-trip on the invertible
monomial subset, and a sub-sampled hyperbox corner check. Pure Python, no sim.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from py_pkg.scenarios.compile import (
    calibration_scalars,
    forward_map,
    _canonical_values,
)
from py_pkg.scenarios.spec.rig import HydrodynamicsSpec

NOMINAL_KNOBS_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "py_pkg/scenarios/library/nominal_knobs.yaml"
)

# Generous sanity bound on |λ|. The axial slots genuinely need large scalars:
# drag_xU ≈ 229 (the -(π/8)ρD²C_D closed form badly underpredicts the canonical
# -108) and added_mass_xx ≈ 14 (k₁ axial AM, no appendage term). The gate exists
# to catch a refactor that flips a sign or blows a scalar to NaN/∞, not to
# enforce that closed-form theory carries the magnitude — see doc §10/§14.2.
SCALAR_BAND = 300.0


@pytest.fixture
def nominal_data():
    return yaml.safe_load(NOMINAL_KNOBS_PATH.read_text())


@pytest.fixture
def nominal_knobs(nominal_data):
    return nominal_data["physics_knobs"]


def slots(spec: HydrodynamicsSpec) -> dict:
    """The 18 numeric SDF slots keyed by the compile.py registry names."""
    return _canonical_values(spec)


# Which slots each knob's closed form actually touches (doc §11). The
# complement is asserted to stay *exactly* fixed — the accidental-coupling
# regression catch. `drag_kP` has no closed-form term (frozen at canonical) so
# it never appears. alpha_stall_* touch only the per-fin alpha_stall field, not
# any numeric slot, so they map to the empty set here.
APPEARS = {
    "L": {
        "added_mass_xx", "added_mass_yy", "added_mass_zz",
        "added_mass_qq", "added_mass_rr",
        "drag_xU", "drag_yV", "drag_zW", "drag_mQ", "drag_nR",
    },
    "D": {
        "added_mass_xx", "added_mass_yy", "added_mass_zz",
        "added_mass_pp", "added_mass_qq", "added_mass_rr", "drag_xU",
    },
    "nabla": {
        "added_mass_xx", "added_mass_yy", "added_mass_zz",
        "added_mass_qq", "added_mass_rr",
        "drag_yV", "drag_zW", "drag_mQ", "drag_nR",
    },
    "b_f": {"added_mass_zz", "added_mass_pp", "added_mass_qq", "horiz_cla", "horiz_area"},
    "c_f": {
        "added_mass_zz", "added_mass_pp", "added_mass_qq",
        "horiz_cla", "horiz_cda", "horiz_area",
    },
    "x_f": {"added_mass_qq"},
    "b_r": {"added_mass_yy", "added_mass_pp", "added_mass_rr", "vert_cla", "vert_area"},
    "c_r": {
        "added_mass_yy", "added_mass_pp", "added_mass_rr",
        "vert_cla", "vert_cda", "vert_area",
    },
    "x_r": {"added_mass_rr"},
    "t_over_c": {"horiz_cda", "vert_cda"},
    "C_d_c": {"drag_yV", "drag_zW", "drag_mQ", "drag_nR"},
    "one_plus_k": {"drag_xU"},
    "C_p_base": {"drag_xU"},
    "C_La_mult": {"horiz_cla", "vert_cla"},
    "alpha_stall_horiz": set(),
    "alpha_stall_rudder": set(),
}

# Sign of the numeric value's response to a +Δ on the knob, for the unambiguous
# high-leverage entries (doc §11/§12.4). +1: value increases; -1: decreases.
# Damping slots are negative, so "more drag" reads as -1 (value gets smaller).
SIGN = {
    "C_d_c": {"drag_yV": -1, "drag_zW": -1, "drag_mQ": -1, "drag_nR": -1},
    "D": {"drag_xU": -1, "added_mass_xx": +1, "added_mass_yy": -1, "added_mass_zz": -1},
    "nabla": {
        "added_mass_xx": +1, "added_mass_yy": +1, "added_mass_zz": +1,
        "added_mass_qq": +1, "added_mass_rr": +1,
        "drag_yV": -1, "drag_zW": -1, "drag_mQ": -1, "drag_nR": -1,
    },
    "L": {
        "added_mass_qq": +1, "added_mass_rr": +1,
        "drag_xU": +1, "drag_yV": -1, "drag_zW": -1, "drag_mQ": -1, "drag_nR": -1,
    },
    "b_f": {
        "added_mass_zz": +1, "added_mass_qq": +1, "added_mass_pp": +1,
        "horiz_cla": +1, "horiz_area": +1,
    },
    "c_f": {"horiz_area": +1, "added_mass_zz": +1, "horiz_cla": -1},
    "x_f": {"added_mass_qq": +1},
    "b_r": {
        "added_mass_yy": +1, "added_mass_rr": +1, "added_mass_pp": +1,
        "vert_cla": +1, "vert_area": +1,
    },
    "c_r": {"vert_area": +1, "added_mass_yy": +1, "vert_cla": -1},
    "x_r": {"added_mass_rr": +1},
    "t_over_c": {"horiz_cda": +1, "vert_cda": +1},
    "C_La_mult": {"horiz_cla": +1, "vert_cla": +1},
    "one_plus_k": {"drag_xU": -1},
    "C_p_base": {"drag_xU": -1},
}


def test_forward_map_identity(nominal_knobs):
    """G1: forward_map(nominal) reproduces the canonical SDF surface."""
    got = slots(forward_map(nominal_knobs))
    canonical = slots(HydrodynamicsSpec())
    for k, v in canonical.items():
        assert got[k] == pytest.approx(v, rel=1e-9, abs=1e-9), k


def test_calibration_scalars():
    """G2: every scalar is non-negative, finite, and within the sanity band."""
    for name, value in calibration_scalars().items():
        assert math.isfinite(value), f"{name} is not finite"
        assert value >= 0.0, f"{name} must be non-negative"
        assert value < SCALAR_BAND, f"{name}={value} exceeds the sanity band"


def test_sign_sanity(nominal_knobs):
    """Random ±20% knobs keep damping ≤ 0 and added mass ≥ 0."""
    rng = np.random.default_rng(42)
    for _ in range(10):
        knobs = {k: v * rng.uniform(0.8, 1.2) for k, v in nominal_knobs.items()}
        s = slots(forward_map(knobs))
        for k, v in s.items():
            if k.startswith("added_mass"):
                assert v >= 0.0, k
            elif k.startswith("drag"):
                assert v <= 0.0, k


def test_no_double_counting(nominal_knobs):
    """G3: fin size feeds body added mass but never body damping.

    Growing the fins must move the AM slots that sum a fin term, yet leave
    every body damping slot untouched — fin lift/drag is the LiftDrag plugin's
    job at runtime, so it must not leak into the body block.
    """
    base = slots(forward_map(nominal_knobs))
    knobs = dict(nominal_knobs)
    knobs["b_f"] *= 1.10
    knobs["b_r"] *= 1.10
    bigger = slots(forward_map(knobs))

    for d in ("drag_xU", "drag_yV", "drag_zW", "drag_kP", "drag_mQ", "drag_nR"):
        assert bigger[d] == base[d], f"{d} moved when only fin spans changed"
    for am in ("added_mass_zz", "added_mass_qq", "added_mass_yy", "added_mass_rr"):
        assert bigger[am] != base[am], f"{am} should track fin span"


def test_single_knob_sweep(nominal_knobs):
    """§14.4: per knob, only the Jacobian-nonzero slots move, with the right sign."""
    base = slots(forward_map(nominal_knobs))

    for knob, appears in APPEARS.items():
        knobs = dict(nominal_knobs)
        knobs[knob] *= 1.05
        res = slots(forward_map(knobs))

        for s in base:
            if s in appears:
                assert res[s] != base[s], f"{knob}: expected {s} to move"
            else:
                # Tolerate float-cancellation noise (D drops out of the
                # cross-flow damping only algebraically — doc §11 note), but a
                # real accidental coupling would move the value far more.
                assert res[s] == pytest.approx(base[s], rel=1e-9, abs=1e-12), (
                    f"{knob}: {s} moved but should not"
                )

        for s, direction in SIGN.get(knob, {}).items():
            if direction > 0:
                assert res[s] > base[s], f"{knob}: {s} should increase"
            else:
                assert res[s] < base[s], f"{knob}: {s} should decrease"


def test_alpha_stall_knobs(nominal_knobs):
    """The split stall knobs scale only their own fin's alpha_stall."""
    base = forward_map(nominal_knobs)

    horiz = forward_map({**nominal_knobs, "alpha_stall_horiz": nominal_knobs["alpha_stall_horiz"] * 1.1})
    assert horiz.left_fin.alpha_stall > base.left_fin.alpha_stall
    assert horiz.right_fin.alpha_stall > base.right_fin.alpha_stall
    assert horiz.top_rudder.alpha_stall == base.top_rudder.alpha_stall
    assert slots(horiz) == slots(base)  # no numeric slot touched

    rudder = forward_map({**nominal_knobs, "alpha_stall_rudder": nominal_knobs["alpha_stall_rudder"] * 1.1})
    assert rudder.top_rudder.alpha_stall > base.top_rudder.alpha_stall
    assert rudder.left_fin.alpha_stall == base.left_fin.alpha_stall
    assert slots(rudder) == slots(base)


def _hoerner(tc: float) -> float:
    return 1.0 + 2.0 * tc + 60.0 * tc**4


def test_round_trip_monomials(nominal_data):
    """§14.5 (informational): recover the invertible monomials from the SDF.

    Fin areas invert exactly (area = λ_area · b·c) and t/c inverts via the
    lower Hoerner root on [0, 0.3], where _hoerner is monotonic so a bisection
    is well-defined.
    """
    nominal_knobs = nominal_data["physics_knobs"]
    fc = nominal_data["fluid_constants"]
    u_ref, nu = float(fc["u_ref"]), float(fc["nu"])
    lam = calibration_scalars()
    rng = np.random.default_rng(7)

    def c_f_from_re(c_len: float) -> float:
        re = u_ref * c_len / nu
        return 0.075 / (math.log10(re) - 2) ** 2

    for _ in range(100):
        knobs = {k: v * rng.uniform(0.95, 1.05) for k, v in nominal_knobs.items()}
        spec = forward_map(knobs)

        # areas: divide out the (near-unity) area scalar to recover b·c.
        assert spec.left_fin.area / lam["horiz_area"] == pytest.approx(
            knobs["b_f"] * knobs["c_f"], rel=1e-9
        )
        assert spec.top_rudder.area / lam["vert_area"] == pytest.approx(
            knobs["b_r"] * knobs["c_r"], rel=1e-9
        )

        # t/c from horiz cda: rendered = λ · 2·C_F(c_f) · hoerner(t/c).
        target = spec.left_fin.cda / (lam["horiz_cda"] * 2.0 * c_f_from_re(knobs["c_f"]))
        lo, hi = 0.0, 0.3
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if _hoerner(mid) < target:
                lo = mid
            else:
                hi = mid
        assert 0.5 * (lo + hi) == pytest.approx(knobs["t_over_c"], rel=1e-2)


def test_hyperbox_corners(nominal_knobs):
    """Sub-sampled ±20% hyperbox corners stay physically signed.

    The full 2^16 product is ~65k Pydantic builds — too slow for the fast tier.
    A seeded random sample of corners catches sign blow-ups just as well.
    """
    keys = list(nominal_knobs)
    rng = np.random.default_rng(123)
    for _ in range(2000):
        mults = rng.choice([0.8, 1.2], size=len(keys))
        knobs = {k: nominal_knobs[k] * m for k, m in zip(keys, mults)}
        s = slots(forward_map(knobs))
        for k, v in s.items():
            if k.startswith("added_mass"):
                assert v >= 0.0, k
            elif k.startswith("drag"):
                assert v <= 0.0, k
            elif k.endswith(("_cla", "_area")):
                assert v > 0.0, k
