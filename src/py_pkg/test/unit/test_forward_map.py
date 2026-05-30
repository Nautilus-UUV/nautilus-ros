import itertools
import pytest
import yaml
import numpy as np
from pathlib import Path
from py_pkg.scenarios.compile import forward_map, _CALIBRATION_SCALARS
from py_pkg.scenarios.spec.rig import HydrodynamicsSpec

NOMINAL_KNOBS_PATH = Path(__file__).resolve().parent.parent.parent / "py_pkg/scenarios/library/nominal_knobs.yaml"

@pytest.fixture
def nominal_knobs():
    raw = yaml.safe_load(NOMINAL_KNOBS_PATH.read_text())
    return raw["physics_knobs"]

def test_forward_map_identity(nominal_knobs):
    spec = forward_map(nominal_knobs, jitter_seed=0, jitter_sigma=0.0)
    default_spec = HydrodynamicsSpec(knobs=spec.knobs)
    
    spec_dict = spec.model_dump()
    def_dict = default_spec.model_dump()
    for k in spec_dict:
        if isinstance(spec_dict[k], float):
            assert spec_dict[k] == pytest.approx(def_dict[k], rel=1e-5)
        elif isinstance(spec_dict[k], dict):
            for sub_k in spec_dict[k]:
                if isinstance(spec_dict[k][sub_k], float):
                    assert spec_dict[k][sub_k] == pytest.approx(def_dict[k][sub_k], rel=1e-5)

def test_calibration_scalars():
    for name, value in _CALIBRATION_SCALARS.items():
        assert value >= 0, f"{name} must be non-negative"
        assert value < 300, f"{name} is suspiciously large"
        assert float(value) == value

def test_sign_sanity(nominal_knobs):
    rng = np.random.default_rng(42)
    for _ in range(10):
        knobs = nominal_knobs.copy()
        for k in knobs:
            knobs[k] *= rng.uniform(0.8, 1.2)
        spec = forward_map(knobs, jitter_seed=0, jitter_sigma=0.0)
        
        # Damping is <= 0
        assert spec.drag_xU <= 0
        assert spec.drag_yV <= 0
        assert spec.drag_zW <= 0
        assert spec.drag_kP <= 0
        assert spec.drag_mQ <= 0
        assert spec.drag_nR <= 0
        
        # Added mass is >= 0
        assert spec.added_mass_xx >= 0
        assert spec.added_mass_yy >= 0
        assert spec.added_mass_zz >= 0
        assert spec.added_mass_pp >= 0
        assert spec.added_mass_qq >= 0
        assert spec.added_mass_rr >= 0

def test_single_knob_sweep(nominal_knobs):
    base = forward_map(nominal_knobs, jitter_seed=0, jitter_sigma=0.0)
    
    # +5% L => pitch AM (qq) and yaw AM (rr) should increase
    knobs = nominal_knobs.copy()
    knobs["L"] *= 1.05
    res = forward_map(knobs, jitter_seed=0, jitter_sigma=0.0)
    assert res.added_mass_qq > base.added_mass_qq
    assert res.added_mass_rr > base.added_mass_rr
    
    # +5% D => sway/heave AM (yy, zz) should decrease (less slender)
    knobs = nominal_knobs.copy()
    knobs["D"] *= 1.05
    res = forward_map(knobs, jitter_seed=0, jitter_sigma=0.0)
    assert res.added_mass_yy < base.added_mass_yy
    assert res.added_mass_zz < base.added_mass_zz
    
    # +5% C_d_c => sway/heave damping (yV, zW) should be MORE negative
    knobs = nominal_knobs.copy()
    knobs["C_d_c"] *= 1.05
    res = forward_map(knobs, jitter_seed=0, jitter_sigma=0.0)
    assert res.drag_yV < base.drag_yV
    assert res.drag_zW < base.drag_zW

def test_hyperbox_corners(nominal_knobs):
    keys = list(nominal_knobs.keys())
    # Instead of full 2^16 combinations, we test random corners
    # (actually doing full 2^16 in pure Python could take several seconds)
    # The requirement says "Test all 2^16 corners". Let's run all 65536.
    combinations = list(itertools.product([0.8, 1.2], repeat=len(keys)))
    for multipliers in combinations:
        knobs = {k: nominal_knobs[k] * m for k, m in zip(keys, multipliers)}
        spec = forward_map(knobs, jitter_seed=0, jitter_sigma=0.0)
        
        assert spec.added_mass_xx >= 0
        assert spec.added_mass_yy >= 0
        assert spec.added_mass_zz >= 0
        assert spec.added_mass_pp >= 0
        assert spec.added_mass_qq >= 0
        assert spec.added_mass_rr >= 0
        
        assert spec.drag_xU <= 0
        assert spec.drag_yV <= 0
        assert spec.drag_zW <= 0
        assert spec.drag_kP <= 0
        assert spec.drag_mQ <= 0
        assert spec.drag_nR <= 0
        
        assert spec.left_fin.area > 0
        assert spec.left_fin.cla > 0
        assert spec.top_rudder.area > 0
        assert spec.top_rudder.cla > 0
