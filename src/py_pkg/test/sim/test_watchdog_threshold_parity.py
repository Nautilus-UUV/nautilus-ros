"""Tier 3 parity: sweep-watchdog viability thresholds == offline classifier's.

Sim-artifact test (house rule: tests importing sim-side packages are
Tier 3 even without Gazebo — marker-gated ``@pytest.mark.sim``). The
canonical MIN_DIVE_M / MIN_RETURN_M live next to ``classify_run`` in
``scripts/analysis/sweep_loader.py``; the in-run watchdog
(``nautilus_hal.sweep_watchdog.plausibility``, dave repo) restates them
because neither side can import the other — analysis hosts have no dave
checkout, and ``scripts/`` is not an installed package inside the sweep
container. This is the tripwire that keeps the early-abort rule and the
offline classifier the same two numbers (the duplicate-literal pattern
``spec/rig.py`` uses for ``FAULT_SCHEDULE_SHAPES``).

``analysis`` resolves via the scripts/ sys.path hook in the shared
``test/conftest.py``.
"""

import pytest

from analysis import sweep_loader
from nautilus_hal.sweep_watchdog import plausibility

pytestmark = pytest.mark.sim


def test_thresholds_match_offline_classifier():
    assert plausibility.MIN_DIVE_M == sweep_loader.MIN_DIVE_M
    assert plausibility.MIN_RETURN_M == sweep_loader.MIN_RETURN_M


def test_classify_run_defaults_on_the_same_numbers():
    # classify_run's keyword defaults are bound from the module constants;
    # a hand-typed default there would dodge the constant-level assert.
    defaults = sweep_loader.classify_run.__kwdefaults__
    assert defaults["min_dive_m"] == plausibility.MIN_DIVE_M
    assert defaults["min_return_m"] == plausibility.MIN_RETURN_M
