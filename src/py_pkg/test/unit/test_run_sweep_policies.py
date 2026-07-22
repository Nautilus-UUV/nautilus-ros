"""Tier 1 for run_sweep's pure per-run policies (scripts/run_sweep.py).

Two policies were added after the v2 campaign drop lost 4 of 5 runs to
infrastructure failures, and each claim is pinned here:

- ``should_retry``: which reaped outcomes requeue the same YAML — infra
  verdicts (abort_init/abort_bad_start/abort_floater/abort_sinker),
  verdict-less crashes, watchdog-mode wall-clock hangs, and missing
  bags; NEVER a completed mission.
- ``truncating_cap_runs``: a --per-run-timeout below a run's scaled
  mission budget is detected up front (the drop ran 1800 s caps against
  ~17 ks budgets and killed every long mission mid-write).

run_sweep.py is a script, not a package module: import it by path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_RUN_SWEEP = Path(__file__).resolve().parents[4] / "scripts" / "run_sweep.py"
if not _RUN_SWEEP.is_file():  # pragma: no cover — repo-layout guard
    pytest.skip(f"run_sweep.py not found at {_RUN_SWEEP}", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("run_sweep_under_test", _RUN_SWEEP)
run_sweep = importlib.util.module_from_spec(_spec)
# Registered before exec: @dataclass resolves its owning module through
# sys.modules to evaluate the string annotations.
sys.modules["run_sweep_under_test"] = run_sweep
_spec.loader.exec_module(run_sweep)


def _retry(**overrides) -> bool:
    kwargs = dict(
        verdict="",
        exit_code=0,
        timed_out=False,
        bag_present=True,
        record=True,
        watchdog=True,
    )
    kwargs.update(overrides)
    return run_sweep.should_retry(**kwargs)


class TestShouldRetry:
    def test_mission_complete_never_retries(self):
        # Even a weird exit code / timeout flag cannot requeue a
        # completed mission — the bag is the product.
        assert not _retry(verdict="mission_complete")
        assert not _retry(verdict="mission_complete", exit_code=1, timed_out=True)

    @pytest.mark.parametrize(
        "verdict",
        ["abort_init", "abort_bad_start", "abort_floater", "abort_sinker"],
    )
    def test_infra_verdicts_always_retry(self, verdict):
        assert _retry(verdict=verdict)

    def test_unknown_nonempty_verdict_does_not_retry(self):
        # Forward-compat: a future conclusive verdict is a conclusion.
        assert not _retry(verdict="some_new_verdict")

    def test_verdictless_crash_retries(self):
        assert _retry(exit_code=1)

    def test_verdictless_timeout_retries_only_with_watchdog(self):
        # Without a watchdog the wall clock is the NORMAL end of a run.
        assert _retry(timed_out=True, watchdog=True)
        assert not _retry(timed_out=True, watchdog=False, exit_code=1)

    def test_missing_bag_retries_only_when_recording(self):
        assert _retry(bag_present=False, record=True)
        assert not _retry(bag_present=False, record=False)

    def test_clean_verdictless_run_with_bag_does_not_retry(self):
        # Legacy watchdog-less sweeps: normal end, bag present.
        assert not _retry(watchdog=False)


class TestTruncatingCapRuns:
    MISSION = {"target_pressure_pa": 294_180.0, "n_oscillations": 2}  # ~30 m

    def test_no_cap_flags_nothing(self):
        assert run_sweep.truncating_cap_runs({"lhs_0000": self.MISSION}, None) == []

    def test_cap_below_scaled_is_flagged_worst_first(self):
        # ~3 m single cycle: scaled ≈ 1.2 ks, under a 1800 s cap — only
        # the deep run is an offender, and it leads the (sorted) list.
        shallow = {"target_pressure_pa": 29_418.0}
        offenders = run_sweep.truncating_cap_runs(
            {"shallow": shallow, "deep": self.MISSION}, 1800.0
        )
        assert [run_id for run_id, _ in offenders] == ["deep"]
        # The v2 failure shape: a 30 m two-cycle mission needs >> 1800 s.
        assert offenders[0][1] > 10_000.0
        # Tighten the cap below the shallow budget and both are flagged,
        # worst first.
        offenders = run_sweep.truncating_cap_runs(
            {"shallow": shallow, "deep": self.MISSION}, 1000.0
        )
        assert [run_id for run_id, _ in offenders] == ["deep", "shallow"]

    def test_adequate_cap_flags_nothing(self):
        scaled = run_sweep._scaled_timeout(self.MISSION, None)
        assert run_sweep.truncating_cap_runs({"deep": self.MISSION}, scaled) == []

    def test_run_without_target_is_exempt(self):
        # No target pressure -> nothing to scale -> the cap applies as-is.
        assert run_sweep.truncating_cap_runs({"x": {"dwell_s": 5.0}}, 1.0) == []


class TestDomainWindow:
    def test_v2_campaign_shape_is_rejected(self):
        # 64 slots at base 50 -> domains 50..113; 102+ map DDS ports into
        # the ephemeral range (7400 + 250*102 = 32900 >= 32768).
        assert run_sweep.domain_window_collides_with_ephemeral(50, 64)

    def test_safe_base_for_64_slots(self):
        # Whole-block rule: domain d owns ports up to 7400 + 250*(d+1) - 1,
        # so the highest fully-safe domain is 100 and base = 100 - 63 = 37.
        base = run_sweep.max_safe_domain_base(64)
        assert base == 37
        assert not run_sweep.domain_window_collides_with_ephemeral(base, 64)
        assert run_sweep.domain_window_collides_with_ephemeral(base + 1, 64)

    def test_default_base_is_safe_up_to_91_slots(self):
        # New default 10: safe through 91 slots (top domain 100), unsafe at 92.
        assert not run_sweep.domain_window_collides_with_ephemeral(10, 91)
        assert run_sweep.domain_window_collides_with_ephemeral(10, 92)
