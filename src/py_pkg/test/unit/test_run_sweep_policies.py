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

run_sweep.py is a script, not a package module; ``_sweep_specs`` owns the repo
walk and the numpy/scipy import policy that makes it importable from a test.
"""

from __future__ import annotations

import pytest
from _sweep_specs import needs_scripts, run_sweep


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


@needs_scripts
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


@needs_scripts
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
        assert run_sweep.truncating_cap_runs({"x": {"n_oscillations": 2}}, 1.0) == []


@needs_scripts
class TestRequeueForRetry:
    """The retry mechanics around the abort_init / probe-failure path.

    v3's interruption at 700/2048 stranded every one of its 189 requeued
    runs at the back of the queue (all rows attempt=0) — front-of-queue
    requeue and verdict parking are what make a probe abort a cheap,
    *diagnosable* retry instead of silent loss.
    """

    def _runner(self, tmp_path):
        return run_sweep.SweepRunner(
            sif=tmp_path / "x.sif",
            scenarios_dir=tmp_path,
            sweep_name="sweepy",
            sim_data_dir=tmp_path / "sim_data",
            launch_file="sawtooth_sim.launch.py",
            extra_launch_args=["watchdog:=true"],
            ros_domain_base=10,
            record=True,
            per_run_timeout=None,
            slots=[run_sweep.Slot(index=0)],
            queue=[("lhs_0001", tmp_path / "lhs_0001.yaml", 0)],
        )

    def _failed_slot(self, runner, tmp_path, with_bag=True):
        slot = runner.slots[0]
        slot.run_id = "lhs_0000"
        slot.yaml_path = tmp_path / "lhs_0000.yaml"
        if with_bag:
            slot.host_bag_path = runner.sweep_dir / "lhs_0000" / "raw"
            slot.host_bag_path.mkdir(parents=True)
            (slot.host_bag_path / "raw_0.mcap").write_text("")
        return slot

    def test_requeue_goes_to_front_and_parks_verdict(self, tmp_path):
        runner = self._runner(tmp_path)
        slot = self._failed_slot(runner, tmp_path)
        verdict_file = slot.host_bag_path.parent / "run_verdict.json"
        verdict_file.write_text(
            '{"verdict": "abort_init", "missing": ["physics:no-hull-response(...)"]}\n'
        )
        runner._requeue_for_retry(slot, "abort_init", verdict_file)

        # Front of the queue, attempt bumped — an interrupted campaign has
        # already executed its retries instead of never reaching them.
        assert runner.queue[0] == ("lhs_0000", slot.yaml_path, 1)
        assert runner.queue[1][0] == "lhs_0001"
        # Bag parked with the verdict riding along; live paths cleared so
        # the retry cannot be misread.
        parked = runner.sweep_dir / "lhs_0000" / "raw.failed0"
        assert (parked / "raw_0.mcap").is_file()
        assert "abort_init" in (parked / "run_verdict.json").read_text()
        assert not verdict_file.exists()
        assert not (runner.sweep_dir / "lhs_0000" / "raw").exists()

    def test_requeue_without_bag_unlinks_verdict(self, tmp_path):
        runner = self._runner(tmp_path)
        slot = self._failed_slot(runner, tmp_path, with_bag=False)
        verdict_file = runner.sweep_dir / "lhs_0000" / "run_verdict.json"
        verdict_file.parent.mkdir(parents=True)
        verdict_file.write_text('{"verdict": "abort_floater"}\n')
        runner._requeue_for_retry(slot, "abort_floater", verdict_file)
        assert runner.queue[0] == ("lhs_0000", slot.yaml_path, 1)
        # Nowhere to park it — gone is correct (a stale verdict would be
        # read as the retry's conclusion).
        assert not verdict_file.exists()


@needs_scripts
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
