"""Tier 3 (sim-artifact, no Gazebo): sim_ready_gate probe + readiness logic.

House rule: anything importing ``nautilus_hal`` is marker-gated
``@pytest.mark.sim`` even without booting Gazebo. This file pins the pure
predicates behind the physics-liveness probe's phase transitions — the
logic that decides whether a frozen-buoyancy run (the v3 campaign's 30 %
yield loss) is caught before ``/sim/ready`` latches:

- ``probe_target_m3``: deflate set point with the bladder-interval floor;
- ``quiescent_baseline_z``: the settle guard — a baseline may only latch
  once the passive spawn settle (which frozen runs perform too) is over,
  else the settle's own ~0.3 m of motion would satisfy the response
  threshold and pass frozen runs;
- ``hull_responded`` / ``within_eps``: the DEFLATE and RESTORE exit
  conditions, including the negative-down z convention;
- ``probe_failure_label``: the abort_init detail line pilot forensics
  greps for;
- the cross-language suppress-topic spelling (Python constant vs the
  literal in HeaveAugmentPlugin.cc) — the only other check of that
  agreement is the minutes-long full-Gazebo smoke test;
- first lock-in cases for the pre-existing ``missing_requirements``.

The dave sister repo may not be built (nautilus-ros-only CI): skip, don't
fail, when ``nautilus_hal`` is absent.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.sim

gate = pytest.importorskip(
    "nautilus_hal.sim_ready_gate",
    reason="dave sister repo not built (nautilus_hal not importable)",
)

# The sweep spawn state: bladder_max policy, float at ~0.44 m depth.
SPAWN_VOL = 2.45e-3
FLOAT_Z = -0.44


class TestProbeTarget:
    def test_nominal_deflate(self):
        # 6.5e-4 below the bladder_max spawn: beats the worst sampled
        # reserve buoyancy (4.45e-4) and stays above bladder_min.
        assert gate.probe_target_m3(SPAWN_VOL, 6.5e-4, 1.0e-3) == pytest.approx(
            1.8e-3
        )

    def test_floor_clamps(self):
        # A shallow spawn volume cannot be probed below the bridge's
        # operating interval — the synthesized tank telemetry must stay
        # inside its calibrated range.
        assert gate.probe_target_m3(1.5e-3, 6.5e-4, 1.0e-3) == 1.0e-3

    def test_exact_floor_is_reachable(self):
        assert gate.probe_target_m3(1.65e-3, 6.5e-4, 1.0e-3) == 1.0e-3


class TestQuiescentBaseline:
    WINDOW_S = 2.0
    PTP_M = 0.03

    @staticmethod
    def _series(t0, t1, z_fn, hz=50.0):
        n = int((t1 - t0) * hz)
        return [(t0 + i / hz, z_fn(t0 + i / hz)) for i in range(n)]

    def test_settling_hull_never_baselines(self):
        # The passive spawn settle: z sliding -0.115 -> -0.44 at ~0.05 m/s.
        # A quiescent verdict here is exactly the false-pass that would let
        # frozen runs through — the trailing 2 s spans 0.1 m >> ptp.
        samples = self._series(0.0, 4.0, lambda t: -0.115 - 0.05 * t)
        assert (
            gate.quiescent_baseline_z(samples, 4.0, self.WINDOW_S, self.PTP_M)
            is None
        )

    def test_settled_hull_baselines_at_median(self):
        samples = self._series(0.0, 3.0, lambda t: FLOAT_Z + 0.005 * ((t * 7) % 2 - 1))
        z0 = gate.quiescent_baseline_z(samples, 3.0, self.WINDOW_S, self.PTP_M)
        assert z0 == pytest.approx(FLOAT_Z, abs=0.01)

    def test_only_trailing_window_counts(self):
        # Settle for 5 s, then 2+ s quiescent at the float: the old motion
        # must not block the baseline.
        settle = self._series(0.0, 5.0, lambda t: -0.115 - 0.065 * t)
        still = self._series(5.0, 7.5, lambda t: FLOAT_Z)
        z0 = gate.quiescent_baseline_z(settle + still, 7.5, self.WINDOW_S, self.PTP_M)
        assert z0 == pytest.approx(FLOAT_Z)

    def test_too_few_samples_is_not_quiescent(self):
        samples = [(0.0, FLOAT_Z), (0.1, FLOAT_Z)]
        assert (
            gate.quiescent_baseline_z(samples, 0.1, self.WINDOW_S, self.PTP_M)
            is None
        )


class TestResponsePredicates:
    def test_sink_meets_threshold(self):
        # Odometry z is negative-down: sinking DECREASES z.
        assert gate.hull_responded(FLOAT_Z, FLOAT_Z - 0.11, 0.10)
        assert not gate.hull_responded(FLOAT_Z, FLOAT_Z - 0.06, 0.10)

    def test_upward_motion_never_responds(self):
        # A hull popping UP 0.1 m is not a deflate response, whatever the
        # magnitude — the sign convention is the contract.
        assert not gate.hull_responded(FLOAT_Z, FLOAT_Z + 0.10, 0.10)

    def test_frozen_hull_never_responds(self):
        assert not gate.hull_responded(FLOAT_Z, FLOAT_Z, 0.10)

    def test_within_eps_either_side(self):
        # The RESTORE checks (volume echo and refloat z) both ride on
        # this; the refloat may overshoot the equilibrium, so both sides
        # count. Probe clearly inside / clearly outside — the exact edge
        # is a float-representation coin flip, not a behavior.
        assert gate.within_eps(SPAWN_VOL - 4.9e-5, SPAWN_VOL, 5.0e-5)
        assert not gate.within_eps(SPAWN_VOL - 5.5e-5, SPAWN_VOL, 5.0e-5)
        assert gate.within_eps(FLOAT_Z + 0.09, FLOAT_Z, 0.10)
        assert not gate.within_eps(FLOAT_Z - 0.11, FLOAT_Z, 0.10)


class TestFailureLabel:
    def test_deflate_label_carries_measurements(self):
        label = gate.probe_failure_label(
            gate.PROBE_DEFLATE, 0.003, 0.10, 25.0, SPAWN_VOL, 1.86e-3
        )
        # The v3 frozen signature reads straight off the line: hull did
        # not move while the volume echo tracked the deflate.
        assert label.startswith("physics:no-hull-response(")
        assert "dz=0.003m<0.100m" in label
        assert "2.4500e-03->1.8600e-03" in label

    def test_restore_label_names_the_epsilon(self):
        # In RESTORE the threshold is a proximity epsilon, not a sink
        # requirement — the label must not read as a threshold miss.
        label = gate.probe_failure_label(
            gate.PROBE_RESTORE, 0.42, 0.10, 40.0, SPAWN_VOL, 1.86e-3
        )
        assert label.startswith("physics:no-restore(")
        assert "eps=0.100m" in label


class TestSuppressTopicParity:
    def test_plugin_literal_matches_python_constant(self):
        """The suppress path is spelled independently in C++ and Python;
        a drift compiles and imports clean, then silently stops landing —
        the servo would fire into every probe sink campaign-wide. This is
        the 1-second lock on that agreement (the only other check is the
        full-Gazebo smoke test)."""
        from nautilus_hal.constants import SimTopics

        # Workspace layout: this file lives in
        # src/nautilus-ros/src/py_pkg/test/sim/; the dave checkout is a
        # sibling repo under the same workspace src/. Absent in a
        # nautilus-ros-only checkout: skip, don't fail.
        plugin_cc = (
            Path(__file__).resolve().parents[5]
            / "dave"
            / "gazebo"
            / "dave_gz_model_plugins"
            / "src"
            / "HeaveAugmentPlugin.cc"
        )
        if not plugin_cc.is_file():
            pytest.skip("dave source checkout not present beside nautilus-ros")
        match = re.search(
            r'"/model/"\s*\+\s*model\.Name\(_ecm\)\s*\+\s*"(/[\w/]+)"',
            plugin_cc.read_text(),
        )
        assert match, "suppress-topic construction not found in HeaveAugmentPlugin.cc"
        assert (
            "/model/{model_name}" + match.group(1) == SimTopics.HEAVE_ENTRY_SUPPRESS
        )


class TestMissingRequirements:
    def test_reports_absent_nodes_and_dead_endpoints_sorted(self):
        missing = gate.missing_requirements(
            ["bcu_node", "attitude_node"],
            {"attitude_node"},
            [("sub:/bcu/rpm (BCU bridge reader)", 0), ("pub:/x", 2)],
        )
        assert missing == ["node:bcu_node", "sub:/bcu/rpm (BCU bridge reader)"]

    def test_empty_when_ready(self):
        assert (
            gate.missing_requirements(["a"], {"a", "b"}, [("pub:/x", 1)]) == []
        )
