"""Tier 3 acceptance test: closed-loop sawtooth vertical velocities match
the 2026-06-24 lake dives.

This is the regression gate for the lake calibration of the sim plant
(fresh-water densities, fitted heave drag zW=-52.2 / zWabsW=-436.7,
trim-mass neutral point V_n=2.097e-3 m^3). It runs the full ROS + DAVE
stack through one SAWTOOTH cycle with the bladder clamps pinned to
dive 4's steady-leg volumes (see test_sawtooth_lake_velocity.yaml) and
measures steady vertical velocity exactly the way the lake analysis
did: depth from external pressure, median of 5 s regression slopes over
the bladder-railed portion of each leg.

Samples are timestamped with SIM time (the ground-truth odometry's own
header stamp, the same source run_watchdog clocks plausibility on), not
the wall clock: the full stack + Gazebo runs this host at RTF ~0.5-0.7,
and wall-clock slopes under-read true sim velocities by exactly that
factor (a 0.121 m/s descent measured 0.077 at RTF 0.63).

Anchors (dive 4, heave_calibration_targets.csv):
  descent: 0.1245 m/s at V_b=0.866e-3   -> assert 0.1245 +/- 0.020
  ascent:  0.0677 m/s at V_b=2.465e-3   -> assert 0.0677 +/- 0.020

The ascent band is anchored like the descent since the
HeaveAugmentPlugin ascent drag relief landed: the symmetric fit
over-damps ascent (fit residuals +26/28%, sim measured ~0.04 m/s where
the real vehicle did 0.068), and the plugin retains only
retain_fraction (nominal 0.66, fitted to the dive-2/4 ascent legs) of
the heave drag on ascent. The ascent branch of the curve-consistency
assertion uses the k-scaled curve accordingly; its tolerance stays at
25% until the pilot quantifies the fin LiftDrag deficit at the new
~0.067 m/s operating point.

The scenario pins entry.peak_speed_mps to 0.0: this test's measurand is
the bladder-railed *steady* window, and the entry transient would only
shorten the railed depth run (faster early descent puts the vehicle
deeper before the rail engages) without touching the steady speed. The
transient has its own gate (test_rate_envelope_sim.py).

Marker-gated ``@pytest.mark.sim``; opt in with ``pytest -m sim``.
``SIM_GUI=1`` shows the Gazebo GUI.
"""

import math
import os
import statistics
import unittest
from collections import deque

import launch_testing
import launch_testing.actions
import launch_testing.asserts
import launch_testing.markers
import pytest
import rclpy
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from py_pkg.path.missions.factory import MissionId
from py_pkg.scenarios.buoyancy import (
    LAKE_FIT_NEUTRAL_VOLUME_M3,
    LAKE_FIT_RHO_G,
    terminal_heave_speed_mps,
)
from py_pkg.scenarios.spec.rig import HydrodynamicsSpec
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Int32

from ._sim_helpers import (
    SimClock,
    mission_command,
    reap_lingering_gz,
    sim_depth_m,
    sim_gui_enabled,
    spin_for,
    spin_until,
)

# --- Lake anchors (dive 4, config B) --------------------------------------
# Provenance: UG-anomaly_detection/lake_test_jun24/investigation/
# heave_calibration_targets.csv + heave_calibration_fit.json; the fit
# anchors themselves live in py_pkg.scenarios.buoyancy next to
# terminal_heave_speed_mps, whose drag coefficients come from the spec
# defaults — those ARE the adopted lake fit, parity-locked to the
# canonical model.sdf — so a re-fit moves the consistency curve here
# automatically. Same for the ascent drag-relief fraction
# (HeaveAugmentPlugin retains k of the heave drag while ascending).
ASCENT_RETAIN_K = HydrodynamicsSpec().ascent_relief.retain_fraction

DESCENT_RAIL_M3 = 0.866e-3  # scenario bladder_min_m3
ASCENT_RAIL_M3 = 2.465e-3  # scenario bladder_max_m3
DESCENT_RAIL_ML = round(DESCENT_RAIL_M3 * 1e6)
ASCENT_RAIL_ML = round(ASCENT_RAIL_M3 * 1e6)

DESCENT_V_BAND = (0.1045, 0.1445)  # 0.1245 +/- 0.020 (real steady leg)
ASCENT_V_BAND = (0.048, 0.088)  # 0.0677 +/- 0.020 (real steady leg, relief on)
DESCENT_CURVE_TOL = 0.15
ASCENT_CURVE_TOL = 0.25

# Deeper than dive 4's 14.68 m peak on purpose: terminal velocity is
# depth-independent in this plant, and the extra runway stretches the
# bladder-railed steady window past the detector's minimum. The rail
# only engages once the pump has swung the bladder ~1.3 L (~85 s after
# the mission starts, by which point the vehicle has sunk to ~10 m) and
# the depth PID starts easing the bladder off the rail ~4 m short of
# the target, so the railed depth run is roughly (target - 14) m. At
# the in-band descent speed (~0.111 m/s) a 22 m target yields a ~70 s
# railed window, mirroring dive 4's real 78.6 s leg; the previous
# 17.5 m target only worked while the mis-trimmed plant descended at
# 0.077 m/s. Seabed is at 95 m — the post-flip overshoot to ~24 m has
# ample clearance.
TARGET_PRESSURE_PA = 215740.0  # 22.0 m * 9806.38 Pa/m
PITCH_RAD = math.radians(30.0)
N_RESURFACES = 1

RAIL_TOL_ML = 20  # "bladder railed" = volume within this of the clamp
EDGE_TRIM_S = 5.0  # drop this much at each end of a railed window
MIN_WINDOW_S = 25.0  # steady window must be at least this long
SLOPE_WIN_S = 5.0  # regression-slope window, mirrors the lake analysis


def _curve_v(dv_m3: float, retain_k: float) -> float:
    """Terminal speed the fitted lake curve predicts for |dV| offset.

    ``retain_k`` scales both drag terms — pass ASCENT_RETAIN_K for the
    ascent leg, where the HeaveAugmentPlugin leaves only that fraction
    of the heave drag in play.
    """
    return terminal_heave_speed_mps(
        LAKE_FIT_RHO_G * abs(dv_m3), retain_fraction=retain_k
    )


def _steady_velocity(
    depth_samples: list[tuple[float, float]],
    t_start: float,
    t_end: float,
) -> float:
    """Median of 5 s regression slopes over [t_start, t_end] (m/s, +down)."""
    window = [(t, d) for t, d in depth_samples if t_start <= t <= t_end]
    slopes = []
    i = 0
    for j in range(len(window)):
        while window[j][0] - window[i][0] > SLOPE_WIN_S:
            i += 1
        chunk = window[i : j + 1]
        if len(chunk) >= 10 and chunk[-1][0] - chunk[0][0] >= 0.8 * SLOPE_WIN_S:
            slopes.append(
                statistics.linear_regression(
                    [t for t, _ in chunk], [d for _, d in chunk]
                ).slope
            )
    return statistics.median(slopes) if slopes else float("nan")


class _RailedWindowTracker:
    """Longest contiguous span with bladder volume within RAIL_TOL_ML of
    the rail, maintained incrementally.

    The test polls readiness from a ``spin_until`` predicate that fires
    per delivered callback, so the update must be O(1) per volume sample
    — not a rescan of the whole telemetry history.

    Samples pass through a median-of-5 debounce first: the volume
    telemetry carries the scenario's lake-fitted sensor noise, and a
    single outlier (e.g. 888 mL in an otherwise railed-at-866 stream)
    must not reset the contiguous span — the lake analysis this test
    mirrors likewise reads the rail from medians, not raw samples.
    """

    _DEBOUNCE_N = 5

    def __init__(self, rail_ml: int):
        self.rail_ml = rail_ml
        self._recent: deque[int] = deque(maxlen=self._DEBOUNCE_N)
        self._span: tuple[float, float] | None = None  # currently open span
        self._best: tuple[float, float] | None = None

    def add(self, t: float, ml: int) -> None:
        self._recent.append(ml)
        ml = statistics.median(self._recent)
        if abs(ml - self.rail_ml) > RAIL_TOL_ML:
            self._span = None
            return
        self._span = (self._span[0], t) if self._span else (t, t)
        if self._best is None or (
            self._span[1] - self._span[0] > self._best[1] - self._best[0]
        ):
            self._best = self._span

    def window(self) -> tuple[float, float] | None:
        """Best span trimmed by EDGE_TRIM_S; None while shorter than
        MIN_WINDOW_S."""
        if self._best is None:
            return None
        lo, hi = self._best[0] + EDGE_TRIM_S, self._best[1] - EDGE_TRIM_S
        if hi - lo < MIN_WINDOW_S:
            return None
        return (lo, hi)


@pytest.mark.launch_test
@pytest.mark.sim
@launch_testing.markers.keep_alive
def generate_test_description():
    reap_lingering_gz()

    sawtooth_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    FindPackageShare("nautilus_hal").find("nautilus_hal"),
                    "launch",
                    "sawtooth_sim.launch.py",
                )
            ]
        ),
        launch_arguments={
            "scenario": os.path.join(
                os.path.dirname(__file__),
                "scenarios",
                "test_sawtooth_lake_velocity.yaml",
            ),
            "headless": "false" if sim_gui_enabled() else "true",
        }.items(),
    )

    return (
        LaunchDescription(
            [
                sawtooth_sim_launch,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {},
    )


class _LakeVelocityDriver(Node):
    """Kicks SAWTOOTH and captures depth + bladder-volume telemetry.

    Depth/volume samples are stamped with the latest ground-truth
    odometry header stamp — Gazebo sim time, ~100 Hz, so the cross-topic
    skew is <= 10 ms — keeping the 5 s regression slopes honest when the
    stack drags RTF below 1.
    """

    def __init__(self):
        super().__init__("lake_velocity_test_driver")
        self.depth_samples: list[tuple[float, float]] = []  # (sim t, depth m)
        self.rail_trackers = {
            ml: _RailedWindowTracker(ml) for ml in (DESCENT_RAIL_ML, ASCENT_RAIL_ML)
        }
        self.imu_msg_count = 0
        self.sim_clock = SimClock(self)

        self.path_pub = create_publisher_for_topic(self, UUVTopics.PATH)
        self.command_pub = create_publisher_for_topic(self, UUVTopics.COMMAND)
        create_subscription_for_topic(self, UUVTopics.IMU, self._on_imu)
        create_subscription_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE, self._on_pressure
        )
        create_subscription_for_topic(self, UUVTopics.BCU_VOLUME, self._on_volume)

    def _on_imu(self, _msg: Imu) -> None:
        self.imu_msg_count += 1

    def _on_pressure(self, msg: Int32) -> None:
        if self.sim_clock.now is None:
            return
        self.depth_samples.append((self.sim_clock.now, sim_depth_m(float(msg.data))))

    def _on_volume(self, msg: Int32) -> None:
        if self.sim_clock.now is None:
            return
        for tracker in self.rail_trackers.values():
            tracker.add(self.sim_clock.now, int(msg.data))

    def publish_mission(self) -> None:
        # shallow_pressure_pa left at 0.0: climb to the surface each dive.
        self.path_pub.publish(
            mission_command(
                MissionId.SAWTOOTH,
                target_pressure_pa=TARGET_PRESSURE_PA,
                angle_rad=PITCH_RAD,
                n_resurfaces=N_RESURFACES,
            )
        )

    def publish_start(self) -> None:
        self.command_pub.publish(Bool(data=True))


@pytest.mark.sim
class SawtoothLakeVelocityTest(unittest.TestCase):
    """Behavior: sawtooth leg velocities land inside the lake envelopes."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.driver = _LakeVelocityDriver()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.driver)

    def tearDown(self):
        self.executor.remove_node(self.driver)
        self.driver.destroy_node()
        self.executor.shutdown()

    def _leg_ready(self, rail_ml: int) -> bool:
        return self.driver.rail_trackers[rail_ml].window() is not None

    def test_leg_velocities_match_lake_dive4(self):
        # Wall-clock TIMEOUTS only (the measurements clock on sim time):
        # sized for RTF ~0.5 so a loaded host doesn't fail the discovery
        # waits before the sim has physically produced the windows.
        startup_timeout_s = 60.0
        descent_budget_s = 600.0
        ascent_budget_s = 900.0

        sim_ready = spin_until(
            self.executor,
            lambda: self.driver.imu_msg_count >= 1
            and self.driver.sim_clock.now is not None,
            timeout_s=startup_timeout_s,
        )
        self.assertTrue(sim_ready, "IMU/odometry never arrived — sim not up?")
        spin_for(self.executor, 2.0)

        self.driver.publish_mission()
        spin_for(self.executor, 0.5)
        self.driver.publish_start()

        # Descent: wait until a full railed-at-min steady window exists.
        got_descent = spin_until(
            self.executor,
            lambda: self._leg_ready(DESCENT_RAIL_ML),
            timeout_s=descent_budget_s,
        )
        self.assertTrue(
            got_descent,
            f"no >= {MIN_WINDOW_S + 2 * EDGE_TRIM_S:.0f}s bladder-railed descent "
            f"window within {descent_budget_s}s — did the BCU saturate at "
            f"bladder_min as expected?",
        )

        # Ascent: keep spinning until the railed-at-max window exists.
        got_ascent = spin_until(
            self.executor,
            lambda: self._leg_ready(ASCENT_RAIL_ML),
            timeout_s=ascent_budget_s,
        )
        self.assertTrue(
            got_ascent,
            f"no bladder-railed ascent window within {ascent_budget_s}s — "
            "did the sawtooth flip to the ascend leg?",
        )

        # --- Measure each leg exactly like the lake analysis ---
        for name, rail_ml, rail_m3, sign, band, curve_tol, retain_k in (
            (
                "descent",
                DESCENT_RAIL_ML,
                DESCENT_RAIL_M3,
                +1.0,
                DESCENT_V_BAND,
                DESCENT_CURVE_TOL,
                1.0,
            ),
            (
                "ascent",
                ASCENT_RAIL_ML,
                ASCENT_RAIL_M3,
                -1.0,
                ASCENT_V_BAND,
                ASCENT_CURVE_TOL,
                ASCENT_RETAIN_K,
            ),
        ):
            window = self.driver.rail_trackers[rail_ml].window()
            self.assertIsNotNone(window, f"{name}: railed window vanished?")
            t0, t1 = window
            v_signed = _steady_velocity(self.driver.depth_samples, t0, t1)
            self.assertFalse(math.isnan(v_signed), f"{name}: no depth slopes")
            v = sign * v_signed  # leg speed, positive in its own direction
            self.assertGreaterEqual(
                v,
                band[0],
                f"{name} steady speed {v:.4f} m/s below lake envelope {band}",
            )
            self.assertLessEqual(
                v,
                band[1],
                f"{name} steady speed {v:.4f} m/s above lake envelope {band}",
            )

            v_curve = _curve_v(rail_m3 - LAKE_FIT_NEUTRAL_VOLUME_M3, retain_k)
            rel = abs(v - v_curve) / v_curve
            self.assertLessEqual(
                rel,
                curve_tol,
                f"{name}: measured {v:.4f} m/s deviates {rel:.0%} from the "
                f"fitted lake curve prediction {v_curve:.4f} m/s "
                f"(tolerance {curve_tol:.0%})",
            )


@launch_testing.post_shutdown_test()
@pytest.mark.sim
class LakeVelocityPostShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        # SIGTERM/SIGINT are the normal ros2/Gazebo shutdown paths.
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )

    def test_reap_lingering_gz_servers(self):
        """Reap gz-sim children launch_testing missed; best-effort."""
        reap_lingering_gz()
