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

Anchors (dive 4, heave_calibration_targets.csv):
  descent: 0.1245 m/s at V_b=0.866e-3   -> assert 0.1245 +/- 0.020
  ascent:  0.0677 m/s at V_b=2.465e-3   -> assert within [0.030, 0.080]

The ascent band is deliberately wide: real ascents are systematically
slower than the symmetric fitted drag curve predicts (fit residuals
+26/28%), and the sim adds extra low-speed damping from the fin
LiftDrag plugins, so the sim ascends at ~0.04 m/s where the real
vehicle did 0.068. That asymmetry is the documented fidelity floor of
the symmetric zW/zWabsW plant model. The curve-consistency assertion
(each leg's (dV, v) point vs the fitted curve: 15% descent / 25%
ascent) keeps both legs honest against the calibration itself.

Marker-gated ``@pytest.mark.sim``; opt in with ``pytest -m sim``.
``SIM_GUI=1`` shows the Gazebo GUI.
"""

import math
import os
import statistics
import time
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
from nautilus_msgs.msg import MissionCommand
from py_pkg.path.missions.factory import MissionId
from py_pkg.physics import gauge_pressure_pa
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

from ._sim_helpers import reap_lingering_gz, sim_gui_enabled, spin_for, spin_until

# --- Lake anchors (dive 4, config B) --------------------------------------
# Provenance: UG-anomaly_detection/lake_test_jun24/investigation/
# heave_calibration_targets.csv + heave_calibration_fit.json.
V_NEUTRAL_M3 = 2.097481e-3
# Drag coefficients come from the spec defaults — those ARE the adopted
# lake fit, and the SDF parity test ties them to the canonical model.sdf
# — so a re-fit moves this consistency curve automatically.
DRAG_D1 = -HydrodynamicsSpec().drag_zW
DRAG_D2 = -HydrodynamicsSpec().drag_zWabsW
# The fit's own force convention (heave_calibration_targets.py uses
# rho*g = 1000*9.8) — deliberately not physics.py's 9.806 gradient.
RHO_G = 1000.0 * 9.8

DESCENT_RAIL_M3 = 0.866e-3  # scenario bladder_min_m3
ASCENT_RAIL_M3 = 2.465e-3  # scenario bladder_max_m3
DESCENT_RAIL_ML = round(DESCENT_RAIL_M3 * 1e6)
ASCENT_RAIL_ML = round(ASCENT_RAIL_M3 * 1e6)

DESCENT_V_BAND = (0.1045, 0.1445)  # 0.1245 +/- 0.020 (real steady leg)
ASCENT_V_BAND = (0.030, 0.080)  # wide: asymmetry fidelity floor, see docstring
DESCENT_CURVE_TOL = 0.15
ASCENT_CURVE_TOL = 0.25

# Depth from the sim's sea-pressure plugin gradient (9.80638 kPa/m).
SIM_PA_PER_M = 9806.38

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


def _curve_v(dv_m3: float) -> float:
    """Terminal speed the fitted lake curve predicts for |dV| offset."""
    force = RHO_G * abs(dv_m3)
    return (-DRAG_D1 + math.sqrt(DRAG_D1**2 + 4.0 * DRAG_D2 * force)) / (2.0 * DRAG_D2)


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
    """Kicks SAWTOOTH and captures depth + bladder-volume telemetry."""

    def __init__(self):
        super().__init__("lake_velocity_test_driver")
        self.depth_samples: list[tuple[float, float]] = []  # (t, depth m)
        self.rail_trackers = {
            ml: _RailedWindowTracker(ml) for ml in (DESCENT_RAIL_ML, ASCENT_RAIL_ML)
        }
        self.imu_msg_count = 0

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
        depth_m = gauge_pressure_pa(float(msg.data)) / SIM_PA_PER_M
        self.depth_samples.append((time.monotonic(), depth_m))

    def _on_volume(self, msg: Int32) -> None:
        t = time.monotonic()
        for tracker in self.rail_trackers.values():
            tracker.add(t, int(msg.data))

    def publish_mission(self) -> None:
        cmd = MissionCommand()
        cmd.mission_id = int(MissionId.SAWTOOTH)
        cmd.target_pressure_pa = float(TARGET_PRESSURE_PA)
        cmd.angle_rad = float(PITCH_RAD)
        cmd.n_resurfaces = int(N_RESURFACES)
        self.path_pub.publish(cmd)

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
        startup_timeout_s = 60.0
        descent_budget_s = 300.0
        ascent_budget_s = 480.0

        sim_ready = spin_until(
            self.executor,
            lambda: self.driver.imu_msg_count >= 1,
            timeout_s=startup_timeout_s,
        )
        self.assertTrue(sim_ready, "IMU never arrived — sim not up?")
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
        for name, rail_ml, rail_m3, sign, band, curve_tol in (
            (
                "descent",
                DESCENT_RAIL_ML,
                DESCENT_RAIL_M3,
                +1.0,
                DESCENT_V_BAND,
                DESCENT_CURVE_TOL,
            ),
            (
                "ascent",
                ASCENT_RAIL_ML,
                ASCENT_RAIL_M3,
                -1.0,
                ASCENT_V_BAND,
                ASCENT_CURVE_TOL,
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

            v_curve = _curve_v(rail_m3 - V_NEUTRAL_M3)
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
