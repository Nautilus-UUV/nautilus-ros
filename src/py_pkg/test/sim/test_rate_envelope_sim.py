"""Tier 3 acceptance test: the sim's d_ext_pressure rate envelope covers
the real nominal dives' entry and ascent rates (A2b's Tier-3 proxy).

The ML pipeline's 5th channel, ``d_ext_pressure``, is the 5 s
box-smoothed per-second rate of external pressure. The v3 campaign
failed acceptance sub-check A2b because real nominal dives 2/4 enter
descents at 0.20-0.24 m/s and ascend at up to 0.113 m/s while the sim
never exceeded 0.131 / 0.057. Two HeaveAugmentPlugin features close the
gap (entry-momentum transient + ascent drag relief); this test is their
end-to-end gate — and the deterministic tripwire for a sign inversion
in the relief force (an inverted relief SLOWS ascent below the band
floor instead of lifting it to ~0.065).

One SAWTOOTH cycle to 15 m (mirroring dive 4's 14.68 m) at the
canonical operating clamps, entry momentum ON at the recalibrated
nominal (0.205). The rate series is measured exactly like the dataset
builder: depth from external pressure, mean-binned to 1 Hz,
``np.gradient`` + 5-tap box convolution — a deliberate reimplementation
of ``UG-anomaly_detection/src/data/windowing.py::_smoothed_rate`` (that
repo is not importable from py_pkg), in metres, positive down.

Samples are timestamped with SIM time (the ground-truth odometry's own
header stamp, run_watchdog's plausibility clock): the full stack drags
this host to RTF ~0.5-0.7, and wall-clock rates under-read sim rates by
exactly that factor. The recorded campaign bags don't have this problem
in the same way — their 1 Hz axis is rosbag receive time and the label
stream shares it — but a rate GATE must measure true sim velocity.

Asserts (rates are 5 s box-smoothed):
  * no spawn-time firing: >= 20 s from mission start to crossing 0.8 m
    (the real pump-out float; the trigger's depth/hold gates must not
    fire while awash);
  * entry-descent peak within [0.18, 0.27] m/s in the first 60 s below
    0.8 m (nominal target 0.205; dive 2 measured 0.225, dive 4 0.184);
  * the transient actually decays: smoothed descent rate <= 0.16 m/s
    from 60 s after the entry peak until the leg's final approach
    (steady descent at the canonical clamps is ~0.11);
  * ascent steady median within [0.048, 0.088] m/s (dive-4 anchor
    0.0677 +/- 0.020; k-scaled curve predicts ~0.065 at these clamps);
  * ascent smoothed peak <= 0.12 m/s (real nominal max 0.113);
  * whole-run |rate| <= 0.30 m/s sanity.

Marker-gated ``@pytest.mark.sim``; opt in with ``pytest -m sim``.
``SIM_GUI=1`` shows the Gazebo GUI.
"""

import math
import os
import unittest

import launch_testing
import launch_testing.actions
import launch_testing.asserts
import launch_testing.markers
import numpy as np
import pytest
import rclpy
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from py_pkg.path.missions.factory import MissionId
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

# --- Bands (m/s, 5 s box-smoothed) -----------------------------------------
ENTRY_PEAK_BAND = (0.18, 0.27)  # nominal 0.205 (d2 0.225 / d4 0.184)
ENTRY_WINDOW_S = 60.0  # peak must occur this soon after crossing DIVE_THR_M
DECAY_LAG_S = 60.0  # rate must be back under the ceiling this long after peak
DECAYED_DESCENT_MAX = 0.16  # steady ~0.11 at canonical clamps + headroom
ASCENT_STEADY_BAND = (0.048, 0.088)  # dive-4 anchor 0.0677 +/- 0.020
ASCENT_PEAK_MAX = 0.12  # real nominal ascents top out at 0.113
RATE_SANITY_MAX = 0.30
MIN_SUBMERGE_S = 20.0  # real pump-out float lasted ~47 s

DIVE_THR_M = 0.8  # in-dive threshold, mirrors real_dives.DIVE_THR_M
SMOOTH_TAPS = 5  # the d_ext_pressure box width at 1 Hz

TARGET_PRESSURE_PA = 147095.7  # 15.0 m * 9806.38 Pa/m (dive 4 peaked 14.68)
PITCH_RAD = math.radians(30.0)
N_RESURFACES = 1


def _smoothed_rate_1hz(
    samples: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """(t, depth) pairs -> (1 Hz depth series, 5 s box-smoothed rate).

    Mean-bin to 1 Hz, then np.gradient + 5-tap box (mode="same") —
    exactly `windowing._smoothed_rate` on a depth series (m/s, +down).
    The binned depth series shares the rate's index axis, so callers
    locate features (e.g. the depth peak) on the same 1 s grid.
    """
    t0 = samples[0][0]
    n_bins = int(samples[-1][0] - t0) + 1
    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for t, d in samples:
        i = min(int(t - t0), n_bins - 1)
        sums[i] += d
        counts[i] += 1
    filled = counts > 0
    depth_1hz = np.interp(
        np.arange(n_bins), np.arange(n_bins)[filled], sums[filled] / counts[filled]
    )
    rate = np.gradient(depth_1hz)
    return depth_1hz, np.convolve(rate, np.ones(SMOOTH_TAPS) / SMOOTH_TAPS, mode="same")


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
                "test_rate_envelope.yaml",
            ),
            # Surfaced start, exactly like the campaign launch args
            # (run_sweep passes z:=-0.115): the entry transient models a
            # dive FROM THE SURFACE FLOAT, and the launch default (-5)
            # would start the run already below the dive threshold.
            "z": "-0.115",
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


class _RateEnvelopeDriver(Node):
    """Kicks SAWTOOTH and captures the external-pressure depth stream."""

    def __init__(self):
        super().__init__("rate_envelope_test_driver")
        self.depth_samples: list[tuple[float, float]] = []  # (sim t, depth m)
        self.imu_msg_count = 0
        self.mission_complete = False
        self.sim_clock = SimClock(self)

        self.path_pub = create_publisher_for_topic(self, UUVTopics.PATH)
        self.command_pub = create_publisher_for_topic(self, UUVTopics.COMMAND)
        create_subscription_for_topic(self, UUVTopics.IMU, self._on_imu)
        create_subscription_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE, self._on_pressure
        )
        create_subscription_for_topic(
            self, UUVTopics.MISSION_COMPLETE, self._on_mission_complete
        )

    def _on_imu(self, _msg: Imu) -> None:
        self.imu_msg_count += 1

    def _on_pressure(self, msg: Int32) -> None:
        if self.sim_clock.now is None:
            return
        self.depth_samples.append((self.sim_clock.now, sim_depth_m(float(msg.data))))

    def _on_mission_complete(self, msg: Bool) -> None:
        self.mission_complete = self.mission_complete or bool(msg.data)

    def publish_mission(self) -> None:
        # shallow_pressure_pa left at 0.0: climb to the surface, complete.
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
class RateEnvelopeTest(unittest.TestCase):
    """Behavior: entry + ascent smoothed rates land in the real envelopes."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.driver = _RateEnvelopeDriver()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.driver)

    def tearDown(self):
        self.executor.remove_node(self.driver)
        self.driver.destroy_node()
        self.executor.shutdown()

    def test_rate_envelope_matches_real_nominal_dives(self):
        # Wall-clock TIMEOUTS only (measurements clock on sim time);
        # sized for RTF ~0.5 on a loaded host.
        startup_timeout_s = 60.0
        mission_budget_s = 1560.0

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
        t_start = self.driver.sim_clock.now

        completed = spin_until(
            self.executor,
            lambda: self.driver.mission_complete,
            timeout_s=mission_budget_s,
        )
        self.assertTrue(
            completed,
            f"SAWTOOTH never latched /mission/complete within {mission_budget_s}s",
        )

        samples = [(t, d) for t, d in self.driver.depth_samples if t >= t_start]
        self.assertGreater(len(samples), 100, "external-pressure stream too thin")

        # --- No spawn-time firing: the surface float must survive -------
        t_dive = next((t for t, d in samples if d >= DIVE_THR_M), None)
        self.assertIsNotNone(t_dive, "vehicle never crossed the dive threshold")
        self.assertGreaterEqual(
            t_dive - t_start,
            MIN_SUBMERGE_S,
            f"crossed {DIVE_THR_M} m only {t_dive - t_start:.1f}s after start — "
            "did the entry transient fire during the surface float?",
        )

        # --- Rate series, dataset-builder style --------------------------
        depth_1hz, rate = _smoothed_rate_1hz(samples)
        t0 = samples[0][0]
        i_dive = int(t_dive - t0)
        i_peak_depth = int(np.argmax(depth_1hz))
        self.assertGreater(i_peak_depth, i_dive, "no descent leg found")

        self.assertLessEqual(
            float(np.max(np.abs(rate))),
            RATE_SANITY_MAX,
            "smoothed |rate| exceeded the sanity ceiling",
        )

        # --- Entry-descent peak ------------------------------------------
        entry_end = min(i_dive + int(ENTRY_WINDOW_S), i_peak_depth)
        i_entry_peak = i_dive + int(np.argmax(rate[i_dive:entry_end]))
        entry_peak = float(rate[i_entry_peak])
        self.assertGreaterEqual(
            entry_peak,
            ENTRY_PEAK_BAND[0],
            f"entry peak {entry_peak:.3f} m/s below {ENTRY_PEAK_BAND} — "
            "entry-momentum servo undershooting its target?",
        )
        self.assertLessEqual(
            entry_peak,
            ENTRY_PEAK_BAND[1],
            f"entry peak {entry_peak:.3f} m/s above {ENTRY_PEAK_BAND}",
        )

        # --- Transient decays back toward the steady descent -------------
        decay_start = i_entry_peak + int(DECAY_LAG_S)
        decay_end = i_peak_depth - 10  # final approach eases off anyway
        if decay_start < decay_end:
            late_descent_max = float(np.max(rate[decay_start:decay_end]))
            self.assertLessEqual(
                late_descent_max,
                DECAYED_DESCENT_MAX,
                f"descent rate {late_descent_max:.3f} m/s at "
                f">{DECAY_LAG_S:.0f}s past the entry peak — transient not decaying?",
            )

        # --- Ascent leg ---------------------------------------------------
        ascent = -rate[i_peak_depth:]
        n = len(ascent)
        self.assertGreater(n, 30, "ascent leg too short to measure")
        ascent_steady = float(np.median(ascent[int(0.2 * n) : int(0.8 * n)]))
        self.assertGreaterEqual(
            ascent_steady,
            ASCENT_STEADY_BAND[0],
            f"ascent steady {ascent_steady:.3f} m/s below {ASCENT_STEADY_BAND} — "
            "relief force inverted or not applied?",
        )
        self.assertLessEqual(
            ascent_steady,
            ASCENT_STEADY_BAND[1],
            f"ascent steady {ascent_steady:.3f} m/s above {ASCENT_STEADY_BAND}",
        )
        ascent_peak = float(np.max(ascent))
        self.assertLessEqual(
            ascent_peak,
            ASCENT_PEAK_MAX,
            f"ascent peak {ascent_peak:.3f} m/s above the real "
            f"nominal ceiling {ASCENT_PEAK_MAX}",
        )


@launch_testing.post_shutdown_test()
@pytest.mark.sim
class RateEnvelopePostShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )

    def test_reap_lingering_gz_servers(self):
        """Reap gz-sim children launch_testing missed; best-effort."""
        reap_lingering_gz()
