"""Tier 3 acceptance test: pump/tank transients match the 2026-06-24 lake fit.

This is the transient-regime counterpart of the steady-state velocity gate
(test_sawtooth_lake_velocity_sim): it starts the vehicle in the real
pre-dive state — floating awash with a full bladder, tank at the empty
rail — replays dive 4's actual deflate command timeline on BCU_RPM, and
asserts the responses the anomaly model sees (tank pressure, external
pressure) land inside the lake-measured bands from
pump_transient_targets.csv.

Hard asserts (fit acceptance PASSED for these):
  - stable awash float before the first command
  - pump spin-up visible on the feedback echo: motion onset inside the
    fitted dead-time band, ramp duration inside the fitted slew band
  - tank rise +10 s inside the measured band
  - t_submerge (0.5 m crossing) and ddepth_30s inside the measured bands

Warn-only (documented residuals, printed not asserted):
  - tank rise +20/+30 s: Stage B of the fit FAILED its 2 kPa gate — the
    adopted gas_free map under-predicts the mid-window rise by ~15%
    (missing pump-slip physics, see pump_transient_findings.md).
  - ddepth_60s: the 1-DOF Stage-C model itself sits ~3% below the band
    floor (volume-scale tension, documented there).

Composed lean (bridges + Gazebo + rendered robot, no controller) so the
test owns BCU_RPM exclusively. Marker-gated ``@pytest.mark.sim``.
``SIM_GUI=1`` shows the Gazebo GUI.
"""

import csv
import os
import time
import unittest

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
from py_pkg.physics import gauge_pressure_pa
from py_pkg.robot_specs import BCU_MOTOR_VALVE_MASK
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Int16, Int32, UInt8

from ._sim_helpers import (
    SIM_PA_PER_M,
    reap_lingering_gz,
    sim_gui_enabled,
    spin_for,
    spin_until,
)

_SCENARIO = os.path.join(
    os.path.dirname(__file__), "scenarios", "test_pump_transient.yaml"
)
_TARGETS_CSV = os.path.join(
    os.path.dirname(__file__), "data", "pump_transient_targets.csv"
)

# Float phase: give the spawn a short settle, then require an awash hold.
SETTLE_S = 20.0
FLOAT_DEPTH_BAND_M = (0.0, 0.4)  # awash equilibrium sits at ~0.12 m
FLOAT_DRIFT_MAX_MPS = 0.01

# Tank sensor noise on the synthesized channel (lake-fitted defaults):
# 600 Pa quantization + 353 Pa sigma. Rises are measured on ~3 s window
# means, so grant one quantization step of margin on the hard band.
TANK_MARGIN_PA = 600.0

# Feedback-echo timing tolerance: the echo publishes at 10 Hz and the
# driver samples asynchronously.
FB_TIMING_MARGIN_S = 0.35

CMD_RATE_HZ = 5.0
SUBMERGE_DEPTH_M = 0.5

# Feedback-echo rpm the spin-up must reach; the ramp band is the time to
# get there at the fitted slew limits.
AT_SPEED_RPM = 2900.0


def _load_targets() -> tuple[dict[str, tuple[float, float, float]], list[dict]]:
    """TARGET rows as name -> (value, lo, hi); TIMELINE rows as a dict."""
    targets: dict[str, tuple[float, float, float]] = {}
    timeline: dict[str, float] = {}
    with open(_TARGETS_CSV) as f:
        rows = csv.DictReader(r for r in f if not r.startswith("#"))
        for row in rows:
            if row["row_type"] == "TARGET":
                targets[row["name"]] = (
                    float(row["value"]),
                    float(row["lo"]),
                    float(row["hi"]),
                )
            elif row["row_type"] == "TIMELINE":
                timeline[row["name"]] = float(row["value"])
    cmds = [
        {k: timeline[f"dive4_cmd{i}_{k}"] for k in ("t_rel_s", "rpm", "duration_s")}
        for i in (1, 2)
    ]
    return targets, cmds


TARGETS, DIVE4_CMDS = _load_targets()


@pytest.mark.launch_test
@pytest.mark.sim
@launch_testing.markers.keep_alive
def generate_test_description():
    # Reap here, not in setUpClass — see test_bcu_sim for the rationale.
    reap_lingering_gz()

    # The scenario carries a hydrodynamics block (spawn volume), so the
    # robot must spawn the rendered SDF, same pattern as
    # test_hydrodynamics_sampling_sim.
    from nautilus_hal.render_sdf import description_file_for_scenario

    rendered_path = description_file_for_scenario(_SCENARIO)

    bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    FindPackageShare("nautilus_hal").find("nautilus_hal"),
                    "launch",
                    "bridge.launch.py",
                )
            ]
        ),
        launch_arguments={"scenario": _SCENARIO}.items(),
    )

    gui_enabled = sim_gui_enabled()
    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    FindPackageShare("dave_demos").find("dave_demos"),
                    "launch",
                    "dave_robot.launch.py",
                )
            ]
        ),
        launch_arguments={
            # Awash float equilibrium for the full-bladder spawn (~1.5 mm
            # freeboard); validated stable in the surfaced-float check.
            "z": "-0.115",
            "roll": "3.141592653589793",
            "yaw": "1.5707963267948966",
            "namespace": "glider_nautilus",
            "world_name": "dave_ocean_waves",
            "description_file": rendered_path,
            "paused": "false",
            "gui": "true",
            "headless": "false" if gui_enabled else "true",
        }.items(),
    )

    return (
        LaunchDescription(
            [
                bridge_launch,
                robot_launch,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {},
    )


class _PumpTransientDriver(Node):
    """Replays the dive-4 deflate timeline; captures tank/depth/feedback."""

    def __init__(self):
        super().__init__("pump_transient_test_driver")
        self.depth_samples: list[tuple[float, float]] = []  # (t, m)
        self.tank_samples: list[tuple[float, float]] = []  # (t, Pa)
        self.fb_samples: list[tuple[float, int]] = []  # (t, rpm)
        self.imu_msg_count = 0
        self.replay_t0: float | None = None

        self.rpm_pub = create_publisher_for_topic(self, UUVTopics.BCU_RPM)
        self.valves_pub = create_publisher_for_topic(self, UUVTopics.BCU_VALVES)
        create_subscription_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE, self._on_pressure
        )
        create_subscription_for_topic(self, UUVTopics.BCU_PRESSURE, self._on_tank)
        create_subscription_for_topic(self, UUVTopics.BCU_FEEDBACK_RPM, self._on_fb)
        create_subscription_for_topic(self, UUVTopics.IMU, self._on_imu)
        self.create_timer(1.0 / CMD_RATE_HZ, self._tick)

    def _on_imu(self, _msg: Imu) -> None:
        self.imu_msg_count += 1

    def _on_pressure(self, msg: Int32) -> None:
        depth_m = gauge_pressure_pa(float(msg.data)) / SIM_PA_PER_M
        self.depth_samples.append((time.monotonic(), depth_m))

    def _on_tank(self, msg: Int32) -> None:
        self.tank_samples.append((time.monotonic(), float(msg.data)))

    def _on_fb(self, msg: Int16) -> None:
        self.fb_samples.append((time.monotonic(), int(msg.data)))

    def start_replay(self) -> None:
        self.replay_t0 = time.monotonic()

    def _commanded_rpm(self, t_rel: float) -> float:
        for cmd in DIVE4_CMDS:
            if cmd["t_rel_s"] <= t_rel < cmd["t_rel_s"] + cmd["duration_s"]:
                return cmd["rpm"]
        return 0.0

    def _tick(self) -> None:
        # Publish 0 during settle too: the bridge holds the last received
        # command, so the replay must start from an explicitly commanded 0.
        t_rel = -1.0 if self.replay_t0 is None else time.monotonic() - self.replay_t0
        rpm = self._commanded_rpm(t_rel) if t_rel >= 0.0 else 0.0
        self.rpm_pub.publish(Int16(data=int(rpm)))
        # Hold valve 2 (motor way) open for the whole replay — the lake fit
        # was made with transfer following the shaft transient, so the
        # bridge's valve gate must never truncate it mid-timeline.
        self.valves_pub.publish(UInt8(data=BCU_MOTOR_VALVE_MASK))


def _mean_near(samples: list[tuple[float, float]], t: float, half_w: float) -> float:
    vals = [v for ts, v in samples if abs(ts - t) <= half_w]
    assert vals, f"no samples within {half_w}s of t={t}"
    return sum(vals) / len(vals)


@pytest.mark.sim
class PumpTransientTest(unittest.TestCase):
    """Behavior: entry-transient responses land inside the lake bands."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.driver = _PumpTransientDriver()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.driver)

    def tearDown(self):
        self.executor.remove_node(self.driver)
        self.driver.destroy_node()
        self.executor.shutdown()

    def _warn(self, name: str, measured: float, lo: float, hi: float) -> None:
        verdict = "in band" if lo <= measured <= hi else "OUT OF BAND"
        print(
            f"[warn-level] {name}: sim {measured:.1f} vs lake [{lo:.1f}, "
            f"{hi:.1f}] -> {verdict} (documented residual, not asserted)"
        )

    def test_entry_transients_match_lake_dive4(self):
        drv = self.driver

        sim_ready = spin_until(
            self.executor, lambda: drv.imu_msg_count >= 1, timeout_s=60.0
        )
        self.assertTrue(sim_ready, "IMU never arrived — sim not up?")

        # ---- Float phase -------------------------------------------------
        spin_for(self.executor, SETTLE_S)
        float_win = [
            (t, d) for t, d in drv.depth_samples if t >= time.monotonic() - 10.0
        ]
        self.assertGreaterEqual(len(float_win), 5, "no depth telemetry at float")
        depths = sorted(d for _, d in float_win)
        median_depth = depths[len(depths) // 2]
        self.assertTrue(
            FLOAT_DEPTH_BAND_M[0] <= median_depth <= FLOAT_DEPTH_BAND_M[1],
            f"not floating awash: median depth {median_depth:.3f} m "
            f"outside {FLOAT_DEPTH_BAND_M}",
        )
        drift = abs(float_win[-1][1] - float_win[0][1]) / max(
            float_win[-1][0] - float_win[0][0], 1e-6
        )
        self.assertLess(
            drift, FLOAT_DRIFT_MAX_MPS, f"float drifting at {drift:.4f} m/s"
        )

        # ---- Replay dive 4's deflate timeline ---------------------------
        drv.start_replay()
        t0 = drv.replay_t0
        last_cmd_end = max(c["t_rel_s"] + c["duration_s"] for c in DIVE4_CMDS)
        # Run until well past the submergence crossing + 60 s depth target.
        horizon_s = last_cmd_end + 80.0
        spin_for(self.executor, horizon_s)

        # ---- Pump spin-up on the feedback echo --------------------------
        _, delay_lo, delay_hi = TARGETS["pump_delay_s"]
        _, slew_lo, slew_hi = TARGETS["pump_slew_rpm_per_s"]
        cmd1_t = t0 + DIVE4_CMDS[0]["t_rel_s"]
        fb = [(t - cmd1_t, rpm) for t, rpm in drv.fb_samples if t >= cmd1_t - 5.0]
        onset = next((t for t, rpm in fb if t >= 0 and abs(rpm) >= 100), None)
        self.assertIsNotNone(onset, "feedback echo never moved after the command")
        self.assertTrue(
            delay_lo - FB_TIMING_MARGIN_S <= onset <= delay_hi + FB_TIMING_MARGIN_S,
            f"spin-up onset {onset:.2f}s outside dead-time band "
            f"[{delay_lo:.2f}, {delay_hi:.2f}] (+/-{FB_TIMING_MARGIN_S})",
        )
        at_speed = next(
            (t for t, rpm in fb if t >= 0 and abs(rpm) >= AT_SPEED_RPM), None
        )
        self.assertIsNotNone(at_speed, f"pump never reached {AT_SPEED_RPM:.0f} rpm")
        ramp_s = at_speed - onset
        ramp_lo, ramp_hi = AT_SPEED_RPM / slew_hi, AT_SPEED_RPM / slew_lo
        self.assertTrue(
            ramp_lo - FB_TIMING_MARGIN_S <= ramp_s <= ramp_hi + FB_TIMING_MARGIN_S,
            f"spin-up ramp {ramp_s:.2f}s outside slew band "
            f"[{ramp_lo:.2f}, {ramp_hi:.2f}] (+/-{FB_TIMING_MARGIN_S})",
        )

        # ---- Tank rise ---------------------------------------------------
        p_base = _mean_near(drv.tank_samples, cmd1_t, 1.5)
        rises = {
            dt_s: _mean_near(drv.tank_samples, cmd1_t + dt_s, 1.5) - p_base
            for dt_s in (10, 20, 30)
        }
        _, lo10, hi10 = TARGETS["tank_rise_10s_pa"]
        self.assertTrue(
            lo10 - TANK_MARGIN_PA <= rises[10] <= hi10 + TANK_MARGIN_PA,
            f"tank rise +10s {rises[10]:.0f} Pa outside "
            f"[{lo10:.0f}, {hi10:.0f}] (+/-{TANK_MARGIN_PA})",
        )
        for dt_s, name in ((20, "tank_rise_20s_pa"), (30, "tank_rise_30s_pa")):
            _, lo, hi = TARGETS[name]
            self._warn(name, rises[dt_s], lo - TANK_MARGIN_PA, hi + TANK_MARGIN_PA)

        # ---- Depth response ---------------------------------------------
        cross = next(
            (
                t - cmd1_t
                for t, d in drv.depth_samples
                if t >= cmd1_t and d >= SUBMERGE_DEPTH_M
            ),
            None,
        )
        self.assertIsNotNone(cross, "vehicle never submerged past 0.5 m")
        _, sub_lo, sub_hi = TARGETS["t_submerge_s"]
        self.assertTrue(
            sub_lo <= cross <= sub_hi,
            f"t_submerge {cross:.1f}s outside [{sub_lo:.1f}, {sub_hi:.1f}]",
        )

        t_cross = cmd1_t + cross
        d30 = _mean_near(drv.depth_samples, t_cross + 30.0, 1.5) - SUBMERGE_DEPTH_M
        _, d30_lo, d30_hi = TARGETS["ddepth_30s_m"]
        self.assertTrue(
            d30_lo <= d30 <= d30_hi,
            f"ddepth_30s {d30:.2f} m outside [{d30_lo:.2f}, {d30_hi:.2f}]",
        )
        d60 = _mean_near(drv.depth_samples, t_cross + 60.0, 1.5) - SUBMERGE_DEPTH_M
        _, d60_lo, d60_hi = TARGETS["ddepth_60s_m"]
        self._warn("ddepth_60s_m", d60, d60_lo, d60_hi)


@launch_testing.post_shutdown_test()
@pytest.mark.sim
class PumpTransientShutdownTest(unittest.TestCase):
    """Sanity-check + cleanup after launched processes are torn down."""

    def test_exit_codes(self, proc_info):
        # SIGTERM/SIGINT are the normal ros2/Gazebo shutdown paths.
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )

    def test_reap_lingering_gz_servers(self):
        """Reap gz-sim children launch_testing missed; best-effort, no assert."""
        reap_lingering_gz()
