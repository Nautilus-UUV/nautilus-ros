# Copied from divetest files: depth_control_node_config.py

from py_pkg.physics import (
    depth_to_pressure_pa,
    gauge_pressure_pa,
)
from py_pkg.robot_specs import (
    BCU_MOTOR_MAX_RPM,
    BLADDER_VOLUME_M3,
)

init_control = {}
# Z-positive-down: shallowest gauge pressure (~20 m), deepest (~70 m).
# Initial setpoint defaults to high_pressure_pa until depth_node receives
# a real POSITION_TARGET; both values are otherwise unused by the loop.
init_control["low_pressure_pa"] = gauge_pressure_pa(depth_to_pressure_pa(20.0))
init_control["high_pressure_pa"] = gauge_pressure_pa(depth_to_pressure_pa(70.0))
init_control["frequency"] = 10
# Single PID, gauge_pa -> q (bladder flow ratio, 1/s).
# kp sized so a ~25 kPa error (~2.5 m) saturates q at 0.01 (the q clamp
# corresponds to ~4000 RPM via q_to_rpm). kd is on derivative-of-measurement
# (Pa/s); the filter is essential at 10 Hz given Int32-Pa quantization
# on EXTERNAL_PRESSURE.
#
# ki + integral_limits are sized small on purpose: the test_trim_neutral_sim*
# 1.5 m step takes ~50 s of saturated drain (max pump rate vs the SDF's
# ~1 L bladder swing) before the glider crosses the target. With a larger
# ki the integral winds up to its clamp during that long descent and the
# glider overshoots ~0.8 m past target before unwinding. ki=1e-8 +
# integral_limits=±0.0025 lets the integral cancel residual buoyancy
# bias near steady state without dominating the transient.
init_control["pid_pressure"] = {
    "kp": 4.0e-7,
    "ki": 1.0e-8,
    "kd": 1.5e-5,
    "integral_limits": (-0.0025, 0.0025),
    "output_limits": (-0.010, 0.010),
    "derivative_filter": 0.3,
}
init_pos = {}
init_pos["x"] = 0.0
init_pos["y"] = 0.0
# z carries the controlled variable (gauge Pa); 0 = surface.
init_pos["z"] = 0.0
init_buoyancy_engine = {}
init_buoyancy_engine["tank_volume"] = BLADDER_VOLUME_M3
init_buoyancy_engine["initial_proportion_full"] = 1.0
init_motor = {}
# depth_node deadband: zero out RPM commands smaller than this. Set to 0
# (no deadband) so the cascaded depth loop can fine-trim near target —
# without it, motor_rpm drops below BCU_MOTOR_MIN_RPM during the settling
# phase, the wire goes to 0, the bladder freezes mid-fill, and the glider
# overshoots on momentum (see test_trim_neutral_sim* trajectory). The
# real-hardware pump's `BCU_MOTOR_MIN_RPM = 1000` reliable-rotation limit
# is enforced separately by the BCU controller node (`bcu/bcu_node.py`)
# downstream of this loop, not here.
init_motor["min_rpm"] = 0
init_motor["max_rpm"] = BCU_MOTOR_MAX_RPM
