from py_pkg.physics import (
    depth_to_pressure_pa,
    gauge_pressure_pa,
)
from py_pkg.robot_specs import (
    BCU_MOTOR_MAX_RPM,
    BCU_MOTOR_MIN_RPM,
    BLADDER_VOLUME_M3,
)

init_control = {}

# Set safe default operating depths (20 meters to 70 meters).
# The system uses these high/low pressure boundaries until it receives
# a specific target depth command from the main controller.
init_control["low_pressure_pa"] = gauge_pressure_pa(depth_to_pressure_pa(20.0))
init_control["high_pressure_pa"] = gauge_pressure_pa(depth_to_pressure_pa(70.0))
init_control["frequency"] = 10

# PID Controller Configuration: Calculates how fast to pump fluid into/out of the
# buoyancy bladder based on the current pressure (depth) error.
#
# Kp (Proportional): Drives the main pump speed. It is tuned so that a 2.5-meter
# error runs the pump at its maximum allowed rate (~4000 RPM).
#
# Kd (Derivative) & Filter: Slows the pump down as we get close to the target to
# prevent overshooting. Because pressure sensors are noisy (jumping between whole
# numbers), taking the derivative of that noise causes erratic motor commands.
# The `derivative_filter` smooths this out.
#
# Ki (Integral) & Limits: Handles tiny residual buoyancy issues (like hovering).
# These values are kept intentionally tiny. Why? During a long, deep dive, the
# sub spends a long time far away from its target. If Ki is too large, it "remembers"
# all that error, keeps the pump running too long, and causes the sub to blast past
# the target depth. Keeping limits small lets it fine-tune hovering without ruining
# long dives.
init_control["pid_pressure"] = {
    "kp": 4.0e-7,
    "ki": 1.0e-8,
    "kd": 1.5e-5,
    "integral_limits": (-0.0025, 0.0025),
    "output_limits": (-0.010, 0.010),
    "derivative_filter": 0.3,
}

init_buoyancy_engine = {}
init_buoyancy_engine["tank_volume"] = BLADDER_VOLUME_M3
init_buoyancy_engine["initial_proportion_full"] = 1.0

init_motor = {}

# Motor Deadband: The smallest RPM command we are allowed to send.
# We set this to 0 here to allow the math to calculate tiny, precise adjustments
# needed for perfect hovering. If we ignored small adjustments here, the pump
# would shut off prematurely as we approached the target, and the sub's momentum
# would carry it past the desired depth.
#
# Note: The physical hardware does have a minimum limit (e.g., the pump physically
# cannot spin slower than 1000 RPM). That hardware limitation is enforced safely
# in a downstream hardware controller, not in this control math.
init_motor["min_rpm"] = BCU_MOTOR_MIN_RPM
init_motor["max_rpm"] = BCU_MOTOR_MAX_RPM
