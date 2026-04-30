"""Physical specifications of the Nautilus glider.

Single source of truth for hardware constants — pump, bladder, motors,
ACU mechanics. Both controllers (`py_pkg`) and the sim HAL
(`nautilus_hal`) import from here so a value change propagates to both.

Environmental constants (gravity, water density, atmospheric pressure)
live in `physics.py`; controller tuning (PID gains, deadbands) lives in
`pid/depth_config.py` and `pid/acu_{pitch,roll}_config.py`.
"""

# ---------------------------------------------------------------------------
# BCU pump
# ---------------------------------------------------------------------------

# Volumetric displacement per revolution (m^3 / rev).
VOLUME_PER_REV_M3 = 0.32e-6

# Pump volumetric efficiency between 1000 and 3000 RPM.
PUMP_EFFICIENCY = 0.93

# ---------------------------------------------------------------------------
# Bladder (buoyancy engine)
# ---------------------------------------------------------------------------

# Nominal bladder volume used by the depth controller's flow-rate model.
BLADDER_VOLUME_M3 = 0.002275

# Mechanical stroke limits of the bladder.
BLADDER_MAX_VOLUME_M3 = 0.0025
BLADDER_MIN_VOLUME_M3 = 0.0010

# ---------------------------------------------------------------------------
# BCU motor (Maxon EPOS4)
# ---------------------------------------------------------------------------

BCU_MOTOR_MAX_RPM = 4000
BCU_MOTOR_MIN_RPM = 1000

# ---------------------------------------------------------------------------
# ACU mechanics
# ---------------------------------------------------------------------------
# Values mirror the Gazebo glider_nautilus model
# (src/dave/models/dave_robot_models/description/glider_nautilus/model.sdf,
# ACU section). Duplicated here so the control stack can import them
# without parsing the SDF; if you change one, change the other.

# Pitch — acu_tilt_joint, prismatic, axis = +x in body frame.
# SDF limit: lower=0, upper=-0.1195 (m) → travel = 0.1195 m.
ACU_PITCH_MASS_KG = 1.83
ACU_PITCH_MAX_TRAVEL_M = 0.1195
ACU_PITCH_MAX_VELOCITY_M_S = 0.5
ACU_PITCH_MAX_EFFORT_N = 10.0

# Soft saturation used by the pitch axis controller (clamped-P output).
# Must satisfy ACU_PITCH_OUTPUT_LIMIT_M <= ACU_PITCH_MAX_TRAVEL_M.
ACU_PITCH_OUTPUT_LIMIT_M = 0.07

# Roll — acu_roll_joint, revolute, axis = +x in body frame.
# SDF limit: lower=-0.5236, upper=0.5236 (rad) = ±30°.
ACU_ROLL_MASS_KG = 4.069
ACU_ROLL_MAX_ANGLE_RAD = 0.5236
ACU_ROLL_MAX_ANGLE_DEG = 30.0
ACU_ROLL_MAX_VELOCITY_RAD_S = 0.5
ACU_ROLL_MAX_EFFORT_NM = 10.0

# Motor-step conversion factors. The ACU motor topics (ACU_PITCH_STEPS /
# ACU_ROLL_STEPS) are Int32 step counts; these constants convert from
# the physical UUV-frame quantity (metres for the pitch mass-shifter,
# degrees for the roll ring) to motor steps.
# TODO: replace with measured values once the mechanics are calibrated.
ACU_PITCH_STEPS_PER_M = 10000.0
ACU_ROLL_STEPS_PER_DEG = 100.0
