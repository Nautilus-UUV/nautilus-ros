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

# ---------------------------------------------------------------------------
# Bladder (buoyancy engine)
# ---------------------------------------------------------------------------

# Full mechanical bladder capacity. Matches the Gazebo BuoyancyEngine plugin's
# <max_volume> on the glider_nautilus SDF (0.0025 m^3 = 2.5 L) and is used by
# the depth controller as the scaling denominator in q_to_rpm: the PID's
# normalized output q (1/s) is multiplied by this volume to get a physical
# flow rate (m^3/s) before inverting through the pump's volume-per-rev.
#
# Operating-range clamps (the bladder min/max actually exposed to the control
# loop, with whatever headroom we want) are tunable from the scenario YAML
# under rig.plant.bladder_min_m3 / bladder_max_m3 — same pattern as the ACU
# pitch axis output_limits.
BLADDER_VOLUME_M3 = 0.0025

# ---------------------------------------------------------------------------
# BCU motor (Maxon EPOS4)
# ---------------------------------------------------------------------------

BCU_MOTOR_MAX_RPM = 4000
BCU_MOTOR_MIN_RPM = 0

# Gauge pressure above which the pump can no longer push oil out into the
# bladder against the surrounding water (inflating only gets harder the
# deeper we are). When we are deeper than this AND the controller wants to
# descend further, we skip the pump entirely — open valve 2 and let the high
# external pressure squeeze oil out of the bladder back into the tank for us,
# free of pump energy. At shallower depths, the pump can still drive flow in
# either direction through valve 1, so no special-casing is needed.

BCU_DEEP_THRESHOLD_PA = 301_534.5  # = 30 m * 1025 kg/m^3 * 9.806 m/s^2

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
ACU_PITCH_MAX_VELOCITY_M_S = 0.011
ACU_PITCH_MAX_EFFORT_N = 10.0

# Roll — acu_roll_joint, revolute, axis = +x in body frame.
# SDF limit: lower=-0.5236, upper=0.5236 (rad) = ±30°.
ACU_ROLL_MASS_KG = 4.069
ACU_ROLL_MAX_ANGLE_RAD = 0.5236
ACU_ROLL_MAX_ANGLE_DEG = 30.0
ACU_ROLL_MAX_VELOCITY_RAD_S = 0.5
ACU_ROLL_MAX_EFFORT_NM = 10.0

# ACU_ROLL wire format: Int16 centidegrees. ±30° -> ±3000 on the topic,
# 0.01° per step. Native rad would collapse the range to {-1, 0, +1}.
# Both the producer (pid/acu_node.py) and the HAL bridge
# (nautilus_hal/acu_sim_bridge.py) read this constant.
ACU_ROLL_CDEG_PER_DEG = 100
