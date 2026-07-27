"""Physical specifications of the Nautilus glider.

Single source of truth for hardware constants — pump, bladder, motors,
ACU mechanics. Both controllers (`py_pkg`) and the sim HAL
(`nautilus_hal`) import from here so a value change propagates to both.

Environmental constants (gravity, water density, atmospheric pressure)
live in `physics.py`; controller tuning lives in the scenario schema
(`scenarios/spec/control.py`), reaching the nodes as ROS parameters.
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
# <max_volume> on the glider_nautilus SDF (0.0025 m^3 = 2.5 L). The bang-bang
# depth controller never scales by it — it commands RPM directly — so this is
# now consumed by the sim plant (rig.plant.bladder_nominal_m3) and by the
# buoyancy derivation.
#
# Operating-range clamps (the bladder min/max actually exposed to the control
# loop, with whatever headroom we want) are tunable from the scenario YAML
# under rig.plant.bladder_min_m3 / bladder_max_m3 — same pattern as the ACU
# pitch axis output_limits.
BLADDER_VOLUME_M3 = 0.0025

# ---------------------------------------------------------------------------
# BCU motor (Maxon EPOS4)
# ---------------------------------------------------------------------------

BCU_MOTOR_MAX_RPM = 3000
BCU_MOTOR_MIN_RPM = 0

# Gauge pressure above which the pump can no longer push oil out into the
# bladder against the surrounding water (inflating only gets harder the
# deeper we are). When we are deeper than this AND the controller wants to
# descend further, we skip the pump entirely by opening valve 1.

BCU_DEEP_THRESHOLD_PA = 301_534.5  # = 30 m * 1025 kg/m^3 * 9.806 m/s^2

# BCU valve wire bitmask. The STM/CAN PDO carries valve state as two bits..
BCU_MOTOR_VALVE_MASK = 0b01  # bit0 -- pump flow path ("valve 2" in the UI)
BCU_FREE_VALVE_MASK = 0b10  # bit1 -- passive free/bypass vent ("valve 1")

# ---------------------------------------------------------------------------
# ACU mechanics (DEPRECATED / NOT IMPLEMENTED IN SIM)
# ---------------------------------------------------------------------------

# Pitch — acu_tilt_joint, prismatic, axis = +x in body frame.
# SDF limit: lower=0, upper=-0.1195 (m) → travel = 0.1195 m.
ACU_PITCH_MASS_KG = 1.83
ACU_PITCH_MAX_TRAVEL_M = 0.1195
ACU_PITCH_MAX_VELOCITY_M_S = 0.011
ACU_PITCH_MAX_EFFORT_N = 10.0

# ACU_PITCH wire format: Int16 millimetres. The specs above are in metres,
# so every producer converts on the way to the topic. Both the producer
# (control/acu_node.py) and the HAL bridge read this constant.
ACU_PITCH_MM_PER_M = 1000

# Roll — acu_roll_joint, revolute, axis = +x in body frame.
# SDF limit: lower=-0.5236, upper=0.5236 (rad) = ±30°.
ACU_ROLL_MASS_KG = 4.069
ACU_ROLL_MAX_ANGLE_RAD = 0.5236
ACU_ROLL_MAX_ANGLE_DEG = 30.0
ACU_ROLL_MAX_VELOCITY_RAD_S = 0.5
ACU_ROLL_MAX_EFFORT_NM = 10.0

# ACU_ROLL wire format: Int16 centidegrees. ±30° -> ±3000 on the topic,
# 0.01° per step. Native rad would collapse the range to {-1, 0, +1}.
# Both the producer (control/acu_node.py) and the HAL bridge
# (nautilus_hal/acu_sim_bridge.py) read this constant.
ACU_ROLL_CDEG_PER_DEG = 100

# ---------------------------------------------------------------------------
# STM32 UART wire scales
# ---------------------------------------------------------------------------
# Per-LSB resolution of the housekeeping telemetry the STM streams up over
# UART. Fixed by the firmware's variable encoding -- if the STM-side scale
# changes, this constant moves in lockstep. Consumed by stm_com_node when
# converting raw words into the absolute-Pa / °C / bitmask contracts of
# the matching uuv_ros_core topics.

# Pressures (var_id 0x2400 ext, 0x2401 tank, 0x2402 int) are uint16 mbar.
STM_PRESSURE_LSB_PA = 100

# Temperatures (var_id 0x2410 ext, 0x2411 int) are int16 centi-°C.
STM_TEMPERATURE_LSB_C = 0.01

# Physical polarity of the BCU pump motor as wired on the hardware. The whole
# control stack speaks one convention -- positive BCU_RPM inflates the bladder
# (vehicle rises), negative deflates (sinks) -- and the simulator honors it
# directly. The bench hardware's motor is wired the opposite way, so this flip
# lives ONLY at the STM/CAN wire boundary.
STM_BCU_RPM_SIGN = -1

# ---------------------------------------------------------------------------
# STM32 IMU (accel + gyro)
# ---------------------------------------------------------------------------
# The STM streams six int16 words per IMU sample -- accel x/y/z on var ids
# 0x2430..0x2432 and gyro x/y/z on 0x2433..0x2435 -- as raw sensor counts (the
# datasheet's "Accel_X_int16" / gyro equivalent). The per-count scales below are
# fixed by how the IMU is configured in firmware.
#
# Accelerometer, ±6 g full scale (ACC_RANGE=1):
#   Accel_mg = counts / 32768 * 1000 * 2**(ACC_RANGE+1) * 1.5
#            = counts / 32768 * 6000           (ACC_RANGE=1 -> 2**2 * 1.5 = 6)
STM_ACCEL_FULLSCALE_MG = 6000.0
STM_ACCEL_MG_PER_LSB = STM_ACCEL_FULLSCALE_MG / 32768.0

# Gyroscope, ±2000 °/s full scale:
#   Gyro_dps = counts / 32768 * 2000
STM_GYRO_FULLSCALE_DPS = 2000.0
STM_GYRO_DPS_PER_LSB = STM_GYRO_FULLSCALE_DPS / 32768.0

# How the IMU is mounted: sensor axes -> the NED body frame the rest of the
# stack:

#   accel:  +ax = right,   +ay = forward,  +az = up   (≈+1 g at rest)
#   gyro:   +gx = pitch-up, +gy = roll-right, +gz = yaw-left

# giving, in the NED body frame (forward / right / down),
#   (a_x, a_y, a_z)_body = (+ay, +ax, -az)_sensor   (down = -up, so a_z ≈ -1 g level)
#   (w_x, w_y, w_z)_body = (+gy, +gx, -gz)_sensor   (NED +pitch about +y is nose-up)
#
IMU_ACCEL_AXIS_MAP = (("y", +1.0), ("x", +1.0), ("z", -1.0))
IMU_GYRO_AXIS_MAP = (("y", +1.0), ("x", +1.0), ("z", -1.0))
