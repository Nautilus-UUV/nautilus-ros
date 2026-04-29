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

# TODO: confirm exact value on the bench. Currently used as a saturation
# cap in trim_and_buoyancy_test.py (300 mm). The pitch axis controller
# uses the tighter ACU_PITCH_OUTPUT_LIMIT_M below as its operational
# stroke clamp.
ACU_PITCH_MAX_TRAVEL_M = 0.3

# Mass-shifter saturation used by the pitch axis controller. The previous
# code used ±0.07 m as a hard cap (and ±0.065 m as a bang-bang setpoint);
# kept here as the single source of truth for the clamped-P pitch output.
# TODO: confirm against bench measurement.
ACU_PITCH_OUTPUT_LIMIT_M = 0.07

# TODO: confirm exact value on the bench. Used as the roll-axis output
# clamp in pid/acu_roll_config.py.
ACU_ROLL_MAX_ANGLE_DEG = 25.0

# Motor-step conversion factors. The ACU motor topics (ACU_PITCH_STEPS /
# ACU_ROLL_STEPS) are Int32 step counts; these constants convert from
# the physical UUV-frame quantity (metres for the pitch mass-shifter,
# degrees for the roll ring) to motor steps.
# TODO: replace with measured values once the mechanics are calibrated.
ACU_PITCH_STEPS_PER_M = 10000.0
ACU_ROLL_STEPS_PER_DEG = 100.0
