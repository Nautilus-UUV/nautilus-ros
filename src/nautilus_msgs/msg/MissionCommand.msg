# Operator-supplied mission selector for /path.
#
# mission_id resolves through py_pkg.path.missions.factory.MissionId.
# The remaining fields are mission-specific; missions ignore fields
# they don't consume. ROS msg defaults are zero, so a partially
# populated message degrades to the zero values for unused fields.

uint8 mission_id

# TRIM_AND_NEUTRAL_BUOYANCY: hold-depth setpoint (gauge Pa).
float32 target_pressure_pa

# SAWTOOTH: glide pitch magnitude in radians (alternates sign per leg).
float32 angle_rad

# SAWTOOTH: number of resurface events before the mission terminates.
uint16 n_resurfaces
