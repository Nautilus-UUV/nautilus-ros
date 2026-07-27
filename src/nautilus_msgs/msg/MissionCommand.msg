# Operator-supplied mission selector for /path.
#
# mission_id resolves through py_pkg.path.missions.factory.MissionId.
# The remaining fields are mission-specific; missions ignore fields
# they don't consume. ROS msg defaults are zero, so a partially
# populated message degrades to the zero values for unused fields.

uint8 mission_id

# TRIM_AND_NEUTRAL_BUOYANCY: hold-depth setpoint (gauge Pa).
# SAWTOOTH: the deep extremum of the glide (gauge Pa).
float32 target_pressure_pa

# SAWTOOTH: the shallow extremum of the glide (gauge Pa). 0 => the
# vehicle climbs to the surface between dives (the legacy profile).
float32 shallow_pressure_pa

# SAWTOOTH: glide pitch magnitude in radians (alternates sign per leg).
float32 angle_rad

# SAWTOOTH: number of dives to the deep extremum before the mission ends
# with a final ascent to the surface. With shallow_pressure_pa = 0 every
# turn IS a resurface, which is the legacy meaning of this field -- hence
# the name, kept so launch args, scenarios and sweeps don't have to move.
uint16 n_resurfaces

# STAIRCASE: number of ladder steps between surface and target_pressure_pa
# (0 is treated as 1). Ignored by every other mission.
uint16 n_steps
