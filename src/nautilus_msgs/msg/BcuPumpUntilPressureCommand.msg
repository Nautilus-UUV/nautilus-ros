# Debug/manual BCU pump command with a tank-pressure stop condition.
# Consumed by bcu_debug_node, which drives /bcu/rpm at `rpm` until the
# tank pressure (/bcu/pressure) crosses `target_pressure_pa` in the
# direction implied by the sign of `rpm`: positive RPM inflates the
# bladder -- tank pressure drops -- stop when tank_pa <= target_pa;
# negative RPM deflates -- tank pressure rises -- stop when tank_pa
# >= target_pa. A hard MAX_PUMP_S safety cap always applies so a stuck
# sensor or unreachable target can't leave the motor running. Bypasses
# pathfinding/depth PID -- bench testing only.

int16 rpm
int32 target_pressure_pa
