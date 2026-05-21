# Debug/manual BCU pump command. Consumed by bcu_debug_node, which
# drives /bcu/rpm at `rpm` for `duration_s` seconds, then publishes 0.
# Bypasses pathfinding/depth PID -- intended for bench testing only.

int16 rpm
float32 duration_s
