# Per-timestamp ground-truth anomaly label, broadcast by the sim-only
# anomaly_label_bridge at a fixed rate and recorded into every bag. The
# dataset builder joins sensor samples against this stream, so every
# timestamp in a run is attributable to a class — including timestamps
# whose anomaly signature is an *absence* (dropout / comms drops): this
# stream is published by a separate node that is never routed through
# any fault or comms gate.
#
# anomaly_class -- constant for the run: "nominal", "bcu_pump",
#   "sensor", "comms", or "biofouling" (mirrors Scenario.anomaly).
# channel / archetype -- set only for anomaly_class == "sensor": which
#   pressure channel ("external_pressure" / "tank_pressure") carries
#   which archetype ("bias" / "drift" / "stuck" / "dropout").
# active -- true while the fault is actually influencing published
#   data. Class semantics:
#     nominal:    always false
#     bcu_pump:   true iff the pump is actuating (last commanded
#                 /bcu/rpm != 0 AND the motor valve is open) — an idle
#                 degraded pump produces nominal-looking data
#     sensor:     true from the first published sample (drift: from
#                 t=0 by FDI convention, even while the ramp is still
#                 sub-noise)
#     comms:      always true (drops are absences in the other streams)
#     biofouling: always true (physical hull state from spawn)

std_msgs/Header header
string anomaly_class
string channel
string archetype
bool active
