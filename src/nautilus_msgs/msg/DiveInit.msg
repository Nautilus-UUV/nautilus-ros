# Operator-registered dive initialization, sent once per deployment from
# the UI Initialize button (nautilus/cmd/init -> /init/dive) and latched
# (TRANSIENT_LOCAL) so latecomer nodes pick the values up without the
# operator re-registering.
#
# surface_pressure_pa -- absolute external pressure read at the surface.
# Replaces the standard-atmosphere constant as the gauge reference in the
# control stack (bcu_node, acu_node, pathfinding), so "0 gauge" means
# the actual surface on dive day, not 101325 Pa.
#
# tank_empty_pa / tank_full_pa -- tank pressure at the drained / full-of-oil
# endpoints, in the tank sensor's own frame (the same stream as
# /bcu/pressure -- register from the live reading, don't hand-convert).
# The tank runs INVERSE to bladder fill: positive bus RPM inflates the
# bladder, draining the tank toward `tank_empty_pa`; negative RPM, or the
# passive free-valve vent, pushes oil back, raising it toward
# `tank_full_pa`. Consumed by bcu_node's output clamp and the
# lifeguard's tank-empty stand-down.

float64 surface_pressure_pa
float64 tank_empty_pa
float64 tank_full_pa
