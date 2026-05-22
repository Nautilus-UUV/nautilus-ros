"""Do-nothing mission: keep the stack live but command the glider not at all.

This is the "control does nothing" profile. Two things happen when it runs:

1. At start the executor emits `CONTROL_RESET`, so depth_node and acu_node
   wipe their setpoint and controller state back to exactly how they sat at
   boot before any mission -- no held target, no integral windup. (Opted in
   via the `resets_control_on_start` flag the executor reads off the mission.)

2. `reference()` returns `None` every tick, so the executor publishes no
   `POSITION_TARGET`. With no setpoint, depth_node emits its zero-RPM /
   valves-closed hold and acu_node skips pitch (roll sits quiet) -- no
   actuation reaches the BCU or ACU. The glider holds whatever trim it has
   and drifts.

Never self-terminates; it ends when the operator sends stop/abort. Operator
parameters from `MissionCommand` are all ignored.
"""

from geometry_msgs.msg import Pose

from .profile import MissionState


class DoNothingMission:
    # Read by pathfinding_node at start: emit CONTROL_RESET so the controllers
    # return to their fresh, no-mission state before we go quiet.
    resets_control_on_start: bool = True

    def start(self, state: MissionState) -> None:
        pass

    def update(self, current_pressure_pa: float) -> None:
        pass

    def reference(self, mission_t: float) -> Pose | None:
        # No setpoint, ever -> executor publishes nothing -> controllers hold
        # their no-target safe state.
        return None

    def is_done(self, mission_t: float) -> bool:
        return False
