"""Mission profiles produced by a registry-based factory.

`/path` carries a mission_id; `pathfinding_node` resolves it via
`create_mission()` and the resulting profile drives `POSITION_TARGET`
at 10 Hz. The Pose's `position.z` is gauge Pa (depth controller's
contract); `orientation` is consumed by the ACU as roll/pitch.
"""

from .do_nothing import DoNothingMission
from .factory import MissionId, create_mission
from .profile import MissionProfile, MissionState
from .sawtooth import SawtoothMission
from .surface import SurfaceMission
from .trim_and_neutral import TrimAndNeutralBuoyancyMission

__all__ = [
    "DoNothingMission",
    "MissionId",
    "MissionProfile",
    "MissionState",
    "SawtoothMission",
    "SurfaceMission",
    "TrimAndNeutralBuoyancyMission",
    "create_mission",
]
