"""Mission profiles produced by a registry-based factory.

`/path` carries a mission_id; `pathfinding_node` resolves it via
`create_mission()` and the resulting profile drives `POSITION_TARGET`
at 10 Hz. The Pose's `position.z` is gauge Pa (depth controller's
contract); `orientation` is consumed by the ACU as roll/pitch.
"""

from .factory import MissionId, create_mission
from .profile import MissionProfile, MissionState
from .sawtooth import SawtoothMission
from .staircase import StaircaseMission
from .surface import SurfaceMission
from .trim_and_neutral import TrimAndNeutralBuoyancyMission

__all__ = [
    "MissionId",
    "MissionProfile",
    "MissionState",
    "SawtoothMission",
    "StaircaseMission",
    "SurfaceMission",
    "TrimAndNeutralBuoyancyMission",
    "create_mission",
]
