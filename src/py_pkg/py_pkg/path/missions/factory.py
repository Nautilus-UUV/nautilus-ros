"""Mission factory: maps mission_id -> MissionProfile instance."""

from enum import IntEnum

from .do_nothing import DoNothingMission
from .profile import MissionProfile
from .sawtooth import SawtoothMission
from .surface import SurfaceMission
from .trim_and_neutral import TrimAndNeutralBuoyancyMission


class MissionId(IntEnum):
    TRIM_AND_NEUTRAL_BUOYANCY = 0
    SAWTOOTH = 1
    SURFACE = 2
    DO_NOTHING = 3


_REGISTRY: dict[int, type[MissionProfile]] = {
    MissionId.TRIM_AND_NEUTRAL_BUOYANCY: TrimAndNeutralBuoyancyMission,
    MissionId.SAWTOOTH: SawtoothMission,
    MissionId.SURFACE: SurfaceMission,
    MissionId.DO_NOTHING: DoNothingMission,
}


def create_mission(mission_id: int) -> MissionProfile:
    cls = _REGISTRY.get(int(mission_id))
    if cls is None:
        known = sorted(_REGISTRY.keys())
        raise ValueError(f"unknown mission_id={mission_id}; known ids: {known}")
    return cls()
