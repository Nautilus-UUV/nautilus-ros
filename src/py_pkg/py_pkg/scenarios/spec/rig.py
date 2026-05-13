"""Rig-side scenario schema.

Authors the simulator plant, fault injection, and world identifiers.
Knows nothing about PID gains or mission setpoints — those live in
`control.py`. Defaults at nominal mirror robot_specs and today's
nautilus_params.yaml, so a RigScenario() with no overrides matches
current behaviour. Phase 7 adds a CI parity test that locks the
nominal mirror to robot_specs.
"""

from __future__ import annotations

from pydantic import Field

from py_pkg.robot_specs import (
    BCU_MOTOR_MAX_RPM,
    BCU_MOTOR_MIN_RPM,
    BLADDER_VOLUME_M3,
    VOLUME_PER_REV_M3,
)

from ._shared import StrictModel


class SimSpec(StrictModel):
    """Sim-only identifiers — the Gazebo model and world."""

    model_name: str = "glider_nautilus"
    # The SDF <world name="..."> string, not the world *file* basename.
    # See dave_worlds/worlds/dave_ocean_waves.world line 20.
    world_name: str = "oceans_waves"


class PlantSpec(StrictModel):
    """Ground-truth plant parameters used by HAL bridges + the sim.

    Mirrors robot_specs.py at nominal. MC perturbs these to model
    uncertainty in the *actual* hardware plant. To perturb the
    controller's *model* of the plant independently, see
    ControlScenario.controllers.depth.plant_model.
    """

    volume_per_rev_m3: float = VOLUME_PER_REV_M3
    # Operating-range clamps the bridge enforces on the simulated bladder.
    # Default to 10% headroom from the SDF's 2.5 L mechanical max so the
    # control loop can swing through a 2.0 L range without ever pinning the
    # plugin clamp; MC sweeps can widen or shrink either end.
    bladder_min_m3: float = 0.00025
    bladder_max_m3: float = 0.00225
    bladder_nominal_m3: float = BLADDER_VOLUME_M3
    bcu_motor_min_rpm: int = BCU_MOTOR_MIN_RPM
    bcu_motor_max_rpm: int = BCU_MOTOR_MAX_RPM


class FaultInjectorSpec(StrictModel):
    """Per-injector knobs. Defaults match BaseFaultInjector hardcoded defaults."""

    probability_per_sec: float = 0.05
    duration_sec: float = 5.0
    # Degraded-state multiplier: BCU applies this to the RPM command.
    # 0.5 = "produce half the commanded flow."
    degraded_factor: float = 0.5
    # Severe-state multiplier: 0.0 = "produce zero flow."
    severe_factor: float = 0.0


class FaultsSpec(StrictModel):
    # Add more injectors here as they appear (acu_pitch, imu_left, ...).
    bcu_rpm: FaultInjectorSpec = Field(default_factory=FaultInjectorSpec)


class NoiseSpec(StrictModel):
    """Placeholder for SDF noise-plugin seeds (Phase 8+ work)."""

    pass


class BcuBridgeSpec(StrictModel):
    """BCU sim-bridge knobs. publish_rate_hz is the rate at which
    pressure/volume telemetry reaches the controllers and so is part
    of the closed-loop data-rate environment a Monte Carlo run sees.
    """

    publish_rate_hz: int = 10


class ExternalSensorBridgeSpec(StrictModel):
    """External-sensor sim-bridge knob. publish_rate_hz drives the
    pressure stream feeding the depth loop.
    """

    publish_rate_hz: int = 10


class BridgesSpec(StrictModel):
    bcu: BcuBridgeSpec = Field(default_factory=BcuBridgeSpec)
    external_sensor: ExternalSensorBridgeSpec = Field(
        default_factory=ExternalSensorBridgeSpec
    )


class RigScenario(StrictModel):
    sim: SimSpec = Field(default_factory=SimSpec)
    plant: PlantSpec = Field(default_factory=PlantSpec)
    faults: FaultsSpec = Field(default_factory=FaultsSpec)
    noise: NoiseSpec = Field(default_factory=NoiseSpec)
    bridges: BridgesSpec = Field(default_factory=BridgesSpec)
