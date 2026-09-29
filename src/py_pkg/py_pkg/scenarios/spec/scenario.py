"""Top-level scenario type.

One YAML on disk drives both halves of a run. `Scenario` is the single
type returned by `load_scenario`; callers reach into `.control` for
controller-side fields and `.rig` for HAL/sim-side fields.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ._shared import StrictModel
from .control import ControlScenario
from .rig import RigScenario


class AnomalyLabelSpec(StrictModel):
    """Ground-truth anomaly label for this run.

    Written by the sweep sampler (or by hand for demo scenarios) and
    broadcast into every bag by the `anomaly_label_bridge` — the
    per-timestamp label stream the dataset builder joins on. Inert for
    control/plant behaviour: severities live in `rig.faults` /
    `rig.hydrodynamics`, and a Scenario-level validator keeps this
    label consistent with them, so a bag can never claim "nominal"
    while a runtime fault is configured.
    """

    anomaly_class: Literal["nominal", "bcu_pump", "sensor", "comms", "biofouling"] = (
        "nominal"
    )
    # Set only for anomaly_class == "sensor": which pressure channel
    # carries the fault, and which archetype it is.
    channel: Literal["", "external_pressure", "tank_pressure"] = ""
    archetype: Literal["", "bias", "drift", "stuck", "dropout"] = ""

    @model_validator(mode="after")
    def _check_sensor_fields(self) -> "AnomalyLabelSpec":
        if self.anomaly_class == "sensor":
            if not self.channel or not self.archetype:
                raise ValueError(
                    "anomaly_class='sensor' requires both channel and archetype"
                )
        elif self.channel or self.archetype:
            raise ValueError(
                "channel/archetype are only valid for anomaly_class='sensor'"
            )
        return self


class Scenario(StrictModel):
    seed: int = 0
    control: ControlScenario = Field(default_factory=ControlScenario)
    rig: RigScenario = Field(default_factory=RigScenario)
    anomaly: AnomalyLabelSpec = Field(default_factory=AnomalyLabelSpec)

    @model_validator(mode="after")
    def _check_label_matches_faults(self) -> "Scenario":
        """Label <-> rig.faults consistency, enforced where checkable.

        bcu_pump/sensor/comms faults are visible in `rig.faults`, so the
        label must match in both directions. Biofouling is a pure
        hydrodynamics overlay (indistinguishable from a hand-authored
        plant change), so its label is trusted as authored — but it must
        not coexist with a configured runtime fault.
        """
        faults = self.rig.faults
        label = self.anomaly.anomaly_class

        pump_active = faults.bcu_pump.effectiveness != 1.0
        # Iterate the SensorFaultsSpec fields so a new channel is
        # covered here automatically (pydantic models yield
        # (name, value) pairs).
        sensor_channels = {
            name: spec for name, spec in faults.sensors if spec.kind != "none"
        }
        comms_active = faults.comms.drop_prob > 0.0

        expected = {
            "bcu_pump": pump_active,
            "sensor": bool(sensor_channels),
            "comms": comms_active,
        }
        for cls, active in expected.items():
            if active and label != cls:
                raise ValueError(
                    f"rig.faults configures a {cls!r} fault but anomaly_class"
                    f" is {label!r}"
                )
            if label == cls and not active:
                raise ValueError(
                    f"anomaly_class={cls!r} but rig.faults carries no such fault"
                )

        if label == "sensor":
            if len(sensor_channels) != 1:
                raise ValueError(
                    "anomaly_class='sensor' requires exactly one faulted channel,"
                    f" got {sorted(sensor_channels)}"
                )
            ((channel, spec),) = sensor_channels.items()
            if self.anomaly.channel != channel:
                raise ValueError(
                    f"anomaly.channel={self.anomaly.channel!r} does not match the"
                    f" faulted channel {channel!r}"
                )
            if self.anomaly.archetype != spec.kind:
                raise ValueError(
                    f"anomaly.archetype={self.anomaly.archetype!r} does not match"
                    f" the configured kind {spec.kind!r}"
                )
        return self

    @property
    def labeled_fault(self):
        """The `rig.faults` block `anomaly` points at, or None.

        Label -> fault-block resolution lives next to the validator that
        already walks it, so consumers (`compile.params_for_anomaly_label`)
        read the mapping rather than restating it. None for the classes
        with no schedulable block: nominal, comms, biofouling.
        """
        if self.anomaly.anomaly_class == "bcu_pump":
            return self.rig.faults.bcu_pump
        if self.anomaly.anomaly_class == "sensor":
            # `_check_label_matches_faults` has already proven this channel
            # exists and is the faulted one.
            return getattr(self.rig.faults.sensors, self.anomaly.channel)
        return None
