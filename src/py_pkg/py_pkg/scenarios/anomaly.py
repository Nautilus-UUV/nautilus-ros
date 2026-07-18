"""Per-run anomaly-mix sampling for validation sweeps.

Pure logic, no ROS, stdlib RNG only (same conventions as
``sensor_noise.py`` / ``buoyancy.py``): the host-side sampler
(``scripts/lhs_sample.py``) consumes this to assign every run of a
sweep a class — ``nominal`` or one of four persistent whole-run
anomalies — and to draw each anomalous run's severity from bands
authored strictly outside the nominal envelope (locked by
``test_anomaly_envelope_separation``).

Design contracts (each Tier-1 tested):

- **Stratified exact counts**: ``assign_classes`` splits ``n`` runs
  into largest-remainder-rounded class counts and shuffles them with a
  dedicated seeded stream — a 320-run sweep at 0.80/0.05x4 carries
  exactly 256/16/16/16/16, deterministically.
- **Decoupling**: the class stream (``"anomaly_class_assignment"``) and
  the per-run severity streams (``"anomaly_severity:<idx>"``) are
  derive_seed children of the sweep seed, independent of the LHS
  matrix (the mix adds zero LHS dimensions). Nominal-assigned runs are
  byte-identical to a no-mix sweep, and editing one class's bands never
  reshuffles assignments or other classes' severities.
- **apply_anomaly** writes only ``rig.faults.*`` / the biofouling
  hydro multipliers / the ``anomaly:`` label block — the label is
  validated against the faults by the Scenario schema at load time.

The biofouling buoyancy offset (fouling mass -> neutral-volume shift)
is computed here (``fouled_neutral_volume``) but *applied* by the
sampler's correlated buoyancy derivation, which owns the trim fields.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal, Optional, get_args

from pydantic import Field, model_validator

from .seed import derive_seed
from .spec._shared import StrictModel
from .spec.rig import SensorFaultKind, SensorFaultsSpec

# Kept explicit (not derived from the AnomalyLabelSpec Literal): this
# tuple's ORDER feeds the seeded class shuffle, so it must never move
# under a wire-schema reorder. test_classes_tuple_is_the_schema_vocabulary
# locks the two spellings together.
ANOMALY_CLASSES = ("nominal", "bcu_pump", "sensor", "comms", "biofouling")

# Derived from the fault schema so the sampler can't drift from
# rig.faults: channels are the SensorFaultsSpec fields, archetypes the
# injectable SensorFaultKind values ("none" is the absence of a fault).
# Membership-only (never drawn from by index), so order is free.
SENSOR_CHANNELS = tuple(SensorFaultsSpec.model_fields)
SENSOR_ARCHETYPES = tuple(k for k in get_args(SensorFaultKind) if k != "none")

# rig.hydrodynamics slots the biofouling overlay multiplies in place.
# All 12 drag diagonals (linear + quadratic — growth roughens the whole
# hull) and the 6 added-mass diagonals. fluid_density, trim masses and
# spawn volume are NEVER touched here: the buoyancy derivation owns
# them (the fouling mass enters through fouled_neutral_volume).
FOULING_DRAG_SLOTS = (
    "drag_xU",
    "drag_yV",
    "drag_zW",
    "drag_kP",
    "drag_mQ",
    "drag_nR",
    "drag_xUabsU",
    "drag_yVabsV",
    "drag_zWabsW",
    "drag_kPabsP",
    "drag_mQabsQ",
    "drag_nRabsR",
)
FOULING_ADDED_MASS_SLOTS = (
    "added_mass_xx",
    "added_mass_yy",
    "added_mass_zz",
    "added_mass_pp",
    "added_mass_qq",
    "added_mass_rr",
)


class Band(StrictModel):
    """Closed severity band [low, high]; low == high pins a fixed value.

    ``signed: true`` flips the drawn magnitude's sign with probability
    1/2 (bias/drift can push either way).
    """

    low: float
    high: float
    signed: bool = False

    @model_validator(mode="after")
    def _check_order(self) -> "Band":
        if self.high < self.low:
            raise ValueError(f"band high ({self.high}) must be >= low ({self.low})")
        return self

    def draw(self, rng: random.Random) -> float:
        v = rng.uniform(self.low, self.high)
        if self.signed and rng.random() < 0.5:
            v = -v
        return float(v)


class BcuPumpMix(StrictModel):
    """Severity bands for the persistent pump fault.

    ``effectiveness`` is mandatory (it is the class marker the schema
    validates the label against). The optional degraded-transient bands
    write the existing ``rig.plant`` knobs — overriding, by design, any
    nominal-envelope draw of the same fields for this run.
    """

    effectiveness: Band
    response_delay_s: Optional[Band] = None
    slew_rpm_per_s: Optional[Band] = None


class SensorMix(StrictModel):
    """One archetype on one pressure channel per anomalous run.

    ``bands[channel][archetype]`` carries the severity band for every
    (channel, archetype) combination that needs one — ``stuck`` has no
    severity, ``dropout`` bands are drop probabilities, ``bias`` /
    ``drift`` bands are channel-native magnitudes (Pa, Pa/s).
    """

    channels: tuple[Literal["external_pressure", "tank_pressure"], ...] = Field(
        min_length=1
    )
    archetypes: tuple[Literal["bias", "drift", "stuck", "dropout"], ...] = Field(
        min_length=1
    )
    bands: dict[str, dict[str, Band]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_complete(self) -> "SensorMix":
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("sensor mix: duplicate channels")
        if len(set(self.archetypes)) != len(self.archetypes):
            raise ValueError("sensor mix: duplicate archetypes")
        for channel in self.bands:
            if channel not in SENSOR_CHANNELS:
                raise ValueError(f"sensor mix: unknown bands channel {channel!r}")
            for archetype in self.bands[channel]:
                if archetype not in SENSOR_ARCHETYPES:
                    raise ValueError(
                        f"sensor mix: unknown bands archetype {archetype!r}"
                    )
        for channel in self.channels:
            for archetype in self.archetypes:
                if archetype == "stuck":
                    continue  # no severity parameter
                if self.bands.get(channel, {}).get(archetype) is None:
                    raise ValueError(
                        f"sensor mix: missing band for ({channel}, {archetype})"
                    )
        return self


class CommsMix(StrictModel):
    drop_prob: Band

    @model_validator(mode="after")
    def _check_range(self) -> "CommsMix":
        if self.drop_prob.signed:
            raise ValueError("comms drop_prob band cannot be signed")
        if not (0.0 < self.drop_prob.low and self.drop_prob.high < 1.0):
            raise ValueError("comms drop_prob band must sit inside (0, 1)")
        return self


class BiofoulingMix(StrictModel):
    """Whole-run hull-growth severity.

    ``drag_mult`` / ``added_mass_mult`` multiply the FOULING_* hydro
    slots after the nominal draw (forward map + jitter), so the fouled
    coefficients sit outside the envelope iff the band floor clears the
    envelope's worst multiplicative extreme. ``fouling_mass_kg`` shifts
    the run's neutral-volume target through the buoyancy derivation;
    the derivation checks viability against the relaxed
    ``min_climb_margin_m3`` (a barely-climbing vehicle IS the
    biofouling signature).
    """

    drag_mult: Band
    added_mass_mult: Band
    fouling_mass_kg: Band
    min_climb_margin_m3: float = 3.0e-5

    @model_validator(mode="after")
    def _check_positive(self) -> "BiofoulingMix":
        for name in ("drag_mult", "added_mass_mult", "fouling_mass_kg"):
            band: Band = getattr(self, name)
            if band.signed or band.low <= 0.0:
                raise ValueError(f"biofouling {name} band must be positive/unsigned")
        for name in ("drag_mult", "added_mass_mult"):
            if getattr(self, name).low <= 1.0:
                raise ValueError(
                    f"biofouling {name} floor must exceed 1.0 (growth adds, "
                    "never removes)"
                )
        return self


class AnomalyMixSpec(StrictModel):
    """The per-run class mix of a validation sweep (e.g. 0.80/0.05x4)."""

    weights: dict[str, float]
    bcu_pump: Optional[BcuPumpMix] = None
    sensor: Optional[SensorMix] = None
    comms: Optional[CommsMix] = None
    biofouling: Optional[BiofoulingMix] = None

    @model_validator(mode="after")
    def _check_weights_and_blocks(self) -> "AnomalyMixSpec":
        unknown = set(self.weights) - set(ANOMALY_CLASSES)
        if unknown:
            raise ValueError(f"anomaly_mix: unknown classes {sorted(unknown)}")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("anomaly_mix: weights must be >= 0")
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"anomaly_mix: weights must sum to 1, got {total}")
        for cls in ("bcu_pump", "sensor", "comms", "biofouling"):
            if self.weights.get(cls, 0.0) > 0.0 and getattr(self, cls) is None:
                raise ValueError(
                    f"anomaly_mix: class {cls!r} has weight > 0 but no config block"
                )
        return self


@dataclass(frozen=True)
class AnomalyAssignment:
    """One run's resolved class + severity draw."""

    anomaly_class: str
    channel: str = ""
    archetype: str = ""
    severity: dict = field(default_factory=dict)

    def record(self) -> dict:
        """Manifest record — explicit for every run, nominal included."""
        return {
            "class": self.anomaly_class,
            "channel": self.channel,
            "archetype": self.archetype,
            "severity": dict(self.severity),
        }


def class_counts(mix: AnomalyMixSpec, n: int) -> dict[str, int]:
    """Largest-remainder rounding of ``weights * n`` to exact counts.

    Ties break in fixed ANOMALY_CLASSES order so the split is a pure
    function of (weights, n).
    """
    quotas = {cls: mix.weights.get(cls, 0.0) * n for cls in ANOMALY_CLASSES}
    counts = {cls: int(quotas[cls]) for cls in ANOMALY_CLASSES}
    leftover = n - sum(counts.values())
    # sorted() is stable over the canonical tuple, so equal remainders
    # keep ANOMALY_CLASSES order without an explicit tie-break key.
    by_remainder = sorted(ANOMALY_CLASSES, key=lambda cls: -(quotas[cls] - counts[cls]))
    for cls in by_remainder[:leftover]:
        counts[cls] += 1
    return counts


def assign_classes(mix: AnomalyMixSpec, parent_seed: int, n: int) -> list[str]:
    """Stratified class assignment: exact counts, seeded permutation."""
    counts = class_counts(mix, n)
    vector = [cls for cls in ANOMALY_CLASSES for _ in range(counts[cls])]
    rng = random.Random(derive_seed(parent_seed, "anomaly_class_assignment"))
    rng.shuffle(vector)
    return vector


def draw_assignment(
    mix: AnomalyMixSpec, parent_seed: int, idx: int, anomaly_class: str
) -> AnomalyAssignment:
    """Severity draw for run ``idx`` given its assigned class.

    Every run gets its own derived RNG stream; the draw order inside a
    class is fixed (documented per branch), so same seed + same mix
    config => identical severities forever.
    """
    if anomaly_class not in ANOMALY_CLASSES:
        raise ValueError(f"unknown anomaly class {anomaly_class!r}")
    if anomaly_class == "nominal":
        return AnomalyAssignment("nominal")

    rng = random.Random(derive_seed(parent_seed, f"anomaly_severity:{idx}"))

    if anomaly_class == "bcu_pump":
        m = _require(mix.bcu_pump, "bcu_pump")
        # Draw order: effectiveness, then the optional transient bands.
        severity = {"effectiveness": m.effectiveness.draw(rng)}
        if m.response_delay_s is not None:
            severity["response_delay_s"] = m.response_delay_s.draw(rng)
        if m.slew_rpm_per_s is not None:
            severity["slew_rpm_per_s"] = m.slew_rpm_per_s.draw(rng)
        return AnomalyAssignment("bcu_pump", severity=severity)

    if anomaly_class == "sensor":
        m = _require(mix.sensor, "sensor")
        # Draw order: channel, archetype, then the severity band.
        channel = m.channels[rng.randrange(len(m.channels))]
        archetype = m.archetypes[rng.randrange(len(m.archetypes))]
        if archetype == "stuck":
            severity = {}
        elif archetype == "dropout":
            severity = {"drop_prob": m.bands[channel][archetype].draw(rng)}
        else:  # bias / drift
            severity = {"magnitude": m.bands[channel][archetype].draw(rng)}
        return AnomalyAssignment(
            "sensor", channel=channel, archetype=archetype, severity=severity
        )

    if anomaly_class == "comms":
        m = _require(mix.comms, "comms")
        return AnomalyAssignment("comms", severity={"drop_prob": m.drop_prob.draw(rng)})

    # biofouling — draw order: drag, added mass, fouling mass.
    m = _require(mix.biofouling, "biofouling")
    return AnomalyAssignment(
        "biofouling",
        severity={
            "drag_mult": m.drag_mult.draw(rng),
            "added_mass_mult": m.added_mass_mult.draw(rng),
            "fouling_mass_kg": m.fouling_mass_kg.draw(rng),
        },
    )


def _require(block, cls: str):
    if block is None:
        raise ValueError(f"anomaly class {cls!r} assigned but mix has no block")
    return block


def apply_anomaly(scenario: dict, assignment: AnomalyAssignment) -> None:
    """Write one run's anomaly into its scenario dict, in place.

    Nominal assignments write nothing (the scenario stays byte-identical
    to a no-mix sweep's output). Anomalous assignments write the
    matching ``rig.faults`` block (or multiply the biofouling hydro
    slots) plus the top-level ``anomaly:`` label the schema validates
    and the label bridge broadcasts.
    """
    if assignment.anomaly_class == "nominal":
        return

    rig = scenario.setdefault("rig", {})
    sev = assignment.severity

    if assignment.anomaly_class == "bcu_pump":
        faults = rig.setdefault("faults", {})
        faults.setdefault("bcu_pump", {})["effectiveness"] = sev["effectiveness"]
        plant = rig.setdefault("plant", {})
        if "response_delay_s" in sev:
            plant["pump_response_delay_s"] = sev["response_delay_s"]
        if "slew_rpm_per_s" in sev:
            plant["pump_slew_rpm_per_s"] = sev["slew_rpm_per_s"]
    elif assignment.anomaly_class == "sensor":
        faults = rig.setdefault("faults", {})
        channel = faults.setdefault("sensors", {}).setdefault(assignment.channel, {})
        channel["kind"] = assignment.archetype
        if assignment.archetype == "dropout":
            channel["drop_prob"] = sev["drop_prob"]
        elif assignment.archetype in ("bias", "drift"):
            channel["magnitude"] = sev["magnitude"]
    elif assignment.anomaly_class == "comms":
        faults = rig.setdefault("faults", {})
        faults.setdefault("comms", {})["drop_prob"] = sev["drop_prob"]
    elif assignment.anomaly_class == "biofouling":
        hydro = rig.get("hydrodynamics")
        if not isinstance(hydro, dict):
            raise ValueError(
                "biofouling requires a rig.hydrodynamics block (physics-mode "
                "sweep) — the drag/added-mass multipliers have nothing to "
                "scale otherwise"
            )
        for slot in FOULING_DRAG_SLOTS:
            hydro[slot] = float(hydro[slot]) * sev["drag_mult"]
        for slot in FOULING_ADDED_MASS_SLOTS:
            hydro[slot] = float(hydro[slot]) * sev["added_mass_mult"]
    else:  # pragma: no cover — draw_assignment already validated
        raise ValueError(f"unknown anomaly class {assignment.anomaly_class!r}")

    label = {"anomaly_class": assignment.anomaly_class}
    if assignment.anomaly_class == "sensor":
        label["channel"] = assignment.channel
        label["archetype"] = assignment.archetype
    scenario["anomaly"] = label


def derivation_overrides(
    mix: Optional[AnomalyMixSpec], assignment: Optional[AnomalyAssignment]
) -> tuple[float, Optional[float]]:
    """Buoyancy-derivation hooks for one run: (extra_mass_kg, climb override).

    Biofouling is the only class that touches the derivation — its
    fouling mass shifts the neutral-volume target and its relaxed
    ``min_climb_margin_m3`` replaces the sweep default. Every other
    assignment (or none) returns ``(0.0, None)``: derivation untouched.
    Lives here so class -> derivation coupling stays in the module that
    owns anomaly semantics; the sampler calls it unconditionally.
    """
    if assignment is not None and assignment.anomaly_class == "biofouling":
        biofouling = _require(mix.biofouling if mix is not None else None, "biofouling")
        return assignment.severity["fouling_mass_kg"], biofouling.min_climb_margin_m3
    return 0.0, None


def fouled_neutral_volume(
    neutral_volume_m3: float, fouling_mass_kg: float, fluid_density: float
) -> float:
    """Neutral-volume target of a fouled hull.

    Growth adds mass; a heavier vehicle needs ``dm / rho`` more bladder
    volume to float neutral. The sampler feeds this shifted target to
    the buoyancy derivation instead of writing trim fields directly.
    """
    if fluid_density <= 0:
        raise ValueError("fluid_density must be positive")
    if fouling_mass_kg < 0:
        raise ValueError("fouling_mass_kg must be >= 0")
    return neutral_volume_m3 + fouling_mass_kg / fluid_density
