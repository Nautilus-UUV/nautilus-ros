#!/usr/bin/env python3
"""Latin Hypercube sampler for Nautilus scenario YAMLs.

Reads a sweep spec describing which scenario dot-paths to perturb (and
how), draws an LHS over the unit hypercube, scales each column to its
distribution, and overlays the perturbations onto a base scenario from
the in-repo `library/`. Writes one YAML per sample plus a manifest
recording the full sample matrix, spec hash, and sampler version so
post-hoc analyses can join bag results back to scenario coordinates
without re-parsing every YAML.

The sampler runs *outside* the SIF — it knows nothing about Pydantic or
the Scenario tree, it just writes nested dicts. Validation happens in
the container via `py_pkg.scenarios.loader.load_scenario` (Pydantic
`extra="forbid"` catches schema typos). `run_sweep.py` does that
pre-flight on the first sample before queueing the rest.

Dimensions whose path starts with `mission.` are *launch dimensions*:
mission knobs (target_pressure_pa, n_oscillations, ...) are launch args,
so their sampled values are recorded in
manifest.json only. A `mission_mix:` block replaces those dimensions
entirely (authoring both is rejected): per-run mission profiles are
assigned/drawn off-matrix (`py_pkg.scenarios.mission_mix`) and their
flat mission.* values ride the manifest the same way.

Dimensions whose path starts with `derive.` are *derivation targets*:
`derive.neutral_volume_m3` is sampled jointly with the row and fed —
together with the run's effective fluid_density — to the correlated
buoyancy derivation (`py_pkg.scenarios.buoyancy`), which computes the
trim masses and bladder spawn volume written into rig.hydrodynamics so
every emitted plant is oscillation-viable by construction. The optional
`buoyancy_derivation:` spec block tunes spawn policy and margins.

Example:
    lhs_sample.py --spec sweeps/example_hydro_faults_pump.yaml \\
                  --out ./scenarios --name lhs_hydro_v1
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src/py_pkg"))
from py_pkg.scenarios import buoyancy
from py_pkg.scenarios.anomaly import (
    AnomalyAssignment,
    AnomalyMixSpec,
    apply_anomaly,
    assign_classes,
    derivation_overrides,
    draw_assignment,
    fouled_neutral_volume,
)
from py_pkg.scenarios.compile import forward_map
from py_pkg.scenarios.mission_mix import (
    MissionAssignment,
    MissionMixSpec,
    assign_profiles,
    draw_mission,
    expected_mission_duration_s,
)
from py_pkg.scenarios.spec.rig import (
    FinAeroSpec,
    HydrodynamicsSpec,
    PhysicsKnobs,
    RigScenario,
)

# 2.0.0: persistent-fault schema (rig.faults.bcu_pump/sensors/comms) +
# per-run anomaly_mix class assignment; the Poisson-ladder schema is gone.
# 2.1.0: additive — per-run mission_mix profile sampling (mission.* values
# + a per-sample `mission` record in the manifest) and sampled fault
# onsets (a rig.faults.*.schedule block on non-immediate anomalous runs).
SAMPLER_VERSION = "2.1.0"

# Dimensions under this prefix are launch args (mission knobs), not scenario
# fields: they are recorded in the manifest but never written into the
# scenario YAML, whose schema (`extra="forbid"`) would reject them.
MISSION_PREFIX = "mission."

# Dimensions under this prefix are *derivation targets*: sampled jointly
# with the rest of the row but consumed by the correlated buoyancy
# derivation (py_pkg.scenarios.buoyancy) rather than written to the
# scenario as-is. The only recognized path is DERIVE_NEUTRAL_VOLUME; the
# derivation computes trim_mass_bow / trim_mass_stern /
# bladder_spawn_volume_m3 from (fluid_density, neutral-volume target) so
# every emitted plant is oscillation-viable by construction.
DERIVE_PREFIX = "derive."
DERIVE_NEUTRAL_VOLUME = "derive.neutral_volume_m3"

# Scenario path -> DerivedBuoyancy attribute the derivation writes. One
# mapping drives both the write-back and the authoring-conflict check:
# sampling any of these paths alongside a derive.* dimension is a
# spec-authoring conflict (two authorities for the same field).
_DERIVED_FIELDS = {
    "rig.hydrodynamics.trim_mass_bow": "trim_mass_bow",
    "rig.hydrodynamics.trim_mass_stern": "trim_mass_stern",
    "rig.hydrodynamics.bladder_spawn_volume_m3": "bladder_spawn_volume_m3",
}

# The bare-name dimensions a "physics" sweep feeds to the deterministic
# forward map. Any dimension whose path is *not* one of these is treated as a
# plain scenario dot-path overlay (e.g. rig.plant.pump_response_delay_s), so a
# sweep can perturb the plant geometry and a scenario knob in one joint LHS.
_PHYSICS_KNOB_FIELDS = frozenset(PhysicsKnobs.model_fields)

SUPPORTED_DISTRIBUTIONS = ("uniform", "loguniform")

# Resolve the in-repo scenario library so a sweep spec can refer to base
# scenarios by short name (`nominal.yaml`) without the author having to
# spell out the full path. The library lives in py_pkg's source tree;
# this script sits two dirs above it under nautilus-ros/.
LIBRARY_DIR = (
    Path(__file__).resolve().parent.parent / "src/py_pkg/py_pkg/scenarios/library"
)


@dataclass(frozen=True)
class Dimension:
    path: str
    distribution: str
    low: float
    high: float
    integer: bool = False

    @classmethod
    def from_dict(cls, raw: dict) -> "Dimension":
        try:
            dist = raw["distribution"]
        except KeyError as e:
            raise ValueError(f"dimension {raw!r} missing 'distribution'") from e
        if dist not in SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                f"dimension {raw['path']!r}: unsupported distribution {dist!r}; "
                f"expected one of {SUPPORTED_DISTRIBUTIONS}"
            )
        low, high = float(raw["low"]), float(raw["high"])
        # A zero-width band (high == low) PINS the value: `scale`
        # degenerates to the constant. Uniform only — a loguniform pin
        # buys nothing over a uniform one, so it keeps the strict check.
        if high < low or (high == low and dist != "uniform"):
            raise ValueError(
                f"dimension {raw['path']!r}: high ({high}) must be > low "
                f"({low}); a low == high pin is allowed for uniform only"
            )
        if dist == "loguniform" and low <= 0:
            raise ValueError(
                f"dimension {raw['path']!r}: loguniform requires low > 0, got {low}"
            )
        return cls(
            path=raw["path"],
            distribution=dist,
            low=low,
            high=high,
            integer=bool(raw.get("integer", False)),
        )

    def scale(self, u: np.ndarray) -> np.ndarray:
        """Map LHS uniforms in [0,1) to the dimension's range."""
        if self.distribution == "uniform":
            scaled = self.low + u * (self.high - self.low)
        else:
            # loguniform: equal density per decade between low and high.
            log_low, log_high = math.log(self.low), math.log(self.high)
            scaled = np.exp(log_low + u * (log_high - log_low))
        # `integer: true` floors, so uniform [1, 4] draws {1, 2, 3} uniformly
        # (u < 1 keeps the high edge exclusive).
        return np.floor(scaled) if self.integer else scaled

    def emit(self, value: float) -> int | float:
        """The value as written to the scenario/manifest (int when declared)."""
        return int(value) if self.integer else float(value)


_SPAWN_POLICIES = ("offset", "bladder_max")


@dataclass(frozen=True)
class BuoyancyDerivationConfig:
    """Tuning for the correlated buoyancy derivation.

    The derivation itself is enabled by the presence of a `derive.*`
    dimension, never by this block — the block only tunes it. Spawn
    policies: "offset" spawns at neutral + spawn_offset_m3 (the gentle
    surface float the canonical SDF encodes); "bladder_max" spawns at
    the run's effective bladder ceiling (bladder-full float).
    """

    spawn_policy: str = "offset"
    spawn_offset_m3: float = buoyancy.SPAWN_OFFSET_M3
    min_dive_margin_m3: float = buoyancy.DEFAULT_MIN_DIVE_MARGIN_M3
    min_climb_margin_m3: float = buoyancy.DEFAULT_MIN_CLIMB_MARGIN_M3

    def __post_init__(self) -> None:
        if self.spawn_policy not in _SPAWN_POLICIES:
            raise ValueError(
                f"buoyancy_derivation: unsupported spawn_policy "
                f"{self.spawn_policy!r}; expected one of {_SPAWN_POLICIES}"
            )

    @classmethod
    def from_dict(cls, raw: dict) -> "BuoyancyDerivationConfig":
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"buoyancy_derivation: unknown keys {sorted(unknown)}")
        return cls(
            **{k: v if k == "spawn_policy" else float(v) for k, v in raw.items()}
        )


@dataclass(frozen=True)
class SweepSpec:
    description: str
    n_samples: int
    seed: int
    base_scenario: str
    dimensions: tuple[Dimension, ...]
    sampling_mode: str = "lhs"
    isotropic_jitter_sigma: float = 0.0
    # Always the *effective* config (defaults when the spec has no
    # block), so consumers never re-resolve a None fallback.
    buoyancy_derivation: BuoyancyDerivationConfig = field(
        default_factory=BuoyancyDerivationConfig
    )
    # Per-run anomaly class mix (validation sweeps). None = every run
    # nominal (legacy sweeps unchanged). Deliberately NOT an LHS
    # dimension: the class/severity streams are derive_seed children of
    # the sweep seed, so adding/editing the mix never reshuffles the
    # nominal LHS draws.
    anomaly_mix: AnomalyMixSpec | None = None
    # Per-run mission-profile mix (v2 sweeps). None = mission knobs ride
    # plain mission.* dimensions (legacy sweeps byte-identical). Same
    # off-matrix seeded-stream design as anomaly_mix; a spec authoring
    # BOTH a mission_mix and mission.* dimensions is rejected at load.
    mission_mix: MissionMixSpec | None = None

    @property
    def derivation_enabled(self) -> bool:
        return any(d.path.startswith(DERIVE_PREFIX) for d in self.dimensions)

    @classmethod
    def load(cls, path: Path) -> "SweepSpec":
        raw = yaml.safe_load(path.read_text())
        try:
            dims = tuple(Dimension.from_dict(d) for d in raw["dimensions"])
        except KeyError as e:
            raise ValueError(f"{path}: sweep spec missing 'dimensions'") from e
        if not dims:
            raise ValueError(f"{path}: 'dimensions' must be non-empty")
        _validate_derive_dims(path, dims)
        spec = cls(
            description=raw.get("description", ""),
            n_samples=int(raw["n_samples"]),
            seed=int(raw.get("seed", 0)),
            base_scenario=raw["base_scenario"],
            dimensions=dims,
            sampling_mode=raw.get("sampling_mode", "lhs"),
            isotropic_jitter_sigma=float(raw.get("isotropic_jitter_sigma", 0.0)),
            buoyancy_derivation=BuoyancyDerivationConfig.from_dict(
                raw.get("buoyancy_derivation") or {}
            ),
            anomaly_mix=(
                AnomalyMixSpec.model_validate(raw["anomaly_mix"])
                if raw.get("anomaly_mix")
                else None
            ),
            mission_mix=(
                MissionMixSpec.model_validate(raw["mission_mix"])
                if raw.get("mission_mix")
                else None
            ),
        )
        if spec.mission_mix is not None:
            conflicts = sorted(
                d.path for d in dims if d.path.startswith(MISSION_PREFIX)
            )
            if conflicts:
                raise ValueError(
                    f"{path}: {conflicts} sampled alongside a mission_mix "
                    "block — two authorities for the run's mission "
                    "(conflicting authorities); drop one side"
                )
        if (
            spec.anomaly_mix is not None
            and spec.anomaly_mix.weights.get("biofouling", 0.0) > 0.0
            and not (spec.sampling_mode == "physics" and spec.derivation_enabled)
        ):
            raise ValueError(
                f"{path}: biofouling anomalies need a physics-mode sweep with "
                f"the correlated buoyancy derivation ({DERIVE_NEUTRAL_VOLUME}) "
                "— the drag multipliers and the fouling-mass neutral shift "
                "have nothing to act on otherwise"
            )
        return spec


def _validate_derive_dims(path: Path, dims: tuple[Dimension, ...]) -> None:
    derive_paths = [d.path for d in dims if d.path.startswith(DERIVE_PREFIX)]
    for p in derive_paths:
        if p != DERIVE_NEUTRAL_VOLUME:
            raise ValueError(
                f"{path}: unrecognized derive dimension {p!r}; the only "
                f"supported derive path is {DERIVE_NEUTRAL_VOLUME!r}"
            )
    if derive_paths:
        conflicts = sorted(d.path for d in dims if d.path in _DERIVED_FIELDS)
        if conflicts:
            raise ValueError(
                f"{path}: {conflicts} sampled alongside "
                f"{DERIVE_NEUTRAL_VOLUME!r} — the derivation owns those "
                "fields (conflicting authorities); drop one side"
            )


def resolve_base_scenario(name_or_path: str) -> Path:
    """Find the base scenario YAML, preferring absolute paths then the library."""
    direct = Path(name_or_path)
    if direct.is_absolute() and direct.is_file():
        return direct
    in_library = LIBRARY_DIR / name_or_path
    if in_library.is_file():
        return in_library
    if direct.is_file():
        return direct.resolve()
    raise FileNotFoundError(
        f"base_scenario {name_or_path!r} not found "
        f"(looked in {LIBRARY_DIR} and as a relative/absolute path)"
    )


def set_dotted(tree: dict, path: str, value: Any) -> None:
    """Write `value` at the dotted `path` in `tree`, creating dicts as needed.

    Mutates in place. Refuses to descend into non-dicts so we don't
    silently overwrite a scalar mid-path — that would almost certainly
    be a typo in the sweep spec.
    """
    parts = path.split(".")
    cursor: Any = tree
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if nxt is None:
            nxt = {}
            cursor[part] = nxt
        elif not isinstance(nxt, dict):
            raise ValueError(
                f"path {path!r}: segment {part!r} is not a mapping in the base scenario"
            )
        cursor = nxt
    cursor[parts[-1]] = value


def draw_samples(spec: SweepSpec) -> np.ndarray:
    """Draw an LHS matrix of shape (n_samples, len(dimensions))."""
    sampler = qmc.LatinHypercube(d=len(spec.dimensions), seed=spec.seed)
    u = sampler.random(n=spec.n_samples)
    scaled = np.empty_like(u)
    for j, dim in enumerate(spec.dimensions):
        scaled[:, j] = dim.scale(u[:, j])
    return scaled


def jitter_hydrodynamics(
    spec: HydrodynamicsSpec, sigma: float, seed: int
) -> HydrodynamicsSpec:
    """Multiply each scalable coefficient by exp(N(0, sigma^2)).

    This is the §12.7 isotropic off-manifold knob, deliberately kept *out* of
    the deterministic `forward_map` and applied only here, for the FDI
    training distribution. Geometry-exact slots are left untouched: fin
    `area` stays exactly b*c, and the SDF-default stalls / a0 aren't jittered.
    So only added mass, hull linear damping, and per-fin cla/cda/alpha_stall
    move.
    """
    if sigma <= 0.0:
        return spec
    rng = np.random.default_rng(seed)

    def n() -> float:
        return math.exp(rng.normal(0.0, sigma))

    # The 12 body coefficients are exactly the float-valued fields on the
    # spec (the fins are sub-models, `knobs` is None), so we don't have to
    # re-list their names here.
    body = {
        name: getattr(spec, name) * n()
        for name in spec.model_fields
        if isinstance(getattr(spec, name), float)
    }

    def jit_fin(
        fin: FinAeroSpec, cla_n: float, cda_n: float, stall_n: float
    ) -> FinAeroSpec:
        return fin.model_copy(
            update={
                "cla": fin.cla * cla_n,
                "cda": fin.cda * cda_n,
                "alpha_stall": fin.alpha_stall * stall_n,
            }
        )

    # The horizontal fins are a mirror pair (identical in the canonical SDF
    # and in the deterministic map), so they share one draw per coefficient —
    # jitter must not invent a left/right asymmetry. The rudder draws its own.
    h_cla, h_cda, h_stall = n(), n(), n()
    return spec.model_copy(
        update={
            **body,
            "left_fin": jit_fin(spec.left_fin, h_cla, h_cda, h_stall),
            "right_fin": jit_fin(spec.right_fin, h_cla, h_cda, h_stall),
            "top_rudder": jit_fin(spec.top_rudder, n(), n(), n()),
        }
    )


def render_scenario(
    base: dict,
    spec: SweepSpec,
    row: Sequence[float],
    idx: int,
    anomaly_class: str | None = None,
    expected_duration_s: float | None = None,
) -> tuple[dict, dict | None, AnomalyAssignment | None]:
    """Deep-copy the base, overlay the row's perturbations, set per-run seed.

    `anomaly_class` is this run's stratified class assignment (None when
    the spec has no anomaly_mix) and `expected_duration_s` its drawn
    mission's optimistic duration (None without a mission_mix) — the
    scale a non-immediate fault onset resolves against. Returns
    (scenario, derived, anomaly) where `derived` is the per-run buoyancy
    derivation record for the manifest (None without derive dimensions)
    and `anomaly` is the resolved AnomalyAssignment (None without a mix).
    """
    scenario = copy.deepcopy(base)
    # Each run gets its own fault-RNG seed so MC outcomes are
    # decorrelated across samples while still being deterministic. Mix
    # the spec seed with the sample index to keep reproducibility.
    scenario["seed"] = (spec.seed * 1_000_003 + idx) & 0xFFFFFFFF

    # Launch dimensions (mission.*) are launch args and derive dimensions
    # (derive.*) are derivation targets — neither is a scenario field, so
    # drop both before writing anything (the schema's `extra="forbid"`
    # would reject them; their sampled values live in the manifest).
    scenario_dims = [
        (dim, value)
        for dim, value in zip(spec.dimensions, row)
        if not dim.path.startswith((MISSION_PREFIX, DERIVE_PREFIX))
    ]

    if spec.sampling_mode == "lhs":
        for dim, value in scenario_dims:
            # `emit` casts through Python int/float so numpy scalars don't
            # end up serialized as `!!python/object/apply`.
            set_dotted(scenario, dim.path, dim.emit(value))
    elif spec.sampling_mode == "physics":
        # Split the row: bare PhysicsKnobs names drive the deterministic
        # forward map; any dotted path is a plain scenario overlay, exactly as
        # in lhs mode. This lets one joint LHS perturb the plant geometry *and*
        # a scenario knob like the fault MTTF together.
        knobs: dict[str, float] = {}
        overlays: list[tuple[Dimension, float]] = []
        for dim, value in scenario_dims:
            if dim.path in _PHYSICS_KNOB_FIELDS:
                knobs[dim.path] = float(value)
            else:
                overlays.append((dim, value))
        hydro_spec = forward_map(knobs)
        if spec.isotropic_jitter_sigma > 0.0:
            hydro_spec = jitter_hydrodynamics(
                hydro_spec, spec.isotropic_jitter_sigma, seed=scenario["seed"]
            )
        scenario.setdefault("rig", {})["hydrodynamics"] = hydro_spec.model_dump()
        # Overlays go in *after* the hydrodynamics block so a dotted path
        # like rig.hydrodynamics.trim_mass_bow perturbs the freshly mapped
        # block instead of being clobbered by it.
        for dim, value in overlays:
            set_dotted(scenario, dim.path, dim.emit(value))
    else:
        raise ValueError(f"Unsupported sampling_mode: {spec.sampling_mode}")

    # Anomaly overlay runs after jitter + dotted overlays (so biofouling
    # multiplies the final nominal-sampled hydro block, and an anomalous
    # pump-transient band overrides the nominal draw of the same field)
    # and before the buoyancy derivation (so the fouled trim is derived,
    # not clobbered).
    assignment = None
    if spec.anomaly_mix is not None:
        if anomaly_class is None:
            raise ValueError("anomaly_mix configured but no class assigned")
        assignment = draw_assignment(
            spec.anomaly_mix,
            spec.seed,
            idx,
            anomaly_class,
            expected_duration_s=expected_duration_s,
        )
        apply_anomaly(scenario, assignment)

    # Correlated buoyancy derivation runs *after* the mode branch so all
    # overlays — including a sampled rig.hydrodynamics.fluid_density and
    # per-run bladder clamps — are already visible in the scenario.
    derived = None
    if spec.derivation_enabled:
        # anomaly.py owns class -> derivation coupling (biofouling's
        # mass shift + relaxed climb margin; identity otherwise).
        extra_mass_kg, min_climb_override = derivation_overrides(
            spec.anomaly_mix, assignment
        )
        derived = _apply_buoyancy_derivation(
            scenario,
            spec,
            row,
            idx,
            extra_mass_kg=extra_mass_kg,
            min_climb_margin_m3=min_climb_override,
        )

    return scenario, derived, assignment


def _apply_buoyancy_derivation(
    scenario: dict,
    spec: SweepSpec,
    row: Sequence[float],
    idx: int,
    extra_mass_kg: float = 0.0,
    min_climb_margin_m3: float | None = None,
) -> dict:
    """Derive trim masses/spawn for one run and write them into the scenario.

    Viability failures HARD-FAIL: the derivation makes every plant
    neutral-by-construction, so a failed margin means the sweep spec's
    bands are mis-authored (e.g. bladder clamps too tight for the
    sampled neutral-volume band). Abort generation — never resample,
    which would silently bias the sweep distribution.

    `extra_mass_kg` / `min_climb_margin_m3` are the biofouling hooks:
    fouling mass shifts the neutral-volume target (heavier hull needs
    more bladder), and the fouled run is checked against the relaxed
    climb margin authored in the anomaly mix — the HARD-FAIL contract
    itself is unchanged.
    """
    v_n_target = next(
        float(value)
        for dim, value in zip(spec.dimensions, row)
        if dim.path == DERIVE_NEUTRAL_VOLUME
    )
    cfg = spec.buoyancy_derivation

    # Validate the overlaid rig subtree through the spec layer, so the
    # effective values here (defaults included) are exactly what the
    # container's load_scenario will hand the launch.
    rig = RigScenario.model_validate(scenario.get("rig") or {})
    if rig.hydrodynamics is None:
        raise ValueError(
            "buoyancy derivation requires physics mode or a base scenario "
            "with rig.hydrodynamics (the derived trim masses and the "
            "fluid_density they balance against live in that block)"
        )
    fluid_density = rig.hydrodynamics.fluid_density
    bladder_min = rig.plant.bladder_min_m3
    bladder_max = rig.plant.bladder_max_m3
    if extra_mass_kg:
        v_n_target = fouled_neutral_volume(v_n_target, extra_mass_kg, fluid_density)

    derived = buoyancy.derive_trim_masses(
        fluid_density,
        v_n_target,
        spawn_volume_m3=bladder_max if cfg.spawn_policy == "bladder_max" else None,
        spawn_offset_m3=cfg.spawn_offset_m3,
        trim_bladder=rig.hydrodynamics.trim_mass_bladder,
    )

    for path, attr in _DERIVED_FIELDS.items():
        set_dotted(scenario, path, getattr(derived, attr))

    viability = buoyancy.check_viability(
        derived,
        bladder_min,
        bladder_max,
        min_dive_margin_m3=cfg.min_dive_margin_m3,
        min_climb_margin_m3=(
            cfg.min_climb_margin_m3
            if min_climb_margin_m3 is None
            else min_climb_margin_m3
        ),
    )
    if not viability.ok:
        raise RuntimeError(
            f"run {idx}: derived plant not oscillation-viable: "
            f"{'; '.join(viability.reasons)} "
            f"[fluid_density={fluid_density:.4f}, "
            f"neutral_volume_m3={v_n_target:.6e}, "
            f"spawn={derived.bladder_spawn_volume_m3:.6e}, "
            f"bladder_min={bladder_min:.6e}, bladder_max={bladder_max:.6e}, "
            f"dive_margin={viability.dive_margin_m3:.6e}, "
            f"climb_margin={viability.climb_margin_m3:.6e}] "
            "HARD-FAIL: the sweep spec's bands are mis-authored; aborting "
            "generation (never resampling)"
        )

    return {
        **asdict(derived),
        "fluid_density": fluid_density,
        "dive_margin_m3": viability.dive_margin_m3,
        "climb_margin_m3": viability.climb_margin_m3,
    }


def write_manifest(
    out_dir: Path,
    spec: SweepSpec,
    spec_path: Path,
    matrix: np.ndarray,
    run_ids: Iterable[str],
    derived_records: Sequence[dict | None],
    anomaly_records: Sequence[AnomalyAssignment | None],
    mission_records: Sequence[MissionAssignment | None] | None = None,
) -> None:
    if mission_records is None:
        mission_records = [None] * len(anomaly_records)
    spec_text = spec_path.read_text()
    manifest = {
        "sampler_version": SAMPLER_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": hashlib.sha256(spec_text.encode()).hexdigest(),
        "n_samples": spec.n_samples,
        "seed": spec.seed,
        "base_scenario": spec.base_scenario,
        "dimensions": [
            {
                "path": d.path,
                "distribution": d.distribution,
                "low": d.low,
                "high": d.high,
                "integer": d.integer,
            }
            for d in spec.dimensions
        ],
        "samples": [
            {
                "run_id": rid,
                # A mission_mix's drawn mission.* values merge into the
                # same flat values dict as the LHS dimensions, so
                # run_sweep.load_mission_args picks them up unchanged.
                "values": {
                    **{d.path: d.emit(v) for d, v in zip(spec.dimensions, row)},
                    **(mission.values if mission is not None else {}),
                },
                **({"derived": derived} if derived is not None else {}),
                # Explicit per-run label, nominal included — downstream
                # dataset builders join on this, never on absence.
                **({"anomaly": anomaly.record()} if anomaly is not None else {}),
                **({"mission": mission.record()} if mission is not None else {}),
            }
            for rid, row, derived, anomaly, mission in zip(
                run_ids, matrix, derived_records, anomaly_records, mission_records
            )
        ],
    }
    if spec.derivation_enabled:
        # Echo the *effective* derivation config (defaults filled in) so
        # post-hoc analyses don't have to reconstruct it from the spec.
        manifest["buoyancy_derivation"] = asdict(spec.buoyancy_derivation)
    if spec.anomaly_mix is not None:
        manifest["anomaly_mix"] = spec.anomaly_mix.model_dump()
    if spec.mission_mix is not None:
        manifest["mission_mix"] = spec.mission_mix.model_dump()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--spec", required=True, type=Path, help="Sweep spec YAML.")
    ap.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output root; YAMLs land in <out>/<name>/.",
    )
    ap.add_argument("--name", help="Sweep name (defaults to the spec file's stem).")
    ap.add_argument(
        "--n-samples",
        type=int,
        help="Override the spec's n_samples (dry runs / smoke checks).",
    )
    args = ap.parse_args(argv)

    spec = SweepSpec.load(args.spec)
    if args.n_samples is not None:
        spec = replace(spec, n_samples=args.n_samples)
    base_path = resolve_base_scenario(spec.base_scenario)
    base = yaml.safe_load(base_path.read_text()) or {}

    name = args.name or args.spec.stem
    out_dir = args.out / name
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = draw_samples(spec)
    width = max(4, len(str(spec.n_samples - 1)))
    run_ids = [f"lhs_{i:0{width}d}" for i in range(spec.n_samples)]

    # Stratified per-run class/profile assignment (exact counts, seeded
    # shuffles) — resolved before rendering so a truncated --n-samples
    # dry run still splits its n exactly by the mix weights.
    classes = (
        assign_classes(spec.anomaly_mix, spec.seed, spec.n_samples)
        if spec.anomaly_mix is not None
        else None
    )
    profiles = (
        assign_profiles(spec.mission_mix, spec.seed, spec.n_samples)
        if spec.mission_mix is not None
        else None
    )

    derived_records: list[dict | None] = []
    anomaly_records: list[AnomalyAssignment | None] = []
    mission_records: list[MissionAssignment | None] = []
    for idx, (run_id, row) in enumerate(zip(run_ids, matrix)):
        mission = (
            draw_mission(spec.mission_mix, spec.seed, idx, profiles[idx])
            if profiles is not None
            else None
        )
        scenario, derived, anomaly = render_scenario(
            base,
            spec,
            row,
            idx,
            anomaly_class=classes[idx] if classes is not None else None,
            expected_duration_s=(
                expected_mission_duration_s(mission.values)
                if mission is not None
                else None
            ),
        )
        derived_records.append(derived)
        anomaly_records.append(anomaly)
        mission_records.append(mission)
        out_path = out_dir / f"{run_id}.yaml"
        out_path.write_text(yaml.safe_dump(scenario, sort_keys=False))

    write_manifest(
        out_dir,
        spec,
        args.spec,
        matrix,
        run_ids,
        derived_records,
        anomaly_records,
        mission_records,
    )

    if profiles is not None:
        print(f"mission mix: {dict(Counter(profiles))}")
    if classes is not None:
        print(f"anomaly mix: {dict(Counter(classes))}")
    print(f"wrote {spec.n_samples} scenarios + manifest.json to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
