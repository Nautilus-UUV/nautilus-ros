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
manifest.json only

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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src/py_pkg"))
from py_pkg.scenarios.compile import forward_map
from py_pkg.scenarios.spec.rig import FinAeroSpec, HydrodynamicsSpec, PhysicsKnobs

SAMPLER_VERSION = "1.1.0"

# Dimensions under this prefix are launch args (mission knobs), not scenario
# fields: they are recorded in the manifest but never written into the
# scenario YAML, whose schema (`extra="forbid"`) would reject them.
MISSION_PREFIX = "mission."

# The bare-name dimensions a "physics" sweep feeds to the deterministic
# forward map. Any dimension whose path is *not* one of these is treated as a
# plain scenario dot-path overlay (e.g. rig.faults.bcu_rpm.mttf_sec), so a
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
        if not (high > low):
            raise ValueError(
                f"dimension {raw['path']!r}: high ({high}) must be > low ({low})"
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


@dataclass(frozen=True)
class SweepSpec:
    description: str
    n_samples: int
    seed: int
    base_scenario: str
    dimensions: tuple[Dimension, ...]
    sampling_mode: str = "lhs"
    isotropic_jitter_sigma: float = 0.0

    @classmethod
    def load(cls, path: Path) -> "SweepSpec":
        raw = yaml.safe_load(path.read_text())
        try:
            dims = tuple(Dimension.from_dict(d) for d in raw["dimensions"])
        except KeyError as e:
            raise ValueError(f"{path}: sweep spec missing 'dimensions'") from e
        if not dims:
            raise ValueError(f"{path}: 'dimensions' must be non-empty")
        return cls(
            description=raw.get("description", ""),
            n_samples=int(raw["n_samples"]),
            seed=int(raw.get("seed", 0)),
            base_scenario=raw["base_scenario"],
            dimensions=dims,
            sampling_mode=raw.get("sampling_mode", "lhs"),
            isotropic_jitter_sigma=float(raw.get("isotropic_jitter_sigma", 0.0)),
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
    base: dict, spec: SweepSpec, row: Sequence[float], idx: int
) -> dict:
    """Deep-copy the base, overlay the row's perturbations, set per-run seed."""
    scenario = copy.deepcopy(base)
    # Each run gets its own fault-RNG seed so MC outcomes are
    # decorrelated across samples while still being deterministic. Mix
    # the spec seed with the sample index to keep reproducibility.
    scenario["seed"] = (spec.seed * 1_000_003 + idx) & 0xFFFFFFFF

    # Launch dimensions (mission.*) are launch args, not scenario fields —
    # they live in the manifest only, so drop them before writing anything.
    scenario_dims = [
        (dim, value)
        for dim, value in zip(spec.dimensions, row)
        if not dim.path.startswith(MISSION_PREFIX)
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

    return scenario


def write_manifest(
    out_dir: Path,
    spec: SweepSpec,
    spec_path: Path,
    matrix: np.ndarray,
    run_ids: Iterable[str],
) -> None:
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
                "values": {d.path: d.emit(v) for d, v in zip(spec.dimensions, row)},
            }
            for rid, row in zip(run_ids, matrix)
        ],
    }
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
    args = ap.parse_args(argv)

    spec = SweepSpec.load(args.spec)
    base_path = resolve_base_scenario(spec.base_scenario)
    base = yaml.safe_load(base_path.read_text()) or {}

    name = args.name or args.spec.stem
    out_dir = args.out / name
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = draw_samples(spec)
    width = max(4, len(str(spec.n_samples - 1)))
    run_ids = [f"lhs_{i:0{width}d}" for i in range(spec.n_samples)]

    for idx, (run_id, row) in enumerate(zip(run_ids, matrix)):
        scenario = render_scenario(base, spec, row, idx)
        out_path = out_dir / f"{run_id}.yaml"
        out_path.write_text(yaml.safe_dump(scenario, sort_keys=False))

    write_manifest(out_dir, spec, args.spec, matrix, run_ids)

    print(f"wrote {spec.n_samples} scenarios + manifest.json to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
