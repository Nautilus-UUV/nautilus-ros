# Hydrodynamic Forward-Map Refactor & Fixes

> Status: implemented 2026-06-01. All five steps below landed; Tier 1 gates
> pass (8/8 in `test/unit/test_forward_map.py`) and the sampler dry-run is
> green. This doc is kept as the rationale record.

## Context

The hydrodynamic-coefficient sampling pipeline (`doc/hydrodynamic_coefficient_sampling_plan.md`)
was fully working — `forward_map` turned the physics knobs into an SDF
hydrodynamic surface, the render path was opt-in, and the CI gates passed —
but carried implementation-level debt that would bite during the v2 ACU /
full-glide extensions:

- The closed-form → spec bridge listed the slot names **three times** (the
  `_compute_closed_form` return dict, the `can_dict` in
  `_compute_calibration_scalars`, and the `HydrodynamicsSpec(...)` constructor),
  with a `horiz`/`vert` vs `left_fin`/`right_fin`/`top_rudder` naming mismatch
  in between. A typo in any copy failed silently or as a deep `KeyError`.
- `forward_map` took a raw `dict[str, float]`, so a bad knob key exploded
  mid-physics instead of at a validation boundary.
- The §12.7 isotropic jitter lived *inside* `forward_map` and hit every slot
  including `area` (which must stay exactly `λ·b·c`), coupling a sampler-only
  noise model into the function the deterministic CI gates call.
- Smaller: a wrong return annotation, an over-broad `except`, a magic `z_r`
  literal mid-physics, import-time file IO + log spew, a 100-line monolith
  function, a slow 2^16 test, two missing gates (G3, G5), and a design doc that
  no longer matched the code (15 vs 16 knobs, stale lever arms, relaxed band).

Outcome: one declarative slot registry, a typed knob boundary, a deterministic
`forward_map`, the full §14 gate set, and a doc that matches reality — with no
change to any rendered SDF at nominal (identity holds to ~1e-14).

## What changed

### compile.py
- `_BODY_SLOTS` (names match `HydrodynamicsSpec` fields) + `_FIN_SLOTS`
  (closed-form key → canonical reader) are the single source of truth;
  `_canonical_values` and the spec builder both drive off them. An import-time
  assert in `_calibration` fails loudly on any closed-form/registry key drift.
- `_compute_closed_form` takes a typed `PhysicsKnobs` and is split into §3–§8
  helpers (`_lamb_factors`, `_hull_added_mass`, `_hull_damping`,
  `_fin_added_mass`, `_fin_lift_slope`, `_fin_profile_drag`).
- `forward_map(knobs: PhysicsKnobs | dict)` coerces + validates at the
  boundary, is deterministic (jitter removed), and builds the spec via
  `_spec_from_slots` (the one place the horiz→Left/Right, vert→TopRudder
  fan-out lives).
- Calibration is lazy via `functools.lru_cache` with a public
  `calibration_scalars()` accessor; no import-time IO/logging. `_load_nominal_data`
  narrows its `except` to `(ImportError, LookupError)`. `z_r` moved to a
  `structural_constants` block in `nominal_knobs.yaml`. Degenerate `L<=D`
  warns instead of silently zeroing.

### lhs_sample.py
- `jitter_hydrodynamics(spec, sigma, seed)` is the §12.7 noise, applied only
  in the sampler. It skips `area` (stays `λ·b·c`) and shares one draw per
  coefficient across the horizontal mirror pair (no invented left/right
  asymmetry). `render_scenario` calls deterministic `forward_map` then this.

### Tests (`test/unit/test_forward_map.py`)
- G1 identity, G2 band (`< 300`, justified by `drag_xU ≈ 229`), sign sanity,
  G3 no-double-counting, the full per-knob single-knob sweep (APPEARS +
  SIGN tables from §11, with float-cancellation tolerance), the §14.5
  round-trip on the monomial subset, and a sub-sampled (2000-corner)
  hyperbox check.

### Doc
- `doc/hydrodynamic_coefficient_sampling_plan.md` reconciled: 16 knobs,
  `alpha_stall_horiz`/`alpha_stall_rudder` split, `x_f=0.70`/`x_r=0.939`,
  scalar band `< 300` with rationale, Saltelli budget at k=16, and the §12.7
  note that `area` and pair-symmetry are preserved under jitter.

## Verification
```bash
cd src/nautilus-ros/src/py_pkg
/usr/bin/python3 -m pytest test/unit/test_forward_map.py -v   # 8/8 pass
```
- Identity holds (worst abs err ~1.4e-14) → no rendered SDF changes at nominal.
- Sampler dry-run on `scripts/sweeps/bcu_hydro_coarse.yaml`: 16 dims, physics
  mode, `area == λ·b·c` exactly under jitter, `left_fin == right_fin`.
- Tier 3 (`pytest -m sim test/sim/test_forward_map_sim.py`) needs a built +
  sourced workspace; jitter-arg call site updated.
