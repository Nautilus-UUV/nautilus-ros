"""Stratified, seeded assignment of a categorical vocabulary over N runs.

The one implementation behind every per-run mix a sweep draws
(``anomaly.py``'s class mix, ``mission_mix.py``'s profile mix): exact
largest-remainder counts plus a permutation on a dedicated derive_seed
stream. Determinism-critical — the tie-break and the shuffle define what
a seed produces, so they live here once rather than once per mix.

Vocabularies are ORDER-LOCKED tuples: the order feeds both the
largest-remainder tie-break (the first member takes the tie) and the
seeded shuffle, so reordering one silently rewrites every sweep drawn
from that mix.
"""

from __future__ import annotations

import random

from .seed import derive_seed


def check_weights(
    weights: dict[str, float],
    vocabulary: tuple[str, ...],
    *,
    block: str,
    noun: str,
) -> None:
    """Validate a mix's weight map: known keys, non-negative, sums to 1.

    ``block`` names the authoring block (``"anomaly_mix"``,
    ``"mission_mix"``, ``"onset"``) and ``noun`` what its keys are
    (``"classes"``, ``"profiles"``, ``"shapes"``), so a bad YAML still
    names the block it came from and what it should have held.
    """
    unknown = set(weights) - set(vocabulary)
    if unknown:
        raise ValueError(f"{block}: unknown {noun} {sorted(unknown)}")
    if any(w < 0 for w in weights.values()):
        raise ValueError(f"{block}: weights must be >= 0")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"{block}: weights must sum to 1, got {total}")


def stratified_counts(
    weights: dict[str, float], vocabulary: tuple[str, ...], n: int
) -> dict[str, int]:
    """Largest-remainder rounding of ``weights * n`` to exact counts.

    Ties break in fixed ``vocabulary`` order, so the split is a pure
    function of (weights, n) and the counts always sum to ``n``.
    """
    quotas = {k: weights.get(k, 0.0) * n for k in vocabulary}
    counts = {k: int(quotas[k]) for k in vocabulary}
    leftover = n - sum(counts.values())
    # sorted() is stable over the canonical tuple, so equal remainders
    # keep `vocabulary` order without an explicit tie-break key.
    by_remainder = sorted(vocabulary, key=lambda k: -(quotas[k] - counts[k]))
    for k in by_remainder[:leftover]:
        counts[k] += 1
    return counts


def stratified_assignment(
    weights: dict[str, float],
    vocabulary: tuple[str, ...],
    parent_seed: int,
    stream: str,
    n: int,
) -> list[str]:
    """Exact stratified counts, shuffled on the ``stream`` derive_seed child.

    A dedicated stream per mix is what decouples the mixes from each
    other and from the LHS matrix: editing one mix never reshuffles
    another.
    """
    counts = stratified_counts(weights, vocabulary, n)
    vector = [k for k in vocabulary for _ in range(counts[k])]
    random.Random(derive_seed(parent_seed, stream)).shuffle(vector)
    return vector
