"""Tier 1 for the persistent sensor/comms fault models (sensor_faults.py).

Every behavior claim of the fault design maps to a test here:
- nominal path is bit-identical (values AND RNG state untouched),
- bias/drift are exact transducer-stage math with a latched drift epoch,
- stuck latches the first *reported* on-comb value and stops drawing RNG,
- MessageDrop is seeded-reproducible with the advertised rate,
- gate_publisher is an identity when inactive.
"""

from __future__ import annotations

import random

import pytest
from py_pkg.sensor_faults import (
    VALUE_FAULT_KINDS,
    FaultyChannel,
    GatedPublisher,
    MessageDrop,
    gate_publisher,
)
from py_pkg.sensor_noise import GaussianQuantizedNoise


def test_value_fault_kinds_mirror_the_schema():
    # sensor_faults stays pydantic-free on purpose, so its vocabulary is
    # spelled out — this locks it to the SensorFaultKind schema (every
    # kind except the transport-level "dropout" is a value fault here).
    from typing import get_args

    from py_pkg.scenarios.spec.rig import SensorFaultKind

    assert set(VALUE_FAULT_KINDS) | {"dropout"} == set(get_args(SensorFaultKind))


def _noise(sigma=0.0, step=0.0, seed=None):
    rng = random.Random(seed) if seed is not None else None
    return GaussianQuantizedNoise(sigma=sigma, quantization_step=step, rng=rng)


# ---------------------------------------------------------------------------
# Nominal path
# ---------------------------------------------------------------------------


def test_none_kind_bitwise_delegates_to_noise():
    bare = _noise(sigma=353.0, step=600.0, seed=11)
    wrapped = FaultyChannel(_noise(sigma=353.0, step=600.0, seed=11), kind="none")
    assert not wrapped.is_active
    for i in range(1000):
        v = 100_000.0 + 7.3 * i
        assert wrapped.sample(v, t_s=0.1 * i) == bare.apply(v)
    # Same RNG state afterwards: the fault layer consumed zero draws.
    assert wrapped.noise.rng.random() == bare.rng.random()


def test_none_kind_int_passthrough_is_bit_identical_when_noise_off():
    chan = FaultyChannel(_noise(), kind="none")
    for v in (0, 101_325, -5):
        assert chan.sample_int(v, t_s=3.0) is v or chan.sample_int(v, t_s=3.0) == v
        assert chan.sample(float(v), t_s=3.0) == float(v)


# ---------------------------------------------------------------------------
# Value archetypes
# ---------------------------------------------------------------------------


def test_bias_exact_math_with_noise_off():
    chan = FaultyChannel(_noise(), kind="bias", magnitude=5000.0)
    assert chan.sample(100_000.0, t_s=0.0) == 105_000.0
    assert chan.sample_int(100_000, t_s=99.0) == 105_000
    neg = FaultyChannel(_noise(), kind="bias", magnitude=-250.5)
    assert neg.sample(1000.0, t_s=1.0) == 749.5


def test_drift_latches_epoch_and_is_linear():
    chan = FaultyChannel(_noise(), kind="drift", magnitude=10.0)
    # First sample defines t0 — no accumulated drift yet.
    assert chan.sample(50_000.0, t_s=100.0) == 50_000.0
    assert chan.sample(50_000.0, t_s=100.0 + 30.0) == 50_000.0 + 10.0 * 30.0
    assert chan.sample_int(50_000, t_s=100.0 + 62.5) == 50_000 + 625


def test_drift_applies_before_quantization():
    chan = FaultyChannel(_noise(step=100.0), kind="drift", magnitude=1.0)
    chan.sample(0.0, t_s=0.0)  # latch epoch
    # 40 s of 1 Pa/s drift = +40 Pa -> quantizes back onto the comb (0),
    # 60 s -> rounds up to 100: the ramp emerges through the digitizer.
    assert chan.sample(0.0, t_s=40.0) == 0.0
    assert chan.sample(0.0, t_s=60.0) == 100.0


def test_stuck_latches_first_reported_value_on_comb():
    chan = FaultyChannel(_noise(sigma=353.0, step=600.0, seed=7), kind="stuck")
    first = chan.sample(120_000.0, t_s=0.0)
    assert first % 600.0 == 0.0  # latched value is post-chain, on the comb
    state_after_latch = chan.noise.rng.getstate()
    for i in range(200):
        assert chan.sample(120_000.0 + 50.0 * i, t_s=float(i)) == first
        assert chan.sample_int(130_000 + i, t_s=float(i)) == int(first)
    # No RNG draws after the latch.
    assert chan.noise.rng.getstate() == state_after_latch


def test_seeded_determinism_of_stuck_and_noise_chain():
    a = FaultyChannel(_noise(sigma=353.0, step=600.0, seed=5), kind="stuck")
    b = FaultyChannel(_noise(sigma=353.0, step=600.0, seed=5), kind="stuck")
    assert a.sample(111_111.0, t_s=0.0) == b.sample(111_111.0, t_s=0.0)


def test_invalid_kind_raises():
    with pytest.raises(ValueError):
        FaultyChannel(_noise(), kind="dropout")  # transport fault, not value
    with pytest.raises(ValueError):
        FaultyChannel(_noise(), kind="spikes")


# ---------------------------------------------------------------------------
# MessageDrop
# ---------------------------------------------------------------------------


def test_message_drop_inactive_never_drops_and_never_draws():
    gate = MessageDrop(p=0.0, rng=random.Random(3))
    state = gate.rng.getstate()
    assert not any(gate.should_drop() for _ in range(1000))
    assert gate.rng.getstate() == state


def test_message_drop_p1_always_drops():
    gate = MessageDrop(p=1.0, rng=random.Random(3))
    assert all(gate.should_drop() for _ in range(1000))


def test_message_drop_seeded_pattern_and_rate():
    a = MessageDrop(p=0.3, rng=random.Random(21))
    b = MessageDrop(p=0.3, rng=random.Random(21))
    pattern_a = [a.should_drop() for _ in range(10_000)]
    pattern_b = [b.should_drop() for _ in range(10_000)]
    assert pattern_a == pattern_b  # reproducible
    rate = sum(pattern_a) / len(pattern_a)
    assert abs(rate - 0.3) < 0.02  # ~4.3 binomial sigmas


@pytest.mark.parametrize("bad", [-0.1, 1.0001, 2.0])
def test_message_drop_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        MessageDrop(p=bad)


# ---------------------------------------------------------------------------
# gate_publisher
# ---------------------------------------------------------------------------


class _StubPublisher:
    def __init__(self):
        self.sent = []
        self.topic_name = "/stub"

    def publish(self, msg):
        self.sent.append(msg)


def test_gate_publisher_identity_when_inactive():
    pub = _StubPublisher()
    assert gate_publisher(pub, MessageDrop(p=0.0)) is pub


def test_gate_publisher_drops_per_message_reproducibly():
    def run(seed):
        pub = _StubPublisher()
        gated = gate_publisher(pub, MessageDrop(p=0.5, rng=random.Random(seed)))
        assert isinstance(gated, GatedPublisher)
        assert gated.topic_name == "/stub"
        for i in range(200):
            gated.publish(i)
        return pub.sent

    assert run(9) == run(9)
    sent = run(9)
    assert 0 < len(sent) < 200  # some dropped, some through
    # Messages that do arrive are unmodified and in order.
    assert sent == sorted(sent)
