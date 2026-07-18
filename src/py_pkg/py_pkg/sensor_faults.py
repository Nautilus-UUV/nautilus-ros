"""Persistent sensor/comms fault models for the HAL sim bridges.

Pure logic, no ROS — same split as ``sensor_noise.py``: the bridges in
``nautilus_hal`` compose these per channel and the math gets Tier 1
coverage without a Gazebo environment.

All faults here are *persistent whole-run archetypes at constant
severity*, active from the first sample (they replace the old
Poisson-MTTF escalation ladder). Three pieces:

- :class:`FaultyChannel` — one value-fault archetype (bias / drift /
  stuck) composed with the channel's calibrated
  :class:`~py_pkg.sensor_noise.GaussianQuantizedNoise` chain. bias and
  drift corrupt the *true* value before noise + quantization (a
  transducer-stage fault; the digitizer always runs last); stuck
  latches the first *reported* post-chain value, so a frozen channel
  repeats a constant on-comb readout, exactly like a stuck ADC.
- :class:`MessageDrop` — per-message Bernoulli drop gate. Sensor
  ``dropout`` uses one on a single stream; the comms fault uses one per
  bridge across every bridged publish.
- :func:`gate_publisher` — wraps a publisher so the gate is consulted
  inside ``publish``; returns the *original* publisher object when the
  gate is inactive, so the nominal path provably has no interception
  layer.

Latency injection is deliberately out of scope: it needs per-topic
buffering plus delayed one-shot timers and adds nondeterministic
ordering — drop probability and the existing publish_rate_hz knobs
span the FDI-relevant comms space.

Determinism contract: same seeds (via ``rng_from_seed``) => identical
per-sample-index fault realizations. Inactive models never consume RNG,
so a nominal run's random streams are bit-identical to a build without
this module in the loop.
"""

from __future__ import annotations

import random

from py_pkg.sensor_noise import GaussianQuantizedNoise

# Value-fault archetypes FaultyChannel implements. "dropout" is a
# transport fault, not a value fault — bridges map it to a MessageDrop.
VALUE_FAULT_KINDS = ("none", "bias", "drift", "stuck")


class FaultyChannel:
    """One persistent value-fault archetype on one sensor channel.

    kind semantics (channel-native units, e.g. Pa for pressure):
      "none"  -> exact delegation to the noise chain (zero extra work)
      "bias"  -> `magnitude` is a signed additive offset
      "drift" -> `magnitude` is a signed ramp rate per second; the epoch
                 is latched on the first sample (bridge start ~= t=0)
      "stuck" -> the first reported (post-noise, on-comb) value is
                 latched and returned forever; no RNG after the latch

    `t_s` is caller-supplied node-clock seconds — this class never
    reads a clock.
    """

    def __init__(
        self,
        noise: GaussianQuantizedNoise,
        kind: str = "none",
        magnitude: float = 0.0,
    ) -> None:
        if kind not in VALUE_FAULT_KINDS:
            raise ValueError(
                f"kind must be one of {VALUE_FAULT_KINDS}, got {kind!r}"
                " (dropout is a MessageDrop, not a value fault)"
            )
        self.noise = noise
        self.kind = kind
        self.magnitude = float(magnitude)
        self.is_active = kind != "none"
        self._t0_s: float | None = None
        self._stuck_value: float | None = None

    def sample(self, value: float, t_s: float) -> float:
        """Faulted + noise-chained reading for the true `value` at `t_s`."""
        if not self.is_active:
            return self.noise.apply(value)
        if self.kind == "stuck":
            if self._stuck_value is None:
                self._stuck_value = self.noise.apply(float(value))
            return self._stuck_value
        v = float(value)
        if self.kind == "bias":
            v += self.magnitude
        else:  # drift
            if self._t0_s is None:
                self._t0_s = t_s
            v += self.magnitude * (t_s - self._t0_s)
        return self.noise.apply(v)

    def sample_int(self, value: int, t_s: float) -> int:
        """Integer-channel variant (e.g. Int32 pressure in Pa)."""
        if not self.is_active:
            return self.noise.apply_int(value)
        return int(round(self.sample(float(value), t_s)))


class MessageDrop:
    """Persistent per-message Bernoulli drop gate.

    `p <= 0` is inactive: `should_drop()` returns False without
    consuming RNG, keeping nominal random streams untouched.
    """

    def __init__(self, p: float = 0.0, rng: random.Random | None = None) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"drop probability must be in [0, 1], got {p}")
        self.p = float(p)
        self.rng = rng if rng is not None else random.Random()
        self.is_active = self.p > 0.0

    def should_drop(self) -> bool:
        if not self.is_active:
            return False
        return self.rng.random() < self.p


class GatedPublisher:
    """Duck-typed publisher wrapper: the drop gate lives inside publish().

    Call sites stay `pub.publish(msg)` — whether a message reaches the
    wire is decided here, one independent Bernoulli draw per message.
    """

    def __init__(self, inner, drop: MessageDrop) -> None:
        self._inner = inner
        self._drop = drop

    def publish(self, msg) -> None:
        if not self._drop.should_drop():
            self._inner.publish(msg)

    @property
    def topic_name(self):
        return self._inner.topic_name


def gate_publisher(pub, drop: MessageDrop):
    """Wrap `pub` behind `drop`; identity when the gate is inactive.

    Returning the original object for an inactive gate is the nominal-
    path guarantee: with default (no-fault) parameters there is no
    interception layer at all.
    """
    return GatedPublisher(pub, drop) if drop.is_active else pub
