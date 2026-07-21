"""Persistent sensor/comms fault models for the HAL sim bridges.

Pure logic, no ROS — same split as ``sensor_noise.py``: the bridges in
``nautilus_hal`` compose these per channel and the math gets Tier 1
coverage without a Gazebo environment.

Faults here are *persistent whole-run archetypes at constant drawn
severity* (they replace the old Poisson-MTTF escalation ladder). An
optional :class:`FaultSchedule` gates WHEN the archetype is felt — a
deterministic envelope ``m(t) in [0, 1]`` giving the fault an onset and
a shape (step / ramp / intermittent); the default schedule is
``m(t) == 1`` from the first sample, i.e. the original whole-run
behavior. Severity itself never changes mid-run — ``m(t)`` scales the
one drawn magnitude. Pieces:

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

# Envelope shapes FaultSchedule implements.
FAULT_SCHEDULE_SHAPES = ("step", "ramp", "intermittent")


class FaultSchedule:
    """Deterministic onset/progression envelope ``m(t)`` for one fault.

    Maps caller-supplied node-clock seconds to a severity multiplier in
    ``[0, 1]`` that scales the fault's one drawn magnitude:

      "step"         -> 0 before ``onset_s`` (relative to the epoch),
                        1 after — the whole-run default when
                        ``onset_s == 0``.
      "ramp"         -> 0 before onset, linear to 1 over ``ramp_s``,
                        then 1.
      "intermittent" -> after onset, 1 during the first
                        ``duty * period_s`` of each period, else 0.

    The epoch is latched by :meth:`start` (bridges call it in setup) or
    lazily on the first :meth:`multiplier` call. No RNG, no clock of its
    own — same determinism contract as :class:`FaultyChannel`.
    """

    def __init__(
        self,
        onset_s: float = 0.0,
        shape: str = "step",
        ramp_s: float = 0.0,
        period_s: float = 0.0,
        duty: float = 0.5,
    ) -> None:
        if shape not in FAULT_SCHEDULE_SHAPES:
            raise ValueError(
                f"shape must be one of {FAULT_SCHEDULE_SHAPES}, got {shape!r}"
            )
        if onset_s < 0.0:
            raise ValueError(f"onset_s must be >= 0, got {onset_s}")
        self.onset_s = float(onset_s)
        self.shape = shape
        self.ramp_s = float(ramp_s)
        self.period_s = float(period_s)
        self.duty = float(duty)
        self._epoch_s: float | None = None

    def start(self, t_s: float) -> None:
        """Latch the epoch the onset counts from."""
        self._epoch_s = float(t_s)

    def multiplier(self, t_s: float) -> float:
        """Severity multiplier at ``t_s`` (epoch latched on first call)."""
        if self._epoch_s is None:
            self._epoch_s = float(t_s)
        rel = t_s - self._epoch_s - self.onset_s
        if rel < 0.0:
            return 0.0
        if self.shape == "ramp" and self.ramp_s > 0.0:
            return min(1.0, rel / self.ramp_s)
        if self.shape == "intermittent" and self.period_s > 0.0:
            return 1.0 if (rel % self.period_s) < self.duty * self.period_s else 0.0
        return 1.0

    def blend(self, healthy: float, severity: float, t_s: float) -> float:
        """Lerp from ``healthy`` to ``severity`` by ``m(t)``.

        The one law for *applying* the envelope, so every consumer reads
        the same way round: a bias is ``blend(0.0, magnitude, t)``, a drop
        probability ``blend(0.0, p, t)``, and a pump's effectiveness
        ``blend(1.0, effectiveness, t)`` — whose healthy value is 1.0, the
        only reason that site looks inverted.
        """
        return healthy + self.multiplier(t_s) * (severity - healthy)

    def active_elapsed(self, t_s: float) -> float:
        """Seconds since the onset fired (0 before). Step-shape helper:
        drift channels accumulate at ``magnitude * active_elapsed`` so the
        drift epoch moves to the onset (drift is step-only by spec)."""
        if self._epoch_s is None:
            self._epoch_s = float(t_s)
        return max(0.0, t_s - self._epoch_s - self.onset_s)


class FaultyChannel:
    """One persistent value-fault archetype on one sensor channel.

    kind semantics (channel-native units, e.g. Pa for pressure):
      "none"  -> exact delegation to the noise chain (zero extra work)
      "bias"  -> `magnitude` is a signed additive offset
      "drift" -> `magnitude` is a signed ramp rate per second; the epoch
                 is latched on the first sample (bridge start ~= t=0)
      "stuck" -> the first reported (post-noise, on-comb) value is
                 latched and returned forever; no RNG after the latch

    An optional `schedule` gates the archetype in time: bias scales by
    `m(t)`; drift accumulates from the schedule's onset instead of the
    first sample (step-only by spec); stuck behaves normally until the
    onset fires, then latches the first post-onset reading. Omitting it
    installs the default `FaultSchedule()` — `m(t) == 1` from the first
    sample, with the onset-relative drift epoch collapsing onto that same
    sample — i.e. exactly the original whole-run behavior, so there is
    only ever one code path.

    `t_s` is caller-supplied node-clock seconds — this class never
    reads a clock.
    """

    def __init__(
        self,
        noise: GaussianQuantizedNoise,
        kind: str = "none",
        magnitude: float = 0.0,
        schedule: FaultSchedule | None = None,
    ) -> None:
        if kind not in VALUE_FAULT_KINDS:
            raise ValueError(
                f"kind must be one of {VALUE_FAULT_KINDS}, got {kind!r}"
                " (dropout is a MessageDrop, not a value fault)"
            )
        self.noise = noise
        self.kind = kind
        self.magnitude = float(magnitude)
        # The inert default IS the unscheduled behavior, so `sample` never
        # has to branch on absence.
        self.schedule = schedule if schedule is not None else FaultSchedule()
        self.is_active = kind != "none"
        self._stuck_value: float | None = None

    def sample(self, value: float, t_s: float) -> float:
        """Faulted + noise-chained reading for the true `value` at `t_s`."""
        if not self.is_active:
            return self.noise.apply(value)
        if self.kind == "stuck":
            if self._stuck_value is None:
                if self.schedule.multiplier(t_s) <= 0.0:
                    return self.noise.apply(float(value))
                self._stuck_value = self.noise.apply(float(value))
            return self._stuck_value
        v = float(value)
        if self.kind == "bias":
            v += self.schedule.blend(0.0, self.magnitude, t_s)
        else:  # drift
            v += self.magnitude * self.schedule.active_elapsed(t_s)
        return self.noise.apply(v)

    def sample_int(self, value: int, t_s: float) -> int:
        """Integer-channel variant (e.g. Int32 pressure in Pa)."""
        if not self.is_active:
            return self.noise.apply_int(value)
        return int(round(self.sample(float(value), t_s)))


class MessageDrop:
    """Persistent per-message Bernoulli drop gate.

    `p <= 0` is inactive: `should_drop()` returns False without
    consuming RNG, keeping nominal random streams untouched. An optional
    `schedule` scales the probability to `m(t) * p` when the caller
    supplies a timestamp; a zero effective probability (pre-onset, or an
    intermittent off-window) likewise draws nothing, so the fault run's
    stream matches the schedule-free draw sequence once the fault is
    fully on.
    """

    def __init__(
        self,
        p: float = 0.0,
        rng: random.Random | None = None,
        schedule: FaultSchedule | None = None,
    ) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"drop probability must be in [0, 1], got {p}")
        self.p = float(p)
        self.rng = rng if rng is not None else random.Random()
        self.schedule = schedule
        self.is_active = self.p > 0.0

    def should_drop(self, t_s: float | None = None) -> bool:
        if not self.is_active:
            return False
        p_eff = self.p
        if self.schedule is not None and t_s is not None:
            p_eff = self.schedule.blend(0.0, self.p, t_s)
            if p_eff <= 0.0:
                return False
        return self.rng.random() < p_eff


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
