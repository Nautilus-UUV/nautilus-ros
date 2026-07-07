"""Gaussian + quantization sensor-noise model for the HAL sim bridges.

Pure logic, no ROS: the bridges in ``nautilus_hal`` compose one
:class:`GaussianQuantizedNoise` per channel and call ``apply`` /
``apply_int`` on each outgoing sample. Keeping the math here (py_pkg)
rather than in the bridge package means it gets Tier 1 unit coverage
without a Gazebo environment -- same split as ``physics.py`` vs the
bridges that consume it.

The model mirrors a real sensor chain: additive Gaussian noise on the
analog quantity first, then quantization by the digitizer. Parameters
default to the values fitted from the 2026-06-24 lake test
(``UG-anomaly_detection/lake_test_jun24/investigation/
noise_characterization.json``) via ``scenarios.spec.rig.NoiseSpec``;
this class itself is value-agnostic.
"""

from __future__ import annotations

import random


def rng_from_seed(seed: int) -> random.Random | None:
    """Map a scenario-compiled seed parameter to the noise/fault RNG.

    ``0`` means "let ``random.Random()`` pick" — non-deterministic ad-hoc
    ``ros2 run`` sessions; anything else is a reproducible stream (the
    scenario compiler injects ``derive_seed`` values). Every bridge RNG
    goes through here so the sentinel convention lives in one place.
    """
    return random.Random(seed) if seed != 0 else None


class GaussianQuantizedNoise:
    """Additive Gaussian noise followed by quantization rounding.

    ``sigma <= 0`` disables the Gaussian term, ``quantization_step <= 0``
    disables the rounding; with both disabled ``apply`` returns its
    input bit-identically (exact passthrough, no float round-trip).
    """

    def __init__(
        self,
        sigma: float = 0.0,
        quantization_step: float = 0.0,
        rng: random.Random | None = None,
    ) -> None:
        self.sigma = float(sigma)
        self.quantization_step = float(quantization_step)
        self.rng = rng if rng is not None else random.Random()
        # Resolved once — apply() runs per sample in stream-rate callbacks.
        self.is_active = self.sigma > 0.0 or self.quantization_step > 0.0

    def apply(self, value: float) -> float:
        """Return ``value`` with Gaussian noise, quantized to the step grid."""
        if not self.is_active:
            return value
        v = float(value)
        if self.sigma > 0.0:
            v += self.rng.gauss(0.0, self.sigma)
        step = self.quantization_step
        if step > 0.0:
            v = round(v / step) * step
        return v

    def apply_int(self, value: int) -> int:
        """Integer-channel variant (e.g. Int32 pressure in Pa).

        Inactive -> the input untouched; active -> noise + quantization
        on the float image, rounded back to int. With an integer step
        the result lands exactly on the step grid.
        """
        if not self.is_active:
            return value
        return int(round(self.apply(value)))
