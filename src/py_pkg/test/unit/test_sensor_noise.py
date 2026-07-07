"""Tier 1 tests for the pure Gaussian+quantization sensor-noise model.

Every behavioral claim the sim noise injection relies on maps to a test
here: exact passthrough when disabled, seed determinism, calibrated
sigma statistics, quantization-comb output, integer-channel round trip,
and the Sheppard relation that ties the analysis-side fitting math
(sigma from quantized residuals) to the injection-side model.
"""

import math
import random
import statistics

from py_pkg.sensor_noise import GaussianQuantizedNoise


def test_disabled_passthrough_exact():
    noise = GaussianQuantizedNoise(sigma=0.0, quantization_step=0.0)
    assert not noise.is_active
    for v in (0.0, -9.80665, 1e-12, 101_325.7):
        assert noise.apply(v) is not None
        assert noise.apply(v) == v  # bit-identical, no float round trip
    assert noise.apply_int(101_325) == 101_325
    assert isinstance(noise.apply_int(101_325), int)


def test_negative_params_treated_as_disabled():
    noise = GaussianQuantizedNoise(sigma=-1.0, quantization_step=-100.0)
    assert not noise.is_active
    assert noise.apply(42.42) == 42.42


def test_seeded_determinism():
    a = GaussianQuantizedNoise(sigma=0.5, rng=random.Random(42))
    b = GaussianQuantizedNoise(sigma=0.5, rng=random.Random(42))
    c = GaussianQuantizedNoise(sigma=0.5, rng=random.Random(43))
    seq_a = [a.apply(0.0) for _ in range(100)]
    seq_b = [b.apply(0.0) for _ in range(100)]
    seq_c = [c.apply(0.0) for _ in range(100)]
    assert seq_a == seq_b
    assert seq_a != seq_c


def test_sigma_statistics():
    sigma = 0.102  # lake-fit accel-z magnitude
    noise = GaussianQuantizedNoise(sigma=sigma, rng=random.Random(7))
    n = 20_000
    samples = [noise.apply(5.0) for _ in range(n)]
    assert abs(statistics.stdev(samples) - sigma) / sigma < 0.03
    assert abs(statistics.fmean(samples) - 5.0) < 4 * sigma / math.sqrt(n)


def test_quantization_comb_with_gaussian():
    noise = GaussianQuantizedNoise(
        sigma=2400.0, quantization_step=600.0, rng=random.Random(3)
    )
    outs = [noise.apply(131_000.0) for _ in range(500)]
    assert all(v % 600.0 == 0.0 for v in outs)
    assert len(set(outs)) > 3  # actually dithering, not stuck


def test_pure_quantization_matches_grid_rounding():
    noise = GaussianQuantizedNoise(sigma=0.0, quantization_step=100.0)
    # Stay off exact half-step ties: banker's rounding on .5 is not part
    # of the modeled behavior.
    for v in (98_212.0, 98_249.0, 98_251.0, -333.0, 101_326.4):
        assert noise.apply(v) == round(v / 100.0) * 100.0


def test_apply_int_stays_on_grid():
    noise = GaussianQuantizedNoise(
        sigma=180.0, quantization_step=100.0, rng=random.Random(11)
    )
    outs = [noise.apply_int(150_000) for _ in range(500)]
    assert all(isinstance(v, int) for v in outs)
    assert all(v % 100 == 0 for v in outs)


def test_quantized_variance_matches_sheppard():
    """Injected (sigma, q) must reproduce sqrt(sigma^2 + q^2/12) overall.

    This is the inverse of the analysis-side Sheppard correction used to
    fit sigma from quantized lake data — locking both directions to the
    same model keeps the round trip honest.
    """
    sigma, q = 100.0, 100.0
    noise = GaussianQuantizedNoise(
        sigma=sigma, quantization_step=q, rng=random.Random(5)
    )
    n = 20_000
    samples = [noise.apply(0.0) for _ in range(n)]
    expected = math.sqrt(sigma**2 + q**2 / 12.0)
    assert abs(statistics.stdev(samples) - expected) / expected < 0.05
