import numpy as np

from TSB_AD.models.SR import SR


def test_saliency_map_is_shift_equivariant():
    # |FFT(x)| does not change when x is rolled, only the phase does, so the
    # averaged log-amplitude spectrum and the spectral residual are unchanged and
    # the saliency map of a rolled series is the rolled saliency map.
    rng = np.random.default_rng(0)
    n, shift = 512, 137
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 64) + 0.05 * rng.standard_normal(n)
    x[300] += 6.0
    x = x.reshape(-1, 1)

    sr = SR(x.copy(), window_size=64)
    sr_rolled = SR(np.roll(x, shift, axis=0).copy(), window_size=64)

    np.testing.assert_allclose(sr_rolled, np.roll(sr, shift), rtol=1e-9, atol=1e-9)
