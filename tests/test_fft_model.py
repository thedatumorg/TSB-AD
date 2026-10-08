import numpy as np
import pytest

from TSB_AD.models.FFT import FFT


def _series(n=1000, period=50, spike_at=400, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / period) + 0.05 * rng.standard_normal(n)
    x[spike_at] += 8.0
    return x.reshape(-1, 1)


def test_reduce_parameters_keeps_conjugate_pairs():
    # Truncating a real signal's spectrum must keep the negative-frequency
    # conjugates, otherwise the inverse transform is not real and the retained
    # oscillating components come back at half amplitude.
    x = _series().ravel()
    k = 5
    y = FFT.reduce_parameters(np.fft.fft(x), k)
    assert np.count_nonzero(y) == 2 * k - 1
    assert np.allclose(np.fft.ifft(y).imag, 0.0, atol=1e-9)


def test_scores_are_invariant_to_a_constant_offset():
    # The zero-frequency term is always retained, so it absorbs a constant
    # offset exactly: the fitted curve shifts by the same constant, the residual
    # is unchanged, and the neighbour deviations data[i] - mean(neighbours) are
    # unchanged. Scores must therefore not depend on the offset.
    x = _series()
    a = FFT(normalize=False).fit(x).decision_scores_
    b = FFT(normalize=False).fit(x + 100.0).decision_scores_
    np.testing.assert_allclose(np.ravel(a), np.ravel(b), rtol=1e-8, atol=1e-8)


def test_multivariate_input_is_rejected_explicitly():
    x = _series()
    xm = np.column_stack([x.ravel(), np.roll(x.ravel(), 7)])
    with pytest.raises(ValueError, match="univariate"):
        FFT(normalize=False).fit(xm)
