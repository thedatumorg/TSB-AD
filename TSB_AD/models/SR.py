"""Spectral Residual
"""
# Author: Andreas Mueller <andreas.mueller@microsoft.com>
import numpy as np

def SR(X, window_size):
    X = (X - X.min()) / (X.max() - X.min())
    X = X.ravel()
    fft = np.fft.fft(X)

    amp = np.abs(fft)
    log_amp = np.log(amp)
    phase = np.angle(fft)
    # split spectrum into bias term and symmetric frequencies
    bias, sym_freq = log_amp[:1], log_amp[1:]
    # select just the first half of the sym_freq
    freq = sym_freq[:(len(sym_freq) + 1) // 2]
    window_amp = 100

    pad_left = (window_amp - 1) // 2
    padded_freq = np.concatenate([np.tile(freq[0], pad_left), freq, np.tile(freq[-1], window_amp - pad_left - 1)])
    conv_amp = np.ones(window_amp) / window_amp
    ma_freq = np.convolve(padded_freq, conv_amp, 'valid')
    # construct moving average log amplitude spectrum
    ma_log_amp = np.concatenate([
        bias,
        ma_freq,
        (ma_freq[:-1] if len(sym_freq) % 2 == 1 else ma_freq)[::-1]
    ])
    assert ma_log_amp.shape[0] == log_amp.shape[0], "`ma_log_amp` size does not match `log_amp` size."
    # compute residual spectrum and transform back to time domain
    res_amp = log_amp - ma_log_amp
    sr = np.abs(np.fft.ifft(np.exp(res_amp + 1j * phase)))
    return sr