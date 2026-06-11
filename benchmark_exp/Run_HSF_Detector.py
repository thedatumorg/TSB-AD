# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import argparse
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from TSB_AD.evaluation.metrics import basic_metricor, generate_curve, get_metrics
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.slidingWindows import find_length_rank


EPS = 1e-12
HSF_MODES = (
    "semisupervised_offline",
    "unsupervised_offline",
    "unsupervised_causal",
)
HSF_MODE_ALIASES = {
    "semisupervised": "semisupervised_offline",
    "unsupervised": "unsupervised_offline",
}
HSF_SCORE_MODES = (
    "default",
    "unsupervised_offline",
    "excitation",
    "hybrid",
    "family_gated",
)
HSF_METRIC_MODES = ("core", "full")
EXCITATION_FAMILIES = ("MITDB", "SVDB", "SED")
DEFAULT_UNI_OVERLAP_TABLE = (
    "benchmark_exp/benchmark_eval_results/uni_mergedTable_VUS-PR.csv"
)


def _as_univariate(X):
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0].reshape(-1)
    if arr.ndim == 2:
        raise ValueError(f"HSF_AD expects univariate input, got shape {arr.shape}")
    raise ValueError(f"HSF_AD expects a 1D or 2D array, got shape {arr.shape}")


def _clean_series(values):
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size == 0:
        return x

    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=float)

    finite_median = float(np.median(x[finite]))
    s = pd.Series(x).replace([np.inf, -np.inf], np.nan)
    s = s.interpolate(method="linear", limit_direction="both")
    s = s.fillna(finite_median)
    out = s.to_numpy(dtype=float)
    return np.nan_to_num(out, nan=finite_median, posinf=finite_median, neginf=finite_median)


def _safe_window(window, n):
    if n <= 1:
        return 1
    window = int(max(1, min(window, n)))
    return window


def _rolling_mean(values, window, center=True):
    x = _clean_series(values)
    window = _safe_window(window, x.size)
    if window <= 1:
        return x.copy()
    return (
        pd.Series(x)
        .rolling(window=window, min_periods=1, center=center)
        .mean()
        .to_numpy(dtype=float)
    )


def _rolling_median(values, window, center=True):
    x = _clean_series(values)
    window = _safe_window(window, x.size)
    if window <= 1:
        return x.copy()
    return (
        pd.Series(x)
        .rolling(window=window, min_periods=1, center=center)
        .median()
        .to_numpy(dtype=float)
    )


def _rolling_std(values, window, center=True):
    x = _clean_series(values)
    window = _safe_window(window, x.size)
    if window <= 1:
        return np.zeros_like(x)
    out = (
        pd.Series(x)
        .rolling(window=window, min_periods=2, center=center)
        .std(ddof=0)
        .to_numpy(dtype=float)
    )
    return _clean_series(out)


def _rolling_corr_lag(values, lag, window, center=True):
    x = _clean_series(values)
    n = x.size
    if n <= 2 or lag <= 0 or lag >= n:
        return np.zeros(n, dtype=float)

    lagged = np.empty(n, dtype=float)
    lagged[:lag] = np.nan
    lagged[lag:] = x[:-lag]
    corr = (
        pd.Series(x)
        .rolling(window=_safe_window(window, n), min_periods=3, center=center)
        .corr(pd.Series(lagged))
        .to_numpy(dtype=float)
    )
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def _shift(values, periods, fill_value=0.0):
    x = np.asarray(values, dtype=float).reshape(-1)
    out = np.empty_like(x)
    if periods == 0 or x.size == 0:
        return x.copy()
    if abs(periods) >= x.size:
        out.fill(fill_value)
        return out
    if periods > 0:
        out[:periods] = fill_value
        out[periods:] = x[:-periods]
    else:
        p = abs(periods)
        out[-p:] = fill_value
        out[:-p] = x[p:]
    return out


def _ewma(values, span):
    x = _clean_series(values)
    span = max(1, int(span))
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy(dtype=float)


def _robust_positive_feature(values):
    x = _clean_series(values)
    if x.size == 0:
        return x
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < EPS:
        scale = float(np.std(x))
    if not np.isfinite(scale) or scale < EPS:
        return np.zeros_like(x, dtype=float)
    z = (x - median) / scale
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(np.maximum(z, 0.0), 0.0, 50.0)


def _combine_proxy_features(feature_list, n):
    scaled = []
    for feature in feature_list:
        arr = np.asarray(feature, dtype=float).reshape(-1)
        if arr.size != n:
            raise ValueError("Proxy feature length mismatch")
        scaled.append(_robust_positive_feature(arr))
    if not scaled:
        return np.zeros(n, dtype=float)
    return _clean_series(np.mean(np.vstack(scaled), axis=0))


def _causal_robust_positive_feature(values, window):
    x = _clean_series(values)
    if x.size == 0:
        return x
    if np.std(x) < EPS:
        return np.zeros_like(x, dtype=float)

    window = _safe_window(window, x.size)
    s = pd.Series(x)
    median = s.rolling(window=window, min_periods=1, center=False).median()
    abs_dev = (s - median).abs()
    mad = abs_dev.rolling(window=window, min_periods=1, center=False).median()
    scale = 1.4826 * mad

    rolling_std = s.rolling(window=window, min_periods=2, center=False).std(ddof=0)
    expanding_std = s.expanding(min_periods=2).std(ddof=0)
    scale = scale.where(scale >= EPS, rolling_std)
    scale = scale.where(scale >= EPS, expanding_std)
    scale = scale.replace([np.inf, -np.inf], np.nan)

    z = (s - median) / scale.replace(0.0, np.nan)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    return np.clip(np.maximum(z, 0.0), 0.0, 50.0)


def _combine_proxy_features_causal(feature_list, n, window):
    scaled = []
    for feature in feature_list:
        arr = np.asarray(feature, dtype=float).reshape(-1)
        if arr.size != n:
            raise ValueError("Proxy feature length mismatch")
        scaled.append(_causal_robust_positive_feature(arr, window))
    if not scaled:
        return np.zeros(n, dtype=float)
    return _clean_series(np.mean(np.vstack(scaled), axis=0))


def _safe_minmax(values):
    x = _clean_series(values)
    if x.size == 0:
        return x
    min_v = float(np.min(x))
    max_v = float(np.max(x))
    if not np.isfinite(min_v) or not np.isfinite(max_v) or max_v - min_v < EPS:
        return np.zeros_like(x, dtype=float)
    return (x - min_v) / (max_v - min_v)


def _safe_expanding_minmax(values):
    x = _clean_series(values)
    if x.size == 0:
        return x
    s = pd.Series(x)
    min_v = s.expanding(min_periods=1).min()
    max_v = s.expanding(min_periods=1).max()
    denom = max_v - min_v
    out = (s - min_v) / denom.replace(0.0, np.nan)
    out = out.where(denom >= EPS, 0.0)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)


def _robust_zscore(values):
    x = _clean_series(values)
    if x.size == 0:
        return x
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < EPS:
        scale = float(np.std(x))
    if not np.isfinite(scale) or scale < EPS:
        return np.zeros_like(x, dtype=float)
    z = (x - median) / scale
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def _scale01_robust(values):
    x = _clean_series(values)
    if x.size == 0 or float(np.std(x)) < EPS:
        return np.zeros_like(x, dtype=float)
    return _safe_minmax(_robust_positive_feature(x))


def _estimate_ar1_phi(values):
    x = _clean_series(values)
    if x.size < 3 or np.std(x) < EPS:
        return 0.0
    centered = x - float(np.median(x))
    prev = centered[:-1]
    curr = centered[1:]
    denom = float(np.dot(prev, prev))
    if denom < EPS:
        return 0.0
    phi = float(np.dot(prev, curr) / denom)
    if not np.isfinite(phi):
        return 0.0
    return float(np.clip(phi, -0.99, 0.99))


def _estimate_period(values, max_lag=256, min_lag=2, threshold=0.35):
    x = _clean_series(values)
    n = x.size
    if n < 12 or np.std(x) < EPS:
        return None

    x = x - float(np.median(x))
    max_lag = int(min(max_lag, max(min_lag, n // 2)))
    if max_lag <= min_lag:
        return None

    best_lag = None
    best_corr = -np.inf
    for lag in range(min_lag, max_lag + 1):
        a = x[:-lag]
        b = x[lag:]
        if a.size < 3 or np.std(a) < EPS or np.std(b) < EPS:
            continue
        corr = float(np.corrcoef(a, b)[0, 1])
        if np.isfinite(corr) and corr > best_corr:
            best_lag = lag
            best_corr = corr

    if best_lag is None or best_corr < threshold:
        return None
    return int(best_lag)


def _spectral_samples(values, window, stride):
    x = _clean_series(values)
    n = x.size
    if n < 4:
        return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float), np.zeros(0, dtype=int)

    window = _safe_window(window, n)
    stride = max(1, int(stride))
    centers = np.arange(0, n, stride, dtype=int)
    if centers.size == 0 or centers[-1] != n - 1:
        centers = np.append(centers, n - 1)

    half = max(1, window // 2)
    distributions = []
    entropies = []

    for center in centers:
        start = max(0, int(center) - half)
        end = min(n, int(center) + half + 1)
        segment = x[start:end]
        if segment.size < 4 or np.std(segment) < EPS:
            distributions.append(np.zeros(3, dtype=float))
            entropies.append(0.0)
            continue

        segment = segment - float(np.mean(segment))
        tapered = segment * np.hanning(segment.size)
        power = np.abs(np.fft.rfft(tapered)) ** 2
        power = power[1:]
        total = float(np.sum(power))
        if power.size == 0 or total < EPS:
            distributions.append(np.zeros(3, dtype=float))
            entropies.append(0.0)
            continue

        bands = np.array_split(power, 3)
        band_energy = np.array([float(np.sum(band)) for band in bands], dtype=float)
        distribution = band_energy / max(float(np.sum(band_energy)), EPS)
        p = power / total
        entropy = -float(np.sum(p * np.log(p + EPS))) / math.log(max(power.size, 2))
        distributions.append(distribution)
        entropies.append(entropy)

    return np.vstack(distributions), np.asarray(entropies, dtype=float), centers


def _interpolate_samples(values, centers, n):
    values = np.asarray(values, dtype=float).reshape(-1)
    centers = np.asarray(centers, dtype=float).reshape(-1)
    if n == 0:
        return np.zeros(0, dtype=float)
    if values.size == 0 or centers.size == 0:
        return np.zeros(n, dtype=float)
    if values.size == 1:
        return np.full(n, float(values[0]), dtype=float)
    return np.interp(np.arange(n, dtype=float), centers, values)


def _hsf_weights(HP=None):
    hp = HP or {}
    weights = hp.get("weights")
    if weights is None:
        weights = [
            hp.get("w1", 1.0),
            hp.get("w2", 1.0),
            hp.get("w3", 1.0),
            hp.get("w4", 1.0),
            hp.get("w5", 1.0),
        ]
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.size != 5 or not np.isfinite(weights).all():
        return np.ones(5, dtype=float)
    return weights


def _default_hsf_windows(n, HP=None):
    hp = HP or {}
    default_window = int(min(256, max(16, n // 20))) if n >= 16 else max(1, n)
    window = _safe_window(hp.get("window", default_window), n)
    short_window = _safe_window(hp.get("short_window", max(5, window // 4)), n)
    long_window = _safe_window(hp.get("long_window", max(window, window * 2)), n)
    spectral_window = _safe_window(hp.get("spectral_window", max(16, window)), n)
    spectral_stride = max(
        1, int(hp.get("spectral_stride", max(1, spectral_window // 4)))
    )
    return window, short_window, long_window, spectral_window, spectral_stride


def _spectral_distribution(segment):
    segment = _clean_series(segment)
    if segment.size < 4 or np.std(segment) < EPS:
        return np.zeros(3, dtype=float), 0.0

    segment = segment - float(np.mean(segment))
    tapered = segment * np.hanning(segment.size)
    power = np.abs(np.fft.rfft(tapered)) ** 2
    power = power[1:]
    total = float(np.sum(power))
    if power.size == 0 or total < EPS:
        return np.zeros(3, dtype=float), 0.0

    bands = np.array_split(power, 3)
    band_energy = np.array([float(np.sum(band)) for band in bands], dtype=float)
    distribution = band_energy / max(float(np.sum(band_energy)), EPS)
    p = power / total
    entropy = -float(np.sum(p * np.log(p + EPS))) / math.log(max(power.size, 2))
    return distribution, entropy


def _rolling_spectral_proxy_causal(x, window, stride, baseline_alpha=0.1):
    x = _clean_series(x)
    n = x.size
    signature_deviation = np.zeros(n, dtype=float)
    entropy_deviation = np.zeros(n, dtype=float)
    high_band_share = np.zeros(n, dtype=float)
    if n < 4:
        return {
            "signature_deviation": signature_deviation,
            "entropy_deviation": entropy_deviation,
            "high_band_share": high_band_share,
        }

    window = _safe_window(window, n)
    stride = max(1, int(stride))
    endpoints = np.arange(0, n, stride, dtype=int)
    if endpoints.size == 0 or endpoints[-1] != n - 1:
        endpoints = np.append(endpoints, n - 1)

    alpha = float(np.clip(baseline_alpha, 0.001, 1.0))
    baseline_signature = None
    baseline_entropy = 0.0
    prev_endpoint = -1
    last_sig_dev = 0.0
    last_entropy_dev = 0.0
    last_high_share = 0.0

    for endpoint in endpoints:
        endpoint = int(endpoint)
        if endpoint > prev_endpoint + 1:
            start_hold = prev_endpoint + 1
            signature_deviation[start_hold:endpoint] = last_sig_dev
            entropy_deviation[start_hold:endpoint] = last_entropy_dev
            high_band_share[start_hold:endpoint] = last_high_share

        start = max(0, endpoint - window + 1)
        distribution, entropy = _spectral_distribution(x[start : endpoint + 1])
        if baseline_signature is None:
            sig_dev = 0.0
            entropy_dev = 0.0
            baseline_signature = distribution.copy()
            baseline_entropy = float(entropy)
        else:
            sig_dev = float(np.sum(np.abs(distribution - baseline_signature)))
            entropy_dev = abs(float(entropy) - baseline_entropy)
            baseline_signature = (
                alpha * distribution + (1.0 - alpha) * baseline_signature
            )
            total = float(np.sum(baseline_signature))
            if total > EPS:
                baseline_signature = baseline_signature / total
            baseline_entropy = alpha * float(entropy) + (1.0 - alpha) * baseline_entropy

        last_sig_dev = sig_dev
        last_entropy_dev = entropy_dev
        last_high_share = float(distribution[-1])
        signature_deviation[endpoint] = last_sig_dev
        entropy_deviation[endpoint] = last_entropy_dev
        high_band_share[endpoint] = last_high_share
        prev_endpoint = endpoint

    return {
        "signature_deviation": signature_deviation,
        "entropy_deviation": entropy_deviation,
        "high_band_share": high_band_share,
    }


def _causal_ar_prediction(x, window, long_window):
    x = _clean_series(x)
    n = x.size
    if n == 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return x.copy()

    baseline = _ewma(x, long_window)
    baseline_prev = _shift(baseline, 1, fill_value=x[0])
    previous = _shift(x, 1, fill_value=x[0])

    pair_prev = previous - baseline_prev
    pair_curr = x - baseline
    numerator = _rolling_mean(pair_prev * pair_curr, window, center=False)
    denominator = _rolling_mean(pair_prev ** 2, window, center=False)
    phi_raw = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > EPS,
    )
    phi_raw = np.clip(np.nan_to_num(phi_raw, nan=0.0, posinf=0.0, neginf=0.0), -0.99, 0.99)
    phi = _shift(phi_raw, 1, fill_value=0.0)
    return baseline_prev + phi * (previous - baseline_prev)


class HSF_AD(BaseDetector):
    """HSF-inspired detector built from robust univariate proxy features."""

    def __init__(self, HP=None, normalize=True):
        super().__init__()
        self.HP = HP or {}
        self.normalize = normalize

    def fit(self, X, y=None):
        x = _clean_series(_as_univariate(X))
        self._configure_from_training(x)
        self.decision_scores_ = self._score_series(x)
        return self

    def decision_function(self, X):
        x = _clean_series(_as_univariate(X))
        return self._score_series(x)

    def _configure_from_training(self, x):
        n = x.size
        default_window = int(min(256, max(16, n // 20))) if n >= 16 else max(1, n)
        self.window_ = _safe_window(self.HP.get("window", default_window), n)
        self.short_window_ = _safe_window(
            self.HP.get("short_window", max(5, self.window_ // 4)), n
        )
        self.long_window_ = _safe_window(
            self.HP.get("long_window", max(self.window_, self.window_ * 2)), n
        )
        self.spectral_window_ = _safe_window(
            self.HP.get("spectral_window", max(16, self.window_)), n
        )
        self.spectral_stride_ = max(
            1, int(self.HP.get("spectral_stride", max(1, self.spectral_window_ // 4)))
        )

        weights = self.HP.get("weights")
        if weights is None:
            weights = [
                self.HP.get("w1", 1.0),
                self.HP.get("w2", 1.0),
                self.HP.get("w3", 1.0),
                self.HP.get("w4", 1.0),
                self.HP.get("w5", 1.0),
            ]
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.size != 5 or not np.isfinite(weights).all():
            weights = np.ones(5, dtype=float)
        self.weights_ = weights

        self.center_ = float(np.median(x)) if n else 0.0
        self.phi_ = _estimate_ar1_phi(x)
        self.period_ = _estimate_period(x, max_lag=self.HP.get("max_period_lag", 256))
        acorr1 = _rolling_corr_lag(x, lag=1, window=self.window_)
        self.acorr1_baseline_ = float(np.median(acorr1)) if acorr1.size else 0.0

        distributions, entropies, _ = _spectral_samples(
            x, self.spectral_window_, self.spectral_stride_
        )
        if distributions.size:
            signature = np.median(distributions, axis=0)
            total = float(np.sum(signature))
            self.spectral_signature_ = (
                signature / total if total > EPS else np.zeros(3, dtype=float)
            )
            self.spectral_entropy_ = float(np.median(entropies))
        else:
            self.spectral_signature_ = np.zeros(3, dtype=float)
            self.spectral_entropy_ = 0.0

    def _score_series(self, x):
        n = x.size
        if n == 0:
            self.components_ = {}
            return np.zeros(0, dtype=float)
        if np.std(x) < EPS:
            self.components_ = {
                "K1_proxy": np.zeros(n, dtype=float),
                "K2_coupling_proxy": np.zeros(n, dtype=float),
                "K3_collective_proxy": np.zeros(n, dtype=float),
                "K4_retuning_proxy": np.zeros(n, dtype=float),
                "C5_closure_proxy": np.zeros(n, dtype=float),
            }
            return np.zeros(n, dtype=float)

        window = _safe_window(getattr(self, "window_", 16), n)
        short_window = _safe_window(getattr(self, "short_window_", 5), n)
        long_window = _safe_window(getattr(self, "long_window_", window), n)

        diff = np.abs(np.diff(x, prepend=x[0]))
        first_diff = np.diff(x, prepend=x[0])
        jerk = np.abs(np.diff(first_diff, prepend=first_diff[0]))
        local_mean = _rolling_mean(x, short_window)
        residual = x - local_mean
        residual_abs = np.abs(residual)
        residual_energy = _rolling_mean(residual ** 2, short_window)
        diff_energy = _rolling_mean(diff ** 2, short_window)

        K1_proxy = _combine_proxy_features(
            [diff, jerk, residual_abs, residual_energy, diff_energy], n
        )

        acorr1 = _rolling_corr_lag(x, lag=1, window=window)
        acorr_baseline = getattr(self, "acorr1_baseline_", 0.0)
        acorr_deviation = np.abs(acorr1 - acorr_baseline)
        acorr_instability = np.abs(np.diff(acorr1, prepend=acorr1[0]))
        signal_center = x - _rolling_mean(x, long_window)
        signal_energy = _rolling_mean(signal_center ** 2, window)
        residual_energy_window = _rolling_mean(residual ** 2, window)
        energy_ratio = residual_energy_window / (signal_energy + EPS)
        quiet = (signal_energy < EPS) & (residual_energy_window < EPS)
        energy_ratio[quiet] = 0.0

        K2_coupling_proxy = _combine_proxy_features(
            [acorr_deviation, acorr_instability, energy_ratio], n
        )

        spectral = self._rolling_spectral_proxy(x)
        K3_collective_proxy = _combine_proxy_features(
            [
                spectral["signature_deviation"],
                spectral["entropy_deviation"],
                spectral["high_band_share"],
            ],
            n,
        )

        impulse = diff + jerk
        delayed_impulse = _shift(_ewma(impulse, short_window), 1, fill_value=0.0)
        recovery_error = delayed_impulse * residual_abs
        residual_persistence = _rolling_mean(residual_abs, long_window)
        decay_lag = min(max(2, short_window), max(2, n - 1))
        acorr_decay = np.abs(_rolling_corr_lag(x, lag=decay_lag, window=long_window))
        fast_residual = _ewma(residual ** 2, short_window)
        slow_residual = _ewma(residual ** 2, long_window)
        tail_energy = np.maximum(slow_residual - fast_residual, 0.0)

        K4_retuning_proxy = _combine_proxy_features(
            [recovery_error, residual_persistence, acorr_decay, tail_energy], n
        )

        previous = _shift(x, 1, fill_value=self.center_)
        ar_prediction = self.center_ + getattr(self, "phi_", 0.0) * (previous - self.center_)
        ar_residual = np.abs(x - ar_prediction)
        ewma_prediction = _shift(_ewma(x, short_window), 1, fill_value=self.center_)
        ewma_residual = np.abs(x - ewma_prediction)
        predictability_error = _rolling_mean(ar_residual ** 2, window)
        period = getattr(self, "period_", None)
        if period is not None and period < n:
            periodic_error = np.abs(x - _shift(x, period, fill_value=self.center_))
            repeatability_error = np.abs(impulse - _shift(impulse, period, fill_value=0.0))
        else:
            periodic_error = ewma_residual
            repeatability_error = _rolling_std(impulse, window) / (
                _rolling_mean(np.abs(impulse), window) + EPS
            )

        C5_closure_proxy = _combine_proxy_features(
            [ar_residual, ewma_residual, predictability_error, periodic_error, repeatability_error],
            n,
        )

        self.components_ = {
            "K1_proxy": K1_proxy,
            "K2_coupling_proxy": K2_coupling_proxy,
            "K3_collective_proxy": K3_collective_proxy,
            "K4_retuning_proxy": K4_retuning_proxy,
            "C5_closure_proxy": C5_closure_proxy,
        }

        stacked = np.vstack(
            [
                K1_proxy,
                K2_coupling_proxy,
                K3_collective_proxy,
                K4_retuning_proxy,
                C5_closure_proxy,
            ]
        )
        score = np.dot(self.weights_, stacked)
        return _clean_series(score)

    def _rolling_spectral_proxy(self, x):
        n = x.size
        distributions, entropies, centers = _spectral_samples(
            x,
            _safe_window(getattr(self, "spectral_window_", 16), n),
            getattr(self, "spectral_stride_", 4),
        )
        if distributions.size == 0:
            return {
                "signature_deviation": np.zeros(n, dtype=float),
                "entropy_deviation": np.zeros(n, dtype=float),
                "high_band_share": np.zeros(n, dtype=float),
            }

        baseline_signature = getattr(self, "spectral_signature_", np.zeros(3, dtype=float))
        baseline_entropy = getattr(self, "spectral_entropy_", 0.0)
        signature_deviation = np.sum(np.abs(distributions - baseline_signature), axis=1)
        entropy_deviation = np.abs(entropies - baseline_entropy)
        high_band_share = distributions[:, -1]

        return {
            "signature_deviation": _interpolate_samples(signature_deviation, centers, n),
            "entropy_deviation": _interpolate_samples(entropy_deviation, centers, n),
            "high_band_share": _interpolate_samples(high_band_share, centers, n),
        }


def run_HSF_AD_Unsupervised(data, HP=None):
    clf = HSF_AD(HP=HP)
    clf.fit(data)
    return _safe_minmax(clf.decision_scores_)


def run_HSF_AD_Semisupervised(data_train, data_test, HP=None):
    clf = HSF_AD(HP=HP)
    clf.fit(data_train)
    return _safe_minmax(clf.decision_function(data_test))


def run_HSF_AD_Causal(data, HP=None):
    x = _clean_series(_as_univariate(data))
    n = x.size
    if n == 0:
        return np.zeros(0, dtype=float)
    if np.std(x) < EPS:
        return np.zeros(n, dtype=float)

    hp = HP or {}
    window, short_window, long_window, spectral_window, spectral_stride = (
        _default_hsf_windows(n, hp)
    )
    weights = _hsf_weights(hp)

    diff = np.abs(np.diff(x, prepend=x[0]))
    first_diff = np.diff(x, prepend=x[0])
    jerk = np.abs(np.diff(first_diff, prepend=first_diff[0]))
    local_mean = _rolling_mean(x, short_window, center=False)
    residual = x - local_mean
    residual_abs = np.abs(residual)
    residual_energy = _rolling_mean(residual ** 2, short_window, center=False)
    diff_energy = _rolling_mean(diff ** 2, short_window, center=False)

    K1_proxy = _combine_proxy_features_causal(
        [diff, jerk, residual_abs, residual_energy, diff_energy], n, window
    )

    acorr1 = _rolling_corr_lag(x, lag=1, window=window, center=False)
    acorr_baseline = _shift(
        _rolling_mean(acorr1, long_window, center=False), 1, fill_value=0.0
    )
    acorr_deviation = np.abs(acorr1 - acorr_baseline)
    acorr_instability = np.abs(np.diff(acorr1, prepend=acorr1[0]))
    signal_center = x - _rolling_mean(x, long_window, center=False)
    signal_energy = _rolling_mean(signal_center ** 2, window, center=False)
    residual_energy_window = _rolling_mean(residual ** 2, window, center=False)
    energy_ratio = residual_energy_window / (signal_energy + EPS)
    quiet = (signal_energy < EPS) & (residual_energy_window < EPS)
    energy_ratio[quiet] = 0.0

    K2_coupling_proxy = _combine_proxy_features_causal(
        [acorr_deviation, acorr_instability, energy_ratio], n, window
    )

    spectral = _rolling_spectral_proxy_causal(
        x,
        spectral_window,
        spectral_stride,
        baseline_alpha=hp.get("spectral_baseline_alpha", 0.1),
    )
    K3_collective_proxy = _combine_proxy_features_causal(
        [
            spectral["signature_deviation"],
            spectral["entropy_deviation"],
            spectral["high_band_share"],
        ],
        n,
        window,
    )

    impulse = diff + jerk
    delayed_impulse = _shift(_ewma(impulse, short_window), 1, fill_value=0.0)
    recovery_error = delayed_impulse * residual_abs
    residual_persistence = _rolling_mean(residual_abs, long_window, center=False)
    decay_lag = min(max(2, short_window), max(2, n - 1))
    acorr_decay = np.abs(
        _rolling_corr_lag(x, lag=decay_lag, window=long_window, center=False)
    )
    fast_residual = _ewma(residual ** 2, short_window)
    slow_residual = _ewma(residual ** 2, long_window)
    tail_energy = np.maximum(slow_residual - fast_residual, 0.0)

    K4_retuning_proxy = _combine_proxy_features_causal(
        [recovery_error, residual_persistence, acorr_decay, tail_energy], n, window
    )

    ar_prediction = _causal_ar_prediction(x, window, long_window)
    ar_residual = np.abs(x - ar_prediction)
    ewma_prediction = _shift(_ewma(x, short_window), 1, fill_value=x[0])
    ewma_residual = np.abs(x - ewma_prediction)
    predictability_error = _rolling_mean(ar_residual ** 2, window, center=False)
    repeatability_error = _rolling_std(impulse, window, center=False) / (
        _rolling_mean(np.abs(impulse), window, center=False) + EPS
    )

    C5_closure_proxy = _combine_proxy_features_causal(
        [ar_residual, ewma_residual, predictability_error, ewma_residual, repeatability_error],
        n,
        window,
    )

    stacked = np.vstack(
        [
            K1_proxy,
            K2_coupling_proxy,
            K3_collective_proxy,
            K4_retuning_proxy,
            C5_closure_proxy,
        ]
    )
    return _safe_expanding_minmax(np.dot(weights, stacked))


def _select_activation_peaks(strength, min_distance, max_peaks=5000):
    strength = _clean_series(strength)
    n = strength.size
    if n < 3 or float(np.std(strength)) < EPS:
        return np.zeros(0, dtype=int)

    median = float(np.median(strength))
    mad = float(np.median(np.abs(strength - median)))
    robust_threshold = median + 1.4826 * mad
    percentile_threshold = float(np.percentile(strength, 90))
    threshold = max(robust_threshold, percentile_threshold)

    left = np.r_[strength[0], strength[:-1]]
    right = np.r_[strength[1:], strength[-1]]
    candidates = np.where(
        (strength >= left) & (strength >= right) & (strength > threshold)
    )[0]
    if candidates.size == 0:
        threshold = float(np.percentile(strength, 95))
        candidates = np.where(strength >= threshold)[0]
    if candidates.size == 0:
        return np.zeros(0, dtype=int)

    min_distance = int(max(1, min_distance))
    selected = []
    for idx in candidates:
        idx = int(idx)
        if not selected or idx - selected[-1] >= min_distance:
            selected.append(idx)
        elif strength[idx] > strength[selected[-1]]:
            selected[-1] = idx

    peaks = np.asarray(selected, dtype=int)
    max_peaks = int(max(1, max_peaks))
    if peaks.size > max_peaks:
        strongest = np.argsort(strength[peaks])[-max_peaks:]
        peaks = np.sort(peaks[strongest])
    return peaks


def _project_event_values(n, peaks, values, radius):
    out = np.zeros(n, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    values = np.asarray(values, dtype=float).reshape(-1)
    radius = int(max(1, radius))
    for peak, value in zip(peaks, values):
        start = max(0, int(peak) - radius)
        end = min(n, int(peak) + radius + 1)
        if start < end:
            out[start:end] = np.maximum(out[start:end], float(value))
    return out


def _project_interval_values(n, peaks, values):
    out = np.zeros(n, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    values = np.asarray(values, dtype=float).reshape(-1)
    if peaks.size < 2 or values.size == 0:
        return out

    usable = min(values.size, peaks.size - 1)
    for idx in range(usable):
        start = int(peaks[idx])
        end = int(peaks[idx + 1]) + 1
        out[start:end] = float(values[idx])
    out[: int(peaks[0])] = float(values[0])
    out[int(peaks[usable]) :] = float(values[usable - 1])
    return out


def compute_inverse_excitation_features(x):
    x = _clean_series(_as_univariate(x))
    n = x.size
    feature_names = [
        "excitation_energy",
        "peak_impulse_score",
        "inter_event_interval_error",
        "local_rhythm_instability",
        "beat_morphology_deviation",
        "recovery_error",
        "rhythm_closure_error",
        "HSF_excitation_score",
    ]
    if n == 0:
        return {name: np.zeros(0, dtype=float) for name in feature_names}
    if float(np.std(x)) < EPS:
        return {name: np.zeros(n, dtype=float) for name in feature_names}

    z = _robust_zscore(x)
    baseline_window = _safe_window(max(31, min(301, n // 100)), n)
    short_window = _safe_window(max(5, min(101, n // 500)), n)
    baseline = _rolling_median(z, baseline_window, center=True)
    residual = _clean_series(z - baseline)
    residual_abs = np.abs(residual)

    local_energy = _rolling_mean(residual ** 2, short_window, center=True)
    excitation_energy = _scale01_robust(local_energy)

    residual_step = np.abs(np.diff(residual, prepend=residual[0]))
    impulse_raw = residual_abs + residual_step
    impulse_strength = _safe_minmax(
        0.6 * _scale01_robust(residual_abs) + 0.4 * _scale01_robust(impulse_raw)
    )
    min_distance = max(5, min(300, n // 1000 if n >= 1000 else n // 20 + 1))
    peaks = _select_activation_peaks(impulse_strength, min_distance=min_distance)

    peak_signal = np.zeros(n, dtype=float)
    if peaks.size:
        peak_signal[peaks] = impulse_strength[peaks]
    peak_impulse_score = _scale01_robust(
        _rolling_mean(peak_signal, max(3, short_window // 2), center=True)
    )

    if peaks.size >= 3:
        intervals = np.diff(peaks).astype(float)
        interval_median = float(np.median(intervals))
        interval_mad = float(np.median(np.abs(intervals - interval_median)))
        interval_scale = max(1.4826 * interval_mad, float(np.std(intervals)), EPS)
        interval_errors = np.clip(np.abs(intervals - interval_median) / interval_scale, 0.0, 50.0)
        interval_axis = _project_interval_values(n, peaks, interval_errors)
        inter_event_interval_error = _scale01_robust(interval_axis)

        local_interval_change = np.abs(
            np.diff(interval_errors, prepend=interval_errors[0])
        )
        local_axis = _project_interval_values(n, peaks, local_interval_change)
        local_rhythm_instability = _scale01_robust(
            _rolling_mean(local_axis, max(3, short_window), center=True)
        )
        median_interval = max(3, int(round(interval_median)))
    else:
        inter_event_interval_error = np.zeros(n, dtype=float)
        local_rhythm_instability = np.zeros(n, dtype=float)
        median_interval = max(3, min_distance * 2)

    beat_half_window = int(max(4, min(120, median_interval // 3)))
    complete_peaks = peaks[
        (peaks - beat_half_window >= 0) & (peaks + beat_half_window + 1 <= n)
    ]
    morphology_axis = np.zeros(n, dtype=float)
    if complete_peaks.size >= 3:
        template_peaks = complete_peaks
        if template_peaks.size > 1000:
            selector = np.linspace(0, template_peaks.size - 1, 1000, dtype=int)
            template_peaks = template_peaks[selector]
        segments = np.vstack(
            [
                residual[p - beat_half_window : p + beat_half_window + 1]
                for p in template_peaks
            ]
        )
        template = np.median(segments, axis=0)
        morphology_values = []
        for peak in complete_peaks:
            segment = residual[peak - beat_half_window : peak + beat_half_window + 1]
            morphology_values.append(float(np.mean(np.abs(segment - template))))
        morphology_axis = _project_event_values(
            n, complete_peaks, morphology_values, radius=beat_half_window
        )
    beat_morphology_deviation = _scale01_robust(morphology_axis)

    recovery_axis = np.zeros(n, dtype=float)
    tail_window = int(max(beat_half_window, min(300, max(short_window, median_interval // 2))))
    if peaks.size:
        for peak in peaks:
            start = int(peak) + 1
            end = min(n, int(peak) + tail_window + 1)
            if start >= end:
                continue
            tail_energy = float(np.mean(residual_abs[start:end]))
            recovery_axis[start:end] = np.maximum(recovery_axis[start:end], tail_energy)
    delayed_activation = _shift(_ewma(peak_impulse_score, max(3, short_window)), 1, 0.0)
    recovery_error = _scale01_robust(
        recovery_axis + delayed_activation * _rolling_mean(residual_abs, short_window, center=True)
    )

    rhythm_closure_error = _scale01_robust(
        0.35 * inter_event_interval_error
        + 0.25 * local_rhythm_instability
        + 0.25 * beat_morphology_deviation
        + 0.15 * peak_impulse_score
    )

    score = _safe_minmax(
        0.25 * excitation_energy
        + 0.20 * inter_event_interval_error
        + 0.20 * beat_morphology_deviation
        + 0.20 * recovery_error
        + 0.15 * rhythm_closure_error
    )

    return {
        "excitation_energy": excitation_energy,
        "peak_impulse_score": peak_impulse_score,
        "inter_event_interval_error": inter_event_interval_error,
        "local_rhythm_instability": local_rhythm_instability,
        "beat_morphology_deviation": beat_morphology_deviation,
        "recovery_error": recovery_error,
        "rhythm_closure_error": rhythm_closure_error,
        "HSF_excitation_score": score,
    }


def run_HSF_AD_Excitation(data, HP=None):
    features = compute_inverse_excitation_features(data)
    return _safe_minmax(features["HSF_excitation_score"])


def _default_hsf_score(data, filename, mode, hp):
    if mode == "semisupervised_offline":
        train_index = _parse_train_index(filename, data.shape[0])
        data_train = data[:train_index, :]
        return run_HSF_AD_Semisupervised(data_train, data, HP=hp)
    if mode == "unsupervised_offline":
        return run_HSF_AD_Unsupervised(data, HP=hp)
    if mode == "unsupervised_causal":
        return run_HSF_AD_Causal(data, HP=hp)
    raise ValueError(f"unsupported mode: {mode}")


def run_HSF_AD_ScoreMode(data, filename, mode, score_mode, HP=None):
    score_mode = _normalize_score_mode(score_mode)
    if score_mode == "default":
        return _default_hsf_score(data, filename, mode, HP)
    if score_mode == "unsupervised_offline":
        return run_HSF_AD_Unsupervised(data, HP=HP)
    if score_mode == "excitation":
        return run_HSF_AD_Excitation(data, HP=HP)
    if score_mode == "hybrid":
        default_score = _safe_minmax(_default_hsf_score(data, filename, mode, HP))
        excitation_score = _safe_minmax(run_HSF_AD_Excitation(data, HP=HP))
        return _safe_minmax(0.65 * default_score + 0.35 * excitation_score)
    if score_mode == "family_gated":
        family = _family_from_filename(filename)
        if family in EXCITATION_FAMILIES:
            return run_HSF_AD_Excitation(data, HP=HP)
        return run_HSF_AD_Unsupervised(data, HP=HP)
    raise ValueError(f"unsupported score_mode: {score_mode}")


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _resolve_path(path_value):
    raw = Path(path_value)
    if raw.is_absolute():
        return raw
    candidates = [
        Path.cwd() / raw,
        _repo_root() / raw,
        Path(__file__).resolve().parent / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return _repo_root() / raw


def _normalize_mode(mode):
    normalized = HSF_MODE_ALIASES.get(mode, mode)
    if normalized not in HSF_MODES:
        valid = ", ".join(HSF_MODES)
        raise ValueError(f"unknown HSF_AD mode {mode!r}; expected one of: {valid}")
    return normalized


def _normalize_score_mode(score_mode):
    if score_mode not in HSF_SCORE_MODES:
        valid = ", ".join(HSF_SCORE_MODES)
        raise ValueError(f"unknown score_mode {score_mode!r}; expected one of: {valid}")
    return score_mode


def _normalize_metric_mode(metric_mode):
    if metric_mode not in HSF_METRIC_MODES:
        valid = ", ".join(HSF_METRIC_MODES)
        raise ValueError(f"unknown metric_mode {metric_mode!r}; expected one of: {valid}")
    return metric_mode


def _parse_family_filter(family_filter):
    if not family_filter:
        return []
    return [
        item.strip().upper()
        for item in str(family_filter).split(",")
        if item.strip()
    ]


def _family_from_filename(filename):
    parts = Path(filename).stem.split("_")
    if len(parts) >= 2:
        return parts[1].upper()
    return ""


def _path_matches_family_filter(path, family_tokens):
    if not family_tokens:
        return True
    name = Path(path).name.upper()
    return any(token in name for token in family_tokens)


def _mode_save_dir(save_root, mode):
    save_root = Path(save_root)
    if save_root.name == mode:
        return save_root
    return save_root / mode


def _run_save_dir(save_root, args):
    save_root = Path(save_root)
    file_list_name = ""
    if getattr(args, "file_list", None):
        file_list_name = Path(str(args.file_list)).stem.lower()
    if file_list_name == "medical_overlap27_files":
        metric_suffix = "vus" if getattr(args, "metric_mode", "full") == "full" else "core"
        medical_name = f"medical_overlap27_{args.score_mode}_{metric_suffix}"
        if save_root.name == medical_name:
            return save_root
        return save_root / medical_name
    if getattr(args, "family_filter", None):
        pilot_root = save_root / "ecg_excitation_pilot"
        run_name = f"{args.score_mode}_{args.metric_mode}"
        return pilot_root / run_name
    if getattr(args, "only_overlap_uni", False):
        overlap_count = getattr(args, "max_files", None) or 350
        if getattr(args, "score_mode", "default") != "default":
            overlap_name = f"overlap{overlap_count}_{args.score_mode}"
        else:
            overlap_name = f"overlap{overlap_count}_{args.mode}"
        if save_root.name == overlap_name:
            return save_root
        return save_root / overlap_name
    return _mode_save_dir(save_root, args.mode)


def _parse_train_index(filename, n_samples):
    parts = Path(filename).stem.split("_")
    try:
        idx = parts.index("tr")
        train_index = int(parts[idx + 1])
    except (ValueError, IndexError):
        try:
            train_index = int(parts[-3])
        except (ValueError, IndexError):
            train_index = max(1, n_samples // 5)
    return int(max(1, min(train_index, n_samples)))


def _display_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(_repo_root()))
    except ValueError:
        return str(path)


def _recursive_csv_paths(root):
    root = Path(root)
    if root.is_file() and root.suffix.lower() == ".csv":
        return [root.resolve()]
    if not root.exists():
        return []
    return sorted(path.resolve() for path in root.rglob("*.csv") if path.is_file())


def _read_file_list(file_list_path):
    file_list_path = _resolve_path(file_list_path)
    if not file_list_path.exists():
        return []
    if file_list_path.suffix.lower() == ".txt":
        lines = file_list_path.read_text(encoding="utf-8").splitlines()
        return [
            line.strip()
            for line in lines
            if line.strip() and line.strip().lower() != "file_name"
        ]
    listed = pd.read_csv(file_list_path)
    column = "file_name" if "file_name" in listed.columns else listed.columns[0]
    return listed[column].dropna().astype(str).tolist()


def _file_list_path_priority(path):
    parts = [part.lower() for part in Path(path).parts]
    old_penalty = 1 if any("old" in part for part in parts) else 0
    canonical_penalty = 0 if "tsb-ad-u" in parts else 1
    return (old_penalty, canonical_penalty, len(parts), str(path).lower())


def _select_candidate_paths(all_csv_paths, filename=None, file_list=None):
    if filename:
        requested = Path(filename)
        if requested.is_absolute() and requested.exists():
            return [requested.resolve()]
        matches = [
            path
            for path in all_csv_paths
            if path.name == filename or str(path).endswith(str(requested))
        ]
        return matches

    if file_list:
        requested = []
        seen_requested = set()
        for item in _read_file_list(file_list):
            name = Path(item).name
            if name and name not in seen_requested:
                requested.append(name)
                seen_requested.add(name)

        candidates_by_name = {}
        requested_names = set(requested)
        for path in all_csv_paths:
            if path.name not in requested_names:
                continue
            current = candidates_by_name.get(path.name)
            if current is None or _file_list_path_priority(path) < _file_list_path_priority(current):
                candidates_by_name[path.name] = path

        return [candidates_by_name[name] for name in requested if name in candidates_by_name]

    return list(all_csv_paths)


def _inspect_csv_file(path):
    row = {
        "path": str(Path(path).resolve()),
        "relative_path": _display_path(path),
        "file": Path(path).name,
        "label_detected": False,
        "data_columns": 0,
        "valid": False,
        "skip_reason": "",
    }

    try:
        preview = pd.read_csv(path, nrows=5)
    except Exception as exc:
        row["skip_reason"] = f"read error: {exc}"
        return row

    if preview.empty:
        row["skip_reason"] = "empty csv"
        return row

    row["label_detected"] = "Label" in preview.columns
    if not row["label_detected"]:
        row["skip_reason"] = "missing Label column"
        return row

    data_columns = [column for column in preview.columns if column != "Label"]
    row["data_columns"] = len(data_columns)
    if len(data_columns) != 1:
        row["skip_reason"] = f"expected 1 data column, found {len(data_columns)}"
        return row

    try:
        pd.to_numeric(preview[data_columns[0]].dropna())
    except Exception:
        row["skip_reason"] = "non-numeric data column"
        return row

    try:
        pd.to_numeric(preview["Label"].dropna())
    except Exception:
        row["skip_reason"] = "non-numeric Label column"
        return row

    row["valid"] = True
    return row


def _build_dataset_audit(args):
    if args.data_root:
        dataset_path = _resolve_path(args.data_root)
    else:
        dataset_path = _resolve_path(args.data_direc)

    all_csv_paths = _recursive_csv_paths(dataset_path)
    candidate_paths = _select_candidate_paths(
        all_csv_paths, filename=args.filename, file_list=args.file_list
    )
    family_tokens = _parse_family_filter(getattr(args, "family_filter", None))
    if family_tokens:
        candidate_paths = [
            path for path in candidate_paths if _path_matches_family_filter(path, family_tokens)
        ]
    inspected = [_inspect_csv_file(path) for path in candidate_paths]
    valid = [row for row in inspected if row["valid"]]
    skipped = [row for row in inspected if not row["valid"]]

    max_files = args.max_files
    if args.smoke and max_files is None:
        max_files = 10
    if max_files is not None:
        limit = max(0, int(max_files))
        limited = valid[:limit]
        for row in valid[limit:]:
            skipped.append({**row, "valid": False, "skip_reason": "max_files limit"})
        valid = limited

    return {
        "dataset_path": dataset_path.resolve(),
        "all_csv_paths": all_csv_paths,
        "candidate_paths": candidate_paths,
        "inspected": inspected,
        "valid": valid,
        "skipped": skipped,
    }


def _score_statistics(score):
    score = _clean_series(score)
    if score.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.min(score)), float(np.max(score)), float(np.median(score))


def _prediction_from_score(score):
    score = _clean_series(score)
    if score.size == 0 or float(np.std(score)) < EPS:
        return np.zeros(score.size, dtype=bool)
    return score > (float(np.mean(score)) + 3.0 * float(np.std(score)))


def _metric_subset(metrics):
    keys = [
        "AUC-PR",
        "AUC-ROC",
        "VUS-PR",
        "VUS-ROC",
        "Standard-F1",
        "PA-F1",
        "Event-based-F1",
        "R-based-F1",
        "Affiliation-F",
    ]
    return {key: metrics.get(key, np.nan) for key in keys}


def _simple_f1_from_prediction(label, pred):
    y = np.asarray(label).astype(bool).reshape(-1)
    p = np.asarray(pred).astype(bool).reshape(-1)
    if y.size != p.size or y.size == 0:
        return np.nan
    tp = float(np.count_nonzero(y & p))
    fp = float(np.count_nonzero(~y & p))
    fn = float(np.count_nonzero(y & ~p))
    denom = 2.0 * tp + fp + fn
    if denom < EPS:
        return 0.0
    return float((2.0 * tp) / denom)


def _get_core_metrics(score, label):
    grader = basic_metricor()
    auc_roc = grader.metric_ROC(label, score)
    auc_pr = grader.metric_PR(label, score)
    pred = _prediction_from_score(score)
    return {
        "AUC-PR": auc_pr,
        "AUC-ROC": auc_roc,
        "Standard-F1": _simple_f1_from_prediction(label, pred),
    }


def _run_one_file(
    csv_path,
    save_dir,
    mode,
    hp,
    no_score_csv=False,
    max_length=None,
    score_mode="default",
    metric_mode="full",
):
    mode = _normalize_mode(mode)
    score_mode = _normalize_score_mode(score_mode)
    metric_mode = _normalize_metric_mode(metric_mode)
    path = Path(csv_path).resolve()
    filename = path.name
    row = {
        "file": filename,
        "path": _display_path(path),
        "family": _family_from_filename(filename),
        "mode": mode,
        "score_mode": score_mode,
        "score_path": "",
        "metric_mode": metric_mode,
        "status": "error",
        "n_samples": 0,
        "original_n_samples": 0,
        "truncated": False,
        "score_length": 0,
        "length_ok": False,
        "label_detected": False,
        "score_nonfinite_after_cleanup": None,
        "score_min": np.nan,
        "score_max": np.nan,
        "score_median": np.nan,
        "runtime_sec": np.nan,
        "error": "",
    }

    if not path.exists():
        row["error"] = f"missing file: {path}"
        return row

    try:
        df = pd.read_csv(path).dropna()
        if "Label" in df.columns:
            row["label_detected"] = True
            data = df.drop(columns=["Label"]).values.astype(float)
            label = df["Label"].astype(int).to_numpy()
        else:
            data = df.values.astype(float)
            label = None
        if data.ndim != 2 or data.shape[1] != 1:
            raise ValueError(f"expected one data column plus optional Label, got {data.shape}")

        row["original_n_samples"] = int(data.shape[0])
        if max_length is not None:
            max_length = int(max_length)
            if max_length <= 0:
                raise ValueError("--max_length must be a positive integer")
            if data.shape[0] > max_length:
                data = data[:max_length, :]
                if label is not None:
                    label = label[:max_length]
                row["truncated"] = True

        row["n_samples"] = int(data.shape[0])

        start_time = time.time()
        if score_mode == "family_gated":
            row["score_path"] = (
                "excitation"
                if row["family"] in EXCITATION_FAMILIES
                else "unsupervised_offline_default"
            )
        else:
            row["score_path"] = score_mode
        score = run_HSF_AD_ScoreMode(data, filename, mode, score_mode, HP=hp)
        row["runtime_sec"] = round(time.time() - start_time, 6)

        score = _clean_series(score)
        row["score_length"] = int(score.size)
        row["length_ok"] = bool(score.size == data.shape[0])
        row["score_nonfinite_after_cleanup"] = int(np.size(score) - np.count_nonzero(np.isfinite(score)))
        row["score_min"], row["score_max"], row["score_median"] = _score_statistics(score)

        score_frame = pd.DataFrame({"score": score})
        if label is not None and label.size == score.size:
            score_frame["Label"] = label
        if not no_score_csv:
            score_frame.to_csv(save_dir / f"{Path(filename).stem}_scores.csv", index=False)

        if label is not None and label.size == score.size:
            sliding_window = find_length_rank(data, rank=1)
            if metric_mode == "core":
                metrics = _get_core_metrics(score, label)
            else:
                pred = _prediction_from_score(score)
                metrics = get_metrics(score, label, slidingWindow=sliding_window, pred=pred)
            row.update(_metric_subset(metrics))

        row["status"] = "ok"
        return row
    except Exception as exc:
        row["error"] = str(exc)
        return row


def _write_report(results, save_dir, command_args):
    report_path = save_dir / "hsf_smoke_report.md"
    ok_rows = [row for row in results if row.get("status") == "ok"]
    error_rows = [row for row in results if row.get("status") != "ok"]
    length_ok = all(bool(row.get("length_ok")) for row in ok_rows) if ok_rows else False
    nonfinite_after = sum(
        int(row.get("score_nonfinite_after_cleanup") or 0) for row in ok_rows
    )

    metric_columns = [
        "AUC-PR",
        "AUC-ROC",
        "VUS-PR",
        "VUS-ROC",
        "Standard-F1",
        "PA-F1",
        "Event-based-F1",
        "R-based-F1",
        "Affiliation-F",
    ]

    lines = [
        "# HSF_AD Smoke Report",
        "",
        f"- Anzahl getesteter Zeitreihen: {len(ok_rows)}",
        f"- Angefragte Dateien: {len(results)}",
        f"- Score-Laenge == Input-Laenge: {length_ok}",
        f"- NaN/inf nach Bereinigung: {nonfinite_after}",
        f"- Modus: {command_args.mode}",
        "",
        "## Score Summary",
        "",
        "| file | n | length_ok | nonfinite | min | max | median | status |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]

    for row in results:
        lines.append(
            "| {file} | {n_samples} | {length_ok} | {nonfinite} | {score_min:.6g} | "
            "{score_max:.6g} | {score_median:.6g} | {status} |".format(
                file=row.get("file", ""),
                n_samples=int(row.get("n_samples") or 0),
                length_ok=row.get("length_ok", False),
                nonfinite=int(row.get("score_nonfinite_after_cleanup") or 0),
                score_min=float(row.get("score_min", np.nan)),
                score_max=float(row.get("score_max", np.nan)),
                score_median=float(row.get("score_median", np.nan)),
                status=row.get("status", ""),
            )
        )

    if ok_rows and any(col in ok_rows[0] for col in metric_columns):
        lines.extend(
            [
                "",
                "## Evaluation",
                "",
                "| file | AUC-PR | AUC-ROC | VUS-PR | VUS-ROC | Standard-F1 | PA-F1 | Event-F1 | R-F1 | Affiliation-F |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in ok_rows:
            lines.append(
                "| {file} | {auc_pr:.6g} | {auc_roc:.6g} | {vus_pr:.6g} | {vus_roc:.6g} | "
                "{std_f1:.6g} | {pa_f1:.6g} | {event_f1:.6g} | {r_f1:.6g} | {aff_f:.6g} |".format(
                    file=row.get("file", ""),
                    auc_pr=float(row.get("AUC-PR", np.nan)),
                    auc_roc=float(row.get("AUC-ROC", np.nan)),
                    vus_pr=float(row.get("VUS-PR", np.nan)),
                    vus_roc=float(row.get("VUS-ROC", np.nan)),
                    std_f1=float(row.get("Standard-F1", np.nan)),
                    pa_f1=float(row.get("PA-F1", np.nan)),
                    event_f1=float(row.get("Event-based-F1", np.nan)),
                    r_f1=float(row.get("R-based-F1", np.nan)),
                    aff_f=float(row.get("Affiliation-F", np.nan)),
                )
            )

    lines.extend(["", "## Fehlerliste", ""])
    if error_rows:
        for row in error_rows:
            lines.append(f"- {row.get('file', '')}: {row.get('error', '')}")
    else:
        lines.append("- Keine offenen Fehler im HSF_AD Runner.")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _mean_metric(rows, metric_name):
    values = []
    for row in rows:
        value = row.get(metric_name, np.nan)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = np.nan
        if np.isfinite(value):
            values.append(value)
    if not values:
        return np.nan
    return float(np.mean(values))


def _write_dataset_audit_report(results, skipped_rows, audit, save_dir, command_args):
    report_path = save_dir / "hsf_dataset_audit_report.md"
    ok_rows = [row for row in results if row.get("status") == "ok"]
    failed_rows = [row for row in results if row.get("status") != "ok"]
    skip_reasons = Counter(row.get("skip_reason", "unknown") for row in skipped_rows)

    metric_means = {
        "mean AUC-PR": _mean_metric(ok_rows, "AUC-PR"),
        "mean AUC-ROC": _mean_metric(ok_rows, "AUC-ROC"),
        "mean VUS-PR": _mean_metric(ok_rows, "VUS-PR"),
        "mean VUS-ROC": _mean_metric(ok_rows, "VUS-ROC"),
    }

    lines = [
        "# HSF_AD Dataset Audit Report",
        "",
        f"- working directory: {Path.cwd()}",
        f"- data_root: {getattr(command_args, 'data_root', None)}",
        f"- dataset_path: {audit['dataset_path']}",
        f"- Anzahl rekursiv gefundener CSVs: {len(audit['all_csv_paths'])}",
        f"- Anzahl verwendeter CSVs: {len(results)}",
        f"- Anzahl erfolgreich bewerteter CSVs: {len(ok_rows)}",
        f"- Anzahl uebersprungener CSVs: {len(skipped_rows)}",
        f"- Anzahl fehlgeschlagener Bewertungslaeufe: {len(failed_rows)}",
        "",
        "## Dataset Diagnose",
        "",
        "Erste Beispiel-Dateipfade aus der rekursiven Suche:",
    ]

    for path in audit["all_csv_paths"][:20]:
        lines.append(f"- {_display_path(path)}")
    if not audit["all_csv_paths"]:
        lines.append("- Keine CSV-Dateien gefunden.")

    lines.extend(["", "## Skip Gruende", ""])
    if skip_reasons:
        for reason, count in skip_reasons.most_common():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- Keine Skip-Gruende.")

    lines.extend(["", "## Erste 20 verwendete Dateien", ""])
    if results:
        for row in results[:20]:
            lines.append(
                f"- {row.get('path', row.get('file'))} "
                f"(labels={row.get('label_detected', False)}, status={row.get('status')})"
            )
    else:
        lines.append("- Keine Dateien verwendet.")

    lines.extend(["", "## Erste 20 uebersprungene Dateien", ""])
    if skipped_rows:
        for row in skipped_rows[:20]:
            lines.append(
                f"- {row.get('relative_path', row.get('path'))} "
                f"(labels={row.get('label_detected', False)}, reason={row.get('skip_reason')})"
            )
    else:
        lines.append("- Keine Dateien uebersprungen.")

    lines.extend(
        [
            "",
            "## Ergebnis Mittelwerte",
            "",
        ]
    )
    for metric_name, value in metric_means.items():
        if np.isfinite(value):
            lines.append(f"- {metric_name}: {value:.6g}")
        else:
            lines.append(f"- {metric_name}: n/a")

    if failed_rows:
        lines.extend(["", "## Fehlerliste", ""])
        for row in failed_rows:
            lines.append(f"- {row.get('path', row.get('file'))}: {row.get('error')}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Running HSF_AD on TSB-AD-U")
    parser.add_argument("--filename", type=str, default=None, help="single CSV file to run")
    parser.add_argument("--data_direc", type=str, default="Datasets/TSB-AD-U/")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--file_list", type=str, default=None)
    parser.add_argument(
        "--family_filter",
        type=str,
        default=None,
        help="comma-separated filename tokens to include, for example MITDB,SVDB,SED",
    )
    parser.add_argument(
        "--only_overlap_uni",
        action="store_true",
        help=(
            "run only files listed in "
            "benchmark_exp/benchmark_eval_results/uni_mergedTable_VUS-PR.csv"
        ),
    )
    parser.add_argument("--AD_Name", type=str, default="HSF_AD")
    parser.add_argument("--save_dir", type=str, default="Results/HSF_AD")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="debug only: truncate each time series to this many samples before scoring",
    )
    parser.add_argument(
        "--no_score_csv",
        action="store_true",
        help="do not write per-file score CSVs",
    )
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help=(
            "write only hsf_summary.csv; implies --no_score_csv and skips markdown reports"
        ),
    )
    parser.add_argument(
        "--core_metrics_only",
        action="store_true",
        help="legacy alias for --metric_mode core",
    )
    parser.add_argument(
        "--metric_mode",
        choices=list(HSF_METRIC_MODES),
        default="full",
        help="core computes AUC-PR, AUC-ROC, and simple F1; full includes TSB-AD VUS metrics",
    )
    parser.add_argument(
        "--score_mode",
        choices=list(HSF_SCORE_MODES),
        default="default",
        help=(
            "default uses current HSF score, unsupervised_offline forces clean "
            "unsupervised HSF, excitation uses inverse excitation features, "
            "hybrid mixes both, family_gated routes MITDB/SVDB/SED to excitation"
        ),
    )
    parser.add_argument("--smoke", action="store_true", help="run at most 10 files")
    parser.add_argument(
        "--mode",
        choices=list(HSF_MODES) + list(HSF_MODE_ALIASES),
        default="semisupervised_offline",
        help=(
            "HSF_AD information mode. Legacy aliases: "
            "semisupervised=semisupervised_offline, "
            "unsupervised=unsupervised_offline."
        ),
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.mode = _normalize_mode(args.mode)
    args.score_mode = _normalize_score_mode(args.score_mode)
    args.metric_mode = _normalize_metric_mode(args.metric_mode)
    if args.only_overlap_uni and args.file_list is None:
        args.file_list = DEFAULT_UNI_OVERLAP_TABLE
    if args.summary_only:
        args.no_score_csv = True
    if args.core_metrics_only:
        args.metric_mode = "core"

    save_root = _resolve_path(args.save_dir)
    save_dir = _run_save_dir(save_root, args)
    save_dir.mkdir(parents=True, exist_ok=True)

    audit = _build_dataset_audit(args)

    print("Dataset diagnosis:")
    print(f"  working_directory: {Path.cwd()}")
    print(f"  dataset_path: {audit['dataset_path']}")
    print(f"  recursive_csv_count: {len(audit['all_csv_paths'])}")
    print(f"  family_filter: {args.family_filter}")
    print(f"  score_mode: {args.score_mode}")
    print(f"  metric_mode: {args.metric_mode}")
    for row in audit["inspected"][:5]:
        status = "use" if row["valid"] else "skip"
        reason = row.get("skip_reason") or "ok"
        print(
            f"  sample: {row['relative_path']} "
            f"labels={row['label_detected']} status={status} reason={reason}"
        )

    hp = {
        "weights": [1.0, 1.0, 1.0, 1.0, 1.0],
    }

    results = []
    run_start = time.time()
    total_files = len(audit["valid"])
    summary_path = save_dir / "hsf_summary.csv"
    partial_summary_path = save_dir / "hsf_summary.partial.csv"
    for index, row in enumerate(audit["valid"], start=1):
        file_name = Path(row["path"]).name
        before_file = time.time()
        print(
            f"[{args.mode}/{args.score_mode}/{args.metric_mode}] {index}/{total_files} start "
            f"elapsed={before_file - run_start:.1f}s file={file_name}",
            flush=True,
        )
        result = _run_one_file(
            row["path"],
            save_dir=save_dir,
            mode=args.mode,
            hp=hp,
            no_score_csv=args.no_score_csv,
            max_length=args.max_length,
            score_mode=args.score_mode,
            metric_mode=args.metric_mode,
        )
        results.append(result)
        pd.DataFrame(results).to_csv(partial_summary_path, index=False)
        after_file = time.time()
        print(
            f"[{args.mode}/{args.score_mode}/{args.metric_mode}] {index}/{total_files} done "
            f"status={result.get('status')} "
            f"file_runtime={result.get('runtime_sec')}s "
            f"elapsed={after_file - run_start:.1f}s file={file_name}",
            flush=True,
        )

    pd.DataFrame(results).to_csv(summary_path, index=False)
    if args.summary_only:
        report_path = None
        audit_report_path = None
    else:
        report_path = _write_report(results, save_dir, args)
        audit_report_path = _write_dataset_audit_report(
            results, audit["skipped"], audit, save_dir, args
        )

    ok_count = sum(1 for row in results if row.get("status") == "ok")
    error_count = len(results) - ok_count
    print(
        f"HSF_AD finished: ok={ok_count}, errors={error_count}, "
        f"skipped={len(audit['skipped'])}"
    )
    print(f"Summary: {summary_path}")
    if report_path is not None:
        print(f"Report: {report_path}")
    if audit_report_path is not None:
        print(f"Dataset audit report: {audit_report_path}")
    if error_count:
        for row in results:
            if row.get("status") != "ok":
                print(f"ERROR {row.get('file')}: {row.get('error')}")


if __name__ == "__main__":
    main()
