"""TSB-AD runner for the CHARM embedding anomaly detector.

Two detectors:

  CHARM_kNN (semi-supervised, fit on the clean train split)
      Two sub-detectors, each run at several window sizes. Every scale's pointwise score
      is z-scored, scales are combined by element-wise max, and the two are summed:
        embedding : channel-mean-pooled CHARM embeddings, cosine kNN, windows {64, 128, 256}
        statistics: per-window [std, range, max, min, mean], L2 kNN, windows {16, ..., 512}
                    (window 128 only for multivariate series with at most 3 channels)

  CHARM_ZS (zero-shot, no train split)
      Bootstrap cosine kNN against a pseudo-clean reference picked by an averaged IsolationForest
      from the series itself, min-max-summed with the per-window std.

Embeddings come from the CHARM endpoint (`CHARM_BASE_URL`, `CHARM_API_KEY`) with
`aggregate=False`; patches are max-pooled over time and channels are kept separate.

Usage:
    python Run_CHARM.py --filename <series>.csv --data_dir Datasets/TSB-AD-U/ --model CHARM_kNN
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial.distance import cdist
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank

HP = {
    "window_size": 128,
    "min_window": 64,
    "k": 3,
    "ref_cap": 10000,
    "emb_scales": (64, 128, 256),
    "stats_scales": (16, 32, 64, 128, 256, 512),
    "stats_gate_max_c": 3,
    "if_estimators": 200,
    "if_max_samples": 256,
    "if_forests": 20,
    "boot_quantile": 0.70,
    "zs_std_weight": 0.40,
}
MAX_REQUEST_POINTS = 500_000
RNG = np.random.RandomState(0)


def zscore(a):
    return (a - a.mean()) / (a.std() + 1e-9)


def minmax(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-12)


def effective_window(length, max_window, k, min_window):
    """Shrink the window (and, if needed, widen the stride) so a series of `length`
    yields enough windows for kNN. Returns (window, stride), or None if too short."""
    if length < min_window:
        return None
    min_windows = max(2 * k, 10)
    ws = min(max_window, length - (min_windows - 1))
    if ws >= min_window:
        return ws, 1
    ws = min(max_window, length)
    if length - ws + 1 >= min_windows:
        return ws, max(1, (length - ws) // (min_windows - 1))
    return ws, 1


def make_windows(x, ws, stride):
    """(L, C) -> (n, ws, C)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if len(x) < ws:
        return x[None]
    return np.moveaxis(sliding_window_view(x, ws, axis=0)[::stride], -1, 1)


def window_stats(windows):
    """(n, ws, C) -> (n, 5): std, range, max, min, mean over each window."""
    w = windows.astype(np.float64)
    mx, mn = w.max(axis=(1, 2)), w.min(axis=(1, 2))
    return np.stack([w.std(axis=(1, 2)), mx - mn, mx, mn, w.mean(axis=(1, 2))], 1)


def to_pointwise(window_scores, ws, stride, length):
    """Average the scores of all windows covering each timestep."""
    n = len(window_scores)
    if stride == 1:
        csum = np.concatenate([[0.0], np.cumsum(window_scores)])
        t = np.arange(length)
        lo, hi = np.clip(t - ws + 1, 0, n), np.clip(t + 1, 0, n)
        return (csum[hi] - csum[lo]) / np.maximum(hi - lo, 1)
    acc, cnt = np.zeros(length), np.zeros(length)
    for i, s in enumerate(window_scores):
        a = i * stride
        b = min(a + ws, length)
        acc[a:b] += s
        cnt[a:b] += 1
    return acc / np.maximum(cnt, 1)


def cap_ref(ref):
    if len(ref) <= HP["ref_cap"]:
        return ref
    return ref[RNG.choice(len(ref), HP["ref_cap"], replace=False)]


def unit_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def knn_score(query, ref, metric, k=HP["k"], chunk=4096):
    """Mean distance to the k nearest reference rows (metric: 'cosine' or 'euclidean')."""
    k = min(k, len(ref))
    if metric == "cosine":
        query, ref = unit_rows(query), unit_rows(ref)
    out = np.empty(len(query), np.float32)
    for i in range(0, len(query), chunk):
        q = query[i:i + chunk]
        d = 1.0 - q @ ref.T if metric == "cosine" else cdist(q, ref)
        out[i:i + chunk] = np.partition(d, k - 1, axis=1)[:, :k].mean(1)
    return out


_client = None


def embed(windows):
    """(n, ws, C) -> (n, C, D): max over time patches, channels kept separate."""
    global _client
    if _client is None:
        from charm import CharmClient
        _client = CharmClient(base_url=os.environ["CHARM_BASE_URL"],
                              api_key=os.environ.get("CHARM_API_KEY", "token"), timeout=300)
    _, ws, C = windows.shape
    batch = max(1, min(4096 // C, MAX_REQUEST_POINTS // (ws * C)))
    names = [f"ch_{c}" for c in range(C)]
    out = []
    for i in range(0, len(windows), max(batch, 2048)):
        chunk = windows[i:i + max(batch, 2048)]
        resp = request_with_retry(chunk.tolist(), [names] * len(chunk), batch)
        out.append(np.nan_to_num(resp.embeds.max(axis=1)).astype(np.float32))
    return np.concatenate(out)


def request_with_retry(ts_array, descriptions, batch, tries=5):
    """Retry transient server errors (e.g. GPU OOM) with a halved batch size."""
    for attempt in range(tries):
        try:
            return _client.embeddings.create(descriptions=descriptions, ts_array=ts_array,
                                             batch_size=batch, return_tensors="np", aggregate=False)
        except Exception as e:
            if attempt == tries - 1 or "Batch too large" in str(e):
                raise
            batch = max(1, batch // 2)
            time.sleep(2 * (attempt + 1))


def scale_curve(train, test, ws, use_stats):
    """Z-scored pointwise kNN score of `test` against `train` at one window size, or None."""
    min_window = min(HP["min_window"], ws)
    fit = effective_window(len(train), ws, HP["k"], min_window)
    query = effective_window(len(test), ws, HP["k"], min_window)
    if fit is None or query is None:
        return None
    ref_w, query_w = make_windows(train, *fit), make_windows(test, *query)
    if use_stats:
        ref, q = window_stats(ref_w), window_stats(query_w)
        mu, sd = ref.mean(0), ref.std(0) + 1e-8
        scores = knn_score((q - mu) / sd, cap_ref((ref - mu) / sd), "euclidean")
    else:
        scores = knn_score(embed(query_w).mean(1), cap_ref(embed(ref_w).mean(1)), "cosine")
    return zscore(to_pointwise(scores, query[0], query[1], len(test)))


def multiscale_max(train, test, scales, use_stats):
    curves = [c for ws in scales if (c := scale_curve(train, test, ws, use_stats)) is not None]
    return np.max(curves, axis=0) if curves else np.zeros(len(test))


def run_CHARM_kNN(train, test):
    n_channels = train.shape[1]
    stats_scales = (HP["window_size"],) if 1 < n_channels <= HP["stats_gate_max_c"] else HP["stats_scales"]
    score = (multiscale_max(train, test, HP["emb_scales"], use_stats=False)
             + multiscale_max(train, test, stats_scales, use_stats=True))
    return minmax(score)


def zero_shot_score(emb, windows, ws, stride, length, n_forests=HP["if_forests"]):
    """Score windows by kNN distance to a pseudo-clean reference (the windows an averaged
    IsolationForest finds least suspicious), plus the per-window std."""
    unit = unit_rows(emb)
    suspicion = np.mean([-IsolationForest(n_estimators=HP["if_estimators"], max_samples=HP["if_max_samples"],
                                          random_state=seed, n_jobs=4).fit(unit).score_samples(unit)
                         for seed in range(n_forests)], axis=0)
    ref = emb[suspicion <= np.quantile(suspicion, HP["boot_quantile"])]
    boot = knn_score(emb, cap_ref(ref), "cosine")
    score = minmax(boot) + HP["zs_std_weight"] * minmax(window_stats(windows)[:, 0])
    return minmax(to_pointwise(score, ws, stride, length))


def run_CHARM_ZS(data):
    fit = effective_window(len(data), HP["window_size"], HP["k"], HP["min_window"])
    if fit is None:
        return np.zeros(len(data))
    windows = make_windows(data, *fit)
    return zero_shot_score(embed(windows).mean(1), windows, *fit, len(data))


DETECTORS = {"CHARM_kNN": run_CHARM_kNN, "CHARM_ZS": run_CHARM_ZS}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--filename", required=True)
    ap.add_argument("--data_dir", default="Datasets/TSB-AD-M/")
    ap.add_argument("--model", default="CHARM_kNN", choices=list(DETECTORS))
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.data_dir, args.filename)).dropna()
    data = df.iloc[:, :-1].values.astype(float)
    label = df["Label"].astype(int).to_numpy()
    train_len = int(args.filename.split("_")[-3])

    if args.model == "CHARM_kNN":
        score = run_CHARM_kNN(data[:train_len], data)
    else:
        score = run_CHARM_ZS(data)

    window = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
    metrics = get_metrics(score, label, slidingWindow=window)
    print(args.model, args.filename, {k: round(v, 4) for k, v in metrics.items()})
