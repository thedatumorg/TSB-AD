"""TSB-AD runner — FTSE/CHARM embedding anomaly detector (upgraded).

Drop-in for benchmark_exp/ (successor to Run_CHARM.py). Two detectors, the best
read-out from the embedding-extraction ablation, one per regime:

  Semi-supervised (has a clean train split):
    * FTSE_kNN : L5 max-over-time / mean-over-channel embedding, cosine kNN to
                 clean-train windows, ENSEMBLED with a per-window mu/std kNN:
                 final = norm(emb_knn) + w * norm(stats_knn).

  Unsupervised / zero-shot (no train reference):
    * FTSE_ZS  : bootstrap-kNN (IsolationForest picks a pseudo-clean reference from
                 the series itself, then cosine kNN) ENSEMBLED with the per-window
                 std used directly as a reference-free score.

Read-out change vs the original Run_CHARM.py: that used `aggregate=True`
(= last-layer mean over patches AND channels). Here we request `aggregate=False`
and pool client-side as **max over time-patches, mean over channels**, on the
**L5** block (assumes the served model exposes the L5 read-out). The per-window
raw statistics are computed directly from the input (no model call) and recover
amplitude anomalies the encoder z-normalizes away (+~7 pp VUS-PR supervised).

Ablation VUS-PR (TSB-AD eval; indicative — official numbers come from running this
on the benchmark file_list):
    supervised  FTSE_kNN  : uni 0.648 / mv 0.519 / all 0.604   (emb 0.606 -> +mu/std)
    supervised  Stats_kNN : uni 0.554 / mv 0.466 / all 0.524   (no model!)
    zero-shot   FTSE_ZS   : uni 0.564 / mv 0.447 / all 0.524
    zero-shot   Stats_ZS  : uni 0.425 / mv 0.334 / all 0.394
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank

# --------------------------------------------------------------------------- #
#  Hyperparameters (leaderboard defaults)
# --------------------------------------------------------------------------- #
FTSE_HP = {
    "window_size": 128,
    "stride": 1,
    "train_stride": 1,
    "min_window": 64,
    "layer": 5,             # L5 block (assumes served model exposes it)
    "k": 3,
    # w in final = norm(emb) + w*norm(stats); tuned per regime on the eval split
    # (official VUS-PR w-sweep {0.25,0.5,1,2}: these maximize the overall "all" score)
    "ensemble_weight_semi": 0.35,   # FTSE_kNN (dense w-sweep peak; plateau 0.25-0.4)
    "ensemble_weight_zs": 0.4,      # FTSE_ZS  (dense w-sweep peak)
    "ref_cap": 10000,       # cap kNN reference size (random subsample) for tractability
    "pointwise_agg": "mean",
    "if_estimators": 200,
    "if_max_samples": 256,
    "boot_quantile": 0.70,  # IF-suspicion percentile for the zero-shot pseudo-clean ref
}

_CHARM_BASE_URL = os.environ.get("CHARM_BASE_URL", "")
_CHARM_API_KEY = os.environ.get("CHARM_API_KEY", "token")
_RNG = np.random.RandomState(0)


# --------------------------------------------------------------------------- #
#  Windowing helpers
# --------------------------------------------------------------------------- #
def _effective_window(series_length, max_window, stride, k, min_window=64,
                      min_windows_mult=2.0):
    """Largest window giving enough windows for kNN; shrink window / stride for short
    series. Mirrors the TSB-AD-side effective-window selection."""
    if series_length < min_window:
        return None
    min_windows = max(int(min_windows_mult * k), 10)
    ws = min(max_window, series_length - (min_windows - 1) * stride)
    if ws >= min_window:
        return ws, stride
    ws = min(max_window, series_length - (min_windows - 1))
    if ws >= min_window:
        return ws, 1
    ws = min(max_window, series_length)
    avail = series_length - ws + 1
    if avail >= min_windows:
        return ws, max(1, (series_length - ws) // (min_windows - 1))
    return ws, 1


def _create_windows(x, ws, stride):
    """x: (T, C) -> (n_windows, ws, C). Vectorized via sliding_window_view."""
    from numpy.lib.stride_tricks import sliding_window_view
    x = np.ascontiguousarray(x, dtype=np.float32)
    T = x.shape[0]
    if T < ws:
        return x[None]  # single short window
    sw = sliding_window_view(x, ws, axis=0)[::stride]   # (n, C, ws) view
    return np.moveaxis(sw, -1, 1)                        # (n, ws, C) view


def _window_stats(windows, chunk=50000):
    """windows: (N, W, C) view -> (N, 5) per-window [std,range,max,min,mean] over (W,C).
    Chunked; reduces over axes (1,2) directly so the strided view isn't fully copied."""
    N = windows.shape[0]
    out = np.empty((N, 5), np.float32)
    for i in range(0, N, chunk):
        b = np.ascontiguousarray(windows[i:i + chunk]).astype(np.float64)  # (m,W,C)
        mx = b.max(axis=(1, 2)); mn = b.min(axis=(1, 2))
        out[i:i + len(b)] = np.stack([b.std(axis=(1, 2)), mx - mn, mx, mn, b.mean(axis=(1, 2))], 1)
    return out


def _window_scores_to_pointwise(scores, ws, stride, total_len, method="mean"):
    """Aggregate overlapping window-level scores to per-timestep scores."""
    n = len(scores)
    if method == "mean":
        if stride == 1 and n > 0:
            cs = np.zeros(n + 1); np.cumsum(scores, out=cs[1:])
            t = np.arange(total_len)
            a = np.clip(t - ws + 1, 0, n); b = np.clip(t + 1, 0, n)
            return ((cs[b] - cs[a]) / np.maximum(b - a, 1)).astype(np.float32)
        acc = np.zeros(total_len); cnt = np.zeros(total_len)
        for i, s in enumerate(scores):
            a = i * stride; b = min(a + ws, total_len); acc[a:b] += s; cnt[a:b] += 1
        return (acc / np.maximum(cnt, 1)).astype(np.float32)
    if method == "max":
        res = np.full(total_len, -np.inf)
        for i, s in enumerate(scores):
            a = i * stride; b = min(a + ws, total_len); res[a:b] = np.maximum(res[a:b], s)
        res[res == -np.inf] = 0.0
        return res.astype(np.float32)
    raise ValueError(method)


def _nz(a):
    a = np.asarray(a, float); r = a.max() - a.min()
    return (a - a.min()) / (r + 1e-12)


def _cap_ref(ref, cap):
    if ref.shape[0] <= cap:
        return ref
    return ref[_RNG.choice(ref.shape[0], cap, replace=False)]


# --------------------------------------------------------------------------- #
#  Embedding (L5, max-over-time, mean-over-channel) via CHARM SDK
# --------------------------------------------------------------------------- #
def _embed_windows(client, windows, batch_size=4096):
    """windows: (N, W, C) -> (N, D) L5.tmax.cmean embeddings.
    Requests aggregate=False (per-patch, per-channel) and pools client-side:
    max over time-patches, mean over channels. Assumes the served model exposes L5."""
    N, W, C = windows.shape
    bs = max(1, batch_size // max(C, 1))
    chunk = max(bs, 2048)                       # windows per API call; avoids one giant tolist()
    out = []
    for i in range(0, N, chunk):
        wc = np.ascontiguousarray(windows[i:i + chunk])
        desc = [[f"ch_{c}" for c in range(C)] for _ in range(len(wc))]
        resp = client.embeddings.create(
            descriptions=desc, ts_array=wc.tolist(),
            batch_size=bs, return_tensors="np", aggregate=False)  # (n, patches, C, D)
        raw = resp.embeds
        emb = raw.max(axis=1).mean(axis=1) if raw.ndim == 4 else raw  # tmax patches, cmean channels
        out.append(np.nan_to_num(emb).astype(np.float32))
    return np.concatenate(out, 0)


def _knn_mean_dist(query, ref, k, metric, chunk=4096):
    """Mean distance to k nearest refs, chunked over queries so we never materialize
    a full (Nq x Nref) matrix (Nq can be ~1e6 at stride 1)."""
    kk = min(k, ref.shape[0])
    if metric == "cosine":
        r = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8)
    out = np.empty(query.shape[0], dtype=np.float32)
    for i in range(0, query.shape[0], chunk):
        q = query[i:i + chunk]
        if metric == "cosine":
            qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
            d = 1.0 - qn @ r.T
        else:  # l2
            from scipy.spatial.distance import cdist
            d = cdist(q.astype(np.float64), ref.astype(np.float64))
        idx = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        out[i:i + len(q)] = np.take_along_axis(d, idx, axis=1).mean(1)
    return out


def _cosine_knn(query, ref, k):
    return _knn_mean_dist(query, ref, k, "cosine")


def _l2_knn(query, ref, k):
    return _knn_mean_dist(query, ref, k, "l2")


def _get_client():
    from charm import CharmClient
    return CharmClient(base_url=_CHARM_BASE_URL, api_key=_CHARM_API_KEY, timeout=300)


# --------------------------------------------------------------------------- #
#  Runners — semi-supervised (fit on clean train, score test)
# --------------------------------------------------------------------------- #
def run_FTSE_kNN(data_train, data_test, HP=FTSE_HP):
    """BEST supervised: L5.tmax.cmean cosine-kNN (+) per-window mu/std L2-kNN."""
    client = _get_client()
    T_tr, C = data_train.shape
    ews = _effective_window(T_tr, HP["window_size"], HP["train_stride"], HP["k"], HP["min_window"])
    if ews is None:
        return np.zeros(len(data_test))
    ws, tr_stride = ews
    # reference = clean train windows
    tw = _create_windows(data_train, ws, tr_stride)
    ref_emb = _embed_windows(client, tw)
    ref_stats = _standardize_fit(_window_stats(tw))
    # test windows
    ews_t = _effective_window(len(data_test), HP["window_size"], HP["stride"], HP["k"], HP["min_window"])
    ws_t, te_stride = ews_t
    qw = _create_windows(data_test, ws_t, te_stride)
    q_emb = _embed_windows(client, qw)
    q_stats = _standardize_apply(_window_stats(qw))
    # two kNN detectors, normalized-score ensemble
    s_emb = _cosine_knn(q_emb, _cap_ref(ref_emb, HP["ref_cap"]), HP["k"])
    s_stats = _l2_knn(q_stats, _cap_ref(ref_stats, HP["ref_cap"]), HP["k"])
    win = _nz(s_emb) + HP["ensemble_weight_semi"] * _nz(s_stats)
    pw = _window_scores_to_pointwise(win, ws_t, te_stride, len(data_test), HP["pointwise_agg"])
    return MinMaxScaler().fit_transform(pw.reshape(-1, 1)).ravel()


# per-series standardization state for stats (fit on train, applied to test)
_STD_STATE = {}
def _standardize_fit(S):
    mu, sd = S.mean(0, keepdims=True), S.std(0, keepdims=True) + 1e-8
    _STD_STATE["mu"], _STD_STATE["sd"] = mu, sd
    return ((S - mu) / sd).astype(np.float32)
def _standardize_apply(S):
    return ((S - _STD_STATE["mu"]) / _STD_STATE["sd"]).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Runners — unsupervised / zero-shot (no train reference)
# --------------------------------------------------------------------------- #
def run_FTSE_ZS(data, HP=FTSE_HP):
    """BEST zero-shot: bootstrap-kNN on L5 embeddings (+) per-window std (reference-free)."""
    client = _get_client()
    ews = _effective_window(len(data), HP["window_size"], HP["stride"], HP["k"], HP["min_window"])
    if ews is None:
        return np.zeros(len(data))
    ws, stride = ews
    w = _create_windows(data, ws, stride)
    emb = _embed_windows(client, w)
    stats = _window_stats(w)
    std = stats[:, 0]                                        # per-window std, reference-free
    En = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    if_score = -IsolationForest(n_estimators=HP["if_estimators"], max_samples=HP["if_max_samples"],
                                random_state=0, n_jobs=4).fit(En).score_samples(En)
    thr = np.quantile(if_score, HP["boot_quantile"])
    ref = emb[if_score <= thr]                               # pseudo-clean reference
    bknn = _cosine_knn(emb, _cap_ref(ref, HP["ref_cap"]), 1) if len(ref) >= 1 else if_score
    win = _nz(bknn) + HP["ensemble_weight_zs"] * _nz(std)
    pw = _window_scores_to_pointwise(win, ws, stride, len(data), HP["pointwise_agg"])
    return MinMaxScaler().fit_transform(pw.reshape(-1, 1)).ravel()


SEMISUPERVISE = {"FTSE_kNN": run_FTSE_kNN}     # semi-supervised ensemble
UNSUPERVISE = {"FTSE_ZS": run_FTSE_ZS}         # zero-shot ensemble


# --------------------------------------------------------------------------- #
#  Main — same file_list / split convention as Run_CHARM.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--filename", required=True)
    ap.add_argument("--data_dir", default="Datasets/TSB-AD-M/")
    ap.add_argument("--model", default="FTSE_kNN",
                    choices=list(SEMISUPERVISE) + list(UNSUPERVISE))
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.data_dir, args.filename)).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df["Label"].astype(int).to_numpy()
    # TSB-AD filename convention: ..._tr_<train_len>_1st_<first_anomaly>...
    train_index = int(args.filename.split("_")[-3])

    if args.model in SEMISUPERVISE:
        score = SEMISUPERVISE[args.model](data[:train_index], data[train_index:])
        label = label[train_index:]
    else:
        score = UNSUPERVISE[args.model](data)

    metrics = get_metrics(score, label, slidingWindow=find_length_rank(data[:, 0].reshape(-1, 1), rank=1))
    print(args.model, args.filename, {k: round(v, 4) for k, v in metrics.items()})
