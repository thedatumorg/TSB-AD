"""
DualRefKNN: window distances to the nearest windows of two references plus a point-wise curve.

Each window of a channel is described in two ways. The first is eight statistics, namely the standard
deviation, range, maximum, minimum and mean of the values and the standard deviation, mean absolute value
and largest absolute value of the first difference. The second is the z-normalised window averaged down to
32 points. A window scores its mean distance to the 3 nearest reference windows and a time step scores the
mean over the windows that cover it. The window sizes are 16 to 512 plus one and two periods (find_length_rank
on column 0), shrunk where the reference is short. Each size is standardised over the series and the maximum
is taken over sizes.

On one channel the two descriptions are scored against two references, the anomaly-free prefix and all windows
of the series (windows that overlap the scored one are left out). The four curves are summed and 4 times a
point-wise curve is added, the largest of three scaled deviations of each point. These are the residual of a
two-sided linear interpolation from its neighbours and the points one period away, the smaller of its two
neighbouring steps and its distance from the median of the 7 points around it. Each absolute deviation is
divided by its scale, reduced by 1 and cut at 0 and 10. The scale is the largest of its 99th percentile over
the series, its 5th largest value in the prefix and a tenth of the robust range of the series.

On several channels two views are summed, each standardised over the series. One is the five window statistics
of all channels together, every channel first standardised over the whole series. The other is the mean of the
per-channel curves over the continuous channels (more than 2 distinct values), with the four curves of one
channel if there are at most 3 of them and else the two against the prefix. Without a continuous channel every
channel that changes is scored against the prefix and the mean is over all channels.

Before that, on several channels, a switch that carries noise is set back to its two values. A channel counts as
such if its values lie in two intervals, each no wider than a tenth of its range, and what is left after taking
away the mean of each interval looks like independent noise (hardly any value repeats, no lag-1 autocorrelation,
no far value). A channel with one or two distinct values comes back unchanged, so a series without noise is
scored exactly as without this step.

There is no pretrained model, no randomness and no label. Only the series and the prefix length are used.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseDetector
from ..utils.slidingWindows import find_length_rank

EPS = 1e-8

# The blocks of the two settings as (description, scored against all windows of the series), summed in this order.
UNI_BLOCKS = (('stats8', False), ('shape', False), ('stats8', True), ('shape', True))
MULTI_BLOCKS = (('stats8', False), ('shape', False))

# The point-wise curve of one channel. Its kinds (merged by the maximum), its weight g, the quantile 1 - alpha that
# scales a deviation, the largest value cap, the floor of the scale as a share of the robust range of the series and
# the count pm of the prefix rule.
POINT = dict(kinds=('lin', 'spike', 'hampel3'), g=4.0, alpha=0.01, cap=10.0, floor=0.1, pm=5)
# On several channels a column with at most KCAT distinct values is a switch and the others are continuous. With at
# most FEW continuous columns their curves use the four blocks of one channel.
KCAT = 2
FEW = 3
# The step that sets a noisy switch back to its two values (several channels only). A channel has two levels if its
# values lie in two intervals of at most `width` times its range. It is set back if it has at most two distinct
# values, or if the remainder inside the levels looks like independent noise: more distinct values than `share`
# times its rows, a lag-1 autocorrelation of at most `bound` over the square root of its rows, and no remainder
# beyond `far` robust standard deviations.
SWITCH = dict(width=0.1, share=0.5, bound=4.0, far=7.0)


def period(x):
    """The benchmark's own period estimate on column 0."""
    return int(find_length_rank(np.asarray(x, dtype=float).reshape(-1, 1), rank=1))


# ------------------------------------------------------------------ rolling features (windows start at every step)
def _csum(v):
    return torch.cat([v.new_zeros(1), torch.cumsum(v, 0)])


def roll_mean_std(x, w):
    sh = x.mean()
    x0 = x - sh
    c1, c2 = _csum(x0), _csum(x0 * x0)
    m = (c1[w:] - c1[:-w]) / w
    var = ((c2[w:] - c2[:-w]) / w - m * m).clamp_min(0)
    return m + sh, var.sqrt()


def roll_max(x, w):
    return F.max_pool1d(x[None, None], w, 1)[0, 0]


def feat_stats5(x, w):
    """std, range, max, min, mean of the window."""
    m, s = roll_mean_std(x, w)
    mx = roll_max(x, w)
    mn = -roll_max(-x, w)
    return torch.stack([s, mx - mn, mx, mn, m], 1)


def feat_dstat(x, w):
    """Statistics of the first difference inside the window (std, mean absolute, largest absolute)."""
    d = torch.diff(x, prepend=x[:1])
    _, s = roll_mean_std(d, w)
    a = d.abs()
    ma, _ = roll_mean_std(a, w)
    return torch.stack([s, ma, roll_max(a, w)], 1)


def feat_stats8(x, w):
    """The five window statistics and the three difference statistics as one description."""
    return torch.cat([feat_stats5(x, w), feat_dstat(x, w)], 1)


def _chunks(x, w, size):
    n_w = x.numel() - w + 1
    for a in range(0, n_w, size):
        b = min(n_w, a + size)
        yield x[a:b + w - 1].unfold(0, w, 1)


def feat_shape(x, w, length):
    """The z-normalised window itself, averaged down to a fixed length."""
    if w < 8:
        return None
    L = min(length, w)
    out = []
    for win in _chunks(x, w, max(256, (1 << 22) // w)):
        z = (win - win.mean(1, keepdim=True)) / (win.std(1, keepdim=True) + EPS)
        out.append(F.adaptive_avg_pool1d(z[:, None, :], L)[:, 0, :] / math.sqrt(L))
    return torch.cat(out)


# ------------------------------------------------------------------ nearest neighbours in the reference
def knn_mean(feats, ref_pos, k, excl, n_train, chunk=8192):
    """Mean distance of every row to its k nearest reference rows. Rows whose window starts within `excl`
    steps of a reference window do not count that window (so a window is never matched to itself)."""
    ref = feats[ref_pos]
    k = min(k, ref.shape[0])
    out = torch.empty(feats.shape[0], dtype=torch.float32, device=feats.device)
    rpos = ref_pos.to(torch.long)
    for a in range(0, feats.shape[0], chunk):
        b = min(feats.shape[0], a + chunk)
        d = torch.cdist(feats[a:b], ref)
        if excl > 0 and a < n_train + excl:
            qpos = torch.arange(a, b, device=feats.device)
            d = d.masked_fill((qpos[:, None] - rpos[None, :]).abs() < excl, float('inf'))
        v = torch.topk(d, k, dim=1, largest=False).values
        fin = torch.isfinite(v)
        v = torch.where(fin, v, torch.zeros_like(v))
        out[a:b] = v.sum(1) / fin.sum(1).clamp_min(1)
    return out


def to_points(ws, w, n):
    """Score of a time step as the mean score of the windows that cover it."""
    c = _csum(ws.double())
    t = torch.arange(n, device=ws.device)
    lo = (t - w + 1).clamp(0, ws.numel())
    hi = (t + 1).clamp(0, ws.numel())
    return ((c[hi] - c[lo]) / (hi - lo).clamp_min(1)).float()


def norm_curve(c):
    """Standardise a curve over the series."""
    return (c - c.mean()) / (c.std() + 1e-9)


def scale_list(p):
    """Window sizes 16 to 512 plus one and two periods."""
    s = [16, 32, 64, 128, 256, 512, p, 2 * p]
    return sorted({int(w) for w in s if w >= 4})


def fit_scales(scales, tr, n, k):
    """Which window sizes the reference can carry. Each size is shrunk so that at least max(2k, 10) reference
    windows remain and is dropped if it would fall below min(64, w)."""
    out = []
    need = max(2 * k, 10)
    for w in scales:
        we = min(w, tr - (need - 1))
        if we >= min(64, w):
            out.append(we)
    out = sorted({w for w in out if 4 <= w < n})
    if not out:
        out = [max(4, min(min(scales), tr // 2, n // 2))]
    return out


def block_curve(x, tr, kind, ref_all, p, k, cap, shape_len):
    """One block on one channel. Per-size curves, each standardised, maximum over sizes."""
    n = x.numel()
    curves = []
    tr_fit = n if ref_all else tr
    for w in fit_scales(scale_list(p), tr_fit, n, k):
        if kind == 'shape':
            f = feat_shape(x, w, shape_len)
        else:
            f = feat_stats8(x, w)
        if f is None:
            continue
        if kind == 'shape':  # already on one scale
            f = f.float()
        n_train = max(tr - w + 1, 1)
        if ref_all:
            n_train = f.shape[0]
        if kind != 'shape':
            r = f[:n_train]
            f = ((f - r.mean(0)) / (r.std(0) + EPS)).float()
        if n_train > cap:
            ref_pos = torch.linspace(0, n_train - 1, cap, device=x.device).long()
        else:
            ref_pos = torch.arange(n_train, device=x.device)
        excl = 0
        if ref_all:  # a window is not matched to the windows that overlap it
            excl = int(1.0 * w)
            excl = min(excl, max(0, (n_train - 2 * k) // 2))
        ws = knn_mean(f, ref_pos, k, excl, n_train)
        curves.append(norm_curve(to_points(ws, w, n)))
    c = torch.stack(curves)
    return c.max(0).values


def score_channel(x_np, tr, blocks, p, k, cap, shape_len, device):
    """The sum of the block curves of one channel."""
    x = torch.as_tensor(np.ascontiguousarray(x_np), dtype=torch.float64, device=device)
    total = None
    for kind, ref_all in blocks:
        c = block_curve(x, tr, kind, ref_all, p, k, cap, shape_len)
        total = c if total is None else total + c
    return total


# ------------------------------------------------------------------ point-wise curve (numpy, one channel)
def dev_hampel(x, h):
    """The value minus the median of the 2h+1 points around it, the series padded with its end values."""
    x = np.asarray(x, dtype=float)
    xp = np.pad(x, h, mode='edge')
    med = np.median(np.lib.stride_tricks.sliding_window_view(xp, 2 * h + 1), axis=1)
    return x - med


def dev_diff(x):
    """The smaller of the two neighbouring steps in absolute value."""
    x = np.asarray(x, dtype=float)
    d = np.abs(np.diff(x, prepend=x[:1]))
    nxt = np.concatenate([d[1:], d[-1:]])
    return np.minimum(d, nxt)   # a spike has a large step on both sides


def exceed(e, alpha, cap, floor, tr, pm):
    """By how many times a point is more extreme than the regular extremes, max(0, |e| / q - 1) cut at cap. The
    scale q is the largest of the (1 - alpha) quantile of |e| over the series, the floor and, with a prefix of at
    least pm points, the pm-th largest |e| of the prefix. Zero everywhere if q is not finite or at most 1e-9 times
    the standard deviation of |e|."""
    a = np.abs(np.asarray(e, dtype=float))
    q = float(np.quantile(a, 1.0 - alpha))
    scale = float(np.std(a)) + 1e-300
    q = max(q, float(floor))
    if tr and pm and tr >= pm:
        # a deviation that the prefix itself shows at least pm times is regular
        q = max(q, float(np.partition(a[:tr], tr - pm)[tr - pm]))
    if q <= 1e-9 * scale or not np.isfinite(q):
        return np.zeros_like(a)
    return np.clip(a / q - 1.0, 0.0, cap)


def spread(x):
    """Robust range of the series, the 99.5th minus the 0.5th percentile."""
    lo, hi = np.quantile(np.asarray(x, dtype=float), [0.005, 0.995])
    return float(hi - lo)


def dev_lin(x, p, near=3, passes=2):
    """Residual of a two-sided linear interpolation. x_t is predicted from its `near` neighbours on each side and,
    where the period allows, from the points p-1, p and p+1 away on each side, by least squares on the whole
    series. After the first pass the points with a large residual are left out of the fit and replaced by their
    prediction where they serve as neighbours, so that one outlier does not echo in the residuals around it."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    lags = list(range(1, near + 1))
    if p > near + 1 and 2 * (p + 1) < n // 4:
        lags += [p - 1, p, p + 1]
    L = max(lags)
    if n < 4 * L + 20:
        return np.zeros(n)
    mu, sd = float(np.mean(x)), float(np.std(x)) + 1e-12
    z = (x - mu) / sd

    def feats(v):
        vp = np.pad(v, L, mode='reflect')
        cols = [vp[L - l:L - l + n] for l in lags] + [vp[L + l:L + l + n] for l in lags]
        return np.stack(cols + [np.ones(n)], axis=1)
    zc = z.copy()
    keep = np.ones(n, dtype=bool)
    e = np.zeros(n)
    for it in range(passes):
        F = feats(zc)
        rows = np.where(keep)[0]
        if len(rows) > 200000:
            rows = rows[np.linspace(0, len(rows) - 1, 200000).astype(int)]
        A = F[rows]
        w = np.linalg.solve(A.T @ A + 1e-6 * len(rows) * np.eye(A.shape[1]), A.T @ z[rows])
        pred = F @ w
        e = z - pred
        if it + 1 < passes:
            s = 1.4826 * float(np.median(np.abs(e - np.median(e)))) + 1e-12
            bad = np.abs(e) > 5.0 * s
            if bad.mean() > 0.05:      # too many, so the plain fit is kept
                break
            keep = ~bad
            zc = np.where(bad, pred, z)
    return e * sd


def point_curve(x, tr, p):
    """The largest of the three point-wise deviations, each scaled by exceed."""
    cs = []
    fl = POINT['floor'] * spread(x)
    for kind in POINT['kinds']:
        if kind == 'lin':
            e = dev_lin(x, p)
        elif kind == 'spike':
            e = dev_diff(x)
        else:
            e = dev_hampel(x, 3)
        cs.append(exceed(e, POINT['alpha'], POINT['cap'], fl, tr, POINT['pm']))
    return np.max(np.stack(cs), axis=0)


# ------------------------------------------------------------------ one channel
def score_uni(x, tr, p, k, cap, shape_len, device):
    """The four block curves summed, plus g times the point-wise curve."""
    s = score_channel(x, tr, UNI_BLOCKS, p, k, cap, shape_len, device)
    base = np.nan_to_num(s.double().cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(base, dtype=float) + POINT['g'] * point_curve(x, tr, p)


# ------------------------------------------------------------------ several channels
def pooled_curve(pooled_in, tr, p, n, k, cap):
    """The five window statistics of all channels together, against the prefix."""
    rm, rq, rmax, rmin = pooled_in
    curves = []
    for w in fit_scales(scale_list(p), tr, n, k):
        m, _ = roll_mean_std(rm, w)
        q, _ = roll_mean_std(rq, w)
        s = (q - m * m).clamp_min(0).sqrt()
        mx = roll_max(rmax, w)
        mn = -roll_max(-rmin, w)
        f = torch.stack([s, mx - mn, mx, mn, m], 1)
        n_train = max(tr - w + 1, 1)
        r = f[:n_train]
        f = ((f - r.mean(0)) / (r.std(0) + EPS)).float()
        ref_pos = torch.linspace(0, n_train - 1, cap, device=f.device).long() if n_train > cap else torch.arange(n_train, device=f.device)
        ws = knn_mean(f, ref_pos, k, 0, n_train)
        curves.append(norm_curve(to_points(ws, w, n)))
    c = torch.stack(curves)
    return c.max(0).values


def _z(c):
    """Standardise a curve over the series in float64."""
    c = np.asarray(c, dtype=np.float64)
    return (c - c.mean()) / (c.std() + 1e-9)


def _standardise(data):
    """Every column standardised over the whole series. A column that never changes becomes 0."""
    mu = data.mean(0)
    sd = data.std(0)
    return np.where(sd > 0, (data - mu) / np.where(sd > 0, sd, 1.0), 0.0)


def _pooled(data, tr, p, k, cap, device):
    """The pooled view of the columns of data, as a float32 array."""
    x = torch.as_tensor(data, dtype=torch.float64, device=device)
    pooled_in = (x.mean(1), (x * x).mean(1), x.max(1).values, x.min(1).values)
    return pooled_curve(pooled_in, tr, p, data.shape[0], k, cap).float().cpu().numpy()


def _channel(x, tr, blocks, p, k, cap, shape_len, device):
    """The standardised curve of one channel with non-finite values set to 0, as a float32 array."""
    s = norm_curve(score_channel(x, tr, blocks, p, k, cap, shape_len, device))
    return torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0).float().cpu().numpy()


def score_multi(data, tr, p, k, cap, shape_len, device):
    """The pooled view of the standardised channels plus the mean of the per-channel curves, each standardised."""
    n, d = data.shape
    nu = np.array([len(np.unique(data[:, c])) for c in range(d)])
    cont = np.where(nu > KCAT)[0]
    pv = _pooled(_standardise(data), tr, p, k, cap, device)
    if len(cont) > 0:
        blocks = UNI_BLOCKS if len(cont) <= FEW else MULTI_BLOCKS
        pc = np.stack([_channel(data[:, c], tr, blocks, p, k, cap, shape_len, device) for c in cont]).astype(np.float64).mean(0)
    else:  # no continuous channel, every channel that changes is scored against the prefix
        rows = np.zeros((d, n), dtype=np.float32)
        for c in range(d):
            if nu[c] > 1:
                rows[c] = _channel(data[:, c], tr, MULTI_BLOCKS, p, k, cap, shape_len, device)
        pc = rows.astype(np.float64).mean(0)
    s = _z(pv) + _z(pc)
    return np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)


# ------------------------------------------------------------------ noisy switches
def _upper_level(col):
    """The smallest value of the upper level if the channel is to be set back to two levels, else None."""
    u = np.unique(col)
    r = float(u[-1] - u[0])
    if not r > 0:
        return None
    w = SWITCH['width'] * r
    j = int(np.searchsorted(u, u[0] + w, side='right'))
    if j >= len(u) or u[-1] > u[j] + w:
        return None
    up = float(u[j])
    if len(u) <= 2:
        return up
    if len(u) <= SWITCH['share'] * len(col):
        return None
    hi = col >= up
    e = np.array(col, dtype=float)
    if hi.any():
        e[hi] -= e[hi].mean()
    if (~hi).any():
        e[~hi] -= e[~hi].mean()
    den = float(np.dot(e, e))
    if not den > 0:
        return up
    a = np.abs(e)
    sd = 1.4826 * float(np.median(a))
    far = float(a.max() / sd) if sd > 0 else float('inf')
    r1 = float(np.dot(e[:-1], e[1:]) / den)
    if abs(r1) <= SWITCH['bound'] / np.sqrt(len(col)) and far <= SWITCH['far']:
        return up
    return None


def restore_switches(data):
    """A copy of data in which every noisy switch takes the smallest value of each of its two levels."""
    x = np.array(data, dtype=float)
    for c in range(x.shape[1]):
        col = x[:, c]
        up = _upper_level(col)
        if up is not None:
            x[:, c] = np.where(col < up, col.min(), up)
    return x


# ------------------------------------------------------------------ detector
class DualRefKNN(BaseDetector):
    """Window descriptions scored by their distance to the nearest windows of two references, plus a point-wise
    curve on one channel.

    Parameters
    ----------
    k : int, optional (default=3)
        The number of nearest reference windows.

    max_ref : int, optional (default=10000)
        The largest number of reference windows per window size. Longer references are thinned evenly.

    shape_len : int, optional (default=32)
        The length that the z-normalised window is averaged down to.

    device : str or None, optional (default=None)
        The torch device. None means cuda if it is available, else cpu.
    """

    def __init__(self, k=3, max_ref=10000, shape_len=32, device=None):
        super().__init__()
        self.k = k
        self.max_ref = max_ref
        self.shape_len = shape_len
        self.device = device

    def fit(self, X, y=None):
        """X is the anomaly-free prefix. Only its length is used, as the scores are computed on the whole series."""
        self.n_train_ = len(X)
        return self

    def decision_function(self, X):
        """X is the whole series with the prefix at its start. Returns one raw score per time step."""
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data[:, None]
        n, d = data.shape
        device = self.device if self.device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        tr = self.n_train_
        if d > 1:
            data = restore_switches(data)
        p = period(data[:, 0])  # one period for every channel
        with torch.no_grad():
            if d == 1:
                s = score_uni(data[:, 0], tr, p, self.k, self.max_ref, self.shape_len, device)
            else:
                s = score_multi(data, tr, p, self.k, self.max_ref, self.shape_len, device)
        self.decision_scores_ = s
        return self.decision_scores_


def run_DualRefKNN_Semisupervised(data_train, data_test, k=3, max_ref=10000, shape_len=32, device=None):
    """data_train is the official training prefix and data_test the whole series. The scores are returned raw."""
    clf = DualRefKNN(k=k, max_ref=max_ref, shape_len=shape_len, device=device)
    clf.fit(data_train)
    return clf.decision_function(data_test)
