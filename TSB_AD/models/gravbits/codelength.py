"""Exact adaptive code lengths (bits), the value metric: smaller = more structure learned.

All models use the Krichevsky-Trofimov estimator p = (count + 1/2) / (total + alphabet/2),
updated after each batch (the decoder sees the same counts), so every figure is a real
lossless code length an arithmetic coder would achieve to within a few bytes.
"""

from __future__ import annotations

import bz2
import lzma
import zlib

import numpy as np

A_KT = 0.5


BUCKETS = 16  # residual context: the previous byte in the same channel, in 16 coarse bins


class WellModel:
    """Statistics per well: transitions between consecutive wells, and residual bytes per
    dim conditioned on the previous byte of that dim (``BUCKETS`` bins). The condition makes
    generation move smoothly inside a well and lets correlated values code cheaply."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.trans = np.zeros((0, 0))
        self.resid = np.zeros((0, dim, BUCKETS, 256))
        self.last = -1  # well of the previous vector, -1 = none yet
        self.last_vec = np.zeros(dim, dtype=np.int64)  # previous vector's bytes

    def _resid_p(self, w: np.ndarray, d: np.ndarray, ctx: np.ndarray) -> np.ndarray:
        """Residual distributions over 256 values for (well, dim, prev-bucket) triples:
        the bucket's counts backed off Witten-Bell style to the well's own residual
        distribution (KT), so a rarely seen bucket never falls back to uniform noise."""
        marg = self.resid[w, d].sum(axis=-2)  # (..., 256) over all buckets
        p_marg = (marg + A_KT) / (marg.sum(axis=-1, keepdims=True) + A_KT * 256)
        c = self.resid[w, d, ctx]
        tot = c.sum(axis=-1, keepdims=True)
        distinct = np.count_nonzero(c, axis=-1)[..., None]
        blended = (c + distinct * p_marg) / np.maximum(tot + distinct, 1e-300)
        return np.asarray(np.where(tot > 0, blended, p_marg))

    def _ctx(self, x: np.ndarray) -> np.ndarray:
        prev = np.vstack([self.last_vec[None, :], x[:-1].astype(np.int64)])
        return np.asarray(prev * BUCKETS // 256)

    def _residual(self, x: np.ndarray, centres: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return np.mod(x.astype(np.int64) - np.rint(centres[idx]).astype(np.int64), 256)

    def bits(
        self, x: np.ndarray, centres: np.ndarray, idx: np.ndarray, per_dim: bool = False
    ) -> np.ndarray:
        """Code length (bits) per vector; ``per_dim`` -> (N, 1 + dim): the well-transition
        cost, then each dim's residual cost (their row sum is the total)."""
        n_wells = len(centres)
        self._grow(n_wells)
        prev = np.concatenate([[self.last], idx[:-1]])
        row_tot = self.trans.sum(axis=1)
        known = prev >= 0
        p_idx = np.full(len(idx), 1.0 / n_wells)
        pk, ik = prev[known], idx[known]
        p_idx[known] = (self.trans[pk, ik] + A_KT) / (row_tot[pk] + A_KT * n_wells)
        r, ctx = self._residual(x, centres, idx), self._ctx(x)
        dims = np.arange(self.dim)[None, :]
        p_all = self._resid_p(idx[:, None], dims, ctx)  # (N, dim, 256)
        p_res = np.take_along_axis(p_all, r[..., None], axis=-1)[..., 0]
        parts = np.column_stack([-np.log2(p_idx), -np.log2(p_res)])
        return parts if per_dim else parts.sum(axis=1)

    def update(self, x: np.ndarray, centres: np.ndarray, idx: np.ndarray) -> None:
        self._grow(len(centres))
        prev = np.concatenate([[self.last], idx[:-1]])
        ok = prev >= 0
        np.add.at(self.trans, (prev[ok], idx[ok]), 1)
        r, ctx = self._residual(x, centres, idx), self._ctx(x)
        dims = np.broadcast_to(np.arange(self.dim), r.shape)
        np.add.at(self.resid, (np.broadcast_to(idx[:, None], r.shape), dims, ctx, r), 1)
        self.last = int(idx[-1])
        self.last_vec = x[-1].astype(np.int64)

    def remap(self, old_to_new: np.ndarray, n_new: int) -> None:
        """Carry counts through merges (-> same target) and pruning (-> -1, dropped)."""
        old = len(self.trans)
        m = np.asarray(old_to_new[:old], dtype=np.int64)
        trans, resid = np.zeros((n_new, n_new)), np.zeros((n_new, self.dim, BUCKETS, 256))
        ii, jj = np.nonzero(self.trans)
        ok = (m[ii] >= 0) & (m[jj] >= 0)
        np.add.at(trans, (m[ii[ok]], m[jj[ok]]), self.trans[ii[ok], jj[ok]])
        rows = np.flatnonzero(m >= 0)
        _, first = np.unique(m[rows], return_index=True)
        resid[m[rows[first]]] = self.resid[rows[first]]  # one copy per surviving well
        for r in np.delete(rows, first):  # the few wells merged into another
            resid[m[r]] += self.resid[r]
        self.trans, self.resid = trans, resid
        self.last = int(m[self.last]) if 0 <= self.last < old else -1

    def _grow(self, n: int) -> None:
        k = len(self.trans)
        if n > k:
            self.trans = np.pad(self.trans, ((0, n - k), (0, n - k)))
            self.resid = np.pad(self.resid, ((0, n - k), (0, 0), (0, 0), (0, 0)))

    def sample(self, n: int, centres: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Order-1 generation: walk the learned well-to-well transitions, then render."""
        self._grow(len(centres))
        w = int(rng.integers(len(centres))) if self.last < 0 else self.last
        ids = np.empty(n, dtype=np.int64)
        for t in range(n):
            p = self.trans[w] + A_KT
            w = ids[t] = int(rng.choice(len(p), p=p / p.sum()))
        return self.render(ids, centres, rng)

    def render(self, ids: np.ndarray, centres: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Well ids -> byte vectors: each well's centre plus residual bytes drawn from its
        learned residual distribution given the previously generated byte of that dim."""
        self._grow(len(centres))
        base = np.rint(centres).astype(np.int64)
        dims = np.arange(self.dim)
        out = np.empty((len(ids), self.dim), dtype=np.uint8)
        prev = self.last_vec.copy()
        for t, w in enumerate(np.asarray(ids, dtype=np.int64)):
            q = self._resid_p(np.full(self.dim, w), dims, prev * BUCKETS // 256)  # (dim, 256)
            cdf = np.cumsum(q, axis=1)
            r = (cdf < rng.random(self.dim)[:, None]).sum(axis=1).clip(0, 255)
            prev = (base[w] + r) % 256
            out[t] = prev
        return out


def byte_model_bits(data: bytes, order: int, batch: int) -> np.ndarray:
    """Per-byte code length of an order-``order`` (0..2) context model, batch-adaptive."""
    if order not in (0, 1, 2):
        raise ValueError(f"order must be 0, 1 or 2, got {order}")
    b = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    ctx = np.zeros(len(b), dtype=np.int64)
    for k in range(1, order + 1):
        ctx[k:] = ctx[k:] * 256 + b[:-k]
    table = np.zeros((256**order, 256))
    bits = np.empty(len(b))
    for lo in range(0, len(b), batch):
        c, s = ctx[lo : lo + batch], b[lo : lo + batch]
        bits[lo : lo + batch] = -np.log2((table[c, s] + A_KT) / (table[c].sum(axis=1) + A_KT * 256))
        np.add.at(table, (c, s), 1)
    return bits


def compressor_bits(data: bytes) -> dict[str, float]:
    """Bits per byte of real general-purpose compressors at their strongest settings."""
    n = max(len(data), 1)
    return {
        "zlib-9": 8 * len(zlib.compress(data, 9)) / n,
        "bz2-9": 8 * len(bz2.compress(data, 9)) / n,
        "xz-9e": 8 * len(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)) / n,
    }
