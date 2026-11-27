"""Gravitational attractors: the wells a cloud of data "masses" pulls to.

Works on (N, D) points of any dimension D (one point per time step or byte window). Every distinct
data point is a mass (its multiplicity). A probe at x feels the softened
(Plummer) gravitational force of all masses,

    F(x) = sum_j m_j (x_j - x) / (|x_j - x|^2 + eps^2)^(3/2).

The update x <- sum_j w_j x_j / sum_j w_j with w_j = m_j / (|x_j - x|^2 + eps^2)^(3/2)
moves x by F(x) / sum_j w_j, so its fixed points are exactly the points of zero net force.
It is the mean-shift procedure whose kernel profile is the softened potential 1/sqrt(r^2+eps^2),
so each probe climbs into a well of the data's gravitational potential (Comaniciu & Meer
2002 prove convergence for convex, decreasing profiles; this one is both). The probes
that settle form the attractors. ``eps`` (softening) is the resolution: a large eps
merges everything into one well, and eps -> 0 makes every point its own attractor.
Unlike k-means, the number of attractors comes from the data; ``k`` only caps it.

Screening (Yukawa, e^(-r/lam)/r) fades the pull beyond a range lam. Pure gravity's
infinite range makes a smooth cloud a single well, which stalls RVQ after stage one;
screening lets local clumps keep their own wells at every stage.
"""

from __future__ import annotations

import numpy as np

from .backend import Backend, CpuBackend, settle

_CHUNK = 4096  # points per distance-matrix block, bounds memory at _CHUNK * K floats


def nearest(points: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Index of the nearest centre (well) for each point (ties -> lowest index)."""
    out = np.empty(len(points), dtype=np.int64)
    sq = (centres**2).sum(axis=1)
    for lo in range(0, len(points), _CHUNK):
        block = points[lo : lo + _CHUNK]
        d2 = sq[None, :] - 2.0 * block @ centres.T  # |p|^2 is constant per row: dropped
        out[lo : lo + _CHUNK] = d2.argmin(axis=1)
    return out


def force(
    x: np.ndarray, pts: np.ndarray, mass: np.ndarray, eps: float, lam: float | None = None
) -> np.ndarray:
    """Net softened (optionally screened) gravitational force at each row of ``x``."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    W, S = CpuBackend().sums(x, pts, mass, eps, lam)
    return np.asarray(S - W[:, None] * x)


def spread(pts: np.ndarray, mass: np.ndarray) -> float:
    """Mass-weighted RMS distance from the centre of mass (floored to stay positive)."""
    centre = (mass[:, None] * pts).sum(axis=0) / mass.sum()
    return max(float(np.sqrt((mass * ((pts - centre) ** 2).sum(axis=1)).sum() / mass.sum())), 1e-6)


def as_points(points: np.ndarray) -> np.ndarray:
    """(N, D) float64; a 1-D array is N points of dimension 1. Empty input fails loudly."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError("gravity needs at least one point")
    return pts.reshape(-1, 1) if pts.ndim == 1 else pts.reshape(len(pts), -1)


def merge(x: np.ndarray, radius: float) -> np.ndarray:
    """Greedy merge: probes that settled within ``radius`` of a kept one join it."""
    x = np.asarray(x)
    covered = np.zeros(len(x), dtype=bool)  # within radius of a kept probe seen earlier
    kept = []
    for i in range(len(x)):
        if not covered[i]:
            kept.append(i)
            covered |= np.linalg.norm(x - x[i], axis=1) < radius
    return x[kept]


def gravity(
    points: np.ndarray,
    k: int,
    seed: int = 0,
    softening: float = 0.05,
    screen: float | None = 0.25,
    probes: int = 512,
    iters: int = 100,
    backend: Backend | None = None,
) -> np.ndarray:
    """Up to ``k`` gravitational attractors of (N, D) points, heaviest first.

    ``softening`` and ``screen`` are fractions of the data's RMS spread, so every RVQ stage
    sees the same physics at its own scale. ``screen=None`` is pure (infinite-range)
    gravity: on a smooth residual cloud it has a single well (shell theorem), so it stops
    refining after the first stage. A finite screen keeps local wells alive.

    Probes start on randomly chosen distinct points and follow the mean-shift step until
    they stop moving (< 1e-4 eps). Probes that end within eps/2 of each other become one
    attractor; its mass is the number of data points for which it is the nearest
    attractor. Returns float64 of shape (A, D) with 1 <= A <= k.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if softening <= 0 or (screen is not None and screen <= 0):
        raise ValueError(f"softening and screen must be > 0, got {softening}, {screen}")
    pts, counts = np.unique(as_points(points), axis=0, return_counts=True)
    mass = counts.astype(np.float64)
    scale = spread(pts, mass)
    eps, lam = softening * scale, None if screen is None else screen * scale

    rng = np.random.default_rng(seed)
    start = pts[rng.choice(len(pts), size=min(probes, len(pts)), replace=False)]
    centres = merge(settle(start, pts, mass, eps, lam, iters, backend), eps / 2)
    owned = np.bincount(nearest(pts, centres), weights=mass, minlength=len(centres))
    return np.asarray(centres[np.argsort(-owned, kind="stable")[:k]])
