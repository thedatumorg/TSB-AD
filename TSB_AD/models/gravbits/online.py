"""Self-learning gravity wells: wells that grow, drift, merge and fade as data streams in.

``OnlineGravity.step(batch)`` first *codes* the batch with the current wells (so the code
length is causal: a decoder running the same algorithm rebuilds every well, nothing is
transmitted) and then *learns* from it:
  1. assign each vector to its nearest well;
  2. spawn wells where data lands farther than one screening length from every well
     (the attractors of those surprising points, found with ``gravity``);
  3. let every well fall in the combined field of the new batch plus the wells' own
     remembered mass (a few mean-shift steps), then merge wells that met;
  4. masses decay by ``decay`` per vector seen and gain the vectors they own; past
     ``k_max`` the lightest wells are dropped.
Learned statistics (well-to-well transitions, per-well residual bytes) follow wells
through merges and pruning via ``remap``. No labels anywhere: the only signal is the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backend import Backend, CpuBackend, settle
from .codelength import WellModel
from .gravity import gravity, merge, nearest


@dataclass
class OnlineGravity:
    dim: int
    k_max: int = 256
    softening: float = 0.05
    screen: float = 0.25
    decay: float = 0.9999
    settle_iters: int = 5
    seed: int = 0
    backend: Backend = field(default_factory=CpuBackend)
    centres: np.ndarray = field(init=False)
    mass: np.ndarray = field(init=False)
    scale: float = field(init=False, default=0.0)
    model: WellModel = field(init=False)
    steps: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.dim < 1 or self.k_max < 1:
            raise ValueError(f"dim and k_max must be >= 1, got {self.dim}, {self.k_max}")
        self.centres = np.zeros((0, self.dim))
        self.mass = np.zeros(0)
        self.model = WellModel(self.dim)

    def step(self, batch: np.ndarray, per_dim: bool = False) -> np.ndarray:
        """Code ``batch`` (N, dim) of byte values, then learn from it. Returns bits per vector
        (``per_dim``: (N, 1 + dim) parts, see ``WellModel.bits``)."""
        bits = self.code(batch, per_dim)
        self._learn(np.asarray(batch, dtype=np.float64))
        self.steps += 1
        return bits

    def code(self, batch: np.ndarray, per_dim: bool = False) -> np.ndarray:
        """Code length of ``batch`` under the current wells, learning nothing (a frozen model
        still tracks the previous vector, so transitions and residual contexts stay exact)."""
        x = np.asarray(batch, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.dim or len(x) == 0:
            raise ValueError(f"expected a non-empty (N, {self.dim}) batch, got {x.shape}")
        if len(self.centres) == 0:  # nothing learned yet: raw bytes
            raw = np.full((len(x), 1 + self.dim), 8.0)
            raw[:, 0] = 0.0
            return raw if per_dim else raw.sum(axis=1)
        idx = nearest(x, self.centres)
        return self.model.bits(x, self.centres, idx, per_dim)

    def follow(self, batch: np.ndarray) -> None:
        """Advance the context (previous well and bytes) past ``batch`` without learning."""
        x = np.asarray(batch, dtype=np.float64)
        if len(self.centres):
            self.model.last = int(nearest(x[-1:], self.centres)[0])
        self.model.last_vec = x[-1].astype(np.int64)

    def _learn(self, x: np.ndarray) -> None:
        spread = float(np.sqrt(((x - x.mean(axis=0)) ** 2).sum(axis=1).mean()))
        self.scale = spread if self.scale == 0 else 0.9 * self.scale + 0.1 * spread
        eps, lam = self.softening * max(self.scale, 1e-6), self.screen * max(self.scale, 1e-6)

        n_old = len(self.centres)
        if n_old:
            far = np.linalg.norm(x - self.centres[nearest(x, self.centres)], axis=1) > lam
            novel = x[far]
        else:
            novel = x
        if len(novel):
            free = max(self.k_max - n_old, 8)
            seeds = gravity(novel, free, seed=self.seed + self.steps, backend=self.backend)
            self.centres = np.vstack([self.centres, seeds])
            self.mass = np.concatenate([self.mass, np.zeros(len(seeds))])

        pts, counts = np.unique(x, axis=0, return_counts=True)
        field_pts = np.vstack([pts, self.centres])
        field_mass = np.concatenate([counts.astype(np.float64), self.mass])
        moved = settle(
            self.centres, field_pts, field_mass, eps, lam, self.settle_iters, self.backend
        )

        merged = merge(moved, eps / 2)
        target = nearest(moved, merged)  # old well -> surviving well
        mass: np.ndarray = np.bincount(target, weights=self.mass, minlength=len(merged))
        mass = mass * self.decay ** len(x) + np.bincount(nearest(x, merged), minlength=len(merged))
        keep = np.argsort(-mass, kind="stable")[: self.k_max]
        new_pos = np.full(len(merged), -1)
        new_pos[keep] = np.arange(len(keep))
        self.model.remap(new_pos[target[:n_old]], len(keep))  # spawned wells start empty
        self.centres, self.mass = merged[keep], mass[keep]
        self.model.update(x, self.centres, nearest(x, self.centres))

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        """Generate ``n`` vectors: walk well transitions, draw residual bytes per well."""
        if len(self.centres) == 0:
            raise ValueError("nothing learned yet")
        return self.model.sample(n, self.centres, np.random.default_rng(seed))
