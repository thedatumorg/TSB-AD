"""The one hot loop: gravitational mean-shift sums, on CPU (numpy) or CUDA (CuPy, fp64).

For probes x_i and masses m_j at p_j, with s = sqrt(|x_i - p_j|^2 + eps^2):
    w_ij = m_j / s^3                                  (pure, lam is None)
    w_ij = m_j e^(-s/lam) (1/s^3 + 1/(lam s^2))       (Yukawa-screened)
    W_i = sum_j w_ij,   S_i = sum_j w_ij p_j          -> mean-shift step x_i <- S_i / W_i
This is an N-body sum (probes x points), the workload fp64 GPUs such as the K80 are built
for. ``settle`` iterates it to the zero-force points. Every backend must match ``CpuBackend``
to 1e-9 relative (tests/test_backend.py), so CPU stays the reference.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

_PROBES, _POINTS = 256, 8192  # CPU block sizes: bound memory at _PROBES * _POINTS doubles


class Backend(Protocol):
    name: str

    def sums(
        self, x: np.ndarray, pts: np.ndarray, mass: np.ndarray, eps: float, lam: float | None
    ) -> tuple[np.ndarray, np.ndarray]: ...


class CpuBackend:
    name = "cpu"

    def sums(
        self, x: np.ndarray, pts: np.ndarray, mass: np.ndarray, eps: float, lam: float | None
    ) -> tuple[np.ndarray, np.ndarray]:
        W = np.zeros(len(x))
        S = np.zeros_like(x, dtype=np.float64)
        p2 = (pts**2).sum(axis=1)
        for lo in range(0, len(x), _PROBES):
            xb = x[lo : lo + _PROBES]
            x2 = (xb**2).sum(axis=1)
            for plo in range(0, len(pts), _POINTS):
                pb, mb = pts[plo : plo + _POINTS], mass[plo : plo + _POINTS]
                r2 = np.maximum(x2[:, None] + p2[None, plo : plo + _POINTS] - 2 * xb @ pb.T, 0)
                s = np.sqrt(r2 + eps * eps)
                w = (
                    mb / s**3
                    if lam is None
                    else mb * np.exp(-s / lam) * (1 / s**3 + 1 / (lam * s * s))
                )
                W[lo : lo + _PROBES] += w.sum(axis=1)
                S[lo : lo + _PROBES] += w @ pb
        return W, S


def get_backend(name: str = "cpu", **cuda_options: object) -> Backend:
    """``cpu`` always works; ``cuda`` needs CuPy and a GPU (see gravbits/cuda_backend.py) and
    takes CudaBackend options (devices, weights, kernel)."""
    if name == "cpu":
        return CpuBackend()
    if name == "cuda":
        from gravbits.cuda_backend import CudaBackend  # optional dependency, imported on demand

        return CudaBackend(**cuda_options)  # type: ignore[arg-type]
    raise ValueError(f"unknown backend {name!r}; expected 'cpu' or 'cuda'")


def settle(
    x: np.ndarray,
    pts: np.ndarray,
    mass: np.ndarray,
    eps: float,
    lam: float | None,
    iters: int = 100,
    backend: Backend | None = None,
) -> np.ndarray:
    """Move every probe along the mean-shift step until all move < 1e-4 eps (or iters)."""
    be = backend or CpuBackend()
    x = np.array(x, dtype=np.float64)
    active = np.ones(len(x), dtype=bool)
    for _ in range(iters):
        if not active.any():
            break
        W, S = be.sums(x[active], pts, mass, eps, lam)
        new = S / W[:, None]
        moved = np.abs(new - x[active]).max(axis=1)
        x[active] = new
        active[np.flatnonzero(active)[moved < 1e-4 * eps]] = False
    return x
