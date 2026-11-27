"""
gravbits: self-learning gravity wells, scored by code length.
Submission by Corneliu Vasiliu. Code: https://github.com/cvasiliu/gravbits (MIT).

Each time step becomes a vector of bytes (every channel z-scored, +-clip sigma mapped onto
0..255). The points attract each other with a softened, Yukawa-screened gravity; the
recurring joint states of the system become wells, learned online from the train prefix
(spawn, settle, merge, decay, prune). The model is then frozen and every step of the series
is scored by its code length in bits: the well-to-well transition plus each channel's
residual byte given its previous byte. A step that is normal channel by channel but a
combination never seen in training is expensive to code. The score is a centred moving
average of the code length. CPU only; numpy is the only dependency.

Step by step, with the functions called:
https://github.com/cvasiliu/gravbits/blob/main/docs/HOW_IT_WORKS.md
"""

import numpy as np

from .base import BaseDetector
from .gravbits.online import OnlineGravity


class GravBits(BaseDetector):
    def __init__(self, k_max=64, softening=0.05, screen=0.25, batch=64, smooth=100, clip=4.0):
        super().__init__()
        self.k_max, self.softening, self.screen = k_max, softening, screen
        self.batch, self.smooth, self.clip = batch, smooth, clip

    def fit(self, X, y=None):
        """Keep the (anomaly-free) train prefix; learning happens in decision_function,
        because the byte scale is taken from the whole scored series."""
        self.X_train = np.asarray(X, dtype=float)
        self.decision_scores_ = np.zeros(len(self.X_train))
        return self

    def _bytes(self, X, mu, sd):
        z = (X - mu) / np.where(sd > 0, sd, 1.0)
        return np.rint((np.clip(z, -self.clip, self.clip) + self.clip) / (2 * self.clip) * 255)

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        mu, sd = X.mean(axis=0), X.std(axis=0)
        q_train, q = self._bytes(self.X_train, mu, sd), self._bytes(X, mu, sd)
        model = OnlineGravity(
            X.shape[1], k_max=self.k_max, softening=self.softening, screen=self.screen
        )
        for lo in range(0, len(q_train), self.batch):  # learn the train prefix
            model.step(q_train[lo : lo + self.batch])
        model.model.last, model.model.last_vec = -1, np.zeros(X.shape[1], dtype=np.int64)
        parts = []
        for lo in range(0, len(q), self.batch):  # then score everything, frozen
            b = q[lo : lo + self.batch]
            parts.append(model.code(b, per_dim=True))
            model.follow(b)
        s = np.vstack(parts).sum(axis=1)
        if self.smooth > 1:
            w = self.smooth
            pad = np.pad(s, (w // 2, w - 1 - w // 2), mode="edge")
            s = np.convolve(pad, np.ones(w) / w, mode="valid")
        return s
