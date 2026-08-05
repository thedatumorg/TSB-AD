"""
This function is adapted from [TimeEval-algorithms] by [CodeLionX&wenig]
Original source: [https://github.com/TimeEval/TimeEval-algorithms]
"""

import numpy as np
from dataclasses import dataclass
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.utility import zscore

class FFT(BaseDetector):

    def __init__(self, ifft_parameters=5, local_neighbor_window=21, local_outlier_threshold=0.6, max_region_size=50, max_sign_change_distance=10, normalize=True):
        super().__init__()

        self.ifft_parameters = ifft_parameters
        self.local_neighbor_window = local_neighbor_window
        self.local_outlier_threshold = local_outlier_threshold
        self.max_region_size = max_region_size
        self.max_sign_change_distance = max_sign_change_distance
        self.normalize = normalize
        self.decision_scores_ = None

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods."""
        n_samples, n_features = X.shape
        if self.normalize: 
            if n_features == 1:
                X = zscore(X, axis=0, ddof=0)
            else:
                X = zscore(X, axis=1, ddof=1)
        self.data = X
        self.decision_scores_ = self.detect_anomalies()  
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector."""
        n_samples, n_features = X.shape
        decision_scores_ = np.zeros(n_samples)
        self.data = X
        local_outliers = self.calculate_local_outliers()
        if not local_outliers:
            print("No local outliers detected.")
            return np.zeros_like(self.data)

        regions = self.calculate_region_outliers(local_outliers)
        anomaly_scores = np.zeros_like(self.data)
        for region in regions:
            start_index = local_outliers[region.start_idx].index
            end_index = local_outliers[region.end_idx].index
            anomaly_scores[start_index:end_index + 1] = region.score

        decision_scores_ = anomaly_scores
        return decision_scores_

    @staticmethod
    def reduce_parameters(f: np.ndarray, k: int) -> np.ndarray:
        transformed = f.copy()
        if k == 1:
            transformed[1:] = 0
        else:
            # keep the k lowest frequencies together with their negative-frequency
            # conjugates, so the inverse transform stays real and the retained
            # components keep their amplitude
            transformed[k:-(k - 1)] = 0
        return transformed

    def calculate_local_outliers(self):
        data = np.asarray(self.data)
        if data.ndim > 1:
            if data.shape[1] != 1:
                raise ValueError(
                    "FFT expects a univariate series, got %d channels" % data.shape[1])
            data = data[:, 0]

        n = len(data)
        k = max(min(self.ifft_parameters, n), 1)
        y = self.reduce_parameters(np.fft.fft(data), k)
        f2 = np.real(np.fft.ifft(y))

        so = np.abs(f2 - data)
        mso = np.mean(so)
        neighbor_c = self.local_neighbor_window // 2

        scores = []
        score_idxs = []
        for i in range(n):
            if so[i] > mso:
                nav = np.mean(data[max(i - neighbor_c, 0):min(i + neighbor_c + 1, n)])
                scores.append(data[i] - nav)
                score_idxs.append(i)

        if not scores:
            return []

        ms = np.mean(scores)
        sds = np.std(scores) + 1e-6  
        z_scores = (np.array(scores) - ms) / sds

        return [self.LocalOutlier(index=score_idxs[i], z_score=z_scores[i])
                for i in range(len(scores)) if abs(z_scores[i]) > self.local_outlier_threshold]

    def calculate_region_outliers(self, local_outliers):
        def distance(a: int, b: int) -> int:
            return abs(local_outliers[b].index - local_outliers[a].index)

        regions = []
        i = 0
        n_l = len(local_outliers) - 1
        while i < n_l:
            start_idx = i
            while i < n_l and distance(i, i + 1) <= self.max_sign_change_distance:
                i += 1
            end_idx = i
            if end_idx > start_idx:
                score = np.mean([abs(local_outliers[j].z_score) for j in range(start_idx, end_idx + 1)])
                regions.append(self.RegionOutlier(start_idx=start_idx, end_idx=end_idx, score=score))
            i += 1

        return regions

    @dataclass
    class LocalOutlier:
        index: int
        z_score: float

        @property
        def sign(self) -> int:
            return np.sign(self.z_score)

    @dataclass
    class RegionOutlier:
        start_idx: int
        end_idx: int
        score: float

    def detect_anomalies(self):
        """Detect anomalies by combining local and regional outliers."""
        local_outliers = self.calculate_local_outliers()
        if not local_outliers:
            print("No local outliers detected.")
            return np.zeros_like(self.data)

        regions = self.calculate_region_outliers(local_outliers)
        anomaly_scores = np.zeros_like(self.data)
        for region in regions:
            start_index = local_outliers[region.start_idx].index
            end_index = local_outliers[region.end_idx].index
            anomaly_scores[start_index:end_index + 1] = region.score

        return anomaly_scores