"""Regression tests for the OCSVM time-series wrapper."""

import unittest

import numpy as np
from sklearn.exceptions import NotFittedError

from TSB_AD.models.OCSVM import OCSVM


class TestOCSVM(unittest.TestCase):
    def test_scores_series_with_different_length_from_training(self):
        train_time = np.arange(32, dtype=float)
        test_time = np.arange(19, dtype=float)
        train = np.column_stack((train_time, 100 * np.sin(train_time)))
        test = np.column_stack((test_time, 100 * np.sin(test_time)))

        detector = OCSVM(slidingWindow=4).fit(train)
        scores = detector.decision_function(test)

        self.assertEqual(scores.shape, (len(test),))
        self.assertTrue(np.isfinite(scores).all())

    def test_scoring_before_fit_raises_not_fitted_error(self):
        detector = OCSVM(slidingWindow=4)
        series = np.arange(10, dtype=float).reshape(-1, 1)

        with self.assertRaises(NotFittedError):
            detector.decision_function(series)
