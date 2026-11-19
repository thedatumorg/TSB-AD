"""Lightweight Anlu interface and temporal aggregation regression tests."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from TSB_AD.models.Anlu import Anlu, _windows
from TSB_AD.model_wrapper import Semisupervise_AD_Pool
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict


class TestAnlu(unittest.TestCase):
    def test_registration_and_fixed_protocol(self):
        self.assertIn('Anlu', Semisupervise_AD_Pool)
        self.assertEqual(Optimal_Uni_algo_HP_dict['Anlu'],
                         dict(window_size=10000, stride=1500, batch_size=8, reference_length=10000))

    def test_unseen_tail_does_not_reweight_existing_overlap(self):
        self.assertEqual(_windows(13, 8, 3), [(0, 8, 0, 8), (3, 11, 3, 11), (5, 13, 11, 13)])
        detector = Anlu(window_size=8, stride=3, batch_size=2, device='cpu').fit(np.arange(4))
        def constant_by_start(model, series, reference, windows):
            return torch.stack([torch.full((end-start,), float(start)) for start,end,_,_ in windows])
        with patch.object(detector, '_model', return_value=None), patch.object(detector, '_forward_batch', side_effect=constant_by_start):
            actual = detector.decision_function(np.arange(13))
        np.testing.assert_array_equal(actual, [0,0,0,1.5,1.5,1.5,1.5,1.5,3,3,3,5,5])
        self.assertIs(detector.decision_scores_, actual)

    def test_reference_copy_and_query_shared_normalization(self):
        original = np.array([10., 12., 14.])
        detector = Anlu(window_size=4, stride=4, reference_length=2, device='cpu').fit(original, [1,0,1])
        original[:] = -100
        seen = {}
        def capture(model, series, reference, windows):
            seen['reference'] = reference.clone();seen['query'] = series.clone()
            return series[:, 0][None]
        with patch.object(detector, '_model', return_value=None), patch.object(detector, '_forward_batch', side_effect=capture):
            detector.decision_function([2.,4.,6.,8.])
        np.testing.assert_allclose(seen['query'].numpy().ravel(), (np.array([2,4,6,8])-5)/np.sqrt(5), rtol=1e-6)
        np.testing.assert_allclose(seen['reference'].numpy().ravel(), (np.array([10,12])-5)/np.sqrt(5), rtol=1e-6)

    def test_invalid_inputs_fail_before_loading_weights(self):
        for kwargs in [dict(stride=0), dict(window_size=0), dict(batch_size=True), dict(reference_length=-1),
                       dict(batch_size=1.5), dict(window_size=10,stride=11)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):Anlu(**kwargs)
        detector=Anlu(device='cpu')
        with self.assertRaises(RuntimeError):detector.decision_function([1,2,3])
        for value in [[], [[1,2],[3,4]], [float('nan')], [float('inf')]]:
            with self.subTest(value=value), self.assertRaises(ValueError):detector.fit(value)
        detector.fit([1,2,3])
        with self.assertRaises(ValueError):detector.decision_function([])


if __name__ == '__main__':unittest.main()
