# -*- coding: utf-8 -*-
# Author: Ken Wu <wooqianghao@gmail.com>
# License: Apache-2.0 License
#
# Tremor: a CPU-only, label-free per-series chooser over MMPAD (n_neighbor=5), Sub-PCA
# (periodicity=1) and an O(n) point-anomaly score (robust z of the first differences,
# max-filtered over the ACF period). On each series it outputs the rank-normalised score of
# one of the three; the choice reads only the detectors' own score arrays (excess kurtosis,
# z-max, top-1% overlap, run count). No training, no pretraining, no GPU.
# On multivariate series the same chooser runs over MMPAD-M and a per-channel point score.
# Code, tests and worklog: https://github.com/KenWuqianghao/tremor
#
# Install the detector (numpy/scipy only; the components come from this repository):
#     pip install git+https://github.com/KenWuqianghao/tremor
# Tremor_HP below is registered verbatim in TSB_AD/HP_list.py, so
#     python Run_Detector_U.py --AD_Name Tremor
# runs exactly this configuration.

import pandas as pd
import numpy as np
import argparse, time, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from tremor import Tremor, run_Tremor_Unsupervised   # pip install git+https://github.com/KenWuqianghao/tremor

# Hyper-parameters (chosen on the TSB-AD Tuning sets only)
Tremor_HP = {'stat': 'kurt', 'power': 'hard', 'prior': (1.0, 0.0, 0.0), 'zfloor': 3.0, 'tau': 0.5, 'flat': 'zmax', 'zprior': 0.0,
             'point_window': 'period', 'flat_only': (2,), 'override': ('pz', 'agree', 'spikes'), 'pzmin': 7.0, 'kfrac': 0.01,
             'athr': 0.5, 'sthr': 10.0, 'cthr': 5, 'fuse_multivariate': False, 'point_m_agg': 'top3z', 'flat_only_m': (1,),
             'override_m': ('pz', 'agree', 'spikes'), 'gate_m': 2, 'cthr_m': 10, 'n_job': 1}

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running Tremor')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='../Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='Tremor')
    args = parser.parse_args()

    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    print('data: ', data.shape)
    print('label: ', label.shape)

    slidingWindow = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)

    start_time = time.time()

    output = run_Tremor_Unsupervised(data, **Tremor_HP)

    end_time = time.time()
    run_time = end_time - start_time

    pred = output > (np.mean(output)+3*np.std(output))
    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)
    print('Run time: %.2fs' % run_time)
