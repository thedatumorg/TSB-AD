import math

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .utils.slidingWindows import find_length_rank

Unsupervise_AD_Pool = ['FFT', 'SR', 'NORMA', 'Series2Graph', 'Sub_IForest', 'IForest', 'LOF', 'Sub_LOF', 'POLY', 'MatrixProfile', 'Sub_PCA', 'PCA', 'HBOS',
                        'Sub_HBOS', 'KNN', 'Sub_KNN','KMeansAD', 'KMeansAD_U', 'KShapeAD', 'COPOD', 'CBLOF', 'COF', 'EIF', 'RobustPCA', 'MMPAD', 'Lag_Llama', 'TimesFM', 'Chronos', 'MOMENT_ZS', 'TSPulse_ZS', 'Time_RCD', 'HSF_U', 'HSF_Causal']
Semisupervise_AD_Pool = ['Left_STAMPi', 'SAND', 'MCD', 'Sub_MCD', 'OCSVM', 'Sub_OCSVM', 'AutoEncoder', 'CNN', 'LSTMAD', 'TranAD', 'USAD', 'OmniAnomaly', 'PatchTST',
                        'AnomalyTransformer', 'TimesNet', 'FITS', 'Donut', 'OFA', 'MOMENT_FT', 'M2N2', 'TSPulse_FT', 'xLSTMAD', 'CHARM', 'StreamVAE', 'HSF', 'PaAno_PAI', 'SHADE', 'TimeRCD_MAFT', 'AxonAD']

def run_Unsupervise_AD(model_name, data, **kwargs):
    try:
        function_name = f'run_{model_name}'
        function_to_call = globals()[function_name]
        results = function_to_call(data, **kwargs)
        return results
    except KeyError:
        error_message = f"Model function '{function_name}' is not defined."
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"An error occurred while running the model '{function_name}': {str(e)}"
        print(error_message)
        return error_message


def run_Semisupervise_AD(model_name, data_train, data_test, **kwargs):
    try:
        function_name = f'run_{model_name}'
        function_to_call = globals()[function_name]
        results = function_to_call(data_train, data_test, **kwargs)
        return results
    except KeyError:
        error_message = f"Model function '{function_name}' is not defined."
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"An error occurred while running the model '{function_name}': {str(e)}"
        print(error_message)
        return error_message

def run_FFT(data, ifft_parameters=5, local_neighbor_window=21, local_outlier_threshold=0.6, max_region_size=50, max_sign_change_distance=10):
    from .models.FFT import FFT
    clf = FFT(ifft_parameters=ifft_parameters, local_neighbor_window=local_neighbor_window, local_outlier_threshold=local_outlier_threshold, max_region_size=max_region_size, max_sign_change_distance=max_sign_change_distance)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Sub_IForest(data, periodicity=1, n_estimators=100, max_features=1, n_jobs=1):
    from .models.IForest import IForest
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = IForest(slidingWindow=slidingWindow, n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_IForest(data, slidingWindow=100, n_estimators=100, max_features=1, n_jobs=1):
    from .models.IForest import IForest
    clf = IForest(slidingWindow=slidingWindow, n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Sub_LOF(data, periodicity=1, n_neighbors=30, metric='minkowski', n_jobs=1):
    from .models.LOF import LOF
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = LOF(slidingWindow=slidingWindow, n_neighbors=n_neighbors, metric=metric, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_LOF(data, slidingWindow=1, n_neighbors=30, metric='minkowski', n_jobs=1):
    from .models.LOF import LOF
    clf = LOF(slidingWindow=slidingWindow, n_neighbors=n_neighbors, metric=metric, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_POLY(data, periodicity=1, power=3, n_jobs=1):
    from .models.POLY import POLY
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = POLY(power=power, window = slidingWindow)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_MatrixProfile(data, periodicity=1, n_jobs=1):
    from .models.MatrixProfile import MatrixProfile
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = MatrixProfile(window=slidingWindow)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Left_STAMPi(data_train, data):
    from .models.Left_STAMPi import Left_STAMPi
    clf = Left_STAMPi(n_init_train=len(data_train), window_size=100)
    clf.fit(data)
    score = clf.decision_function(data)
    return score.ravel()

def run_SAND(data_train, data_test, periodicity=1):
    from .models.SAND import SAND
    slidingWindow = find_length_rank(data_test, rank=periodicity)
    clf = SAND(pattern_length=slidingWindow, subsequence_length=4*(slidingWindow))
    clf.fit(data_test.squeeze(), online=True, overlaping_rate=int(1.5*slidingWindow), init_length=len(data_train), alpha=0.5, batch_size=max(5*(slidingWindow), int(0.1*len(data_test))))
    score = clf.decision_scores_
    return score.ravel()

def run_KShapeAD(data, periodicity=1):
    from .models.SAND import SAND
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = SAND(pattern_length=slidingWindow, subsequence_length=4*(slidingWindow))
    clf.fit(data.squeeze(), overlaping_rate=int(1.5*slidingWindow))
    score = clf.decision_scores_
    return score.ravel()

def run_Series2Graph(data, periodicity=1):
    from .models.Series2Graph import Series2Graph
    slidingWindow = find_length_rank(data, rank=periodicity)

    data = data.squeeze()
    s2g = Series2Graph(pattern_length=slidingWindow)
    s2g.fit(data)
    query_length = 2*slidingWindow
    s2g.score(query_length=query_length,dataset=data)

    score = s2g.decision_scores_
    score = np.array([score[0]]*math.ceil(query_length//2) + list(score) + [score[-1]]*(query_length//2))
    return score.ravel()

def run_Sub_PCA(data, periodicity=1, n_components=None, n_jobs=1):
    from .models.PCA import PCA
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = PCA(slidingWindow = slidingWindow, n_components=n_components)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_PCA(data, slidingWindow=100, n_components=None, n_jobs=1):
    from .models.PCA import PCA
    clf = PCA(slidingWindow = slidingWindow, n_components=n_components)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_NORMA(data, periodicity=1, clustering='hierarchical', n_jobs=1):
    from .models.NormA import NORMA
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = NORMA(pattern_length=slidingWindow, nm_size=3*slidingWindow, clustering=clustering)
    clf.fit(data)
    score = clf.decision_scores_
    score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
    if len(score) > len(data):
        start = len(score) - len(data)
        score = score[start:]
    return score.ravel()

def run_Sub_HBOS(data, periodicity=1, n_bins=10, tol=0.5, n_jobs=1):
    from .models.HBOS import HBOS
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = HBOS(slidingWindow=slidingWindow, n_bins=n_bins, tol=tol)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_HBOS(data, slidingWindow=1, n_bins=10, tol=0.5, n_jobs=1):
    from .models.HBOS import HBOS
    clf = HBOS(slidingWindow=slidingWindow, n_bins=n_bins, tol=tol)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Sub_OCSVM(data_train, data_test, kernel='rbf', nu=0.5, periodicity=1, n_jobs=1):
    from .models.OCSVM import OCSVM
    slidingWindow = find_length_rank(data_test, rank=periodicity)
    clf = OCSVM(slidingWindow=slidingWindow, kernel=kernel, nu=nu)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_OCSVM(data_train, data_test, kernel='rbf', nu=0.5, slidingWindow=1, n_jobs=1):
    from .models.OCSVM import OCSVM
    clf = OCSVM(slidingWindow=slidingWindow, kernel=kernel, nu=nu)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_Sub_MCD(data_train, data_test, support_fraction=None, periodicity=1, n_jobs=1):
    from .models.MCD import MCD
    slidingWindow = find_length_rank(data_test, rank=periodicity)
    clf = MCD(slidingWindow=slidingWindow, support_fraction=support_fraction)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_MCD(data_train, data_test, support_fraction=None, slidingWindow=1, n_jobs=1):
    from .models.MCD import MCD
    clf = MCD(slidingWindow=slidingWindow, support_fraction=support_fraction)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_Sub_KNN(data, n_neighbors=10, method='largest', periodicity=1, n_jobs=1):
    from .models.KNN import KNN
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = KNN(slidingWindow=slidingWindow, n_neighbors=n_neighbors,method=method, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_KNN(data, slidingWindow=1, n_neighbors=10, method='largest', n_jobs=1):
    from .models.KNN import KNN
    clf = KNN(slidingWindow=slidingWindow, n_neighbors=n_neighbors, method=method, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_KMeansAD(data, n_clusters=20, window_size=20, n_jobs=1):
    from .models.KMeansAD import KMeansAD
    clf = KMeansAD(k=n_clusters, window_size=window_size, stride=1, n_jobs=n_jobs)
    score = clf.fit_predict(data)
    return score.ravel()

def run_KMeansAD_U(data, n_clusters=20, periodicity=1,n_jobs=1):
    from .models.KMeansAD import KMeansAD
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = KMeansAD(k=n_clusters, window_size=slidingWindow, stride=1, n_jobs=n_jobs)
    score = clf.fit_predict(data)
    return score.ravel()

def run_COPOD(data, n_jobs=1):
    from .models.COPOD import COPOD
    clf = COPOD(n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_CBLOF(data, n_clusters=8, alpha=0.9, n_jobs=1):
    from .models.CBLOF import CBLOF
    clf = CBLOF(n_clusters=n_clusters, alpha=alpha, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_COF(data, n_neighbors=30):
    from .models.COF import COF
    clf = COF(n_neighbors=n_neighbors)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_EIF(data, n_trees=100):
    from .models.EIF import EIF
    clf = EIF(n_trees=n_trees)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_RobustPCA(data, max_iter=1000):
    from .models.RobustPCA import RobustPCA
    clf = RobustPCA(max_iter=max_iter)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_SR(data, periodicity=1):
    from .models.SR import SR
    slidingWindow = find_length_rank(data, rank=periodicity)
    return SR(data, window_size=slidingWindow)

def run_AutoEncoder(data_train, data_test, window_size=100, hidden_neurons=[64, 32], n_jobs=1):
    from .models.AE import AutoEncoder
    clf = AutoEncoder(slidingWindow=window_size, hidden_neurons=hidden_neurons, batch_size=128, epochs=50)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_CNN(data_train, data_test, window_size=100, num_channel=[32, 32, 40], lr=0.0008, n_jobs=1):
    from .models.CNN import CNN
    clf = CNN(window_size=window_size, num_channel=num_channel, feats=data_test.shape[1], lr=lr, batch_size=128)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_LSTMAD(data_train, data_test, window_size=100, lr=0.0008):
    from .models.LSTMAD import LSTMAD
    clf = LSTMAD(window_size=window_size, pred_len=1, lr=lr, feats=data_test.shape[1], batch_size=128)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_TranAD(data_train, data_test, win_size=10, lr=1e-3):
    from .models.TranAD import TranAD
    clf = TranAD(win_size=win_size, feats=data_test.shape[1], lr=lr)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_AnomalyTransformer(data_train, data_test, win_size=100, lr=1e-4, batch_size=128):
    from .models.AnomalyTransformer import AnomalyTransformer
    clf = AnomalyTransformer(win_size=win_size, input_c=data_test.shape[1], lr=lr, batch_size=batch_size)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_PatchTST(data_train, data_test, win_size=100, lr=1e-4, batch_size=128):
    from .models.PatchTST import PatchTST
    clf = PatchTST(win_size=win_size, input_c=data_test.shape[1], lr=lr, batch_size=batch_size)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_OmniAnomaly(data_train, data_test, win_size=100, lr=0.002):
    from .models.OmniAnomaly import OmniAnomaly
    clf = OmniAnomaly(win_size=win_size, feats=data_test.shape[1], lr=lr)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_USAD(data_train, data_test, win_size=5, lr=1e-4):
    from .models.USAD import USAD
    clf = USAD(win_size=win_size, feats=data_test.shape[1], lr=lr)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_Donut(data_train, data_test, win_size=120, lr=1e-4, batch_size=128):
    from .models.Donut import Donut
    clf = Donut(win_size=win_size, input_c=data_test.shape[1], lr=lr, batch_size=batch_size)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_TimesNet(data_train, data_test, win_size=96, lr=1e-4):
    from .models.TimesNet import TimesNet
    clf = TimesNet(win_size=win_size, enc_in=data_test.shape[1], lr=lr, epochs=50)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_FITS(data_train, data_test, win_size=100, lr=1e-3):
    from .models.FITS import FITS
    clf = FITS(win_size=win_size, input_c=data_test.shape[1], lr=lr, batch_size=128)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_OFA(data_train, data_test, win_size=100, batch_size = 64):
    from .models.OFA import OFA
    clf = OFA(win_size=win_size, enc_in=data_test.shape[1], epochs=10, batch_size=batch_size)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_Lag_Llama(data, win_size=96, batch_size=64):
    from .models.Lag_Llama import Lag_Llama
    clf = Lag_Llama(win_size=win_size, input_c=data.shape[1], batch_size=batch_size)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Chronos(data, win_size=50, batch_size=64):
    from .models.Chronos import Chronos
    clf = Chronos(win_size=win_size, prediction_length=1, input_c=data.shape[1], model_size='base', batch_size=batch_size)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_TimesFM(data, win_size=96):
    from .models.TimesFM import TimesFM
    clf = TimesFM(win_size=win_size)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_MOMENT_ZS(data, win_size=256):
    from .models.MOMENT import MOMENT
    clf = MOMENT(win_size=win_size, input_c=data.shape[1])

    # Zero shot
    clf.zero_shot(data)
    score = clf.decision_scores_
    return score.ravel()


def run_Time_RCD(data,
                 win_size=15000,
                 batch_size=64,
                 device=None,
                 checkpoint=None,
                 model_id="thu-sail-lab/Time-RCD",
                 cache_dir=None):
    from .models.Time_RCD import Time_RCD
    clf = Time_RCD(
        win_size=win_size,
        input_c=data.shape[1],
        batch_size=batch_size,
        device=device,
        checkpoint=checkpoint,
        model_id=model_id,
        cache_dir=cache_dir,
    )
    clf.zero_shot(data)
    score = clf.decision_scores_
    return score.ravel()

def run_MOMENT_FT(data_train, data_test, win_size=256):
    from .models.MOMENT import MOMENT
    clf = MOMENT(win_size=win_size, input_c=data_test.shape[1])

    # Finetune
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_M2N2(
        data_train, data_test, win_size=12, stride=12,
        batch_size=64, epochs=100, latent_dim=16,
        lr=1e-3, ttlr=1e-3, normalization='Detrend',
        gamma=0.99, th=0.9, valid_size=0.2, infer_mode='online'
    ):
    from .models.M2N2 import M2N2
    clf = M2N2(
        win_size=win_size, stride=stride,
        num_channels=data_test.shape[1],
        batch_size=batch_size, epochs=epochs,
        latent_dim=latent_dim,
        lr=lr, ttlr=ttlr,
        normalization=normalization,
        gamma=gamma, th=th, valid_size=valid_size,
        infer_mode=infer_mode
    )
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()


def run_TSPulse_ZS(data,
                   model="ibm-granite/granite-timeseries-tspulse-r1",
                   win_size=96,
                   batch_size=256,
                   smoothing_window=8,
                   prediction_mode="time",
                   **kwargs,
                   ):
    from TSB_AD.models.TSPulse import TSPulsePipeline
    num_input_channels = data.shape[1]
    clf = TSPulsePipeline(
            model_path=model,
            num_input_channels=num_input_channels,
            batch_size=batch_size,
            aggr_win_size=win_size,
            smoothing_window=smoothing_window,
            prediction_mode=prediction_mode,
        )
    score = clf.decision_function(data)
    return score.ravel()

def run_TSPulse_FT(data_train,
                   data_test,
                   model="ibm-granite/granite-timeseries-tspulse-r1",
                   win_size=96,
                   batch_size=256,
                   smoothing_window=8,
                   prediction_mode="time",
                   decoder_mode="common_channels",
                   num_epochs=20,
                   freeze_backbone=False,
                   validation_fraction=0.2,
                   lr=1e-4,
                   **kwargs):
    from TSB_AD.models.TSPulse import TSPulsePipeline
    num_input_channels = data_train.shape[1]

    clf = TSPulsePipeline(
            model_path=model,
            num_input_channels=num_input_channels,
            batch_size=batch_size,
            aggr_win_size=win_size,
            smoothing_window=smoothing_window,
            prediction_mode=prediction_mode,
            finetune_decoder_mode=decoder_mode,
            finetune_validation=validation_fraction,
            finetune_freeze_backbone=freeze_backbone,
            finetune_epochs=num_epochs,
            finetune_lr=lr,
        )
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()


def run_xLSTMAD(data_train, data_test, window_size=100, lr=0.005, batch_size=32, embedding_dim=40):
    from .models.xLSTMAD import xLSTMAD, xLSTMADModule
    model = xLSTMADModule(embedding_dim=embedding_dim, window_size=window_size, lr=lr, features_no=data_test.shape[1])
    clf = xLSTMAD(model=model, window_size=window_size, batch_size=batch_size)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()


def run_MMPAD(data, periodicity=1, n_dim=None, n_neighbor=1,
              sorting_place='pre', mode='discord', post_processing=2, 
              n_job=None, backend=None):
    from .models.MMPAD import MMPAD
    clf = MMPAD(periodicity=periodicity, n_dim=n_dim, n_neighbor=n_neighbor,
                sorting_place=sorting_place, mode=mode, post_processing=post_processing, 
                n_job=n_job, backend=backend)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_StreamVAE(data_train, data_test, win_size=100, latent_dim=64, batch_size=128, epochs=50,
                  patience=10, lr=1e-3, validation_size=0.2, target_kl=100.0, event_l1_weight=1e-3):
    from .models.StreamVAE import StreamVAE
    eps = 1e-8
    mu = data_train.mean(axis=0, keepdims=True)
    sd = data_train.std(axis=0, keepdims=True)
    sd = np.where(sd == 0, eps, sd)
    data_train_n = (data_train - mu) / sd
    data_test_n = (data_test - mu) / sd
    clf = StreamVAE(
        win_size=win_size, feats=data_test.shape[1], latent_dim=latent_dim,
        batch_size=batch_size, epochs=epochs, patience=patience, lr=lr,
        validation_size=validation_size, target_kl=target_kl, event_l1_weight=event_l1_weight,
    )
    clf.fit(data_train_n)
    score = clf.decision_function(data_test_n)
    return score.ravel()


def run_SHADE(data_train, data_test, **kwargs):
    from .models.SHADE_AD import run_SHADE_AD_Semisupervised
    return run_SHADE_AD_Semisupervised(data_train, data_test, HP=kwargs).ravel()

def run_PaAno_PAI(data_train, data_test, **kwargs):
    from .models.PaAno_PAI import run_PaAno_PAI_Semisupervised
    return run_PaAno_PAI_Semisupervised(data_train, data_test, kwargs).ravel()

def run_HSF(data_train, data_test, **kwargs):
    from .models.HSF_AD import run_HSF_AD_Semisupervised
    return run_HSF_AD_Semisupervised(data_train, data_test, HP=kwargs).ravel()

def run_HSF_U(data, **kwargs):
    from .models.HSF_AD import run_HSF_AD_Unsupervised
    return run_HSF_AD_Unsupervised(data, HP=kwargs).ravel()

def run_HSF_Causal(data, **kwargs):
    from .models.HSF_AD import run_HSF_AD_Causal
    return run_HSF_AD_Causal(data, HP=kwargs).ravel()


def run_CHARM(
    data_train,
    data_test,
    window_size=128,
    k=3,
    pointwise_agg="mean",
    stride=1,
    train_stride=1,
    min_window=64,
):
    from .models.CHARM import CHARM_AD
    """Semisupervised runner — matches TSB-AD's run_Semisupervise_AD dispatcher."""
    clf = CHARM_AD(
        HP={},
        window_size=window_size,
        stride=stride,
        k=k,
        pointwise_agg=pointwise_agg,
        train_stride=train_stride,
        min_window=min_window,
    )
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = (
        MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    )
    return score

def run_TimeRCD_MAFT_FT(
    data_train,
    data_test,
    win_size=64,
    weight=0.5,
    lr_adapter=1e-3,
    epochs_adapter=5,
    adapter_mode="train",
    device="cpu",
    adapter_checkpoint_dir="checkpoints/MAFT",
    fusion="add",
    timercd_win_size=15000,
    timercd_checkpoint="checkpoints/time-rcd/pretrain_checkpoint_best_uni.pth",
    filename=None,
    **kwargs,
):
    from .models.TimeRCD_MAFT import run_maft

    if filename is None:
        filename = f"official_tr_{len(data_train)}_1st_0.csv"

    return run_maft(
        filename=filename,
        data=data_test,
        win_size=win_size,
        weight=weight,
        lr_adapter=lr_adapter,
        epochs_adapter=epochs_adapter,
        mode=adapter_mode,
        device=device,
        adapter_checkpoint_dir=adapter_checkpoint_dir,
        fusion=fusion,
        timercd_win_size=timercd_win_size,
        timercd_checkpoint=timercd_checkpoint,
    ).ravel()


def run_TimeRCD_MAFT(*args, **kwargs):
    return run_TimeRCD_MAFT_FT(*args, **kwargs)

def run_AxonAD(
        data_train, data_test, win_size=100, d_model=64, num_heads=4,
        batch_size=128, epochs=50, lr=0.001, validation_size=0.2,
        kl_tail_k=5, forecast_steps=1
    ):
    from .models.AxonAD import TCNTransformer
    clf = TCNTransformer(
        win_size=win_size,
        feats=data_test.shape[1],
        d_model=d_model,
        num_heads=num_heads,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        validation_size=validation_size,
        kl_tail_k=kl_tail_k,
        forecast_steps=forecast_steps
    )
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()
