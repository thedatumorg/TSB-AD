
Multi_algo_HP_dict = {
    'IForest': {
        'n_estimators': [25, 50, 100, 150, 200],
        'max_features': [0.2, 0.4, 0.6, 0.8, 1.0]
    },
    'LOF': {
        'n_neighbors': [10, 20, 30, 40, 50],
        'metric': ['minkowski', 'manhattan', 'euclidean']
    },    
    'PCA': {
        'n_components': [0.25, 0.5, 0.75, None]
    },        
    'HBOS': {
        'n_bins': [5, 10, 20, 30, 40],
        'tol': [0.1, 0.3, 0.5, 0.7]
    },
    'OCSVM': {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'nu': [0.1, 0.3, 0.5, 0.7]
    },        
    'MCD': {
        'support_fraction': [0.2, 0.4, 0.6, 0.8, None]
    },
    'KNN': {
        'n_neighbors': [10, 20, 30, 40, 50],
        'method': ['largest', 'mean', 'median']
    },        
    'KMeansAD': {
        'n_clusters': [10, 20, 30, 40],
        'window_size': [10, 20, 30, 40]
    },
    'COPOD': {
        'HP': [None]
    },    
    'CBLOF': {
        'n_clusters': [4, 8, 16, 32],
        'alpha': [0.6, 0.7, 0.8, 0.9]
    },
    'EIF': {
        'n_trees': [25, 50, 100, 200]
    },   
    'RobustPCA': {
        'max_iter': [500, 1000, 1500]
    },
    'AutoEncoder': {
        'hidden_neurons': [[64, 32], [32, 16], [128, 64]]
    },
    'CNN': {
        'window_size': [50, 100, 150],
        'num_channel': [[32, 32, 40], [16, 32, 64]]
    },
    'LSTMAD': {
        'window_size': [50, 100, 150],
        'lr': [0.0004, 0.0008]
    },  
    'TranAD': {
        'win_size': [5, 10, 50],
        'lr': [1e-3, 1e-4]
    },  
    'AnomalyTransformer': {
        'win_size': [50, 100, 150],
        'lr': [1e-3, 1e-4, 1e-5]
    },  
    'OmniAnomaly': {
        'win_size': [5, 50, 100],
        'lr': [0.002, 0.0002]
    },
    'USAD': {
        'win_size': [5, 50, 100],
        'lr': [1e-3, 1e-4, 1e-5]
    },  
    'Donut': {
        'win_size': [60, 90, 120],
        'lr': [1e-3, 1e-4, 1e-5]
    },  
    'TimesNet': {
        'win_size': [32, 96, 192],
        'lr': [1e-3, 1e-4, 1e-5]
    },
    'FITS': {
        'win_size': [100, 200],
        'lr': [1e-3, 1e-4, 1e-5]
    },    
    'OFA': {
        'win_size': [50, 100, 150]
    },
    'Time_RCD': {
        'win_size': [15000]
    },
    'TimeRCD_MAFT': {
        'win_size': [64, 512],
        'fusion': ['mul', 'add'],
        'weight': [0.01, 0.1, 0.2, 0.5, 0.6, 0.99],
        'lr_adapter': [1e-3, 5e-4],
    },
    'xLSTMAD': {
        'window_size': [50, 100, 150],
        'lr': [0.001],
        'embedding_dim': [20, 40],
    },
    'MMPAD': {
        'n_dim': [1, 0.1, 0.3, 0.5, 0.7],
        'n_neighbor': [1, 5, 10, 15],
    },
    'PaAno_PAI': {
        'patch_size': [32, 64, 128],
    },
}


Optimal_Multi_algo_HP_dict = {
    'IForest': {'n_estimators': 25, 'max_features': 0.8},
    'LOF': {'n_neighbors': 50, 'metric': 'euclidean'},    
    'PCA': {'n_components': 0.25},        
    'HBOS': {'n_bins': 30, 'tol': 0.5},
    'OCSVM': {'kernel': 'rbf', 'nu': 0.1},        
    'MCD': {'support_fraction': 0.8},
    'KNN': {'n_neighbors': 50, 'method': 'mean'},        
    'KMeansAD': {'n_clusters': 10, 'window_size': 40},
    'KShapeAD': {'n_clusters': 20, 'window_size': 40},
    'COPOD': {'n_jobs':1},    
    'CBLOF': {'n_clusters': 4, 'alpha': 0.6},
    'EIF': {'n_trees': 50},   
    'RobustPCA': {'max_iter': 1000},
    'AutoEncoder': {'hidden_neurons': [128, 64]},
    'CNN': {'window_size': 50, 'num_channel': [32, 32, 40]},
    'LSTMAD': {'window_size': 150, 'lr': 0.0008},  
    'TranAD': {'win_size': 10, 'lr': 0.001},  
    'AnomalyTransformer': {'win_size': 50, 'lr': 0.001},  
    'PatchTST': {},
    'OmniAnomaly': {'win_size': 100, 'lr': 0.002},
    'USAD': {'win_size': 100, 'lr': 0.001},  
    'Donut': {'win_size': 60, 'lr': 0.001},  
    'TimesNet': {'win_size': 96, 'lr': 0.0001},
    'FITS': {'win_size': 100, 'lr': 0.001},
    'OFA': {'win_size': 50},
    'Time_RCD': {'win_size': 15000, 'batch_size': 64},
    'xLSTMAD': {'window_size': 50, 'lr': 0.001, 'embedding_dim': 40},
    'MMPAD': {'n_dim': 0.7, 'n_neighbor': 15},
    'CHARM': {"window_size": 128, "k": 3, "pointwise_agg": "mean", "stride": 1, "train_stride": 1, "min_window": 64},
    'StreamVAE': {'win_size': 100, 'latent_dim': 64, 'batch_size': 128, 'epochs': 50, 'patience': 10, 'lr': 1e-3, 'validation_size': 0.2, 'target_kl': 100.0, 'event_l1_weight': 1e-3},
    'PaAno_PAI': {'score_recipe': 'a28', 'train_zscore': False, 'patch_size': 64, 'stride': 1, 'num_iters': 200, 'batch_size': 512, 'embed_batch_size': 512, 'distance_batch_size': 512, 'lr': 1e-4, 'seed': 2021, 'calib_seed': 42, 'calib_frac': 0.20, 'bank_seed': 42, 'pretext_step': 64, 'num_rand_patches': 5, 'temperature': 1.0, 'alpha': 10.0, 'ema_weight': 0.5, 'triplet_margin': 0.5, 'positive_radius': 2, 'embed_dim': 64, 'projection_dim': 256, 'use_revin': True, 'use_teacher': True, 'bank_k': 1024, 'top_k': 3, 'mag_weight_unit_eu': 0.667, 't2_weight_unit_eu': 0.2, 't2_window': 32, 'device': 'auto'},
    'SHADE': {'base_url': 'https://shade.avara-ai.com', 'split': 'eval', 'prefix_precision_guard': True, 'prefix_precision_guard_min_length': 50000, 'prefix_precision_guard_anchors': 1},
    'AxonAD': {'win_size': 100, 'd_model': 128, 'num_heads': 8, 'lr': 0.0005, 'kl_tail_k': 10, 'forecast_steps': 1}
}


Uni_algo_HP_dict = {
    'Sub_IForest': {
        'periodicity': [1, 2, 3],
        'n_estimators': [25, 50, 100, 150, 200]
    },
    'IForest': {
        'n_estimators': [25, 50, 100, 150, 200]
    },
    'Sub_LOF': {
        'periodicity': [1, 2, 3],
        'n_neighbors': [10, 20, 30, 40, 50]
    }, 
    'LOF': {
        'n_neighbors': [10, 20, 30, 40, 50]
    }, 
    'POLY': {
        'periodicity': [1, 2, 3],
        'power': [1, 2, 3, 4]
    },
    'MatrixProfile': {
        'periodicity': [1, 2, 3]
    },
    'NORMA': {
        'periodicity': [1, 2, 3],
        'clustering': ['hierarchical', 'kshape']
    },
    'SAND': {
        'periodicity': [1, 2, 3]
    }, 
    'Series2Graph': {
        'periodicity': [1, 2, 3]
    },
    'Sub_PCA': {
        'periodicity': [1, 2, 3],
        'n_components': [0.25, 0.5, 0.75, None]
    },
    'Sub_HBOS': {
        'periodicity': [1, 2, 3],
        'n_bins': [5, 10, 20, 30, 40]
    },
    'Sub_OCSVM': {
        'periodicity': [1, 2, 3],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    },
    'Sub_MCD': {
        'periodicity': [1, 2, 3],
        'support_fraction': [0.2, 0.4, 0.6, 0.8, None]
    },
    'Sub_KNN': {
        'periodicity': [1, 2, 3],
        'n_neighbors': [10, 20, 30, 40, 50],
    },
    'KMeansAD_U': {
        'periodicity': [1, 2, 3],
        'n_clusters': [10, 20, 30, 40],
    },
    'KShapeAD': {
        'periodicity': [1, 2, 3]
    },
    'AutoEncoder': {
        'window_size': [50, 100, 150],
        'hidden_neurons': [[64, 32], [32, 16], [128, 64]]
    },
    'CNN': {
        'window_size': [50, 100, 150],
        'num_channel': [[32, 32, 40], [16, 32, 64]]
    },
    'LSTMAD': {
        'window_size': [50, 100, 150],
        'lr': [0.0004, 0.0008]
    },  
    'TranAD': {
        'win_size': [5, 10, 50],
        'lr': [1e-3, 1e-4]
    },
    'AnomalyTransformer': {
        'win_size': [50, 100, 150],
        'lr': [1e-3, 1e-4, 1e-5]
    },  
    'OmniAnomaly': {
        'win_size': [5, 50, 100],
        'lr': [0.002, 0.0002]
    },
    'USAD': {
        'win_size': [5, 50, 100],
        'lr': [1e-3, 1e-4, 1e-5]
    },  
    'Donut': {
        'win_size': [60, 90, 120],
        'lr': [1e-3, 1e-4, 1e-5]
    },  
    'TimesNet': {
        'win_size': [32, 96, 192],
        'lr': [1e-3, 1e-4, 1e-5]
    },
    'FITS': {
        'win_size': [100, 200],
        'lr': [1e-3, 1e-4, 1e-5]
    },
    'OFA': {
        'win_size': [50, 100, 150]
    },    
    'Lag_Llama': {
        'win_size': [32, 64, 96]
    },    
    'Chronos': {
        'win_size': [50, 100, 150]
    },
    'TimesFM': {
        'win_size': [32, 64, 96]
    },
    'MOMENT_ZS': {
        'win_size': [64, 128, 256]
    },
    'MOMENT_FT': {
        'win_size': [64, 128, 256]
    },
    'Time_RCD': {
        'win_size': [15000]
    },
    'xLSTMAD': {
        'window_size': [50, 100, 150],
        'lr': [0.0005, 0.001],
        'embedding_dim': [20, 40],
    },
    'MMPAD': {
        'n_neighbor': [1, 5, 10, 15],
    },
    'HSF': {
        'window': [64, 128, 256],
    },
    'HSF_U': {
        'window': [64, 128, 256],
    },
    'HSF_Causal': {
        'window': [64, 128, 256],
    },
    'PaAno_PAI': {
        'patch_size': [32, 64, 128],
    },
}

Optimal_Uni_algo_HP_dict = {
    'Sub_IForest': {'periodicity': 1, 'n_estimators': 150},
    'IForest': {'n_estimators': 200},
    'Sub_LOF': {'periodicity': 2, 'n_neighbors': 30},
    'LOF': {'n_neighbors': 50},
    'POLY': {'periodicity': 1, 'power': 4},
    'MatrixProfile': {'periodicity': 1},
    'NORMA': {'periodicity': 1, 'clustering': 'kshape'},
    'SAND': {'periodicity': 1},
    'Series2Graph': {'periodicity': 1},
    'SR': {'periodicity': 1},
    'Sub_PCA': {'periodicity': 1, 'n_components': None},        
    'Sub_HBOS': {'periodicity': 1, 'n_bins': 10},
    'Sub_OCSVM': {'periodicity': 2, 'kernel': 'rbf'},        
    'Sub_MCD': {'periodicity': 3, 'support_fraction': None},
    'Sub_KNN': {'periodicity': 2, 'n_neighbors': 50}, 
    'KMeansAD_U': {'periodicity': 2, 'n_clusters': 10},
    'KShapeAD': {'periodicity': 1},
    'FFT': {},
    'Left_STAMPi': {},
    'AutoEncoder': {'window_size': 100, 'hidden_neurons': [128, 64]},
    'CNN': {'window_size': 50, 'num_channel': [32, 32, 40]},
    'LSTMAD': {'window_size': 100, 'lr': 0.0008},  
    'TranAD': {'win_size': 10, 'lr': 0.0001},
    'AnomalyTransformer': {'win_size': 50, 'lr': 0.001},  
    'PatchTST': {},
    'OmniAnomaly': {'win_size': 5, 'lr': 0.002},
    'USAD': {'win_size': 100, 'lr': 0.001},
    'Donut': {'win_size': 60, 'lr': 0.0001},  
    'TimesNet': {'win_size': 32, 'lr': 0.0001},
    'FITS': {'win_size': 100, 'lr': 0.0001},
    'OFA': {'win_size': 50},
    'Lag_Llama': {'win_size': 96},
    'Chronos': {'win_size': 100},
    'TimesFM': {'win_size': 96},
    'MOMENT_ZS': {'win_size': 64},
    'MOMENT_FT': {'win_size': 64},
    'M2N2': {},
    'Time_RCD': {'win_size': 15000, 'batch_size': 64},
    'TimeRCD_MAFT': {'win_size': 512, 'weight': 0.2, 'fusion': 'mul', 'lr_adapter': 1e-3, 'epochs_adapter': 5},
    'TSPulse_ZS': {'win_size': 96, 
                   'prediction_mode': 'time'},
    'TSPulse_FT': {'win_size': 96, 
                   'prediction_mode': 'time',
                   'lr': 1e-4},
    'xLSTMAD': {'window_size': 50, 'lr': 0.001, 'embedding_dim': 40},
    'MMPAD': {'n_neighbor': 5},
    'CHARM': {"window_size": 128, "k": 3, "pointwise_agg": "mean", "stride": 1, "train_stride": 1, "min_window": 64},
    'StreamVAE': {'win_size': 100, 'latent_dim': 64, 'batch_size': 128, 'epochs': 50, 'patience': 10, 'lr': 1e-3, 'validation_size': 0.2, 'target_kl': 100.0, 'event_l1_weight': 1e-3},
    'HSF': {},
    'HSF_U': {},
    'HSF_Causal': {},
    'PaAno_PAI': {'score_recipe': 'a28', 'train_zscore': False, 'patch_size': 64, 'stride': 1, 'num_iters': 200, 'batch_size': 512, 'embed_batch_size': 512, 'distance_batch_size': 512, 'lr': 1e-4, 'seed': 2021, 'calib_seed': 42, 'calib_frac': 0.20, 'bank_seed': 42, 'pretext_step': 64, 'num_rand_patches': 5, 'temperature': 1.0, 'alpha': 10.0, 'ema_weight': 0.5, 'triplet_margin': 0.5, 'positive_radius': 2, 'embed_dim': 64, 'projection_dim': 256, 'use_revin': True, 'use_teacher': True, 'bank_k': 1024, 'top_k': 3, 'mag_weight_unit_eu': 0.667, 't2_weight_unit_eu': 0.2, 't2_window': 32, 'device': 'auto'},
    'SHADE': {'base_url': 'https://shade.avara-ai.com', 'split': 'eval', 'prefix_precision_guard': True, 'prefix_precision_guard_min_length': 50000, 'prefix_precision_guard_anchors': 1},
    'AxonAD': {'win_size': 100, 'd_model': 128, 'num_heads': 8, 'lr': 0.0005, 'kl_tail_k': 10, 'forecast_steps': 1},
}
