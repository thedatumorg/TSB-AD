# ChakraTSAD anomaly detector

`Run_ChakraTSAD.py` scores TSB-AD series with ChakraTSAD, a hosted anomaly-detection model from YHat Labs served through the
YHat Labs API (`api.yhatlabs.com`). The script sends each series and its train length; the API returns one anomaly score per
time step. The script does no preprocessing and sets no hyper-parameters.

## Results

VUS-PR on the TSB-AD eval set (350 uni, 180 mv, 530 total):

| detector | uni | mv | all |
|---|---|---|---|
| `ChakraTSAD` | 0.715 | 0.549 | 0.659 |

Per-series VUS-PR is in `leaderboard_results/Uni_ChakraTSAD.csv` and `leaderboard_results/Multi_ChakraTSAD.csv`.

## Run

From the repository root:

```
pip install "pyarrow==20.0.0"
export YHAT_API_KEY=...
python benchmark_exp/Run_ChakraTSAD.py --set U --dataset_dir Datasets/TSB-AD-U/ --file_list Datasets/File_List/TSB-AD-U-Eva.csv --save
python benchmark_exp/Run_ChakraTSAD.py --set M --dataset_dir Datasets/TSB-AD-M/ --file_list Datasets/File_List/TSB-AD-M-Eva.csv --save
```

The protocol is the same as `Run_Detector_U.py` and `Run_Detector_M.py`: the train length is the `tr_` value in the file name,
the full series is scored, and the scores are evaluated on the full labels with TSB-AD's `get_metrics`. The output is one
`.npy` of scores per series and `Uni_ChakraTSAD.csv` / `Multi_ChakraTSAD.csv` with the standard metric columns, under
`eval/score/ChakraTSAD/`. The scoring step is `run_ChakraTSAD_Semisupervised(data_train, data_test)`, TSB-AD's
semi-supervised runner form, so it can also be called from another harness.

Series with more than 500,000 values are uploaded once as a parquet file, hence `pyarrow` (newer releases need numpy 2, which
TSB-AD does not use). Failed requests are retried; a rerun reuses the scores already saved and only calls the API for the
series still missing. The maintainers can get an API key for the evaluation from the authors.
